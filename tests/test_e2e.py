# File: tests/test_e2e.py
# Purpose: True Black-Box E2E API tests using testcontainers for real Qdrant and Redis infrastructure.

import asyncio
import uuid
from collections.abc import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import get_password_hash
from app.db.models import Document, Role, User
from app.dependencies import get_db_session, get_qdrant, get_redis
from app.main import app
from app.rate_limit import limiter

# Valid plain text content for testing ingestion without triggering heavy PDF hi_res OCR
TXT_CONTENT = b"EMPLOYMENT AGREEMENT\nEmployee Name: Alice Smith\nEmail: alice.smith@example.com\nBase salary is $150,000 per year."


@pytest_asyncio.fixture
async def e2e_client(
    db_session: AsyncSession, mock_redis: Redis, mock_qdrant: AsyncQdrantClient
) -> AsyncGenerator[AsyncClient]:
    """Test client overriding the DB session and injecting external API mocks into app state."""
    limiter.enabled = False
    # 1. Override FastAPI Depends() injections
    app.dependency_overrides[get_db_session] = lambda: db_session
    app.dependency_overrides[get_redis] = lambda: mock_redis
    app.dependency_overrides[get_qdrant] = lambda: mock_qdrant

    # 2. Inject directly into app.state (since ASGITransport skips lifespan startup events)
    app.state.redis = mock_redis
    app.state.qdrant = mock_qdrant

    # 3. Mock session_factory for background tasks / health checks that use app.state.session_factory
    class DummyContextManager:
        async def __aenter__(self):
            return db_session

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    app.state.session_factory = lambda: DummyContextManager()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as ac:
        yield ac

    app.dependency_overrides.clear()
    limiter.enabled = True


@pytest_asyncio.fixture
async def setup_e2e_users(db_session: AsyncSession) -> dict[str, str]:
    """Creates roles and users for E2E scenarios."""
    hr_role = Role(name="hr_manager", permissions=["view_unmasked", "upload_docs"])
    viewer_role = Role(name="viewer", permissions=["view_masked"])
    db_session.add_all([hr_role, viewer_role])

    hr_user = User(
        id=uuid.uuid4(),
        email="hr@example.com",
        hashed_password=get_password_hash("Pass123!"),
        role_name="hr_manager",
        department_id="hr_dept",
        is_active=True,
    )
    viewer_user = User(
        id=uuid.uuid4(),
        email="viewer@example.com",
        hashed_password=get_password_hash("Pass123!"),
        role_name="viewer",
        department_id="hr_dept",
        is_active=True,
    )
    other_hr_user = User(
        id=uuid.uuid4(),
        email="other_hr@example.com",
        hashed_password=get_password_hash("Pass123!"),
        role_name="hr_manager",
        department_id="finance_dept",
        is_active=True,
    )

    db_session.add_all([hr_user, viewer_user, other_hr_user])
    await db_session.commit()

    return {
        "hr_email": "hr@example.com",
        "viewer_email": "viewer@example.com",
        "other_hr_email": "other_hr@example.com",
        "password": "Pass123!",
    }


@pytest.mark.asyncio
@patch("app.graph.nodes.get_llm_response")
async def test_e2e_full_flow_hr_manager(
    mock_llm: MagicMock,
    e2e_client: AsyncClient,
    setup_e2e_users: dict[str, str],
) -> None:
    """E2E: Login -> Real Upload/Ingest -> Real Qdrant/Redis RAG Query -> Demasked Response"""
    # 1. Login as HR
    login_resp = await e2e_client.post(
        "/auth/login",
        data={
            "username": setup_e2e_users["hr_email"],
            "password": setup_e2e_users["password"],
        },
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Upload Document (Real ingestion pipeline will execute against Testcontainers)
    upload_resp = await e2e_client.post(
        "/ingest/",
        data={"department_id": "hr_dept", "access_level": 1},
        files={"file": ("contract.txt", TXT_CONTENT, "text/plain")},
        headers=headers,
    )
    assert upload_resp.status_code == 202
    doc_id = upload_resp.json()["document_id"]

    # Wait for the background ingestion task to complete
    for _ in range(50):
        status_resp = await e2e_client.get(f"/ingest/{doc_id}/status", headers=headers)
        if status_resp.json()["status"] == "done":
            break
        await asyncio.sleep(0.5)
    else:
        pytest.fail("Ingestion background task timed out in E2E test")

    # 3. Query the System
    # Mock LLM to return exactly what it would receive from the masked context
    mock_llm.return_value = (
        "The email is [EMAIL_ADDRESS_1] and salary is [FINANCIAL_1]."
    )

    query_resp = await e2e_client.post(
        "/query/",
        json={"question": "What is the email and salary?"},
        headers=headers,
    )

    assert query_resp.status_code == 200
    answer = query_resp.json()["answer"]

    # HR Manager has 'view_unmasked' -> Should see real data restored from real Redis container
    assert "[EMAIL_ADDRESS_1]" not in answer
    assert "alice.smith@example.com" in answer


@pytest.mark.asyncio
@patch("app.graph.nodes.get_llm_response")
async def test_e2e_viewer_gets_masked_response(
    mock_llm: MagicMock,
    e2e_client: AsyncClient,
    setup_e2e_users: dict[str, str],
) -> None:
    """E2E: Viewer role -> Real Query -> Masked Response (RBAC blocks demasking)"""
    # 1. Login as HR to upload the document
    hr_login = await e2e_client.post(
        "/auth/login",
        data={
            "username": setup_e2e_users["hr_email"],
            "password": setup_e2e_users["password"],
        },
    )
    hr_headers = {"Authorization": f"Bearer {hr_login.json()['access_token']}"}

    upload_resp = await e2e_client.post(
        "/ingest/",
        data={"department_id": "hr_dept", "access_level": 1},
        files={"file": ("contract_viewer.txt", TXT_CONTENT, "text/plain")},
        headers=hr_headers,
    )
    doc_id = upload_resp.json()["document_id"]

    # Wait for ingestion
    for _ in range(50):
        status_resp = await e2e_client.get(
            f"/ingest/{doc_id}/status", headers=hr_headers
        )
        if status_resp.json()["status"] == "done":
            break
        await asyncio.sleep(0.5)

    # 2. Login as Viewer
    viewer_login = await e2e_client.post(
        "/auth/login",
        data={
            "username": setup_e2e_users["viewer_email"],
            "password": setup_e2e_users["password"],
        },
    )
    viewer_headers = {"Authorization": f"Bearer {viewer_login.json()['access_token']}"}

    mock_llm.return_value = "The employee is [PERSON_1]."

    # 3. Query
    query_resp = await e2e_client.post(
        "/query/",
        json={"question": "Who is the employee?"},
        headers=viewer_headers,
    )

    assert query_resp.status_code == 200
    answer = query_resp.json()["answer"]

    # Viewer lacks 'view_unmasked' -> PII must remain masked
    assert "[PERSON_1]" in answer
    assert "Alice" not in answer


@pytest.mark.asyncio
async def test_e2e_viewer_cannot_escalate_access_level(
    e2e_client: AsyncClient,
    setup_e2e_users: dict[str, str],
) -> None:
    """E2E: Viewer attempts IDOR by requesting access_level=5. Qdrant payload filters must block it."""
    # 1. Login as HR and upload a SECRET document
    hr_login = await e2e_client.post(
        "/auth/login",
        data={
            "username": setup_e2e_users["hr_email"],
            "password": setup_e2e_users["password"],
        },
    )
    hr_headers = {"Authorization": f"Bearer {hr_login.json()['access_token']}"}

    secret_content = b"TOP SECRET: Project X is launching tomorrow."
    upload_resp = await e2e_client.post(
        "/ingest/",
        data={"department_id": "hr_dept", "access_level": 5},  # High access level
        files={"file": ("secret.txt", secret_content, "text/plain")},
        headers=hr_headers,
    )
    doc_id = upload_resp.json()["document_id"]

    for _ in range(50):
        if (
            await e2e_client.get(f"/ingest/{doc_id}/status", headers=hr_headers)
        ).json()["status"] == "done":
            break
        await asyncio.sleep(0.5)

    # 2. Login as Viewer
    viewer_login = await e2e_client.post(
        "/auth/login",
        data={
            "username": setup_e2e_users["viewer_email"],
            "password": setup_e2e_users["password"],
        },
    )
    viewer_headers = {"Authorization": f"Bearer {viewer_login.json()['access_token']}"}

    # 3. Execute Malicious Query Attempt
    query_resp = await e2e_client.post(
        "/query/",
        json={
            "question": "What is Project X?",
            "filters": {"access_level": 5},  # IDOR Attack
        },
        headers=viewer_headers,
    )

    assert query_resp.status_code == 200

    # Validation: Graph node drops access_level to 1 -> Real Qdrant returns 0 results -> Pipeline short-circuits
    assert query_resp.json()["sources"] == []
    assert "Information not found" in query_resp.json()["answer"]
