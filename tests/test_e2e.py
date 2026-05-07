# File: tests/test_e2e.py
# Purpose: End-to-end API tests using FastAPI's dependency overrides and httpx.


import uuid
from collections.abc import AsyncGenerator
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import get_password_hash
from app.db.models import Document, Role, User
from app.dependencies import get_db_session, get_qdrant, get_redis
from app.main import app
from app.rate_limit import limiter

# Valid PDF magic bytes to bypass magic validation
PDF_MAGIC_BYTES = (
    b"%PDF-1.4\n%\xaa\xbb\xcc\xdd test content with email john.doe@example.com"
)


@pytest_asyncio.fixture
async def e2e_client(
    db_session: AsyncSession, mock_redis, mock_qdrant
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
@patch("app.ingestion.pipeline.run_ingestion")
@patch("app.graph.nodes.get_llm_response")
async def test_e2e_full_flow_hr_manager(
    mock_llm,
    mock_ingest,
    e2e_client: AsyncClient,
    setup_e2e_users: dict[str, str],
    db_session: AsyncSession,
    mock_redis,
) -> None:
    """E2E: Login -> Upload -> Query -> Demasked Response"""
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

    # 2. Upload Document
    upload_resp = await e2e_client.post(
        "/ingest/",
        data={"department_id": "hr_dept", "access_level": 1},
        files={"file": ("contract.pdf", PDF_MAGIC_BYTES, "application/pdf")},
        headers=headers,
    )
    assert upload_resp.status_code == 202
    doc_id = upload_resp.json()["document_id"]

    # Simulate ingestion background task completing
    doc = await db_session.get(Document, uuid.UUID(doc_id))
    doc.status = "done"  # type: ignore
    await db_session.commit()

    # Manually store PII mapping in Redis to simulate masking engine output
    from cryptography.fernet import Fernet

    from app.config import get_settings

    key = get_settings().redis_encryption_key.get_secret_value().encode()
    cipher = Fernet(key)
    encrypted_val = cipher.encrypt(b"john.doe@example.com").decode()
    await mock_redis.setex(f"pii:{doc_id}:[EMAIL_1]", 3600, encrypted_val)

    # 3. Query the System
    # Mock the LLM to return a response with the masked token
    mock_llm.return_value = "The email is [EMAIL_1]."

    # Mock Qdrant retriever node to return our document id
    with patch("app.graph.nodes.retrieve_context") as mock_retrieve:
        from app.vectorstore.retriever import RetrievedChunk

        mock_retrieve.return_value = [
            RetrievedChunk(text="Context", metadata={"document_id": doc_id}, score=0.9)
        ]

        query_resp = await e2e_client.post(
            "/query/",
            json={"question": "What is the email?"},
            headers=headers,
        )

    assert query_resp.status_code == 200
    answer = query_resp.json()["answer"]

    # HR Manager should see the DEMASKED data
    assert "[EMAIL_1]" not in answer
    assert "john.doe@example.com" in answer


@pytest.mark.asyncio
@patch("app.graph.nodes.get_llm_response")
async def test_e2e_viewer_gets_masked_response(
    mock_llm,
    e2e_client: AsyncClient,
    setup_e2e_users: dict[str, str],
    db_session: AsyncSession,
    mock_redis,
) -> None:
    """E2E: Viewer role -> Query -> Masked Response (no demasking allowed)"""
    # 1. Login as Viewer
    login_resp = await e2e_client.post(
        "/auth/login",
        data={
            "username": setup_e2e_users["viewer_email"],
            "password": setup_e2e_users["password"],
        },
    )
    assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Simulate existing document and mapping
    doc_id = str(uuid.uuid4())
    from cryptography.fernet import Fernet

    from app.config import get_settings

    key = get_settings().redis_encryption_key.get_secret_value().encode()
    cipher = Fernet(key)
    await mock_redis.setex(
        f"pii:{doc_id}:[PERSON_1]", 3600, cipher.encrypt(b"Alice").decode()
    )

    mock_llm.return_value = "The employee is [PERSON_1]."

    with patch("app.graph.nodes.retrieve_context") as mock_retrieve:
        from app.vectorstore.retriever import RetrievedChunk

        mock_retrieve.return_value = [
            RetrievedChunk(text="Context", metadata={"document_id": doc_id}, score=0.9)
        ]

        query_resp = await e2e_client.post(
            "/query/",
            json={"question": "Who is the employee?"},
            headers=headers,
        )

    assert query_resp.status_code == 200
    answer = query_resp.json()["answer"]

    # Viewer must NEVER see demasked data
    assert "[PERSON_1]" in answer
    assert "Alice" not in answer
