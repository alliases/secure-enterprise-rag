"""
File: tests/test_api_ingest.py
Task: 2.5 - Ingest API Endpoints Tests
"""

import uuid
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document
from app.dependencies import get_db_session, get_qdrant, get_redis
from app.main import app

# Valid PDF magic bytes to bypass magic validation
PDF_MAGIC_BYTES = b"%PDF-1.4\n%\xaa\xbb\xcc\xdd fake pdf content"
# Invalid EXE magic bytes
EXE_MAGIC_BYTES = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff"


@pytest_asyncio.fixture
async def client(
    db_session: AsyncSession, mock_redis: Any, mock_qdrant: Any
) -> AsyncGenerator[AsyncClient]:
    app.dependency_overrides[get_db_session] = lambda: db_session
    app.dependency_overrides[get_redis] = lambda: mock_redis
    app.dependency_overrides[get_qdrant] = lambda: mock_qdrant

    class DummySessionFactory:
        def __call__(self) -> "DummySessionFactory":
            return self

        async def __aenter__(self) -> AsyncSession:
            return db_session

        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

    app.state.session_factory = DummySessionFactory()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as ac:
        yield ac

    app.dependency_overrides.clear()
    if hasattr(app.state, "session_factory"):
        delattr(app.state, "session_factory")


@pytest.mark.asyncio
async def test_upload_requires_authentication(client: AsyncClient) -> None:
    """Missing JWT header -> 401 Unauthorized."""
    response = await client.post(
        "/ingest/",
        data={"department_id": "hr", "access_level": "1"},
        files={"file": ("test.pdf", PDF_MAGIC_BYTES, "application/pdf")},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_upload_viewer_role_rejected(
    client: AsyncClient, viewer_user: dict[str, str]
) -> None:
    """Viewer role attempts to upload -> 403 Forbidden."""
    # We patch get_current_user to simulate the viewer token injection
    from app.dependencies import get_current_user

    app.dependency_overrides[get_current_user] = lambda: viewer_user

    response = await client.post(
        "/ingest/",
        data={"department_id": "hr", "access_level": "1"},
        files={"file": ("test.pdf", PDF_MAGIC_BYTES, "application/pdf")},
    )
    assert response.status_code == 403
    app.dependency_overrides.clear()


@pytest.mark.asyncio
@patch("app.api.endpoints.ingest.run_ingestion")
async def test_upload_success_creates_record(
    mock_run_ingestion: Any,
    client: AsyncClient,
    hr_user: dict[str, str],
    db_session: AsyncSession,
) -> None:
    """Valid PDF uploaded by HR Manager -> 202 Accepted, Document created."""
    import uuid

    from app.db.models import Role, User
    from app.dependencies import get_current_user

    dummy_hash: str = "dummy_hash_for_testing"

    user_id = uuid.UUID(hr_user["user_id"])
    test_role = Role(name=hr_user["role"], permissions=["view_unmasked", "upload_docs"])
    db_session.add(test_role)

    test_user = User(
        id=user_id,
        email="hr_uploader@example.com",
        hashed_password=dummy_hash,
        role_name=hr_user["role"],
        department_id=hr_user["department_id"],
        is_active=True,
    )
    db_session.add(test_user)
    await db_session.commit()

    app.dependency_overrides[get_current_user] = lambda: hr_user

    response = await client.post(
        "/ingest/",
        data={"department_id": "hr_dept", "access_level": "1"},
        files={"file": ("test.pdf", PDF_MAGIC_BYTES, "application/pdf")},
    )

    assert response.status_code == 202
    data = response.json()
    assert "document_id" in data
    assert data["status"] == "pending"

    # Verify document is in DB
    from app.db.models import Document

    doc_id = uuid.UUID(data["document_id"])
    doc = await db_session.get(Document, doc_id)
    assert doc is not None
    assert doc.filename == "test.pdf"
    assert doc.status == "pending"
    assert doc.uploaded_by == user_id  # Додаткова перевірка цілісності

    # Verify background task was called
    mock_run_ingestion.assert_called_once()
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_upload_unsupported_magic_bytes(
    client: AsyncClient, hr_user: dict[str, str]
) -> None:
    """File extension is PDF, but content is EXE -> 400 Bad Request."""
    from app.dependencies import get_current_user

    app.dependency_overrides[get_current_user] = lambda: hr_user

    response = await client.post(
        "/ingest/",
        data={"department_id": "hr_dept", "access_level": "1"},
        files={"file": ("malware.pdf", EXE_MAGIC_BYTES, "application/pdf")},
    )

    assert response.status_code == 400
    assert "Invalid or unsupported file content type" in response.json()["detail"]
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_status_returns_correct_data(
    client: AsyncClient, hr_user: dict[str, str], db_session: AsyncSession
) -> None:
    """Verify status endpoint returns document state."""
    from app.dependencies import get_current_user

    app.dependency_overrides[get_current_user] = lambda: hr_user

    # Setup DB
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        filename="status_test.pdf",
        department_id="hr_dept",
        access_level=1,
        status="done",
        chunk_count=10,
    )
    db_session.add(doc)
    await db_session.commit()

    response = await client.get(f"/ingest/{doc_id}/status")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "done"
    assert data["chunk_count"] == 10
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_status_not_found_returns_404(
    client: AsyncClient, hr_user: dict[str, str]
) -> None:
    """Querying non-existent document -> 404."""
    from app.dependencies import get_current_user

    app.dependency_overrides[get_current_user] = lambda: hr_user

    random_id = uuid.uuid4()
    response = await client.get(f"/ingest/{random_id}/status")

    assert response.status_code == 404
    app.dependency_overrides.clear()
