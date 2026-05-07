# File: tests/test_ingestion.py
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, Role, User
from app.ingestion.chunker import Chunk
from app.ingestion.pipeline import run_ingestion
from app.masking.presidio_engine import MaskedResult


@pytest.mark.asyncio
async def test_run_ingestion_success(
    db_session: AsyncSession,
    mock_redis: Any,
    mock_qdrant: Any,
    tmp_path: Path,
    hr_user: dict[str, str],
) -> None:
    """Happy path: Document is parsed, chunks are generated and masked, status updated to 'done'."""
    user_id = uuid.UUID(hr_user["user_id"])

    # 1. Seed prerequisites
    test_role = Role(name=hr_user["role"], permissions=["upload_docs"])
    db_session.add(test_role)
    await db_session.flush()  # Enforce explicit insert order for foreign keys

    test_user = User(
        id=user_id,
        email="ingest_test@example.com",
        hashed_password="secure_dummy_hash",
        role_name=hr_user["role"],
        department_id=hr_user["department_id"],
    )
    db_session.add(test_user)
    await db_session.flush()  # Enforce explicit insert order for foreign keys

    doc_id = uuid.uuid4()
    test_doc = Document(
        id=doc_id,
        filename="test.txt",
        department_id=hr_user["department_id"],
        access_level=1,
        status="pending",
        uploaded_by=user_id,
    )
    db_session.add(test_doc)
    await db_session.commit()

    temp_file = tmp_path / "test.txt"
    temp_file.write_text("John Doe works at Acme Corp.", encoding="utf-8")

    with (
        patch("app.ingestion.pipeline.parse_document") as mock_parse,
        patch("app.ingestion.pipeline.chunk_text") as mock_chunk,
        patch("app.ingestion.pipeline.analyze_text") as mock_analyze,
        patch("app.ingestion.pipeline.mask_text") as mock_mask,
        patch(
            "app.ingestion.pipeline.embed_texts", new_callable=AsyncMock
        ) as mock_embed,
        patch("app.ingestion.pipeline.store_mappings", new_callable=AsyncMock),
        patch("app.ingestion.pipeline.upsert_chunks", new_callable=AsyncMock),
    ):
        mock_parse.return_value.text = "John Doe works at Acme Corp."
        mock_chunk.return_value = [
            Chunk(
                text="John Doe works at Acme Corp.",
                metadata={"document_id": str(doc_id)},
                chunk_index=0,
            )
        ]

        mock_analyze.return_value = []

        mock_mask.return_value = MaskedResult(
            masked_text="[PERSON_1] works at Acme Corp.",
            mappings={"[PERSON_1]": "John Doe"},
        )

        mock_embed.return_value = [[0.1] * 1536]

        def mock_session_factory() -> Any:
            class DummyContextManager:
                async def __aenter__(self) -> AsyncSession:
                    return db_session

                async def __aexit__(
                    self, exc_type: Any, exc_val: Any, exc_tb: Any
                ) -> None:
                    pass

            return DummyContextManager()

        await run_ingestion(
            file_path=temp_file,
            file_name="test.txt",
            file_type="txt",
            document_id=str(doc_id),
            department_id=hr_user["department_id"],
            access_level=1,
            user_id=str(user_id),
            redis=mock_redis,
            qdrant=mock_qdrant,
            session_factory=mock_session_factory,
        )

    await db_session.refresh(test_doc)
    assert test_doc.status == "done"
    assert test_doc.chunk_count == 1
    assert not temp_file.exists()


@pytest.mark.asyncio
async def test_run_ingestion_redis_failure(
    db_session: AsyncSession,
    mock_redis: Any,
    mock_qdrant: Any,
    tmp_path: Path,
    hr_user: dict[str, str],
) -> None:
    """Error path: Redis mapping store fails during chunk processing."""
    user_id = uuid.UUID(hr_user["user_id"])

    # 1. Seed prerequisites
    test_role = Role(name=hr_user["role"], permissions=["upload_docs"])
    db_session.add(test_role)
    await db_session.flush()

    test_user = User(
        id=user_id,
        email="ingest_error@example.com",
        hashed_password="secure_dummy_hash",
        role_name=hr_user["role"],
        department_id=hr_user["department_id"],
    )
    db_session.add(test_user)
    await db_session.flush()

    doc_id = uuid.uuid4()
    test_doc = Document(
        id=doc_id,
        filename="test.txt",
        department_id=hr_user["department_id"],
        access_level=1,
        status="pending",
        uploaded_by=user_id,
    )
    db_session.add(test_doc)
    await db_session.commit()

    temp_file = tmp_path / "test.txt"
    temp_file.write_text("Data", encoding="utf-8")

    with (
        patch("app.ingestion.pipeline.parse_document"),
        patch("app.ingestion.pipeline.chunk_text") as mock_chunk,
        patch("app.ingestion.pipeline.analyze_text"),
        patch("app.ingestion.pipeline.mask_text"),
        patch(
            "app.ingestion.pipeline.store_mappings",
            new_callable=AsyncMock,
            side_effect=ConnectionError("Redis down"),
        ),
    ):
        mock_chunk.return_value = [
            Chunk(text="Data", metadata={"document_id": str(doc_id)}, chunk_index=0)
        ]

        def mock_session_factory() -> Any:
            class DummyContextManager:
                async def __aenter__(self) -> AsyncSession:
                    return db_session

                async def __aexit__(
                    self, exc_type: Any, exc_val: Any, exc_tb: Any
                ) -> None:
                    pass

            return DummyContextManager()

        await run_ingestion(
            file_path=temp_file,
            file_name="test.txt",
            file_type="txt",
            document_id=str(doc_id),
            department_id=hr_user["department_id"],
            access_level=1,
            user_id=str(user_id),
            redis=mock_redis,
            qdrant=mock_qdrant,
            session_factory=mock_session_factory,
        )

    await db_session.refresh(test_doc)
    assert test_doc.status == "error"
