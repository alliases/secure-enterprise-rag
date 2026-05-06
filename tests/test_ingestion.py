"""
File: tests/test_ingestion.py
Task: 2.1 - Pipeline Tests
"""

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document
from app.ingestion.pipeline import run_ingestion


@pytest.mark.asyncio
async def test_run_ingestion_success(
    db_session: AsyncSession, mock_redis, mock_qdrant, tmp_path: Path
) -> None:
    """Happy path: Document is parsed, chunks are generated and masked, status updated to 'done'."""
    # 1. Setup mock document in DB
    doc_id = uuid.uuid4()
    test_doc = Document(
        id=doc_id,
        filename="test.txt",
        department_id="hr_dept",
        access_level=1,
        status="pending",
    )
    db_session.add(test_doc)
    await db_session.commit()

    # 2. Setup a temporary dummy file
    temp_file = tmp_path / "test.txt"
    temp_file.write_text("John Doe works at Acme Corp.")

    # 3. Patch out internal dependencies to isolate pipeline logic
    with (
        patch("app.ingestion.pipeline.parse_document") as mock_parse,
        patch("app.ingestion.pipeline.chunk_text") as mock_chunk,
        patch("app.ingestion.pipeline.analyze_text") as mock_analyze,
        patch("app.ingestion.pipeline.mask_text") as mock_mask,
        patch("app.ingestion.pipeline.embed_texts") as mock_embed,
    ):
        # Configure mocks
        mock_parse.return_value.text = "John Doe works at Acme Corp."

        from app.ingestion.chunker import Chunk

        mock_chunk.return_value = [
            Chunk(
                text="John Doe works at Acme Corp.",
                metadata={"document_id": str(doc_id)},
                chunk_index=0,
            )
        ]

        mock_analyze.return_value = []

        from app.masking.presidio_engine import MaskedResult

        mock_mask.return_value = MaskedResult(
            masked_text="[PERSON_1] works at Acme Corp.",
            mappings={"[PERSON_1]": "John Doe"},
        )

        mock_embed.return_value = [[0.1] * 1536]

        # 4. Execute pipeline (simulate the session factory behavior passing the current isolated session)
        async def mock_session_factory():
            # Create a dummy async context manager that returns our test session
            class DummyContextManager:
                async def __aenter__(self):
                    return db_session

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass

            return DummyContextManager()

        await run_ingestion(
            file_path=temp_file,
            file_name="test.txt",
            file_type="txt",
            document_id=str(doc_id),
            department_id="hr_dept",
            access_level=1,
            user_id=str(uuid.uuid4()),
            redis=mock_redis,
            qdrant=mock_qdrant,
            session_factory=mock_session_factory,
        )

    # 5. Assertions
    # Verify DB status updated
    await db_session.refresh(test_doc)
    assert test_doc.status == "done"
    assert test_doc.chunk_count == 1

    # Verify temp file is cleaned up
    assert not temp_file.exists()


@pytest.mark.asyncio
async def test_run_ingestion_parse_error(
    db_session: AsyncSession, mock_redis, mock_qdrant, tmp_path: Path
) -> None:
    """Error path: Parsing fails (e.g., corrupted file), status updated to 'error'."""
    doc_id = uuid.uuid4()
    test_doc = Document(
        id=doc_id,
        filename="corrupted.pdf",
        department_id="hr_dept",
        access_level=1,
        status="pending",
    )
    db_session.add(test_doc)
    await db_session.commit()

    temp_file = tmp_path / "corrupted.pdf"
    temp_file.write_bytes(b"not a real pdf")

    with patch(
        "app.ingestion.pipeline.parse_document", side_effect=Exception("Parsing failed")
    ):

        async def mock_session_factory():
            class DummyContextManager:
                async def __aenter__(self):
                    return db_session

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass

            return DummyContextManager()

        await run_ingestion(
            file_path=temp_file,
            file_name="corrupted.pdf",
            file_type="pdf",
            document_id=str(doc_id),
            department_id="hr_dept",
            access_level=1,
            user_id=str(uuid.uuid4()),
            redis=mock_redis,
            qdrant=mock_qdrant,
            session_factory=mock_session_factory,
        )

    await db_session.refresh(test_doc)
    assert test_doc.status == "error"
    assert not temp_file.exists()


@pytest.mark.asyncio
async def test_run_ingestion_redis_failure(
    db_session: AsyncSession, mock_redis, mock_qdrant, tmp_path: Path
) -> None:
    """Error path: Redis mapping store fails during chunk processing."""
    doc_id = uuid.uuid4()
    test_doc = Document(
        id=doc_id,
        filename="test.txt",
        department_id="hr_dept",
        access_level=1,
        status="pending",
    )
    db_session.add(test_doc)
    await db_session.commit()

    temp_file = tmp_path / "test.txt"
    temp_file.write_text("Data")

    with (
        patch("app.ingestion.pipeline.parse_document"),
        patch("app.ingestion.pipeline.chunk_text") as mock_chunk,
        patch("app.ingestion.pipeline.analyze_text"),
        patch("app.ingestion.pipeline.mask_text"),
        patch(
            "app.ingestion.pipeline.store_mappings",
            side_effect=ConnectionError("Redis down"),
        ),
    ):
        from app.ingestion.chunker import Chunk

        mock_chunk.return_value = [
            Chunk(text="Data", metadata={"document_id": str(doc_id)}, chunk_index=0)
        ]

        async def mock_session_factory():
            class DummyContextManager:
                async def __aenter__(self):
                    return db_session

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass

            return DummyContextManager()

        await run_ingestion(
            file_path=temp_file,
            file_name="test.txt",
            file_type="txt",
            document_id=str(doc_id),
            department_id="hr_dept",
            access_level=1,
            user_id=str(uuid.uuid4()),
            redis=mock_redis,
            qdrant=mock_qdrant,
            session_factory=mock_session_factory,
        )

    await db_session.refresh(test_doc)
    assert test_doc.status == "error"
