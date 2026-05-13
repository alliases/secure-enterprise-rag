# File: app/ingestion/deduplicator.py
import asyncio
import hashlib
from collections.abc import Coroutine
from typing import Any

from fastapi import UploadFile
from qdrant_client import AsyncQdrantClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document
from app.logging_config.setup import get_logger

logger = get_logger(__name__)

SEMANTIC_THRESHOLD = 0.985


async def compute_file_hash_stream(file: UploadFile, chunk_size: int = 65536) -> str:
    """
    Calculates SHA-256 using an async generator to prevent OOM on large files.
    Reads in 64KB chunks by default.
    """
    hasher = hashlib.sha256()
    await file.seek(0)
    while chunk := await file.read(chunk_size):
        hasher.update(chunk)
    await file.seek(0)
    return hasher.hexdigest()


async def check_exact_duplicate(
    db: AsyncSession,
    content_hash: str,
    department_id: str,
) -> str | None:
    """Level 1: DB Lookup for exact hash match within the same department."""
    stmt = (
        select(Document.id)
        .where(Document.file_hash == content_hash)
        .where(Document.department_id == department_id)
        .where(Document.status == "done")
        .limit(1)
    )
    result = await db.execute(stmt)
    row = result.scalar_one_or_none()
    return str(row) if row else None


async def update_chunk_metadata(
    qdrant: AsyncQdrantClient,
    point_id: str | int,
    existing_payload: dict[str, Any],
    new_document_id: str,
    new_department_id: str,
    new_access_level: int,
) -> None:
    """Patches payload of a single specific chunk/point in Qdrant."""
    doc_ids = existing_payload.get(
        "document_ids", [existing_payload.get("document_id")]
    )
    departments = existing_payload.get(
        "department_ids", [existing_payload.get("department_id")]
    )
    access_levels = existing_payload.get(
        "access_levels", [existing_payload.get("access_level")]
    )

    # Clean None values from legacy records
    doc_ids = [i for i in doc_ids if i]
    departments = [d for d in departments if d]
    access_levels = [a for a in access_levels if a]

    if new_document_id not in doc_ids:
        doc_ids.append(new_document_id)
    if new_department_id not in departments:
        departments.append(new_department_id)
    if new_access_level not in access_levels:
        access_levels.append(new_access_level)

    await qdrant.set_payload(
        collection_name="documents",
        payload={
            "document_ids": doc_ids,  # CRITICAL FOR REDIS DEMASKING
            "department_ids": departments,
            "access_levels": access_levels,
            "document_id": existing_payload.get("document_id"),  # Backward compat
            "department_id": existing_payload.get("department_id"),
            "access_level": existing_payload.get("access_level"),
        },
        points=[point_id],
    )


async def process_chunk_deduplication(
    qdrant: AsyncQdrantClient,
    masked_chunks_data: list[dict[str, Any]],
    vectors: list[list[float]],
    document_id: str,
    department_id: str,
    access_level: int,
    update_tasks: list[Coroutine[Any, Any, None]] | None = None,
) -> tuple[list[dict[str, Any]], list[list[float]], int, str | None]:
    """
    Evaluates each chunk individually for semantic duplicates.
    Returns (unique_chunks, unique_vectors, duplicate_count, canonical_doc_id).
    """
    from app.vectorstore.qdrant_client import search_similar

    if update_tasks is None:
        update_tasks = []

    unique_chunks: list[dict[str, Any]] = []
    unique_vectors: list[list[float]] = []
    duplicate_count = 0
    canonical_doc_id: str | None = None

    for chunk, vector in zip(masked_chunks_data, vectors, strict=True):
        results = await search_similar(
            client=qdrant,
            collection_name="documents",
            query_vector=vector,
            department_id=department_id,
            access_level=access_level,
            top_k=1,
        )

        is_duplicate = False
        if results:
            score = float(results[0]["score"])
            if score >= SEMANTIC_THRESHOLD:
                payload = results[0].get("payload", {})
                point_id = str(results[0]["id"])

                doc_ids = payload.get("document_ids", [payload.get("document_id")])
                str_doc_ids = [str(d) for d in doc_ids if d]

                # Double-check to prevent self-duplication
                if document_id not in str_doc_ids:
                    is_duplicate = True
                    duplicate_count += 1

                    if not canonical_doc_id and str_doc_ids:
                        canonical_doc_id = str_doc_ids[0]

                    update_tasks.append(
                        update_chunk_metadata(
                            qdrant=qdrant,
                            point_id=point_id,
                            existing_payload=payload,
                            new_document_id=document_id,
                            new_department_id=department_id,
                            new_access_level=access_level,
                        )
                    )

        if not is_duplicate:
            unique_chunks.append(chunk)
            unique_vectors.append(vector)

    if update_tasks:
        await asyncio.gather(*update_tasks)

    return unique_chunks, unique_vectors, duplicate_count, canonical_doc_id
