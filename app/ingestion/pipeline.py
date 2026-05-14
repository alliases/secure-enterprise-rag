# File: app/ingestion/pipeline.py
# Purpose: Orchestration of the document ingestion flow.

import traceback
import uuid
from pathlib import Path
from typing import Any

import anyio
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import AuditLog, Document
from app.ingestion.chunker import chunk_text
from app.ingestion.deduplicator import process_chunk_deduplication
from app.ingestion.parser import parse_document
from app.logging_config.setup import get_logger
from app.masking.mapping_store import store_mappings
from app.masking.presidio_engine import analyze_text, mask_text
from app.metrics import INGESTION_TOTAL
from app.vectorstore.embedder import embed_texts
from app.vectorstore.qdrant_client import upsert_chunks

logger = get_logger(__name__)


async def run_ingestion(
    file_path: Path,
    file_name: str,
    file_type: str,
    document_id: str,
    department_id: str,
    access_level: int,
    user_id: str,
    redis: Redis,
    qdrant: AsyncQdrantClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """
    Executes the full ingestion pipeline in the background.
    Spawns its own DB session to avoid conflicts with closed HTTP request contexts.
    """
    async with session_factory() as db_session:
        try:
            # 1. Update status to processing
            doc = await db_session.get(Document, uuid.UUID(document_id))
            if not doc:
                logger.error("Document not found in DB", document_id=document_id)
                return

            doc.status = "processing"
            await db_session.commit()

            # 2. Parse document
            parsed_doc = parse_document(file_path, file_name, file_type)

            # 3. Chunk text
            chunks = chunk_text(
                text=parsed_doc.text,
                document_id=document_id,
                department_id=department_id,
                access_level=access_level,
                source_filename=file_name,
            )

            if not chunks:
                logger.warning(
                    "No chunks generated from document", document_id=document_id
                )
                doc.status = "error"
                await db_session.commit()
                return

            masked_chunks_data: list[dict[str, Any]] = []
            total_pii_found = 0

            # Global state for Stateful Document PII Mapping across all chunks
            global_entity_counters: dict[str, int] = {}

            # 4. Mask and store PII mappings
            for chunk in chunks:
                analyzer_results = analyze_text(chunk.text)
                total_pii_found += len(analyzer_results)
                masked_result = mask_text(
                    text=chunk.text,
                    analyzer_results=analyzer_results,
                    entity_counters=global_entity_counters,
                )

                # Push generated mappings to Redis
                await store_mappings(redis, document_id, masked_result.mappings)

                masked_chunks_data.append(
                    {
                        "id": str(uuid.uuid4()),
                        "text": masked_result.masked_text,
                        "metadata": chunk.metadata,
                    }
                )

            # 5. Embed fully masked texts (Batch execution for speed)
            # CRITICAL: Apply token normalization ONLY for embeddings to prevent index-shift false negatives
            from app.masking.presidio_engine import normalize_for_embedding

            normalized_texts_for_embedding = [
                normalize_for_embedding(c["text"]) for c in masked_chunks_data
            ]
            vectors = await embed_texts(normalized_texts_for_embedding)

            # Level 2: Chunk-Level Deduplication
            (
                unique_chunks,
                unique_vectors,
                duplicate_count,
                canonical_doc_id,
            ) = await process_chunk_deduplication(
                qdrant=qdrant,
                masked_chunks_data=masked_chunks_data,
                vectors=vectors,
                document_id=document_id,
                department_id=department_id,
                access_level=access_level,
            )

            logger.info(
                "Chunk-level deduplication completed",
                document_id=document_id,
                unique_chunks=len(unique_chunks),
                duplicate_chunks=duplicate_count,
            )

            if duplicate_count > 0 and len(unique_chunks) == 0:
                doc.dedup_strategy = "semantic_full"
                if canonical_doc_id:
                    doc.canonical_document_id = uuid.UUID(canonical_doc_id)
            elif duplicate_count > 0 and len(unique_chunks) > 0:
                doc.dedup_strategy = "semantic_partial"
                if canonical_doc_id:
                    doc.canonical_document_id = uuid.UUID(canonical_doc_id)
                await upsert_chunks(
                    client=qdrant,
                    collection_name="documents",
                    chunks=unique_chunks,
                    vectors=unique_vectors,
                )
            else:
                doc.dedup_strategy = "none"
                await upsert_chunks(
                    client=qdrant,
                    collection_name="documents",
                    chunks=unique_chunks,
                    vectors=unique_vectors,
                )

            # 7. Update status to done
            doc.status = "done"
            doc.chunk_count = len(unique_chunks)
            # 8. Audit log generation
            audit_entry = AuditLog(
                user_id=uuid.UUID(user_id),
                action="ingest",
                details={
                    "document_id": document_id,
                    "chunk_count": len(chunks),
                    "pii_entities_found": total_pii_found,
                },
                ip_address="internal_background_task",
            )
            db_session.add(audit_entry)
            INGESTION_TOTAL.labels(status="done").inc()
            await db_session.commit()

            logger.info("Ingestion completed successfully", document_id=document_id)

        except Exception as e:
            logger.error(
                "Ingestion pipeline failed",
                document_id=document_id,
                error=str(e),
                trace=traceback.format_exc(),
            )
            INGESTION_TOTAL.labels(status="error").inc()
            await db_session.rollback()
            # Fallback status update
            doc = await db_session.get(Document, uuid.UUID(document_id))
            if doc:
                doc.status = "error"
                await db_session.commit()
        finally:
            # 9. Cleanup temp file to prevent disk exhaustion
            async_path = anyio.Path(file_path)
            if await async_path.exists():
                await async_path.unlink()
