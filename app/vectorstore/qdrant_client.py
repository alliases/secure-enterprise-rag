# File: app/vectorstore/qdrant_client.py
# Purpose: Qdrant vector database operations (collections, upserts, searches).
# === File: app/vectorstore/qdrant_client.py ===
from typing import Any, Protocol, cast

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,  # Added explicit Enum for schema types
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from app.logging_config.setup import get_logger

logger = get_logger(__name__)


# Protocol bypass for untyped .search() from previous step
class AsyncQdrantSearcher(Protocol):
    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        query_filter: Filter | None = None,
        limit: int = 10,
    ) -> list[ScoredPoint]: ...


async def init_collection(
    client: AsyncQdrantClient, collection_name: str, vector_size: int = 1536
) -> None:
    """
    Idempotent creation of a Qdrant collection with payload indices for RBAC.
    """
    exists = await client.collection_exists(collection_name)
    if exists:
        logger.info(
            "Collection already exists, skipping initialization",
            collection=collection_name,
        )
        return

    logger.info(
        "Creating new Qdrant collection", collection=collection_name, size=vector_size
    )
    await client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    # Use strict Enum types from Qdrant models instead of raw strings or literals
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="department_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="access_level",
        field_schema=PayloadSchemaType.INTEGER,
    )


async def upsert_chunks(
    client: AsyncQdrantClient,
    collection_name: str,
    chunks: list[dict[str, Any]],
    vectors: list[list[float]],
) -> int:
    """
    Performs a batch upsert of vectorized chunks with their associated metadata.
    """
    points: list[PointStruct] = []

    for chunk, vector in zip(chunks, vectors, strict=True):
        points.append(
            PointStruct(
                id=chunk["id"],
                vector=vector,
                payload={"text": chunk["text"], **chunk["metadata"]},
            )
        )

    operation_info = await client.upsert(collection_name=collection_name, points=points)

    logger.info(
        "Upserted points to Qdrant", count=len(points), status=operation_info.status
    )
    return len(points)


async def search_similar(
    client: AsyncQdrantClient,
    collection_name: str,
    query_vector: list[float],
    department_id: str,
    access_level: int,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Executes a semantic search with strict metadata filtering.
    """
    query_filter = Filter(
        must=[
            FieldCondition(key="department_id", match=MatchValue(value=department_id)),
            FieldCondition(key="access_level", match=MatchValue(value=access_level)),
        ]
    )

    # 3. Cast the client to our local Protocol to ensure 100% Type Safety
    search_client = cast(AsyncQdrantSearcher, client)

    search_result: list[ScoredPoint] = await search_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=top_k,
    )

    results: list[dict[str, Any]] = []
    for hit in search_result:
        results.append(
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload or {},
            }
        )

    return results
