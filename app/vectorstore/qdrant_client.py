# File: app/vectorstore/qdrant_client.py
# Purpose: Qdrant vector database operations (collections, upserts, searches).
from typing import Any, Protocol, cast

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from app.logging_config.setup import get_logger

logger = get_logger(__name__)


# 1. Define a custom protocol for QueryResponse to fix missing '.points' stub in Qdrant
class QueryResponseProtocol(Protocol):
    points: list[ScoredPoint]


# 2. Update protocol to return our strictly typed Response Protocol
class AsyncQdrantSearcher(Protocol):
    async def query_points(
        self,
        collection_name: str,
        query: list[float],
        query_filter: Filter | None = None,
        limit: int = 10,
    ) -> QueryResponseProtocol: ...


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

    search_client = cast(AsyncQdrantSearcher, client)

    # query_response is strictly typed thanks to our Protocol
    query_response = await search_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
    )

    results: list[dict[str, Any]] = []
    for hit in query_response.points:
        results.append(
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload or {},
            }
        )

    return results


async def check_semantic_duplicate(
    client: AsyncQdrantClient,
    collection_name: str,
    query_vector: list[float],
    department_id: str,
    access_level: int,
    threshold: float = 0.98,
) -> bool:
    """
    Level 2 Deduplication: Checks for semantically identical documents using cosine similarity.
    Returns True if a highly similar chunk exists within the same department/access level.
    """
    results = await search_similar(
        client=client,
        collection_name=collection_name,
        query_vector=query_vector,
        department_id=department_id,
        access_level=access_level,
        top_k=1,
    )

    if results and results[0]["score"] >= threshold:
        logger.info(
            "Semantic duplicate detected in Qdrant",
            score=results[0]["score"],
            threshold=threshold,
        )
        return True

    return False
