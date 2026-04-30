# File: app/vectorstore/retriever.py
# Purpose: Retrieval logic with filtering and reranking based on Qdrant.

from dataclasses import dataclass
from typing import Any, cast

from qdrant_client import AsyncQdrantClient

from app.logging_config.setup import get_logger
from app.vectorstore.embedder import embed_query
from app.vectorstore.qdrant_client import search_similar

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """
    Data container for a retrieved text chunk from the vector database.
    """

    text: str
    metadata: dict[str, Any]
    score: float


async def retrieve_context(
    masked_query: str,
    department_id: str,
    access_level: int,
    qdrant: AsyncQdrantClient,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """
    Embeds the masked query and retrieves relevant chunks from Qdrant,
    applying strict RBAC filtering and a minimum score threshold.
    """
    logger.info("Retrieving context for query", department_id=department_id)

    # 1. Embed the masked query
    # The query is already masked by Presidio before reaching this point
    query_vector = await embed_query(masked_query)

    # 2. Search in Qdrant with hardware-accelerated metadata filters
    raw_results = await search_similar(
        client=qdrant,
        collection_name="documents",
        query_vector=query_vector,
        department_id=department_id,
        access_level=access_level,
        top_k=top_k,
    )

    processed_chunks: list[RetrievedChunk] = []

    if not raw_results:
        logger.debug("No results found in Qdrant for the given filters.")
        return processed_chunks

    # 3. Threshold Check
    # Explicitly type the result of .get() to avoid Unknown type propagation
    first_score_val: Any = raw_results[0].get("score", 0.0)
    highest_score = float(first_score_val)

    if highest_score < 0.3:
        logger.info(
            "Highest search score below threshold, discarding results",
            highest_score=highest_score,
            threshold=0.3,
        )
        return processed_chunks

    # 4. Parse payload and map to data classes
    for hit in raw_results:
        raw_payload: Any = hit.get("payload", {})
        if not isinstance(raw_payload, dict):
            continue

        # Cast the built-in dict[Unknown, Unknown] to dict[str, Any] to satisfy Strict Mode
        payload = cast(dict[str, Any], raw_payload)

        raw_text: Any = payload.get("text", "")
        text = str(raw_text)

        # Explicitly declare types for dictionary comprehension variables
        metadata: dict[str, Any] = {
            str(k): v for k, v in payload.items() if str(k) != "text"
        }

        raw_score: Any = hit.get("score", 0.0)
        score = float(raw_score)

        processed_chunks.append(
            RetrievedChunk(
                text=text,
                metadata=metadata,
                score=score,
            )
        )

    logger.info("Successfully retrieved chunks", count=len(processed_chunks))
    return processed_chunks
