# File: app/vectorstore/retriever.py
# Purpose: Retrieval logic with filtering and reranking based on Qdrant.

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError
from qdrant_client import AsyncQdrantClient

from app.logging_config.setup import get_logger
from app.vectorstore.embedder import embed_query
from app.vectorstore.qdrant_client import search_similar

logger = get_logger(__name__)


class QdrantPayload(BaseModel):
    """
    Pydantic schema for strict runtime validation of Qdrant payloads.
    Ensures all required RBAC and mapping fields are present to prevent downstream crashes.
    """

    text: str
    document_id: str
    department_id: str
    access_level: int
    source_filename: str
    chunk_index: int


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

    query_vector = await embed_query(masked_query)

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

    # search_similar explicitly returns list[dict[str, Any]] with a guaranteed "score" float key
    highest_score = float(raw_results[0]["score"])

    if highest_score < 0.3:
        logger.info(
            "Highest search score below threshold, discarding results",
            highest_score=highest_score,
            threshold=0.3,
        )
        return processed_chunks

    for hit in raw_results:
        try:
            # Runtime validation: instantly catches missing fields or type mismatches from DB
            validated_payload = QdrantPayload.model_validate(hit["payload"])
        except ValidationError as e:
            logger.error(
                "Invalid payload format from Qdrant", error=str(e), hit_id=hit["id"]
            )
            continue

        # Extract metadata dynamically while safely omitting the text field
        metadata = validated_payload.model_dump(exclude={"text"})
        score = float(hit["score"])

        processed_chunks.append(
            RetrievedChunk(
                text=validated_payload.text,
                metadata=metadata,
                score=score,
            )
        )

    logger.info("Successfully retrieved chunks", count=len(processed_chunks))
    return processed_chunks
