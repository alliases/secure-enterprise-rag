# File: app/api/endpoints/query.py
# Purpose: Execution endpoint for the RAG pipeline.

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AuditLog
from app.dependencies import get_current_user, get_db_session, get_qdrant, get_redis
from app.graph.graph_builder import rag_graph
from app.logging_config.setup import get_logger
from app.rate_limit import limiter

logger = get_logger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    """Payload schema for incoming RAG queries."""

    question: str = Field(..., min_length=3, max_length=2000)
    filters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("question")
    @classmethod
    def sanitize_question(cls, v: str) -> str:
        """Removes null bytes and strips whitespace to prevent basic prompt injections."""
        v = v.replace("\x00", "").strip()
        if not v:
            raise ValueError("Question cannot be empty after sanitization")
        return v


@router.post("/", status_code=status.HTTP_200_OK)
@limiter.limit("30/minute")  # type: ignore[reportUntypedFunctionDecorator, reportUnknownMemberType]
async def ask_question(
    request: Request,
    payload: QueryRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
) -> dict[str, Any]:
    """
    Processes a user query through the secure RAG graph.
    Injects external database clients via LangGraph's RunnableConfig.
    """
    initial_state = {
        "original_query": payload.question,
        "user": current_user,
        "filters": payload.filters,
    }

    # Pass connection clients via config to maintain serializable State dictionary
    config = {
        "configurable": {
            "redis": redis,
            "qdrant": qdrant,
        }
    }

    try:
        final_state = await rag_graph.ainvoke(initial_state, config=config)
    except Exception as e:
        logger.error("LangGraph execution failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal processing error during query execution",
        ) from e

    # Construct secure audit log entry without leaking original PII
    client_ip = request.client.host if request.client else "unknown"

    audit_entry = AuditLog(
        user_id=uuid.UUID(current_user["user_id"]),
        action="query",
        details={
            "masked_query": final_state.get("masked_query", ""),
            "chunk_count": len(final_state.get("retrieved_chunks", [])),
            "error_detected": final_state.get("error") is not None,
        },
        ip_address=client_ip,
    )
    db.add(audit_entry)
    await db.commit()

    # Handle short-circuit scenario where no context was retrieved
    if not final_state.get("retrieved_chunks"):
        return {
            "answer": "Information not found in the available documents.",
            "sources": [],
        }

    return {
        "answer": final_state.get("final_response", ""),
        "sources": final_state.get("document_ids", []),
    }
