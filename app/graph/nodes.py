# File: app/graph/nodes.py
# Purpose: LangGraph execution nodes for the RAG pipeline.

from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from app.auth.rbac import check_permission
from app.graph.state import RAGState
from app.llm.prompts import RAG_SYSTEM_PROMPT
from app.llm.provider import get_llm_response
from app.logging_config.setup import get_logger
from app.masking.demasker import demask_response
from app.masking.presidio_engine import analyze_text, mask_text
from app.vectorstore.retriever import retrieve_context

logger = get_logger(__name__)


async def query_analyzer_node(state: RAGState) -> dict[str, Any]:
    """
    Analyzes the incoming query for PII and masks it before any external API calls.
    """
    original_query = str(state.get("original_query", ""))

    analyzer_results = analyze_text(original_query)
    masked_result = mask_text(original_query, analyzer_results)

    logger.info("Query analysis complete", pii_found=len(analyzer_results))

    return {
        "masked_query": masked_result.masked_text,
        "pii_mappings": masked_result.mappings,
    }


async def retriever_node(state: RAGState, config: RunnableConfig) -> dict[str, Any]:
    """
    Retrieves relevant context from Qdrant using the masked query and RBAC filters.
    """
    masked_query = str(state.get("masked_query", ""))

    # Вилучено cast, оскільки RAGState вже гарантує тип dict[str, Any]
    user = state.get("user", {})
    filters = state.get("filters", {})

    configurable = config.get("configurable", {})
    qdrant_instance = configurable.get("qdrant")

    if qdrant_instance is None:
        raise RuntimeError("Qdrant client not provided in RunnableConfig")

    qdrant = cast(AsyncQdrantClient, qdrant_instance)

    department_id = str(user.get("department_id", ""))

    # Secure access level: Enforce maximum allowed level based on user role.
    # Prevents IDOR (Insecure Direct Object Reference) via payload.filters.
    user_max_access = int(
        user.get(
            "access_level", 5 if user.get("role") in ["admin", "hr_manager"] else 1
        )
    )
    requested_access = int(filters.get("access_level", 1))

    # Cap the requested access level to the user's maximum authorized level
    access_level = min(requested_access, user_max_access)

    retrieved_chunks = await retrieve_context(
        masked_query=masked_query,
        department_id=department_id,
        access_level=access_level,
        qdrant=qdrant,
        top_k=5,
    )
    logger.info(
        "Context retrieval complete",
        chunk_count=len(retrieved_chunks),
        department_id=department_id,
    )

    document_ids = list(
        {str(chunk.metadata.get("document_id", "")) for chunk in retrieved_chunks}
    )

    return {
        "retrieved_chunks": retrieved_chunks,
        "document_ids": document_ids,
    }


async def synthesizer_node(state: RAGState) -> dict[str, Any]:
    """
    Generates a response using the LLM based strictly on the retrieved masked context.
    """
    masked_query = str(state.get("masked_query", ""))

    # Вилучено cast, RAGState контролює тип списку
    chunks = state.get("retrieved_chunks", [])

    context_texts = (
        [str(getattr(chunk, "text", "")) for chunk in chunks]
        if chunks
        else ["No context available."]
    )

    response = await get_llm_response(
        system_prompt=RAG_SYSTEM_PROMPT,
        user_message=masked_query,
        context_chunks=context_texts,
    )
    logger.info("LLM generation complete", response_length=len(response))
    return {"llm_response": response}


async def validator_node(state: RAGState) -> dict[str, Any]:
    """
    Heuristic validation to detect prompt injection leakage.
    """
    response = str(state.get("llm_response", ""))

    if "STRICT CONSTRAINTS:" in response or "You are a highly secure" in response:
        logger.warning("System prompt leakage detected in LLM response")
        return {
            "llm_response": "Security Error: Invalid response generated.",
            "error": "prompt_leakage",
        }

    return {}


async def demasking_node(state: RAGState, config: RunnableConfig) -> dict[str, Any]:
    """
    Restores PII from Redis and local state mappings for authorized users.
    """
    llm_response = str(state.get("llm_response", ""))

    # Вилучено всі cast, делегуємо перевірку типів еталонному RAGState
    document_ids = state.get("document_ids", [])
    user = state.get("user", {})
    query_mappings = state.get("pii_mappings", {})

    configurable = config.get("configurable", {})
    redis_instance = configurable.get("redis")

    if redis_instance is None:
        raise RuntimeError("Redis client not provided in RunnableConfig")

    redis = cast(Redis, redis_instance)
    department_id = str(user.get("department_id", ""))

    demasked_response = await demask_response(
        response_text=llm_response,
        document_ids=document_ids,
        target_department_id=department_id,
        redis=redis,
        user=user,
    )

    # Conditional Query De-masking: Do not restore user query PII if they lack 'view_unmasked'
    has_unmask_access = check_permission(
        user=user, target_department_id=department_id, action="view_unmasked"
    )

    if has_unmask_access:
        for token, original in query_mappings.items():
            demasked_response = demasked_response.replace(str(token), str(original))
    else:
        logger.warning(
            "User lacks view_unmasked permission. Query PII restoration skipped.",
            user_id=user.get("user_id"),
            role=user.get("role"),
        )

    return {"final_response": demasked_response}
