# File: app/graph/state.py
# Purpose: TypedDict definition for LangGraph application state.

from typing import Any, TypedDict


class RAGState(TypedDict, total=False):
    """
    Represents the state of the RAG pipeline throughout the LangGraph execution.
    The 'total=False' flag allows nodes to selectively update keys during execution
    without throwing missing key errors.
    """

    # Input phase
    original_query: str
    masked_query: str
    pii_mappings: dict[str, str]

    # Request context
    user: dict[str, Any]
    filters: dict[str, Any]

    # Retrieval phase
    retrieved_chunks: list[dict[str, Any]]
    document_ids: list[str]

    # Generation phase
    llm_response: str
    final_response: str
    error: str | None
