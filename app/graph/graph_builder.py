from typing import Any, Protocol, cast

# Ignore missing stubs for external library
from langgraph.graph import END, START, StateGraph  # type: ignore[import-untyped]
from langgraph.graph.state import CompiledStateGraph  # type: ignore[import-untyped]

from app.graph.nodes import (
    demasking_node,
    query_analyzer_node,
    retriever_node,
    synthesizer_node,
    validator_node,
)
from app.graph.state import RAGState
from app.logging_config.setup import get_logger

logger = get_logger(__name__)


# 1. Define a strict Protocol to shield our app from LangGraph's complex internal types
class StateGraphProtocol(Protocol):
    def add_node(self, node: str, action: Any) -> Any: ...
    def add_edge(self, start_key: Any, end_key: Any) -> Any: ...
    def add_conditional_edges(
        self, start_key: Any, condition: Any, conditional_edge_mapping: dict[str, Any]
    ) -> Any: ...
    def compile(self) -> Any: ...


def check_retrieval(state: RAGState) -> str:
    """
    Conditional edge router. If no chunks were retrieved from Qdrant,
    bypass the LLM to save tokens and prevent hallucination.
    """
    if not state.get("retrieved_chunks"):
        logger.info("No context retrieved, short-circuiting graph execution")
        return "end"
    return "synthesize"


# Use 'Any' return type because the compiled graph type is highly dynamic
def build_rag_graph() -> Any:
    """
    Assembles the RAG execution pipeline.
    """
    # 2. Cast the untyped graph to our strict Protocol
    workflow = cast(StateGraphProtocol, StateGraph(RAGState))

    # 1. Add computation nodes
    workflow.add_node("analyze_query", query_analyzer_node)
    workflow.add_node("retrieve", retriever_node)
    workflow.add_node("synthesize", synthesizer_node)
    workflow.add_node("validate", validator_node)
    workflow.add_node("demask", demasking_node)

    # 2. Define standard linear execution edges
    workflow.add_edge(START, "analyze_query")
    workflow.add_edge("analyze_query", "retrieve")

    # 3. Define conditional branching
    workflow.add_conditional_edges(
        "retrieve",
        check_retrieval,
        {
            "synthesize": "synthesize",
            "end": END,
        },
    )

    # 4. Complete the post-generation path
    workflow.add_edge("synthesize", "validate")
    workflow.add_edge("validate", "demask")
    workflow.add_edge("demask", END)

    return workflow.compile()


rag_graph = build_rag_graph()
