# File: tests/test_graph.py
# Purpose: Integration tests for the LangGraph orchestrator.

import pytest
from pytest_mock import MockerFixture

from app.graph.graph_builder import rag_graph
from app.vectorstore.retriever import RetrievedChunk


@pytest.mark.asyncio
async def test_full_flow_authorized(mocker: MockerFixture) -> None:
    """
    Tests the complete graph execution for an authorized user.
    Ensures that the LLM is called and the response is correctly de-masked.
    """
    # 1. Mock Retriever: Return a dummy chunk with a PII token
    mock_retrieve = mocker.patch("app.graph.nodes.retrieve_context")
    mock_retrieve.return_value = [
        RetrievedChunk(
            text="The salary of [PERSON_1] is $100k.",
            metadata={"document_id": "doc-1"},
            score=0.9,
        )
    ]

    # 2. Mock LLM: Return a generated response containing the PII token
    mock_llm = mocker.patch("app.graph.nodes.get_llm_response")
    mock_llm.return_value = "Based on the documents, the salary of [PERSON_1] is $100k."

    # 3. Mock Mapping Store: Simulate Redis returning the real name for the token
    mocker.patch(
        "app.masking.demasker.retrieve_mappings",
        return_value={"[PERSON_1]": "Alice"},
    )

    initial_state = {
        "original_query": "What is the salary of Alice?",
        "user": {"role": "hr_manager", "department_id": "hr", "user_id": "user-1"},
        "filters": {"access_level": 1},
    }

    # Pass mock clients to the graph config
    config = {
        "configurable": {
            "redis": mocker.AsyncMock(),
            "qdrant": mocker.AsyncMock(),
        }
    }

    final_state = await rag_graph.ainvoke(initial_state, config=config)

    # Assertions
    assert final_state["document_ids"] == ["doc-1"]
    mock_llm.assert_called_once()

    # The final response should have restored "Alice" from "[PERSON_1]"
    assert "Alice" in final_state["final_response"]
    assert "[PERSON_1]" not in final_state["final_response"]


@pytest.mark.asyncio
async def test_full_flow_unauthorized(mocker: MockerFixture) -> None:
    """
    Tests the graph execution for an unauthorized user.
    Ensures that DB-level PII remains masked if the user doesn't have access.
    """
    mock_retrieve = mocker.patch("app.graph.nodes.retrieve_context")
    mock_retrieve.return_value = [
        RetrievedChunk(
            text="Performance issue for [PERSON_1].",
            metadata={"document_id": "doc-2"},
            score=0.8,
        )
    ]

    mock_llm = mocker.patch("app.graph.nodes.get_llm_response")
    mock_llm.return_value = "There is a performance issue for [PERSON_1]."

    mocker.patch(
        "app.masking.demasker.retrieve_mappings",
        return_value={"[PERSON_1]": "Bob"},
    )

    # FIX: The user asks a generic question WITHOUT knowing the PII.
    initial_state = {
        "original_query": "What are the recent performance issues?",
        "user": {"role": "viewer", "department_id": "hr", "user_id": "user-2"},
        "filters": {},
    }

    config = {
        "configurable": {
            "redis": mocker.AsyncMock(),
            "qdrant": mocker.AsyncMock(),
        }
    }

    final_state = await rag_graph.ainvoke(initial_state, config=config)

    # The final response MUST contain the token, NOT the real name, due to RBAC
    assert "Bob" not in final_state["final_response"]
    assert "[PERSON_1]" in final_state["final_response"]


@pytest.mark.asyncio
async def test_no_results_short_circuit(mocker: MockerFixture) -> None:
    """
    Tests the conditional edge in the graph.
    If no context is found, it should end immediately without calling the LLM.
    """
    mock_retrieve = mocker.patch("app.graph.nodes.retrieve_context")
    # Simulate Qdrant returning no relevant results
    mock_retrieve.return_value = []

    mock_llm = mocker.patch("app.graph.nodes.get_llm_response")

    initial_state = {
        "original_query": "What is the secret project?",
        "user": {"role": "admin", "department_id": "it", "user_id": "user-3"},
        "filters": {},
    }

    config = {
        "configurable": {
            "redis": mocker.AsyncMock(),
            "qdrant": mocker.AsyncMock(),
        }
    }

    final_state = await rag_graph.ainvoke(initial_state, config=config)

    # Assertions
    assert len(final_state.get("retrieved_chunks", [])) == 0
    # CRITICAL: The LLM should NEVER be called if there is no context (Cost/Hallucination protection)
    mock_llm.assert_not_called()
    # The final response won't exist because it bypassed the synthesizer and demasker nodes
    assert "final_response" not in final_state


@pytest.mark.asyncio
async def test_retriever_node_prevents_access_level_escalation(
    mocker: MockerFixture,
) -> None:
    """
    Tests that IDOR attempts are blocked by capping the requested access_level
    to the user's maximum authorized access_level.
    """
    from app.graph.nodes import retriever_node

    mock_retrieve = mocker.patch("app.graph.nodes.retrieve_context")
    mock_retrieve.return_value = []

    # User is a viewer (max access level 1) but tries to request access level 5 via payload
    state = {
        "masked_query": "Test query",
        "user": {"role": "viewer", "department_id": "hr", "user_id": "user-4"},
        "filters": {"access_level": 5},
    }

    config = {
        "configurable": {
            "qdrant": mocker.AsyncMock(),
        }
    }

    await retriever_node(state, config)  # type: ignore

    # Verify that retrieve_context was called with access_level 1, NOT 5
    mock_retrieve.assert_called_once()
    assert mock_retrieve.call_args.kwargs["access_level"] == 1


@pytest.mark.asyncio
async def test_demasking_node_respects_rbac_for_query_tokens(
    mocker: MockerFixture,
) -> None:
    """
    Tests that query PII mappings are NOT restored if the user lacks the 'view_unmasked' permission.
    """
    from app.graph.nodes import demasking_node

    mocker.patch(
        "app.masking.demasker.demask_response",
        return_value="The salary of [PERSON_1] is $100k.",
    )

    # Viewer asks about Alice. The query analyzer generated a mapping.
    state = {
        "llm_response": "The salary of [PERSON_1] is $100k.",
        "document_ids": ["doc-1"],
        "user": {"role": "viewer", "department_id": "hr", "user_id": "user-5"},
        "pii_mappings": {"[PERSON_1]": "Alice"},
    }

    config = {
        "configurable": {
            "redis": mocker.AsyncMock(),
        }
    }

    result = await demasking_node(state, config)  # type: ignore

    # The final response MUST STILL contain [PERSON_1] because the user is a viewer
    assert "[PERSON_1]" in result["final_response"]
    assert "Alice" not in result["final_response"]
