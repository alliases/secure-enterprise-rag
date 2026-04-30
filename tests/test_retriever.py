# File: tests/test_retriever.py
# Purpose: Unit tests for vector retrieval logic, scoring thresholds, and payload validation.

import pytest
from pytest_mock import MockerFixture

from app.vectorstore.retriever import retrieve_context

# Constants for mock metadata to ensure Pydantic validation passes
VALID_PAYLOAD = {
    "text": "Confidential salary report for Alice.",
    "document_id": "doc-999",
    "department_id": "dept_hr_1",
    "access_level": 3,
    "source_filename": "salary_Q3.pdf",
    "chunk_index": 0,
}


@pytest.mark.asyncio
async def test_retrieve_context_success(mocker: MockerFixture) -> None:
    """Verifies successful context retrieval when results exceed the score threshold."""
    # Mock the embedder to return a dummy vector
    mock_embed = mocker.patch(
        "app.vectorstore.retriever.embed_query", return_value=[0.1, 0.2, 0.3]
    )

    # Mock the Qdrant search to return a high-scoring result
    mock_search = mocker.patch("app.vectorstore.retriever.search_similar")
    mock_search.return_value = [
        {
            "id": "chunk-uuid-1",
            "score": 0.85,  # 0.85 > 0.3 (Threshold)
            "payload": VALID_PAYLOAD,
        }
    ]

    mock_qdrant = mocker.AsyncMock()

    results = await retrieve_context(
        masked_query="salary report",
        department_id="dept_hr_1",
        access_level=3,
        qdrant=mock_qdrant,
        top_k=5,
    )

    # Assertions
    assert len(results) == 1
    assert results[0].text == VALID_PAYLOAD["text"]
    assert results[0].score == 0.85
    # Ensure 'text' was successfully popped out of metadata
    assert "text" not in results[0].metadata
    assert results[0].metadata["document_id"] == "doc-999"

    mock_embed.assert_called_once_with("salary report")
    mock_search.assert_called_once()


@pytest.mark.asyncio
async def test_retrieve_context_below_threshold(mocker: MockerFixture) -> None:
    """Ensures results with a score below 0.3 are strictly discarded to prevent hallucination."""
    mocker.patch("app.vectorstore.retriever.embed_query", return_value=[0.1])
    mock_search = mocker.patch("app.vectorstore.retriever.search_similar")

    # Score is 0.2, which is < 0.3 threshold
    mock_search.return_value = [
        {
            "id": "chunk-uuid-2",
            "score": 0.2,
            "payload": VALID_PAYLOAD,
        }
    ]

    mock_qdrant = mocker.AsyncMock()

    results = await retrieve_context(
        masked_query="unrelated query",
        department_id="dept_hr_1",
        access_level=3,
        qdrant=mock_qdrant,
    )

    # The retriever should discard the low-scoring result
    assert len(results) == 0


@pytest.mark.asyncio
async def test_retrieve_context_no_results(mocker: MockerFixture) -> None:
    """Verifies behavior when Qdrant returns an empty list (no match for RBAC filters)."""
    mocker.patch("app.vectorstore.retriever.embed_query", return_value=[0.1])
    mocker.patch("app.vectorstore.retriever.search_similar", return_value=[])

    mock_qdrant = mocker.AsyncMock()

    results = await retrieve_context(
        masked_query="secret project",
        department_id="dept_hr_1",
        access_level=3,
        qdrant=mock_qdrant,
    )

    assert len(results) == 0


@pytest.mark.asyncio
async def test_retrieve_context_invalid_payload(mocker: MockerFixture) -> None:
    """Verifies that Pydantic validation catches and drops corrupted DB records."""
    mocker.patch("app.vectorstore.retriever.embed_query", return_value=[0.1])
    mock_search = mocker.patch("app.vectorstore.retriever.search_similar")

    # Missing required fields like 'document_id' and 'department_id'
    corrupted_payload = {"text": "Corrupted chunk data"}

    mock_search.return_value = [
        {
            "id": "chunk-uuid-3",
            "score": 0.9,
            "payload": corrupted_payload,
        }
    ]

    mock_qdrant = mocker.AsyncMock()

    results = await retrieve_context(
        masked_query="test",
        department_id="dept_hr_1",
        access_level=3,
        qdrant=mock_qdrant,
    )

    # The corrupted chunk should be dropped, avoiding a crash
    assert len(results) == 0
