"""
File: tests/test_qdrant_client.py
Task: 2.2 - Qdrant Client Tests
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.vectorstore.qdrant_client import (
    init_collection,
    search_similar,
    upsert_chunks,
)


@pytest.mark.asyncio
async def test_init_collection_creates_new() -> None:
    """Collection does not exist -> it should be created with indices."""
    mock_client = AsyncMock()
    mock_client.collection_exists.return_value = False

    await init_collection(mock_client, "test_collection", 1536)

    mock_client.create_collection.assert_called_once()
    assert mock_client.create_payload_index.call_count == 2

    # Verify index fields
    call_args_list = mock_client.create_payload_index.call_args_list
    fields_indexed = {call.kwargs["field_name"] for call in call_args_list}
    assert "department_id" in fields_indexed
    assert "access_level" in fields_indexed


@pytest.mark.asyncio
async def test_init_collection_idempotent() -> None:
    """Collection already exists -> should skip creation."""
    mock_client = AsyncMock()
    mock_client.collection_exists.return_value = True

    await init_collection(mock_client, "test_collection")

    mock_client.create_collection.assert_not_called()
    mock_client.create_payload_index.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_chunks_batch() -> None:
    """Verifies that chunks are correctly mapped to Qdrant PointStructs."""
    mock_client = AsyncMock()
    mock_op_info = MagicMock(status="completed")
    mock_client.upsert.return_value = mock_op_info

    chunks = [
        {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "text": "Masked test content",
            "metadata": {"department_id": "hr", "access_level": 1, "chunk_index": 0},
        }
    ]
    vectors = [[0.1, 0.2, 0.3]]

    result_count = await upsert_chunks(mock_client, "test_collection", chunks, vectors)

    assert result_count == 1
    mock_client.upsert.assert_called_once()

    # Inspect the point structure passed to the mock
    call_kwargs = mock_client.upsert.call_args.kwargs
    points = call_kwargs["points"]
    assert len(points) == 1
    assert points[0].id == "123e4567-e89b-12d3-a456-426614174000"
    assert points[0].payload["department_id"] == "hr"
    assert points[0].payload["text"] == "Masked test content"


@pytest.mark.asyncio
async def test_search_similar_with_filters() -> None:
    """Verifies RBAC filters are correctly applied during vector search."""
    mock_client = AsyncMock()

    # Mock the specific returned structure from Qdrant
    mock_point = MagicMock(id="uuid-1", score=0.92)
    mock_point.payload = {"text": "found context", "department_id": "hr"}

    mock_response = MagicMock()
    mock_response.points = [mock_point]
    mock_client.query_points.return_value = mock_response

    query_vector = [0.1, 0.5]
    results = await search_similar(
        client=mock_client,
        collection_name="test_collection",
        query_vector=query_vector,
        department_id="hr",
        access_level=2,
        top_k=5,
    )

    assert len(results) == 1
    assert results[0]["id"] == "uuid-1"
    assert results[0]["score"] == 0.92
    assert results[0]["payload"]["department_id"] == "hr"

    # Verify that the filter was correctly constructed
    call_kwargs = mock_client.query_points.call_args.kwargs
    qf = call_kwargs["query_filter"]
    assert qf is not None
    assert len(qf.must) == 2  # Must contain both department and access level filters


@pytest.mark.asyncio
async def test_search_similar_empty_result() -> None:
    """Verifies behavior when no chunks match the vector query."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.points = []
    mock_client.query_points.return_value = mock_response

    results = await search_similar(
        client=mock_client,
        collection_name="test_collection",
        query_vector=[0.0],
        department_id="finance",
        access_level=1,
    )

    assert results == []
