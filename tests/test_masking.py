# File: tests/test_masking.py
# Purpose: Unit tests for PII identification, masking, and de-masking flows.

import pytest
from redis.asyncio import Redis

from app.masking.demasker import demask_response
from app.masking.mapping_store import delete_mappings, retrieve_mappings, store_mappings
from app.masking.presidio_engine import analyze_text, mask_text


def test_mask_person() -> None:
    """Verifies that standard PERSON entities are masked correctly."""
    text = "John Doe is a new employee."
    results = analyze_text(text)
    masked = mask_text(text, results)

    assert "John Doe" not in masked.masked_text
    assert "[PERSON_1]" in masked.masked_text
    assert masked.mappings["[PERSON_1]"] == "John Doe"


def test_mask_email() -> None:
    """Verifies that EMAIL_ADDRESS entities are masked correctly."""
    text = "Contact at test.user@example.com immediately."
    results = analyze_text(text)
    masked = mask_text(text, results)

    assert "test.user@example.com" not in masked.masked_text
    assert "[EMAIL_ADDRESS_1]" in masked.masked_text
    assert masked.mappings["[EMAIL_ADDRESS_1]"] == "test.user@example.com"


def test_custom_regex_employee_id() -> None:
    """Verifies that the custom regex recognizer catches internal IDs."""
    text = "The internal identifier is 4500-1234."
    results = analyze_text(text)
    masked = mask_text(text, results)

    assert "4500-1234" not in masked.masked_text
    assert "[EMPLOYEE_ID_1]" in masked.masked_text
    assert masked.mappings["[EMPLOYEE_ID_1]"] == "4500-1234"


def test_mask_multiple_entities() -> None:
    """Verifies incremental token generation for multiple entities."""
    text = "Alice and Bob sent an email."
    results = analyze_text(text)
    masked = mask_text(text, results)

    assert "Alice" not in masked.masked_text
    assert "Bob" not in masked.masked_text
    assert "[PERSON_1]" in masked.masked_text
    assert "[PERSON_2]" in masked.masked_text
    assert len(masked.mappings) == 2


@pytest.mark.asyncio
async def test_store_retrieve_delete_mappings(mock_redis: Redis) -> None:
    """End-to-end test for Redis mapping store CRUD operations."""
    doc_id = "doc-123"
    mappings = {"[PERSON_1]": "Alice", "[EMAIL_ADDRESS_1]": "alice@test.com"}

    # Test Store
    stored_count = await store_mappings(mock_redis, doc_id, mappings)
    assert stored_count == 2

    # Test Retrieve
    retrieved = await retrieve_mappings(mock_redis, doc_id)
    assert retrieved == mappings

    # Test Delete
    deleted_count = await delete_mappings(mock_redis, doc_id)
    assert deleted_count == 2
    assert await retrieve_mappings(mock_redis, doc_id) == {}


@pytest.mark.asyncio
async def test_demask_authorized_user(
    mock_redis: Redis, hr_user: dict[str, str]
) -> None:
    """Verifies that an HR Manager can view unmasked data for their department."""
    doc_id = "doc-456"
    mappings = {"[PERSON_1]": "Bob"}
    await store_mappings(mock_redis, doc_id, mappings)

    masked_response = "The salary of [PERSON_1] is high."
    demasked = await demask_response(
        response_text=masked_response,
        document_ids=[doc_id],
        target_department_id="hr_dept",  # Matches the hr_user fixture
        redis=mock_redis,
        user=hr_user,
    )

    assert demasked != masked_response  # Fixed variable name
    assert demasked == "The salary of Bob is high."


@pytest.mark.asyncio
async def test_demask_unauthorized_user(
    mock_redis: Redis, viewer_user: dict[str, str]
) -> None:
    """Verifies that a Viewer receives masked data regardless of the department."""
    doc_id = "doc-789"
    mappings = {"[PERSON_1]": "Charlie"}
    await store_mappings(mock_redis, doc_id, mappings)

    masked_response = "The salary of [PERSON_1] is secret."
    demasked = await demask_response(
        response_text=masked_response,
        document_ids=[doc_id],
        target_department_id="dept_hr_1",
        redis=mock_redis,
        user=viewer_user,
    )

    # The string should remain exactly the same
    assert demasked == masked_response
    assert "Charlie" not in demasked
