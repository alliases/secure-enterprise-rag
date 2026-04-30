# File: tests/test_e2e.py
# Purpose: End-to-end API tests using FastAPI's dependency overrides and httpx.

import uuid
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pytest_mock import MockerFixture

from app.auth.jwt_handler import create_access_token
from app.dependencies import get_current_user, get_db_session, get_qdrant, get_redis
from app.main import app


async def override_get_current_user() -> dict[str, Any]:
    """Simulates an authenticated HR Manager."""
    return {
        "user_id": str(uuid.uuid4()),
        "email": "hr@example.com",
        "role": "hr_manager",
        "department_id": "dept_hr_1",
    }


@pytest.fixture
def test_app(mocker: MockerFixture) -> Any:
    """Returns a FastAPI app instance with mocked dependencies."""

    # Define the DB override inside the fixture to cleanly access 'mocker'
    async def override_get_db_session() -> Any:
        mock_session = mocker.AsyncMock()
        mock_session.add = mocker.MagicMock()
        mock_result = mocker.MagicMock()

        mock_user = mocker.MagicMock()
        mock_user.id = uuid.uuid4()
        mock_user.email = "hr@example.com"
        mock_user.hashed_password = "mocked_hash"
        mock_user.role_name = "hr_manager"
        mock_user.department_id = "dept_hr_1"
        mock_user.is_active = True

        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result

        # Using return instead of yield for standard dependency mock injection
        return mock_session

    async def override_get_redis() -> Any:
        return mocker.AsyncMock()

    async def override_get_qdrant() -> Any:
        return mocker.AsyncMock()

    # Apply overrides
    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis] = override_get_redis
    app.dependency_overrides[get_qdrant] = override_get_qdrant

    yield app

    # Clean up overrides after tests to prevent state leakage
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_health_check(test_app: FastAPI) -> None:
    """Verifies the API is up and running."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_query_endpoint(test_app: FastAPI, mocker: MockerFixture) -> None:
    """Verifies the /query endpoint successfully routes to the LangGraph pipeline."""
    # Mock the LangGraph execution to prevent actual DB/LLM calls
    mock_graph = mocker.patch("app.api.endpoints.query.rag_graph.ainvoke")
    mock_graph.return_value = {
        "masked_query": "What is the salary of [PERSON_1]?",
        "retrieved_chunks": ["chunk1"],
        "document_ids": ["doc-1"],
        "final_response": "The salary is $100k.",
        "error": None,
    }

    # Generate a valid token so OAuth2PasswordBearer doesn't block the request
    token = create_access_token({"sub": str(uuid.uuid4()), "role": "hr_manager"})

    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/query/",
            json={"question": "What is the salary of John?", "filters": {}},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The salary is $100k."
    assert data["sources"] == ["doc-1"]
    mock_graph.assert_called_once()


@pytest.mark.asyncio
async def test_unauthorized_access(test_app: FastAPI) -> None:
    """Verifies that endpoints are protected by OAuth2."""
    # Clear the user override to force the app to validate the token
    test_app.dependency_overrides.pop(get_current_user, None)

    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        # Requesting without Authorization header
        response = await client.post(
            "/query/",
            json={"question": "What is the salary?", "filters": {}},
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"
