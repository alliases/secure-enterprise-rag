# File: tests/conftest.py
# Purpose: Global pytest fixtures for dependency injection in tests.

from collections.abc import AsyncGenerator
from typing import Any

import fakeredis.aioredis  # type: ignore[import-untyped]
import pytest
import pytest_asyncio
from redis.asyncio import Redis


@pytest_asyncio.fixture
async def mock_redis() -> AsyncGenerator[Redis]:
    """
    Provides an isolated, in-memory fake Redis instance for each test.
    Simulates async Redis behavior without requiring a real database connection.
    """
    server = fakeredis.FakeServer()
    # Decode responses=False aligns with our app.main implementation
    client = fakeredis.aioredis.FakeRedis(server=server, decode_responses=False)

    yield client

    await client.aclose()


@pytest.fixture
def hr_manager_user() -> dict[str, Any]:
    """Fixture for an authorized HR Manager."""
    return {
        "user_id": "11111111-1111-1111-1111-111111111111",
        "role": "hr_manager",
        "department_id": "dept_hr_1",
    }


@pytest.fixture
def viewer_user() -> dict[str, Any]:
    """Fixture for an unauthorized Viewer."""
    return {
        "user_id": "22222222-2222-2222-2222-222222222222",
        "role": "viewer",
        "department_id": "dept_hr_1",
    }


@pytest.fixture
def admin_user() -> dict[str, Any]:
    """Fixture for an Admin with global access."""
    return {
        "user_id": "33333333-3333-3333-3333-333333333333",
        "role": "admin",
        "department_id": "dept_it_1",
    }
