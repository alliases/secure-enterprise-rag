"""
File: tests/conftest.py
Task: 2.7 - Pytest fixtures and mocks
"""

from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import AsyncMock, patch

import fakeredis.aioredis
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.auth.jwt_handler import create_access_token
from app.db.models import Base


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession]:
    """
    Creates a fresh SQLite in-memory database for each test.
    Ensures complete isolation of database states between tests.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    async with session_factory() as session:
        yield session
        await session.rollback()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture
async def mock_redis() -> AsyncGenerator[fakeredis.aioredis.FakeRedis]:
    """
    Provides a fully functional in-memory Redis mock.
    """
    redis = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield redis
    await redis.aclose()


@pytest.fixture
def mock_qdrant() -> AsyncMock:
    """
    Mocks the AsyncQdrantClient to prevent network calls to the vector DB.
    """
    client = AsyncMock()
    client.collection_exists.return_value = True
    client.upsert.return_value = AsyncMock(status="completed")
    return client


@pytest.fixture
def mock_openai() -> Generator[AsyncMock]:
    """
    Patches the AsyncOpenAI client globally to prevent real API billing charges.
    Returns mocked responses for chat completions and embeddings.
    """
    with patch("app.llm.provider.AsyncOpenAI") as mock:
        mock_client = AsyncMock()

        # Mock ChatCompletion
        mock_message = AsyncMock(
            content="Mocked LLM response generated strictly from context."
        )
        mock_choice = AsyncMock(message=mock_message)
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        # Mock Embeddings (OpenAI 1536 dims)
        mock_data = AsyncMock(embedding=[0.1] * 1536)
        mock_client.embeddings.create.return_value.data = [mock_data]

        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def hr_user() -> dict[str, str]:
    return {
        "user_id": "00000000-0000-0000-0000-000000000001",
        "role": "hr_manager",
        "department_id": "hr_dept",
    }


@pytest.fixture
def viewer_user() -> dict[str, str]:
    return {
        "user_id": "00000000-0000-0000-0000-000000000002",
        "role": "viewer",
        "department_id": "hr_dept",
    }


@pytest.fixture
def hr_token(hr_user: dict[str, str]) -> str:
    """
    Generates a valid JWT for the HR Manager fixture.
    """
    return create_access_token(
        data={
            "sub": hr_user["user_id"],
            "role": hr_user["role"],
            "department_id": hr_user["department_id"],
        }
    )
