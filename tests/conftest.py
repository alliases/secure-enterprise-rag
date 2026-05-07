"""
File: tests/conftest.py
Task: 2.7 - Pytest fixtures and mocks
"""

import os
from collections.abc import AsyncGenerator, Generator
from typing import Any

from cryptography.fernet import Fernet
from httpx import ASGITransport, AsyncClient

os.environ["REDIS_ENCRYPTION_KEY"] = Fernet.generate_key().decode("utf-8")
from unittest.mock import AsyncMock, patch

import fakeredis.aioredis
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]

from app.auth.jwt_handler import create_access_token
from app.db.models import Base
from app.dependencies import get_db_session, get_qdrant, get_redis
from app.main import create_app


@pytest.fixture(scope="session")
def postgres_container() -> Generator[PostgresContainer]:
    """
    Spins up a real PostgreSQL container once per test session.
    Ensures Dev/Prod parity for tests involving DB-specific features (UUIDs, JSONB).
    """
    with PostgresContainer("postgres:16-alpine") as postgres:
        yield postgres


@pytest_asyncio.fixture
async def db_session(
    postgres_container: PostgresContainer,
) -> AsyncGenerator[AsyncSession]:
    """
    Provides an isolated database session.
    Creates and drops tables dynamically per test using the session-scoped container.
    """
    # testcontainers returns psycopg2 URL by default, swap it to asyncpg
    sync_url = postgres_container.get_connection_url()
    async_url = sync_url.replace("psycopg2", "asyncpg")

    engine = create_async_engine(async_url, echo=False)

    # Initialize clean schema for the specific test
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    async with session_factory() as session:
        yield session
        await session.rollback()

    # Teardown schema to ensure complete isolation for the next test
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


@pytest_asyncio.fixture
async def async_client(
    db_session: AsyncSession, mock_redis: Any, mock_qdrant: Any
) -> AsyncGenerator[AsyncClient]:
    """
    Test client for FastAPI that overrides dependencies to use
    our test database, fakeredis, and mock qdrant.
    """
    app = create_app()

    async def override_get_db() -> AsyncGenerator[AsyncSession]:
        yield db_session

    async def override_get_redis() -> Any:
        return mock_redis

    async def override_get_qdrant() -> Any:
        return mock_qdrant

    app.dependency_overrides[get_db_session] = override_get_db
    app.dependency_overrides[get_redis] = override_get_redis
    app.dependency_overrides[get_qdrant] = override_get_qdrant

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client
