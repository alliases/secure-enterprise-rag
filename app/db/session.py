# File: app/db/session.py
# Purpose: Async SQLAlchemy engine and session factory configuration.
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def create_engine(dsn: str) -> AsyncEngine:
    """
    Creates an asynchronous SQLAlchemy engine.
    Connection pool parameters are tuned for a high-concurrency microservice.
    """
    return create_async_engine(
        dsn,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,  # Protects against dropped connections
        echo=False,  # Must be False in production to avoid logging sensitive data
    )


def get_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """
    Returns an async session factory bound to the provided engine.
    expire_on_commit=False is crucial for async workflows.
    """
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
