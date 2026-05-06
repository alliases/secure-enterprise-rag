# File: app/main.py
# Purpose: FastAPI application entry point and lifespan management.

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient
from redis.asyncio import from_url  # type: ignore[reportUnknownVariableType]
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.router import api_router
from app.config import get_settings
from app.db.session import create_engine, get_session_factory
from app.logging_config.setup import configure_logging, get_logger
from app.rate_limit import limiter
from app.vectorstore.qdrant_client import init_collection

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """
    Manages global application state and resources.
    Executes startup logic before the server starts receiving requests,
    and shutdown logic when the server is terminating.
    """
    # Startup Phase
    configure_logging()
    settings = get_settings()
    logger.info("Initializing Secure Enterprise RAG components...")

    # Initialize PostgreSQL Engine & Session Factory
    engine = create_engine(str(settings.postgres_dsn))
    app.state.session_factory = get_session_factory(engine)

    # Initialize Redis pool
    app.state.redis = from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=False,
    )

    # Initialize Qdrant Client
    app.state.qdrant = AsyncQdrantClient(
        host=settings.qdrant_host, port=settings.qdrant_port
    )

    # Pre-warm connection and ensure collection structure exists
    await init_collection(app.state.qdrant, "documents", 1536)

    yield  # Server is running and handling requests

    # Shutdown Phase
    logger.info("Shutting down components gracefully...")
    await engine.dispose()
    await app.state.redis.aclose()
    await app.state.qdrant.close()  # Added closing Qdrant connection


def create_app() -> FastAPI:
    """
    Application factory method.
    """
    app = FastAPI(
        title="Secure Enterprise RAG",
        description="RAG system with real-time PII masking and RBAC",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Register rate limiter and its specific exception handler
    app.state.limiter = limiter
    app.add_exception_handler(
        RateLimitExceeded,
        _rate_limit_exceeded_handler,  # type: ignore[reportArgumentType]
    )

    app.include_router(api_router)

    return app


app = create_app()
