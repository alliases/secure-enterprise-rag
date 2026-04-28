# File: app/main.py
# Purpose: FastAPI application entry point and lifespan management.

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.config import get_settings
from app.db.session import create_engine, get_session_factory
from app.logging_config.setup import configure_logging, get_logger

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

    # TODO: Initialize Redis pool and Qdrant client here

    yield  # Server is running and handling requests

    # Shutdown Phase
    logger.info("Shutting down components gracefully...")
    await engine.dispose()
    # TODO: Close Redis and Qdrant connections here


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

    app.include_router(api_router)

    return app


app = create_app()
