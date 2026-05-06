# File: app/api/endpoints/health.py
# Purpose: Health check endpoint for Docker and monitoring systems.

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.logging_config.setup import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check(request: Request) -> JSONResponse:
    """
    Performs active ping checks against PostgreSQL, Redis, and Qdrant.
    Returns HTTP 503 if any critical service is degraded.
    """
    checks: dict[str, Any] = {}
    overall = "healthy"

    # 1. PostgreSQL Check
    try:
        session_factory = request.app.state.session_factory
        async with session_factory() as session:
            await session.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as e:
        logger.error("Health check failed for PostgreSQL", error=str(e))
        checks["postgres"] = "error"
        overall = "degraded"

    # 2. Redis Check
    try:
        await request.app.state.redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        logger.error("Health check failed for Redis", error=str(e))
        checks["redis"] = "error"
        overall = "degraded"

    # 3. Qdrant Check
    try:
        await request.app.state.qdrant.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:
        logger.error("Health check failed for Qdrant", error=str(e))
        checks["qdrant"] = "error"
        overall = "degraded"

    status_code = 200 if overall == "healthy" else 503

    return JSONResponse(
        content={"status": overall, "version": "0.1.0", "services": checks},
        status_code=status_code,
    )
