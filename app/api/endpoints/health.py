# File: app/api/endpoints/health.py
# Purpose: Health check endpoint for Docker and monitoring systems.

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health", status_code=200)
async def health_check(request: Request) -> dict[str, str]:
    """
    Returns the overall health status of the API and its dependencies.
    TODO: Add ping checks for PostgreSQL, Redis, and Qdrant later.
    """
    return {"status": "healthy", "version": "0.1.0"}
