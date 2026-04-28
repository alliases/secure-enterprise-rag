# File: app/api/router.py
# Purpose: Main API router aggregator.

from fastapi import APIRouter

from app.api.endpoints import health

api_router = APIRouter()

api_router.include_router(health.router, tags=["Monitoring"])
