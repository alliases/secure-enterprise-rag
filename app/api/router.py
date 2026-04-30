# File: app/api/router.py
# Purpose: Main API router aggregator.

from fastapi import APIRouter

from app.api.endpoints import auth, health, ingest, query

api_router = APIRouter()

api_router.include_router(health.router, tags=["Monitoring"])
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])
api_router.include_router(query.router, prefix="/query", tags=["Query"])
