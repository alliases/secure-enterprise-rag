# File: app/api/router.py
# Purpose: Main API router aggregator.

from fastapi import APIRouter

from app.api.endpoints import auth, health, ingest

api_router = APIRouter()

api_router.include_router(health.router, tags=["Monitoring"])
# Connect the authentication router
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
# Connect the document ingestion router
api_router.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])
