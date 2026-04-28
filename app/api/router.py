# File: app/api/router.py
# Purpose: Main API router aggregator.

from fastapi import APIRouter

from app.api.endpoints import auth, health

api_router = APIRouter()

api_router.include_router(health.router, tags=["Monitoring"])
# Connect the authentication router
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
