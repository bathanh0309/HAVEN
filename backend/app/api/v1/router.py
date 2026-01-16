"""
Main API router - aggregates all v1 endpoints
"""
from fastapi import APIRouter
from .health import router as health_router
from .adl import router as adl_router
from .camera import router as camera_router
from .media import router as media_router
from .pipeline import router as pipeline_router
from .stats import router as stats_router

api_router = APIRouter(prefix="/api/v1")

# Include all sub-routers
api_router.include_router(health_router, tags=["Health"])
api_router.include_router(pipeline_router, tags=["Pipeline"])
api_router.include_router(adl_router, tags=["ADL Events"])
api_router.include_router(camera_router, tags=["Cameras"])
api_router.include_router(media_router, tags=["Media"])
api_router.include_router(stats_router, tags=["Statistics"])
