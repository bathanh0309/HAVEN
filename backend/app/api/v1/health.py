"""
Health check endpoints
GET /health - System health status
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any
from datetime import datetime

from app.models.database import get_db
from app.core.capture.rtsp_handler import RTSPConnectionPool
from app.services.pipeline_service import PipelineService

router = APIRouter()


@router.get("/health")
async def health_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Health check endpoint
    
    Returns:
        {
            "status": "healthy" | "degraded" | "unhealthy",
            "timestamp": "2026-01-16T10:30:00Z",
            "components": {
                "database": {"status": "up", "latency_ms": 5},
                "rtsp_pool": {"status": "up", "active_connections": 2},
                "pipeline": {"status": "running", "cameras_active": 2}
            },
            "version": "1.0.0"
        }
    """
    components = {}
    overall_status = "healthy"
    
    # Check database
    try:
        start = datetime.utcnow()
        db.execute("SELECT 1")
        latency = (datetime.utcnow() - start).total_seconds() * 1000
        components["database"] = {"status": "up", "latency_ms": round(latency, 2)}
    except Exception as e:
        components["database"] = {"status": "down", "error": str(e)}
        overall_status = "unhealthy"
    
    # Check RTSP pool (pseudo-code)
    try:
        pool = RTSPConnectionPool.get_instance()
        components["rtsp_pool"] = {
            "status": "up",
            "active_connections": pool.active_count()
        }
    except Exception as e:
        components["rtsp_pool"] = {"status": "down", "error": str(e)}
        overall_status = "degraded"
    
    # Check pipeline service
    try:
        pipeline = PipelineService.get_instance()
        components["pipeline"] = {
            "status": "running" if pipeline.is_running() else "stopped",
            "cameras_active": pipeline.active_cameras_count()
        }
    except Exception as e:
        components["pipeline"] = {"status": "error", "error": str(e)}
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "components": components,
        "version": "1.0.0"
    }
