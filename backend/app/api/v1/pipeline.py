"""
Pipeline control endpoints
POST /pipeline/start - Start processing pipeline
POST /pipeline/stop - Stop processing pipeline
GET /pipeline/status - Get pipeline status
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any

from app.models.database import get_db
from app.services.pipeline_service import PipelineService
from app.models.schemas import PipelineStatusResponse

router = APIRouter()


@router.post("/pipeline/start")
async def start_pipeline(
    camera_ids: list[int] = None,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Start the ADL processing pipeline
    
    Body (optional):
        {
            "camera_ids": [1, 2]  // If null, start all enabled cameras
        }
    
    Returns:
        {
            "status": "started",
            "cameras_started": [1, 2],
            "message": "Pipeline started successfully"
        }
    """
    service = PipelineService.get_instance()
    
    try:
        started_cameras = service.start(camera_ids=camera_ids, db=db)
        return {
            "status": "started",
            "cameras_started": started_cameras,
            "message": f"Pipeline started for {len(started_cameras)} camera(s)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start pipeline: {str(e)}")


@router.post("/pipeline/stop")
async def stop_pipeline(
    camera_ids: list[int] = None
) -> Dict[str, Any]:
    """
    Stop the ADL processing pipeline
    
    Body (optional):
        {
            "camera_ids": [1]  // If null, stop all cameras
        }
    
    Returns:
        {
            "status": "stopped",
            "cameras_stopped": [1, 2],
            "message": "Pipeline stopped successfully"
        }
    """
    service = PipelineService.get_instance()
    
    try:
        stopped_cameras = service.stop(camera_ids=camera_ids)
        return {
            "status": "stopped",
            "cameras_stopped": stopped_cameras,
            "message": f"Pipeline stopped for {len(stopped_cameras)} camera(s)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop pipeline: {str(e)}")


@router.get("/pipeline/status", response_model=PipelineStatusResponse)
async def get_pipeline_status() -> PipelineStatusResponse:
    """
    Get current pipeline status
    
    Returns:
        {
            "is_running": true,
            "cameras": [
                {
                    "camera_id": 1,
                    "status": "running",
                    "fps": 15.2,
                    "queue_size": 3,
                    "last_frame_at": "2026-01-16T10:30:45Z"
                }
            ],
            "total_events_today": 42,
            "uptime_seconds": 3600
        }
    """
    service = PipelineService.get_instance()
    return service.get_status()
