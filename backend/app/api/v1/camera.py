"""
Camera management endpoints
GET /cameras - List all cameras
POST /cameras - Add new camera
PUT /cameras/{camera_id} - Update camera
DELETE /cameras/{camera_id} - Remove camera
GET /video/{camera_id} - MJPEG stream
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List

from app.models.database import get_db
from app.models.schemas import CameraResponse, CameraCreate, CameraUpdate
from app.services.camera_service import CameraService
from app.core.capture.mjpeg_streamer import MJPEGStreamer

router = APIRouter()


@router.get("/cameras", response_model=List[CameraResponse])
async def list_cameras(db: Session = Depends(get_db)) -> List[CameraResponse]:
    """
    List all registered cameras
    
    Returns:
        [
            {
                "camera_id": 1,
                "name": "Living Room",
                "rtsp_url": "rtsp://192.168.1.100:554/stream",
                "location": "living_room",
                "enabled": true,
                "status": "connected",
                "fps": 15,
                "resolution": "1280x720"
            },
            ...
        ]
    """
    service = CameraService(db)
    return service.list_cameras()


@router.post("/cameras", response_model=CameraResponse, status_code=201)
async def create_camera(
    camera: CameraCreate,
    db: Session = Depends(get_db)
) -> CameraResponse:
    """
    Register a new camera
    
    Body:
        {
            "name": "Kitchen Camera",
            "rtsp_url": "rtsp://192.168.1.101:554/stream",
            "location": "kitchen",
            "enabled": true
        }
    """
    service = CameraService(db)
    return service.create_camera(camera)


@router.get("/video/{camera_id}")
async def stream_video(camera_id: int):
    """
    MJPEG video stream endpoint
    
    Usage:
        <img src="/api/v1/video/1" />
    
    Returns: multipart/x-mixed-replace MJPEG stream
    """
    streamer = MJPEGStreamer.get_streamer(camera_id)
    
    if not streamer:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    return StreamingResponse(
        streamer.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
