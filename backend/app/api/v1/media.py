"""
Media serving endpoints
GET /media/{path} - Serve snapshots and clips
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from app.config import settings

router = APIRouter()


@router.get("/media/{media_type}/{filename}")
async def serve_media(media_type: str, filename: str):
    """
    Serve media files (snapshots, clips)
    
    Examples:
        GET /media/snapshots/event_123.jpg
        GET /media/clips/event_123.mp4
    
    Returns: Image or video file
    """
    # Validate media type
    if media_type not in ["snapshots", "clips"]:
        raise HTTPException(status_code=400, detail="Invalid media type")
    
    # Build file path (pseudo)
    file_path = Path(settings.DATA_DIR) / media_type / filename
    
    # Security: prevent path traversal
    if not file_path.resolve().is_relative_to(Path(settings.DATA_DIR).resolve()):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine media type
    content_type = "image/jpeg" if media_type == "snapshots" else "video/mp4"
    
    return FileResponse(file_path, media_type=content_type)
