"""
ADL (Activities of Daily Living) endpoints
GET /adl/events - List ADL events with filters
GET /adl/event/{event_id} - Get specific event details
"""
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from app.models.database import get_db
from app.models.schemas import ADLEventResponse, ADLEventDetail
from app.services.adl_service import ADLService

router = APIRouter()


@router.get("/adl/events", response_model=List[ADLEventResponse])
async def list_adl_events(
    from_time: Optional[datetime] = Query(None, description="Start time filter"),
    to_time: Optional[datetime] = Query(None, description="End time filter"),
    camera_id: Optional[int] = Query(None, description="Camera ID filter"),
    label: Optional[str] = Query(None, description="Activity label filter"),
    min_confidence: Optional[float] = Query(0.0, ge=0.0, le=1.0),
    severity: Optional[str] = Query(None, description="low|medium|high|critical"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
) -> List[ADLEventResponse]:
    """
    List ADL events with optional filters
    
    Query Parameters:
        - from_time: ISO8601 datetime (e.g., 2026-01-16T00:00:00Z)
        - to_time: ISO8601 datetime
        - camera_id: Filter by camera
        - label: eating, drinking, reading, sleeping, phone, fall, stroke_like
        - min_confidence: Minimum confidence score (0.0-1.0)
        - severity: Alert severity level
        - limit: Max results (default 100)
        - offset: Pagination offset
    
    Returns:
        [
            {
                "event_id": 123,
                "camera_id": 1,
                "person_id": null,
                "label": "fall",
                "confidence": 0.95,
                "severity": "critical",
                "start_time": "2026-01-16T10:25:30Z",
                "end_time": "2026-01-16T10:25:35Z",
                "duration_seconds": 5,
                "snapshot_url": "/media/snapshots/event_123.jpg"
            },
            ...
        ]
    """
    service = ADLService(db)
    events = service.get_events(
        from_time=from_time,
        to_time=to_time,
        camera_id=camera_id,
        label=label,
        min_confidence=min_confidence,
        severity=severity,
        limit=limit,
        offset=offset
    )
    return events


@router.get("/adl/event/{event_id}", response_model=ADLEventDetail)
async def get_adl_event(
    event_id: int,
    db: Session = Depends(get_db)
) -> ADLEventDetail:
    """
    Get detailed information about a specific ADL event
    
    Returns:
        {
            "event_id": 123,
            "camera_id": 1,
            "camera_location": "Living Room",
            "person_id": null,
            "label": "fall",
            "confidence": 0.95,
            "severity": "critical",
            "start_time": "2026-01-16T10:25:30Z",
            "end_time": "2026-01-16T10:25:35Z",
            "duration_seconds": 5,
            "bbox": {"x": 100, "y": 200, "w": 80, "h": 150},
            "keypoints": [...],  // 17 COCO keypoints
            "snapshot_url": "/media/snapshots/event_123.jpg",
            "clip_url": "/media/clips/event_123.mp4",
            "alert_sent": true,
            "created_at": "2026-01-16T10:25:35Z"
        }
    """
    service = ADLService(db)
    event = service.get_event_by_id(event_id)
    
    if not event:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found")
    
    return event
