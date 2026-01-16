"""
Statistics and analytics endpoints
GET /adl/stats - Get ADL statistics by day/range
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from datetime import date, datetime
from typing import Dict, Any

from app.models.database import get_db
from app.services.adl_service import ADLService

router = APIRouter()


@router.get("/adl/stats")
async def get_adl_stats(
    day: date = Query(default_factory=date.today, description="Target date (YYYY-MM-DD)"),
    camera_id: int = Query(None, description="Filter by camera"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get ADL statistics for a specific day
    
    Query Parameters:
        - day: Target date (default: today)
        - camera_id: Optional camera filter
    
    Returns:
        {
            "date": "2026-01-16",
            "camera_id": null,
            "total_events": 42,
            "events_by_label": {
                "eating": 5,
                "drinking": 8,
                "reading": 12,
                "sleeping": 3,
                "phone": 10,
                "fall": 2,
                "stroke_like": 0
            },
            "events_by_severity": {
                "low": 35,
                "medium": 5,
                "high": 1,
                "critical": 1
            },
            "hourly_distribution": {
                "00": 0, "01": 0, ..., "10": 8, ..., "23": 1
            },
            "average_confidence": 0.87,
            "critical_alerts_sent": 1
        }
    """
    service = ADLService(db)
    stats = service.get_daily_stats(day=day, camera_id=camera_id)
    return stats


@router.get("/adl/stats/range")
async def get_adl_stats_range(
    from_date: date = Query(..., description="Start date"),
    to_date: date = Query(..., description="End date"),
    camera_id: int = Query(None),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get ADL statistics for a date range
    
    Returns:
        {
            "from_date": "2026-01-10",
            "to_date": "2026-01-16",
            "total_events": 294,
            "daily_breakdown": [
                {"date": "2026-01-10", "total": 38, "by_label": {...}},
                {"date": "2026-01-11", "total": 45, "by_label": {...}},
                ...
            ],
            "top_activities": [
                {"label": "phone", "count": 78},
                {"label": "reading", "count": 65},
                ...
            ]
        }
    """
    service = ADLService(db)
    stats = service.get_range_stats(
        from_date=from_date,
        to_date=to_date,
        camera_id=camera_id
    )
    return stats
