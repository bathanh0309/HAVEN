"""
WebSocket and REST API routes for multi-camera streaming.

Updated endpoints:
- /ws/stream           Legacy (defaults to cam1)
- /ws/stream/{camera_id}  New multi-camera endpoint
- GET /api/cameras     List all camera statuses
"""

import asyncio
import logging
import time
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse

from core.stream_hub import get_stream_hub
from core.stream_worker import FramePacket

logger = logging.getLogger(__name__)

router = APIRouter()

# Configuration (adjust as needed or move to config file)
FPS_LIMIT = 10  # Max FPS to send to WebSocket clients
DEFAULT_CAMERA_ID = "cam1"


@router.websocket("/ws/stream")
async def websocket_stream_legacy(websocket: WebSocket):
    """
    Legacy WebSocket endpoint (backward compatible).
    Defaults to cam1 if no camera_id specified.
    """
    await websocket.accept()
    logger.info(f"Client connected to legacy /ws/stream (routing to {DEFAULT_CAMERA_ID})")
    
    try:
        await _stream_camera_to_websocket(websocket, DEFAULT_CAMERA_ID)
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from /ws/stream")
    except Exception as e:
        logger.error(f"Error in legacy stream: {e}", exc_info=True)
    finally:
        try:
            await websocket.close()
        except:
            pass


@router.websocket("/ws/stream/{camera_id}")
async def websocket_stream_camera(websocket: WebSocket, camera_id: str):
    """
    Multi-camera WebSocket endpoint.
    Streams frames from specified camera_id.
    """
    await websocket.accept()
    logger.info(f"Client connected to /ws/stream/{camera_id}")
    
    try:
        await _stream_camera_to_websocket(websocket, camera_id)
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from {camera_id}")
    except Exception as e:
        logger.error(f"Error streaming {camera_id}: {e}", exc_info=True)
    finally:
        try:
            await websocket.close()
        except:
            pass


async def _stream_camera_to_websocket(websocket: WebSocket, camera_id: str):
    """
    Core streaming logic (shared by both endpoints).
    
    Reads latest packet from StreamWorker (non-destructive).
    No inference here - that happens in worker thread.
    Just sends pre-processed packets to client.
    """
    hub = get_stream_hub()
    worker = hub.get_worker(camera_id)
    
    if not worker:
        error_msg = f"Camera {camera_id} not found. Available: {hub.list_cameras()}"
        logger.error(error_msg)
        await websocket.send_json({
            "type": "error",
            "message": error_msg
        })
        return
    
    # Calculate frame interval for FPS limiting
    frame_interval = 1.0 / FPS_LIMIT if FPS_LIMIT > 0 else 0
    last_send_time = 0
    
    logger.info(f"Streaming {camera_id} to client (FPS limit: {FPS_LIMIT})")
    
    while True:
        try:
            # Get latest packet (non-destructive read)
            packet = worker.get_latest_packet()
            
            if packet is None:
                # Worker hasn't produced any frames yet
                await asyncio.sleep(0.05)
                continue
            
            # Apply FPS throttling (limit send rate to client)
            current_time = time.time()
            elapsed = current_time - last_send_time
            
            if elapsed < frame_interval:
                await asyncio.sleep(frame_interval - elapsed)
                continue
            
            last_send_time = time.time()
            
            # Build WebSocket payload
            payload = {
                "type": "frame",
                "data": packet.frame_base64,
                "metadata": {
                    **packet.metadata,
                    "detections": packet.detections,
                    "packet_timestamp": packet.timestamp,
                }
            }
            
            # Send to client
            await websocket.send_json(payload)
            
        except WebSocketDisconnect:
            raise  # Re-raise to trigger cleanup
        except Exception as e:
            logger.error(f"Error sending frame from {camera_id}: {e}")
            await asyncio.sleep(0.1)


@router.get("/api/cameras")
async def get_cameras():
    """
    REST endpoint: List all registered cameras with status.
    
    Returns:
        List of camera statuses with stats (FPS, uptime, etc.)
    """
    hub = get_stream_hub()
    statuses = hub.get_all_statuses()
    
    return JSONResponse(content={
        "cameras": statuses,
        "total_cameras": len(statuses),
    })


@router.get("/api/cameras/{camera_id}/status")
async def get_camera_status(camera_id: str):
    """
    REST endpoint: Get status for specific camera.
    
    Returns:
        Camera status dict or 404 if not found
    """
    hub = get_stream_hub()
    worker = hub.get_worker(camera_id)
    
    if not worker:
        raise HTTPException(
            status_code=404,
            detail=f"Camera {camera_id} not found. Available: {hub.list_cameras()}"
        )
    
    return JSONResponse(content=worker.get_status())


# Optional: Health check endpoint
@router.get("/api/health")
async def health_check():
    """Simple health check endpoint."""
    hub = get_stream_hub()
    statuses = hub.get_all_statuses()
    
    connected_cameras = sum(1 for s in statuses if s["connected"])
    
    return JSONResponse(content={
        "status": "healthy",
        "cameras_registered": len(statuses),
        "cameras_connected": connected_cameras,
    })

