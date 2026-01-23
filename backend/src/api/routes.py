"""
HAVEN API Routes
================
FastAPI endpoints for camera streaming and control.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import asyncio
import base64
import time
import logging
from typing import Literal, Optional
from datetime import datetime

# Import dependencies
from ..core.stream_manager import StreamManager, get_stream_manager
from ..core.ai_engine import AIEngine, get_ai_engine
from ..config.camera_config import CameraSettings, load_camera_config

logger = logging.getLogger(__name__)
router = APIRouter()

# ==================
# Singleton Cache for Dependencies
# ==================
_cached_config = None
_cached_stream_mgr = None
_cached_ai_engine = None

def get_deps():
    """
    Get singleton instances of dependencies.
    Config and managers are loaded ONCE and cached.
    """
    global _cached_config, _cached_stream_mgr, _cached_ai_engine
    
    if _cached_config is None:
        _cached_config = load_camera_config()
    
    if _cached_stream_mgr is None:
        _cached_stream_mgr = get_stream_manager(_cached_config)
    
    if _cached_ai_engine is None:
        _cached_ai_engine = get_ai_engine(_cached_config)
    
    return _cached_stream_mgr, _cached_ai_engine, _cached_config


# ==================
# Health Check
# ==================
@router.get("/api/health")
async def health_check():
    stream, _, config = get_deps()
    stats = stream.get_stats()
    return {
        "status": "ok",
        "camera": config.CAMERA_NAME,
        "details": stats
    }


# ==================
# Stream Switch
# ==================
@router.post("/api/stream/switch")
async def switch_stream(stream: Literal["HD", "SD"] = Query(...)):
    stream_mgr, _, _ = get_deps()
    success = stream_mgr.switch_stream(stream)
    return {"success": success, "current_stream": stream}


# ==================
# MJPEG Video Feed (Fallback)
# ==================
@router.get("/video_feed")
async def video_feed(stream: Optional[str] = None, ai: bool = True):
    """
    MJPEG streaming endpoint.
    
    Query params:
    - stream: "HD" or "SD" (optional)
    - ai: Enable AI inference (default True, set False for raw stream)
    """
    stream_mgr, ai_engine, config = get_deps()
    
    if stream:
        stream_mgr.switch_stream(stream.upper())

    def frame_gen():
        """Synchronous generator for MJPEG frames."""
        error_count = 0
        max_errors = 10
        
        while True:
            try:
                # Get latest frame from queue
                data = stream_mgr.get_latest_frame()
                
                if data is None:
                    # No frame available, short sleep to avoid busy loop
                    time.sleep(0.01)
                    continue
                
                frame, meta = data
                error_count = 0  # Reset on success
                
                # AI Inference (optional, with fail-safe)
                if ai and ai_engine is not None:
                    try:
                        frame = ai_engine.process_frame(frame)
                    except Exception as e:
                        # AI failed, continue with raw frame
                        logger.warning(f"AI inference failed: {e}")
                
                # Encode JPEG
                ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
                if not ret:
                    continue
                
                # Yield MJPEG frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                
                # FPS limiting (if configured)
                if config.FPS_LIMIT > 0:
                    time.sleep(1.0 / config.FPS_LIMIT)
                    
            except GeneratorExit:
                # Client disconnected
                logger.info("MJPEG client disconnected")
                break
            except Exception as e:
                error_count += 1
                logger.error(f"MJPEG frame error: {e}")
                if error_count >= max_errors:
                    logger.error("Too many errors, stopping MJPEG stream")
                    break
                time.sleep(0.1)  # Brief pause before retry

    return StreamingResponse(
        frame_gen(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ==================
# WebSocket Stream (Low Latency)
# ==================
@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket streaming endpoint.
    Lower latency than MJPEG, sends base64 encoded frames.
    """
    await websocket.accept()
    stream_mgr, ai_engine, config = get_deps()
    logger.info("WebSocket client connected")
    
    error_count = 0
    max_errors = 10
    
    try:
        while True:
            try:
                # Get latest frame
                data = stream_mgr.get_latest_frame()
                
                if data is None:
                    await asyncio.sleep(0.01)
                    continue
                
                frame, meta = data
                error_count = 0
                
                # AI Inference (with fail-safe)
                if ai_engine is not None:
                    try:
                        frame = ai_engine.process_frame(frame)
                    except Exception as e:
                        logger.warning(f"AI inference failed: {e}")
                
                # Encode JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
                if not ret:
                    continue
                
                # Convert to base64
                b64_data = base64.b64encode(buffer).decode('utf-8')
                
                # Build payload
                payload = {
                    "type": "frame",
                    "data": b64_data,
                    "metadata": {
                        "fps": meta.get('fps', 0),
                        "w": meta.get('w', 0),
                        "h": meta.get('h', 0),
                        "ts": datetime.now().isoformat()
                    }
                }
                
                # Send to client
                await websocket.send_json(payload)
                
            except Exception as e:
                error_count += 1
                if error_count >= max_errors:
                    logger.error(f"Too many WebSocket errors: {e}")
                    break
                await asyncio.sleep(0.1)
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass


# ==================
# Route Initialization (called from main.py)
# ==================
def init_routes(mgr, cfg):
    """
    Initialize routes with pre-configured dependencies.
    Called during app startup to ensure singletons are ready.
    """
    global _cached_config, _cached_stream_mgr, _cached_ai_engine
    
    _cached_config = cfg
    _cached_stream_mgr = mgr
    
    # Lazy load AI engine (first frame will trigger load)
    # This avoids blocking startup if model is large
    _cached_ai_engine = get_ai_engine(cfg)
    
    logger.info("Routes initialized with dependencies")