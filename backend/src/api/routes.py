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
import json

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
    Lấy các instance singleton của các phụ thuộc (dependencies).
    Cấu hình và các trình quản lý (StreamManager, AIEngine) chỉ được tải MỘT LẦN và lưu vào cache.
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
    """
    Kiểm tra sức khỏe hệ thống API.
    Trả về thông tin kết nối camera, FPS hiện tại và cấu hình.
    """
    stream, _, config = get_deps()
    stats = stream.get_stats()
    return {
        "status": "ok",
        "camera": config.CAMERA_NAME,
        "details": stats,
        "config": config.to_public_info()
    }


# ==================
# Stream Switch
# ==================
@router.post("/api/stream/switch")
async def switch_stream(stream: Literal["HD", "SD"] = Query(...)):
    """
    Chuyển đổi luồng video giữa HD và SD.
    Yêu cầu StreamManager thực hiện việc chuyển đổi kết nối RTSP.
    """
    stream_mgr, _, _ = get_deps()
    success = stream_mgr.switch_stream(stream)
    return {"success": success, "current_stream": stream}


# ==================
# MJPEG Video Feed (Fallback)
# ==================
@router.get("/video_feed")
async def video_feed(stream: Optional[str] = None, ai: bool = True):
    """
    Endpoint streaming MJPEG (dùng cho các trình duyệt/client cũ).
    Lưu ý: Endpoint này không hỗ trợ gửi metadata (box, confidence) tách biệt như WebSocket.
    """
    stream_mgr, ai_engine, config = get_deps()
    
    if stream:
        stream_mgr.switch_stream(stream.upper())

    def frame_gen():
        """Generator đồng bộ để tạo các frame MJPEG."""
        error_count = 0
        max_errors = 10
        
        while True:
            try:
                # Get latest frame from queue
                data = stream_mgr.get_latest_frame()
                
                if data is None:
                    time.sleep(0.01)
                    continue
                
                frame, meta = data
                error_count = 0  # Reset on success
                
                # AI Inference
                if ai and ai_engine is not None:
                    try:
                        # Process frame returns (annotated_frame, detections_list)
                        # MJPEG only cares about the image
                        frame, _ = ai_engine.process_frame(frame)
                    except Exception as e:
                        logger.warning(f"AI inference failed: {e}")
                
                # Encode JPEG
                ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
                if not ret:
                    continue
                
                # Yield MJPEG frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                
                # FPS limiting
                if config.FPS_LIMIT > 0:
                    time.sleep(1.0 / config.FPS_LIMIT)
                    
            except GeneratorExit:
                logger.info("MJPEG client disconnected")
                break
            except Exception as e:
                error_count += 1
                logger.error(f"MJPEG frame error: {e}")
                if error_count >= max_errors:
                    break
                time.sleep(0.1)

    return StreamingResponse(
        frame_gen(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ==================
# WebSocket Stream (Low Latency + Metadata)
# ==================
@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Endpoint streaming qua WebSocket.
    Gửi gói tin JSON chứa hình ảnh (Base64) VÀ Metadata (Detections, FPS, TS).
    Đây là phương thức chính để truyền dữ liệu cho Frontend mới.
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
                detections = []
                
                # AI Inference
                if ai_engine is not None:
                    try:
                        # Unpack tuple: (annotated_frame, detections_list)
                        frame, detections = ai_engine.process_frame(frame)
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
                        "ts": datetime.now().isoformat(),
                        "detections": detections, # List of {label, conf, box}
                        "roi_active": config.ROI_ENABLED,
                        "fps": meta.get('fps', 0)
                    }
                }
                
                # Send to client
                await websocket.send_json(payload)
                
                # FPS Limit enforced by asyncio sleep if needed, 
                # but 'await' on send_json acts as natural backpressure often.
                # Explicit limit:
                if config.FPS_LIMIT > 0:
                    await asyncio.sleep(1.0 / config.FPS_LIMIT)
                
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
# Route Initialization
# ==================
def init_routes(mgr, cfg):
    """
    Khởi tạo các singleton dependency từ main.py.
    """
    global _cached_config, _cached_stream_mgr, _cached_ai_engine
    _cached_config = cfg
    _cached_stream_mgr = mgr
    _cached_ai_engine = get_ai_engine(cfg)
    logger.info("Routes initialized with dependencies")