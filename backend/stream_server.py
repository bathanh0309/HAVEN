"""
HAVEN - FastAPI Camera Stream Server with WebSocket
Supports ONVIF discovery and RTSP streaming for Tapo C210
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import asyncio
import json
import time
import logging
from typing import Dict, Optional
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Camera Configuration
CAMERA_CONFIG = {
    'ip': '10.0.14.14',
    'onvif_port': 2020,
    'rtsp_port': 554,  # Standard RTSP port discovered via ONVIF
    'username': 'bathanh0309',
    'password': 'bathanh0309'
}

# Build RTSP URLs
def get_rtsp_url(stream: str = 'stream1') -> str:
    """Generate RTSP URL for camera stream"""
    return f"rtsp://{CAMERA_CONFIG['username']}:{CAMERA_CONFIG['password']}@{CAMERA_CONFIG['ip']}:{CAMERA_CONFIG['rtsp_port']}/{stream}"

RTSP_URL_HD = get_rtsp_url('stream1')  # 1080p
RTSP_URL_SD = get_rtsp_url('stream2')  # 640x480

class StreamManager:
    """Manages camera streams and WebSocket connections"""
    
    def __init__(self):
        self.streams: Dict[str, cv2.VideoCapture] = {}
        self.active_connections: list[WebSocket] = []
        self.running = False
        
    def get_stream(self, cam_id: str = 'main') -> Optional[cv2.VideoCapture]:
        """Get or create video capture for camera"""
        if cam_id not in self.streams:
            url = RTSP_URL_SD  # Use SD for better performance
            logger.info(f"Connecting to camera: {url}")
            cap = cv2.VideoCapture(url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if cap.isOpened():
                self.streams[cam_id] = cap
                logger.info(f"Camera {cam_id} connected successfully")
            else:
                logger.error(f"Failed to connect to camera {cam_id}")
                return None
                
        return self.streams.get(cam_id)
    
    async def connect(self, websocket: WebSocket):
        """Add new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass
                
    def release_all(self):
        """Release all camera streams"""
        for cap in self.streams.values():
            cap.release()
        self.streams.clear()

# Global stream manager
manager = StreamManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("HAVEN Camera Server starting...")
    yield
    logger.info("Shutting down...")
    manager.release_all()

# Create FastAPI app
app = FastAPI(
    title="HAVEN Camera Stream",
    description="Real-time camera streaming with WebSocket support",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== HTTP Endpoints ====================

@app.get("/")
async def root():
    """API info"""
    return {
        "service": "HAVEN Camera Stream",
        "camera": CAMERA_CONFIG['ip'],
        "endpoints": {
            "video_feed": "/video_feed",
            "websocket": "/ws/stream",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Health check"""
    cap = manager.get_stream('health_check')
    status = "connected" if cap and cap.isOpened() else "disconnected"
    return {
        "status": "ok",
        "camera": CAMERA_CONFIG['ip'],
        "stream_status": status,
        "active_connections": len(manager.active_connections)
    }

def generate_frames():
    """Generator for MJPEG streaming (fallback method)"""
    cap = manager.get_stream('mjpeg')
    
    while True:
        if not cap or not cap.isOpened():
            cap = manager.get_stream('mjpeg')
            time.sleep(0.1)
            continue
            
        success, frame = cap.read()
        if not success:
            continue
            
        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    """MJPEG streaming endpoint (HTTP fallback)"""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ==================== WebSocket Endpoints ====================

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time video streaming"""
    await manager.connect(websocket)
    cap = manager.get_stream('ws')
    
    try:
        while True:
            if not cap or not cap.isOpened():
                await websocket.send_json({"error": "Camera not connected"})
                await asyncio.sleep(1)
                cap = manager.get_stream('ws')
                continue
            
            success, frame = cap.read()
            if not success:
                await asyncio.sleep(0.01)
                continue
            
            # Encode frame as base64 JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                import base64
                frame_data = base64.b64encode(buffer).decode('utf-8')
                await websocket.send_json({
                    "type": "frame",
                    "data": frame_data,
                    "timestamp": time.time()
                })
            
            await asyncio.sleep(0.033)  # ~30 FPS
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """WebSocket endpoint for ADL events (future AI integration)"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Placeholder for ADL events
            # Will be populated when AI models are integrated
            await asyncio.sleep(1)
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": time.time()
            })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 50)
    logger.info("HAVEN Camera Stream Server")
    logger.info("=" * 50)
    logger.info(f"Camera IP: {CAMERA_CONFIG['ip']}")
    logger.info(f"RTSP URL (HD): {RTSP_URL_HD}")
    logger.info(f"RTSP URL (SD): {RTSP_URL_SD}")
    logger.info("")
    logger.info("Endpoints:")
    logger.info("  HTTP:      http://localhost:8000")
    logger.info("  Video:     http://localhost:8000/video_feed")
    logger.info("  WebSocket: ws://localhost:8000/ws/stream")
    logger.info("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
