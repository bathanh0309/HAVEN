"""
HAVEN Backend - Main Entry Point
=================================
FastAPI application with secure camera streaming.

Run with: python src/main.py
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from .config.camera_config import load_camera_config
from .core.stream_manager import get_stream_manager
from .api.routes import router, init_routes

# ==================
# Logging Configuration
# ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("haven.log")
    ]
)
logger = logging.getLogger(__name__)

# ==================
# Global State
# ==================
stream_manager = None
camera_config = None


# ==================
# Lifecycle Management
# ==================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.
    
    Startup:
    1. Load camera config from .env
    2. Initialize StreamManager
    3. Start background capture thread
    
    Shutdown:
    1. Stop StreamManager
    2. Release camera resources
    """
    global stream_manager, camera_config
    
    # ==================
    # STARTUP
    # ==================
    logger.info("=" * 60)
    logger.info("HAVEN Backend Starting...")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        logger.info("Loading camera configuration...")
        camera_config = load_camera_config()
        
        if camera_config.DEBUG_MODE:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")
        
        # Initialize StreamManager
        logger.info("Initializing StreamManager...")
        stream_manager = get_stream_manager(camera_config)
        
        # Initialize API routes with dependencies
        init_routes(stream_manager, camera_config)
        
        # Start capture thread
        logger.info("Starting camera capture...")
        stream_manager.start()
        
        logger.info("=" * 60)
        logger.info("HAVEN Backend Ready!")
        logger.info(f"Camera: {camera_config.CAMERA_NAME}")
        logger.info(f"Default stream: {camera_config.DEFAULT_STREAM}")
        logger.info(f"Access at: http://localhost:8000")
        logger.info("=" * 60)
        
        yield  # Application runs here
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        logger.error("Check that:")
        logger.error("   1. .env exists at project root and is valid")
        logger.error("   2. Camera is powered on and connected")
        logger.error("   3. RTSP is enabled in Tapo app")
        logger.error("   4. Laptop and camera are on same network")
        raise
    
    # ==================
    # SHUTDOWN
    # ==================
    logger.info("=" * 60)
    logger.info("HAVEN Backend Shutting Down...")
    logger.info("=" * 60)
    
    if stream_manager:
        logger.info("Stopping StreamManager...")
        stream_manager.stop()
    
    logger.info("Shutdown complete")
    logger.info("=" * 60)


# ==================
# FastAPI Application
# ==================
app = FastAPI(
    title="HAVEN Camera API",
    description="Secure RTSP camera streaming with HD/SD support",
    version="1.0.0",
    lifespan=lifespan
)

# ==================
# CORS Configuration
# ==================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://localhost:8090",
        "http://localhost:8091",
        "http://localhost:5500",  # Live Server
        "http://127.0.0.1:8000",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================
# Include API Routes
# ==================
app.include_router(router)

# ==================
# Static Files (Frontend)
# ==================
frontend_path = Path(__file__).parent.parent.parent / "frontend" / "public"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
    logger.info(f"[DIR] Serving frontend from: {frontend_path}")


# ==================
# Root Endpoint (Serve Frontend)
# ==================
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve frontend HTML"""
    index_file = frontend_path / "index.html"
    
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding='utf-8'), status_code=200)
    else:
        return HTMLResponse(
            content="""
            <html>
                <head><title>HAVEN Camera</title></head>
                <body>
                    <h1>HAVEN Camera Backend</h1>
                    <p>Backend is running</p>
                    <p>Frontend not found. Check frontend/public/ directory.</p>
                    <ul>
                        <li><a href="/docs">API Documentation</a></li>
                        <li><a href="/api/health">Health Check</a></li>
                        <li><a href="/video_feed">MJPEG Stream</a></li>
                    </ul>
                </body>
            </html>
            """,
            status_code=200
        )


# ==================
# Health Check (Simple)
# ==================
@app.get("/ping")
async def ping():
    """Simple ping endpoint"""
    return {"status": "ok", "message": "pong"}


# ==================
# Development Entry Point
# ==================
if __name__ == "__main__":
    import uvicorn
    
    # Check .env exists at PROJECT ROOT
    # main.py is in backend/src, so we go up 3 levels to get to root
    project_root = Path(__file__).resolve().parent.parent.parent
    env_file = project_root / ".env"
    
    if not env_file.exists():
        logger.error("=" * 60)
        logger.error("ERROR: .env file not found at project root!")
        logger.error(f"Expected at: {env_file}")
        logger.error("=" * 60)
        logger.error("Setup steps:")
        logger.error("1. Copy .env.example to .env:")
        logger.error("   copy .env.example .env")
        logger.error("")
        logger.error("2. Edit .env with your camera details")
        logger.error("")
        logger.error("3. Restart server")
        logger.error("=" * 60)
        sys.exit(1)
    
    # Run server
    uvicorn.run(
        "backend.src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )