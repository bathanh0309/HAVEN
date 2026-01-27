"""
Unified Configuration for HAVEN Project
========================================
This config supports both RTSP camera streams and video file processing.
All AI detection features are configured here.
"""

import os
from pathlib import Path
from typing import Literal

# ================================
# BASE PATHS
# ================================
BASE_DIR = Path(__file__).parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
MODELS_DIR = BACKEND_DIR / "models"
OUTPUTS_DIR = BACKEND_DIR / "outputs"
DATABASE_DIR = BACKEND_DIR / "database"
DATA_DIR = BACKEND_DIR / "data"

# ================================
# SOURCE CONFIGURATION
# ================================
SOURCE_TYPE: Literal["rtsp", "video"] = os.getenv("SOURCE_TYPE", "rtsp")

# RTSP Camera Configuration
RTSP_CONFIG = {
    "camera_name": os.getenv("CAMERA_NAME", "tapo_c210"),
    "camera_ip": os.getenv("CAMERA_IP", "192.168.1.100"),
    "camera_port": int(os.getenv("CAMERA_PORT", "554")),
    "camera_user": os.getenv("CAMERA_USER", "admin"),
    "camera_password": os.getenv("CAMERA_PASSWORD", "password"),
    "stream_hd": os.getenv("RTSP_STREAM_HD", "stream1"),
    "stream_sd": os.getenv("RTSP_STREAM_SD", "stream2"),
    "default_stream": os.getenv("DEFAULT_STREAM", "SD"),
}

# Video File Configuration
VIDEO_CONFIG = {
    "video_path": os.getenv("VIDEO_PATH", str(DATA_DIR / "video/walking.mp4")),
    "loop_video": bool(os.getenv("LOOP_VIDEO", "False") == "True"),
}

# ================================
# DETECTION FEATURES (Feature Flags)
# ================================
FEATURE_FLAGS = {
    "enable_pose_detection": True,       # ADL pose estimation
    "enable_object_detection": False,    # Object/item detection
}

# ================================
# MODEL PATHS
# ================================
MODEL_PATHS = {
    # Pose Detection (ADL)
    "pose_model": str(MODELS_DIR / "yolo11n-pose.pt"),
    
    # Object Detection (placeholder - to be implemented)
    "object_detector": str(MODELS_DIR / "object_detector.pt"),
}

# ================================
# PERFORMANCE PARAMETERS
# ================================
PERFORMANCE_PARAMS = {
    # General
    "conf_threshold": float(os.getenv("CONF_THRES", "0.25")),
    "iou_threshold": float(os.getenv("IOU_THRES", "0.7")),
    "img_size": int(os.getenv("IMG_SIZE", "640")),
    "max_det": int(os.getenv("MAX_DET", "10")),
    
    # Streaming
    "fps_limit": int(os.getenv("FPS_LIMIT", "15")),
    "frame_buffer_size": int(os.getenv("FRAME_BUFFER_SIZE", "1")),
    "jpeg_quality": int(os.getenv("JPEG_QUALITY", "80")),
    "resize_width": int(os.getenv("RESIZE_WIDTH", "0")),
    
    # Connection
    "stream_timeout": int(os.getenv("STREAM_TIMEOUT_SECONDS", "30")),
    "reconnect_backoff": int(os.getenv("RECONNECT_BACKOFF_SECONDS", "5")),
    "max_reconnect_attempts": int(os.getenv("MAX_RECONNECT_ATTEMPTS", "10")),
}

# ================================
# OBJECT DETECTION CONFIG
# ================================
OBJECT_DETECTION_CONFIG = {
    "object_classes": [
        "chair", "bed", "table", "phone", "remote", "cup", "bottle",
        "keyboard", "mouse", "book", "clock", "vase"
    ],
    "class_thresholds": {  # Per-class confidence thresholds
        "phone": 0.4,
        "default": 0.3,
    }
}

# ================================
# OUTPUT CONFIGURATION
# ================================
OUTPUT_CONFIG = {
    "save_video": bool(os.getenv("SAVE_OUTPUT_VIDEO", "False") == "True"),
    "output_dir": str(OUTPUTS_DIR),
    "log_events": True,
    "event_log_path": str(OUTPUTS_DIR / "events.csv"),
}

# ================================
# ADVANCED OPTIONS
# ================================
DEBUG_MODE = bool(os.getenv("DEBUG_MODE", "False") == "True")
WS_PING_INTERVAL = int(os.getenv("WS_PING_INTERVAL", "30"))

# ================================
# HELPER FUNCTIONS
# ================================
def get_rtsp_url(stream: Literal["HD", "SD"] = "SD") -> str:
    """Generate RTSP URL based on configuration."""
    stream_path = RTSP_CONFIG["stream_hd"] if stream == "HD" else RTSP_CONFIG["stream_sd"]
    return (
        f"rtsp://{RTSP_CONFIG['camera_user']}:{RTSP_CONFIG['camera_password']}"
        f"@{RTSP_CONFIG['camera_ip']}:{RTSP_CONFIG['camera_port']}/{stream_path}"
    )

def is_feature_enabled(feature: str) -> bool:
    """Check if a detection feature is enabled."""
    return FEATURE_FLAGS.get(f"enable_{feature}", False)

def get_model_path(model_name: str) -> str:
    """Get absolute path to a model file."""
    return MODEL_PATHS.get(model_name, "")
