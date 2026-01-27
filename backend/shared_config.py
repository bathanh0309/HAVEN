"""
HAVEN - Shared Pose + ADL Configuration
========================================
File config dùng chung cho tất cả các test scripts:
- test-video/pose-adl.py (test với video file)
- test-rtsp-pose/pose-adl.py (test với RTSP camera)

Import trong các script:
    from backend.shared_config import *
    hoặc
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from shared_config import *
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# PATHS
# ============================================================
CONFIG_DIR = Path(__file__).parent          # backend/
PROJECT_ROOT = CONFIG_DIR.parent            # HAVEN/
MODELS_DIR = CONFIG_DIR / "models"
OUTPUTS_DIR = CONFIG_DIR / "outputs"

def get_next_output_path(base_name="pose-adl", ext=".gif"):
    """
    Tự động sinh tên file tiếp theo: pose-adl-verX.gif
    Kiểm tra trong thư mục output để tìm số lớn nhất hiện tại.
    """
    if not OUTPUTS_DIR.exists():
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Tìm version cao nhất hiện tại
    max_ver = 0
    for file in OUTPUTS_DIR.glob(f"{base_name}-ver*{ext}"):
        try:
            # Lấy phần số sau "-ver"
            parts = file.stem.split("-ver")
            if len(parts) > 1:
                ver = int(parts[-1])
                if ver > max_ver:
                    max_ver = ver
        except ValueError:
            continue
            
    # Tăng version lên 1
    next_ver = max_ver + 1
    return str(OUTPUTS_DIR / f"{base_name}-ver{next_ver}{ext}")

# ============================================================
# MODEL SETTINGS
# ============================================================
# Model path - tất cả scripts đều load từ backend/models/
DEFAULT_MODEL_PATH = str(MODELS_DIR / "yolo11s-pose.pt")
MODEL_PATH = os.getenv("AI_MODEL_PATH", DEFAULT_MODEL_PATH)
CONF_THRES = float(os.getenv("AI_CONF_THRES", "0.25"))
IOU_THRES = float(os.getenv("AI_IOU_THRES", "0.45"))
IMG_SIZE = int(os.getenv("AI_IMG_SIZE", "640"))

# ============================================================
# RTSP CAMERA SETTINGS (for test-rtsp-pose)
# ============================================================
RTSP_CONFIG = {
    "camera_ip": os.getenv("CAMERA_IP", "192.168.1.100"),
    "camera_port": int(os.getenv("CAMERA_PORT", "554")),
    "camera_user": os.getenv("CAMERA_USER", "admin"),
    "camera_password": os.getenv("CAMERA_PASSWORD", "password"),
    "stream_hd": os.getenv("RTSP_STREAM_HD", "stream1"),
    "stream_sd": os.getenv("RTSP_STREAM_SD", "stream2"),
    "default_stream": os.getenv("DEFAULT_STREAM", "SD"),
}

def get_rtsp_url(stream: str = "SD") -> str:
    """Generate RTSP URL based on configuration."""
    stream_path = RTSP_CONFIG["stream_hd"] if stream == "HD" else RTSP_CONFIG["stream_sd"]
    return (
        f"rtsp://{RTSP_CONFIG['camera_user']}:{RTSP_CONFIG['camera_password']}"
        f"@{RTSP_CONFIG['camera_ip']}:{RTSP_CONFIG['camera_port']}/{stream_path}"
    )

# ============================================================
# CHỈ SỐ KEYPOINT (ĐỊNH DẠNG COCO - 17 điểm)
# ============================================================
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

# ============================================================
# MÀU SẮC SKELETON (BGR) - CHÍNH XÁC 100%
# ============================================================
SKELETON_COLORS = {
    "head":      (0, 0, 255),      # Red
    "torso":     (255, 0, 255),    # Magenta (rất nổi, khác hẳn đỏ)
    "upper_arm": (255, 255, 0),    # Cyan
    "lower_arm": (255, 0, 0),      # Blue (đậm, tách khỏi cyan)
    "upper_leg": (0, 165, 255),    # Orange
    "lower_leg": (0, 255, 128),    # Spring Green (thay vì vàng để không “dính” cam/cyan)
}

# ============================================================
# MÀU SẮC CHO CÁC TƯ THẾ (BGR)
# ============================================================
POSTURE_COLORS = {
    "STANDING": (0, 255, 0),       # Green (OK)
    "WALKING":  (255, 255, 0),     # Cyan (khác STANDING)
    "SITTING":  (0, 165, 255),     # Orange
    "LAYING":   (0, 0, 255),       # Red
    "UNKNOWN":  (128, 128, 128),   # Gray
}

# ============================================================
# CÁC KẾT NỐI XƯƠNG - PHÂN CHIA CHÍNH XÁC
# ============================================================
SKELETON_CONNECTIONS = [
    # Đầu (Red)
    (KP_NOSE, KP_LEFT_EYE, "head"),
    (KP_NOSE, KP_RIGHT_EYE, "head"),
    (KP_LEFT_EYE, KP_LEFT_EAR, "head"),
    (KP_RIGHT_EYE, KP_RIGHT_EAR, "head"),
    
    # Thân (Pink)
    (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, "torso"),
    (KP_LEFT_SHOULDER, KP_LEFT_HIP, "torso"),
    (KP_RIGHT_SHOULDER, KP_RIGHT_HIP, "torso"),
    (KP_LEFT_HIP, KP_RIGHT_HIP, "torso"),
    
    # Tay trên (Green) - vai → cùi chỏ
    (KP_LEFT_SHOULDER, KP_LEFT_ELBOW, "upper_arm"),
    (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW, "upper_arm"),
    
    # Tay dưới/Bàn tay (Dark Green) - cùi chỏ → cổ tay
    (KP_LEFT_ELBOW, KP_LEFT_WRIST, "lower_arm"),
    (KP_RIGHT_ELBOW, KP_RIGHT_WRIST, "lower_arm"),
    
    # Chân trên (Orange) - hông → đầu gối
    (KP_LEFT_HIP, KP_LEFT_KNEE, "upper_leg"),
    (KP_RIGHT_HIP, KP_RIGHT_KNEE, "upper_leg"),
    
    # Chân dưới/Bàn chân (Yellow) - đầu gối → mắt cá
    (KP_LEFT_KNEE, KP_LEFT_ANKLE, "lower_leg"),
    (KP_RIGHT_KNEE, KP_RIGHT_ANKLE, "lower_leg"),
]

# ============================================================
# NGƯỠNG PHÁT HIỆN KEYPOINT
# ============================================================
KEYPOINT_CONF_THRESHOLD = 0.3

# ============================================================
# TRACKING SETTINGS
# ============================================================
TRACKER_IOU_THRESHOLD = 0.3
TRACKER_MAX_AGE = 30
POSITION_HISTORY_MAXLEN = 30
POSTURE_HISTORY_MAXLEN = 10
EVENT_HISTORY_MAXLEN = 5

# ============================================================
# POSTURE CLASSIFICATION THRESHOLDS
# ============================================================
TORSO_ANGLE_LAYING_THRESHOLD = 50
ASPECT_RATIO_LAYING_THRESHOLD = 1.2
KNEE_ANGLE_SITTING_THRESHOLD = 130
MOVEMENT_THRESHOLD_RATIO = 0.08
MOVEMENT_WALKING_MULTIPLIER = 1.5
DEFAULT_KNEE_ANGLE = 180

# ============================================================
# EVENT DETECTION SETTINGS
# ============================================================
HAND_RAISE_FRAMES_THRESHOLD = 15
POSTURE_VOTING_MIN_FRAMES = 5
ASSUMED_FPS = 30

# ============================================================
# DISPLAY SETTINGS
# ============================================================
WINDOW_NAME = "HAVEN Pose + ADL"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Font settings
FONT_FACE = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LABEL = 0.6
FONT_SCALE_INFO = 0.8
FONT_SCALE_EVENT = 0.5
FONT_THICKNESS = 2

# Skeleton drawing
SKELETON_LINE_THICKNESS = 2
SKELETON_KEYPOINT_RADIUS = 4

# BBox drawing
BBOX_THICKNESS = 2

# ============================================================
# GIF EXPORT SETTINGS
# ============================================================
GIF_RESIZE_RATIO = 0.5
GIF_DURATION_MS = 66
GIF_LOOP = 0
GIF_OPTIMIZE = True

# ============================================================
# UI TEXT MESSAGES
# ============================================================
CONTROLS_TEXT = {
    "quit": "Q = Thoát",
    "pause": "Space = Tạm dừng/Tiếp tục",
    "loop": "L = Bật/Tắt lặp video",
    "gif": "G = Bắt đầu/Dừng ghi GIF",
}

EVENT_NAMES = {
    "fall_down": "FALL_DOWN",
    "sitting_down": "SITTING_DOWN",
    "standing_up": "STANDING_UP",
    "left_hand_raised": "LEFT_HAND_RAISED",
    "right_hand_raised": "RIGHT_HAND_RAISED",
    "zone_intrusion": "ZONE_INTRUSION",
    "dangerous_object": "DANGEROUS_OBJECT",
}

# ============================================================
# ZONE INTRUSION SETTINGS
# ============================================================
ZONE_ENABLED = bool(os.getenv("ZONE_ENABLED", "True") == "True")
# Format: list of dicts with name and coords (x1, y1, x2, y2) as percentages (0.0-1.0)
FORBIDDEN_ZONES = [
    {"name": "Kitchen", "coords": (0.6, 0.0, 1.0, 0.5)},  # Top-right area
]
ZONE_COLOR = (0, 0, 255)  # Red for forbidden zones

# ============================================================
# DANGEROUS OBJECT DETECTION SETTINGS
# ============================================================
OBJECT_DETECTION_ENABLED = bool(os.getenv("OBJECT_DETECTION_ENABLED", "True") == "True")
OBJECT_MODEL_PATH = str(MODELS_DIR / "yolo11s.pt")  # COCO pretrained
OBJECT_CONF_THRES = 0.3

# COCO class IDs for dangerous objects
# Note: gun is NOT in COCO dataset, would need custom model
DANGEROUS_OBJECT_CLASSES = {
    "cell phone": 67,   # COCO class 67
    "knife": 43,        # COCO class 43
    # "gun": N/A - not in COCO, needs custom training
}
DANGEROUS_OBJECT_COLOR = (0, 0, 255)  # Red for dangerous objects

