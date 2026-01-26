"""
HAVEN - Pose + ADL Configuration File
========================================
Tất cả các cấu hình, hằng số, màu sắc, góc độ, ngưỡng cho hệ thống phát hiện tư thế và ADL
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# MODEL & VIDEO SETTINGS
# ============================================================
MODEL_PATH = os.getenv("AI_MODEL_PATH", "models/cpu_laptop/yolo11s-pose.pt")
VIDEO_PATH = os.getenv("TEST_VIDEO_PATH", "data/video/walking.mp4")
CONF_THRES = float(os.getenv("AI_CONF_THRES", "0.25"))

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
    "head": (0, 0, 255),        # Đỏ (Red)
    "torso": (180, 105, 255),   # Hồng (Pink)
    "upper_arm": (0, 255, 0),   # Xanh lá (Green) - vai đến cùi chỏ
    "lower_arm": (0, 100, 0),   # Xanh lá đậm (Dark Green) - cùi chỏ đến cổ tay (BÀN TAY)
    "upper_leg": (0, 165, 255), # Cam (Orange) - hông đến đầu gối
    "lower_leg": (0, 255, 255), # Vàng (Yellow) - đầu gối đến mắt cá (BÀN CHÂN)
}

# ============================================================
# MÀU SẮC CHO CÁC TƯ THẾ (BGR)
# ============================================================
POSTURE_COLORS = {
    "STANDING": (0, 255, 0),      # Xanh lá
    "WALKING": (255, 255, 0),     # Xanh lơ (Cyan)
    "SITTING": (0, 255, 255),     # Vàng
    "LAYING": (0, 0, 255),        # Đỏ
    "UNKNOWN": (128, 128, 128),   # Xám
}

# ============================================================
# CÁC KẾT NỐI XƯƠNG - PHÂN CHIA CHÍNH XÁC
# ============================================================
# Format: (start_keypoint, end_keypoint, part_name)
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
KEYPOINT_CONF_THRESHOLD = 0.3  # Ngưỡng confidence cho keypoint

# ============================================================
# TRACKING SETTINGS
# ============================================================
TRACKER_IOU_THRESHOLD = 0.3     # Ngưỡng IOU cho tracking
TRACKER_MAX_AGE = 30            # Số frame tối đa trước khi xóa track
POSITION_HISTORY_MAXLEN = 30    # Số lượng vị trí lưu trữ
POSTURE_HISTORY_MAXLEN = 10     # Số lượng tư thế lưu cho majority voting
EVENT_HISTORY_MAXLEN = 5        # Số lượng sự kiện lưu trữ

# ============================================================
# POSTURE CLASSIFICATION THRESHOLDS
# ============================================================
# Ngưỡng góc thân người (Torso Angle)
TORSO_ANGLE_LAYING_THRESHOLD = 50  # Độ - nếu > threshold thì đang nằm

# Ngưỡng tỷ lệ khung hình (Aspect Ratio)
ASPECT_RATIO_LAYING_THRESHOLD = 1.2  # nếu w/h > threshold thì đang nằm

# Ngưỡng góc đầu gối (Knee Angle)
KNEE_ANGLE_SITTING_THRESHOLD = 130  # Độ - nếu < threshold thì đang ngồi

# Ngưỡng di chuyển (Movement) - tính theo % chiều cao frame
MOVEMENT_THRESHOLD_RATIO = 0.08      # 8% chiều cao frame
MOVEMENT_WALKING_MULTIPLIER = 1.5    # Hệ số cho phát hiện đi bộ

# Góc đầu gối mặc định khi không phát hiện được
DEFAULT_KNEE_ANGLE = 180

# ============================================================
# EVENT DETECTION SETTINGS
# ============================================================
# Phát hiện giơ tay
HAND_RAISE_FRAMES_THRESHOLD = 15  # Số frame liên tiếp để xác nhận giơ tay (0.5s @ 30fps)

# Majority voting cho tư thế
POSTURE_VOTING_MIN_FRAMES = 5     # Số frame tối thiểu để bỏ phiếu

# Velocity calculation (FPS assumed)
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
GIF_RESIZE_RATIO = 0.5      # Giảm 50% kích thước để tiết kiệm dung lượng
GIF_DURATION_MS = 66        # 66ms ≈ 15fps
GIF_LOOP = 0                # 0 = loop vô hạn
GIF_OPTIMIZE = True         # Tối ưu hóa GIF

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
    "fall_down": "⚠️ FALL_DOWN",
    "sitting_down": "SITTING_DOWN",
    "standing_up": "STANDING_UP",
    "left_hand_raised": "👋 LEFT_HAND_RAISED",
    "right_hand_raised": "👋 RIGHT_HAND_RAISED",
}
