"""
HAVEN - Video Test Configuration
=================================
Import config chung từ shared_config + thêm cấu hình riêng cho video testing.
"""

import sys
from pathlib import Path

# Add parent directory to path để import shared_config
SCRIPT_DIR = Path(__file__).parent  # test-video/
BACKEND_DIR = SCRIPT_DIR.parent     # backend/
sys.path.insert(0, str(BACKEND_DIR))

# Import all shared configurations
from shared_config import *

# ============================================================
# VIDEO-SPECIFIC SETTINGS
# ============================================================
import os
# Đường dẫn đến video trong backend/data/video/
DEFAULT_VIDEO_PATH = str(BACKEND_DIR / "data" / "video" / "walking.mp4")
VIDEO_PATH = os.getenv("TEST_VIDEO_PATH", DEFAULT_VIDEO_PATH)
