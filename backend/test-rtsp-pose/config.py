"""
HAVEN - RTSP Camera Pose + ADL Test Configuration
==================================================
Import config chung từ shared_config + thêm cấu hình riêng cho RTSP testing.
"""

import sys
from pathlib import Path

# Add parent directory to path để import shared_config
SCRIPT_DIR = Path(__file__).parent  # test-rtsp-pose/
BACKEND_DIR = SCRIPT_DIR.parent     # backend/
sys.path.insert(0, str(BACKEND_DIR))

# Import all shared configurations
from shared_config import *

# RTSP_CONFIG và get_rtsp_url() đã được định nghĩa trong shared_config
# Không cần định nghĩa lại ở đây
