"""
HAVEN Camera Configuration Loader
=================================
Secure configuration management using Pydantic Settings.
"""

from typing import Literal, Optional
from pathlib import Path
from pydantic import Field, field_validator, computed_field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import logging
import os

logger = logging.getLogger(__name__)


class CameraSettings(BaseSettings):
    """
    Camera configuration with security-first design.
    """
    
    # ==================
    # Network Configuration
    # ==================
    CAMERA_NAME: str = Field(default="tapo_c210")
    CAMERA_IP: str = Field(...)
    CAMERA_PORT: int = Field(default=554, ge=1, le=65535)
    CAMERA_USER: str = Field(...)
    CAMERA_PASSWORD: SecretStr = Field(...)
    
    # ==================
    # Stream Configuration
    # ==================
    RTSP_STREAM_HD: str = Field(default="stream1")
    RTSP_STREAM_SD: str = Field(default="stream2")
    DEFAULT_STREAM: Literal["HD", "SD"] = Field(default="SD")
    
    # ==================
    # AI Model Configuration
    # ==================
    AI_MODEL_PATH: str = Field(default="models/cpu_laptop/yolo11s-pose.pt")
    AI_CONF_THRES: float = Field(default=0.25, ge=0.01, le=1.0, description="Confidence threshold")
    AI_IOU_THRES: float = Field(default=0.45, ge=0.01, le=1.0, description="IOU threshold for NMS")
    AI_IMG_SIZE: int = Field(default=640, ge=320, le=1280, description="Inference image size")
    AI_MAX_DET: int = Field(default=10, ge=1, le=100, description="Maximum detections per frame")
    AI_SKIP_FRAMES: int = Field(default=3, ge=1, le=10, description="Run AI every N frames")
    
    # ==================
    # ROI Mode (Region of Interest)
    # ==================
    ROI_ENABLED: bool = Field(default=False, description="Enable ROI mode")
    ROI_X1: float = Field(default=0.5, ge=0.0, le=1.0, description="ROI left edge (0-1)")
    ROI_Y1: float = Field(default=0.0, ge=0.0, le=1.0, description="ROI top edge (0-1)")
    ROI_X2: float = Field(default=1.0, ge=0.0, le=1.0, description="ROI right edge (0-1)")
    ROI_Y2: float = Field(default=0.7, ge=0.0, le=1.0, description="ROI bottom edge (0-1)")
    
    # ==================
    # AI Stream & Debug
    # ==================
    AI_USE_HD_STREAM: bool = Field(default=False, description="Use HD stream for AI inference")
    AI_DEBUG_OVERLAY: bool = Field(default=True, description="Show debug info on frame")
    
    # ==================
    # Performance Tuning
    # ==================
    FRAME_BUFFER_SIZE: int = Field(default=1, ge=1, le=10)
    RECONNECT_BACKOFF_SECONDS: int = Field(default=5, ge=1, le=60)
    MAX_RECONNECT_ATTEMPTS: int = Field(default=10, ge=1, le=100)
    STREAM_TIMEOUT_SECONDS: int = Field(default=30, ge=5, le=300)
    FPS_LIMIT: int = Field(default=15, ge=0, le=30)
    JPEG_QUALITY: int = Field(default=80, ge=1, le=100)
    RESIZE_WIDTH: int = Field(default=0, ge=0)
    DEBUG_MODE: bool = Field(default=False)
    WS_PING_INTERVAL: int = Field(default=30, ge=5, le=300)
    
    # ==================
    # Pydantic Settings Config
    # ==================
    model_config = SettingsConfigDict(
        env_file=[
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), '.env'),
            '.env'
        ],
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )
    
    # ==================
    # Validators
    # ==================
    @field_validator("CAMERA_IP")
    @classmethod
    def validate_ip(cls, v: str) -> str:
        """Kiểm tra định dạng địa chỉ IP (xxx.xxx.xxx.xxx)."""
        parts = v.split(".")
        if len(parts) != 4:
            raise ValueError("IP must be in format: xxx.xxx.xxx.xxx")
        for part in parts:
            try:
                num = int(part)
                if not 0 <= num <= 255:
                    raise ValueError(f"IP octet must be 0-255, got {num}")
            except ValueError:
                raise ValueError(f"Invalid IP format: {v}")
        return v
    
    @field_validator("DEFAULT_STREAM")
    @classmethod
    def validate_stream_type(cls, v: str) -> str:
        """Chuẩn hóa loại luồng thành chữ in hoa (HD/SD)."""
        return v.upper()
    
    # ==================
    # Computed Properties (RTSP URLs)
    # ==================
    @computed_field
    @property
    def rtsp_url_hd(self) -> str:
        password = self.CAMERA_PASSWORD.get_secret_value()
        return f"rtsp://{self.CAMERA_USER}:{password}@{self.CAMERA_IP}:{self.CAMERA_PORT}/{self.RTSP_STREAM_HD}"
    
    @computed_field
    @property
    def rtsp_url_sd(self) -> str:
        password = self.CAMERA_PASSWORD.get_secret_value()
        return f"rtsp://{self.CAMERA_USER}:{password}@{self.CAMERA_IP}:{self.CAMERA_PORT}/{self.RTSP_STREAM_SD}"
    
    # ==================
    # Helper Methods
    # ==================
    def get_stream_url(self, stream_type: Literal["HD", "SD"]) -> str:
        """
        Lấy URL RTSP đầy đủ (bao gồm user/pass) cho loại luồng tương ứng.
        """
        stream_type = stream_type.upper()
        if stream_type == "HD":
            return self.rtsp_url_hd
        elif stream_type == "SD":
            return self.rtsp_url_sd
        else:
            raise ValueError(f"Invalid stream type: {stream_type}")
    
    def get_roi_coords(self, frame_width: int, frame_height: int) -> tuple:
        """
        Tính toán tọa độ pixel của vùng ROI từ các giá trị phần trăm (0-1).
        Trả về (x1, y1, x2, y2).
        """
        x1 = int(self.ROI_X1 * frame_width)
        y1 = int(self.ROI_Y1 * frame_height)
        x2 = int(self.ROI_X2 * frame_width)
        y2 = int(self.ROI_Y2 * frame_height)
        return (x1, y1, x2, y2)
    
    def to_public_info(self) -> dict:
        """
        Xuất cấu hình an toàn (không chứa mật khẩu) để gửi về frontend.
        """
        return {
            "name": self.CAMERA_NAME,
            "available_streams": ["HD", "SD"],
            "default_stream": self.DEFAULT_STREAM,
            "ai_config": {
                "conf_thres": self.AI_CONF_THRES,
                "iou_thres": self.AI_IOU_THRES,
                "img_size": self.AI_IMG_SIZE,
                "max_det": self.AI_MAX_DET,
                "skip_frames": self.AI_SKIP_FRAMES,
                "debug_overlay": self.AI_DEBUG_OVERLAY,
                "roi_enabled": self.ROI_ENABLED,
            }
        }
    
    def __repr__(self) -> str:
        return (
            f"CameraSettings(name='{self.CAMERA_NAME}', ip='{self.CAMERA_IP}', "
            f"ai_conf={self.AI_CONF_THRES}, roi={self.ROI_ENABLED})"
        )


# ==================
# Global Config Instance
# ==================
def load_camera_config() -> CameraSettings:
    """
    Tải cấu hình từ file .env.
    Nếu không thấy file .env sẽ báo lỗi log.
    """
    try:
        config = CameraSettings()
        logger.info(f"Camera config loaded: {config.CAMERA_NAME}")
        logger.info(f"AI Config: conf={config.AI_CONF_THRES}, iou={config.AI_IOU_THRES}, "
                   f"img_size={config.AI_IMG_SIZE}, roi={config.ROI_ENABLED}")
        return config
    except FileNotFoundError:
        logger.error(".env file not found. Copy .env.example to .env")
        raise
    except Exception as e:
        logger.error(f"Failed to load camera config: {e}")
        raise
