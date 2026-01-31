"""
Video Stream Abstraction

Supports:
- Single video file
- Folder with multiple video chunks (played sequentially)
- RTSP stream
"""

import cv2
import glob
import os
from typing import Optional, Tuple, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VideoStream:
    """
    Unified video stream interface.
    
    Handles:
    - Single video file
    - Folder with multiple video files (auto-concatenated)
    - RTSP stream
    """
    
    def __init__(
        self,
        camera_id: int,
        camera_name: str,
        source_type: str,
        source_path: str,
        data_root: Optional[str] = None,
        pattern: str = "*.mp4",
        fps_override: Optional[float] = None,
        resize_width: Optional[int] = None,
        skip_frames: int = 0
    ):
        """
        Initialize video stream.
        
        Args:
            camera_id: Unique camera ID
            camera_name: Camera name
            source_type: "video_file" | "video_folder" | "rtsp"
            source_path: Path to video/folder or RTSP URL
            data_root: Root directory for relative paths
            pattern: File pattern for folder source (e.g., "*.mp4")
            fps_override: Override FPS (None = auto-detect)
            resize_width: Resize width (None = original)
            skip_frames: Skip every N frames (0 = no skip)
        """
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.source_type = source_type
        self.resize_width = resize_width
        self.skip_frames = skip_frames
        self.fps_override = fps_override
        
        # Resolve path
        if data_root and not os.path.isabs(source_path):
            self.source_path = os.path.join(data_root, source_path)
        else:
            self.source_path = source_path
        
        self.pattern = pattern
        
        # State
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_files: List[str] = []
        self.current_file_idx = 0
        self.frame_count = 0
        self.total_frames = 0
        self.fps = 0.0
        self.width = 0
        self.height = 0
        
        # Initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize video source"""
        if self.source_type == "video_file":
            self._init_video_file()
        elif self.source_type == "video_folder":
            self._init_video_folder()
        elif self.source_type == "rtsp":
            self._init_rtsp()
        else:
            raise ValueError(f"Unknown source_type: {self.source_type}")
    
    def _init_video_file(self):
        """Initialize single video file"""
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"Video file not found: {self.source_path}")
        
        self.video_files = [self.source_path]
        self._open_next_file()
    
    def _init_video_folder(self):
        """Initialize video folder with multiple chunks"""
        # Check if source_path is a file
        if os.path.isfile(self.source_path):
            # Treat as single file
            self.video_files = [self.source_path]
            self._open_next_file()
            return
        
        # It's a folder
        if not os.path.isdir(self.source_path):
            raise NotADirectoryError(f"Folder not found: {self.source_path}")
        
        # Find all matching video files
        pattern_path = os.path.join(self.source_path, self.pattern)
        video_files = glob.glob(pattern_path)
        
        if not video_files:
            raise FileNotFoundError(f"No videos found in {self.source_path} with pattern {self.pattern}")
        
        # Sort by name (assumes naming like video_001.mp4, video_002.mp4, etc.)
        self.video_files = sorted(video_files)
        
        logger.info(f"[cam{self.camera_id}] Found {len(self.video_files)} video files in folder")
        for i, vf in enumerate(self.video_files):
            logger.debug(f"  [{i+1}] {os.path.basename(vf)}")
        
        # Open first file
        self._open_next_file()
    
    def _init_rtsp(self):
        """Initialize RTSP stream"""
        self.cap = cv2.VideoCapture(self.source_path)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open RTSP stream: {self.source_path}")
        
        self.fps = self.fps_override or self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(
            f"[cam{self.camera_id}] RTSP stream opened: "
            f"{self.width}x{self.height} @ {self.fps:.1f} FPS"
        )
    
    def _open_next_file(self) -> bool:
        """Open next video file in sequence"""
        if self.current_file_idx >= len(self.video_files):
            return False
        
        # Close previous
        if self.cap is not None:
            self.cap.release()
        
        # Open next
        video_path = self.video_files[self.current_file_idx]
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            logger.error(f"[cam{self.camera_id}] Failed to open: {video_path}")
            return False
        
        # Get properties
        self.fps = self.fps_override or self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        file_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(
            f"[cam{self.camera_id}] Opened file {self.current_file_idx + 1}/{len(self.video_files)}: "
            f"{os.path.basename(video_path)} "
            f"({self.width}x{self.height}, {file_frames} frames @ {self.fps:.1f} FPS)"
        )
        
        self.total_frames += file_frames
        self.current_file_idx += 1
        
        return True
    
    def read(self) -> Tuple[bool, Optional[any]]:
        """
        Read next frame.
        
        Returns:
            (success, frame)
        """
        if self.cap is None:
            return False, None
        
        # Handle frame skipping
        for _ in range(self.skip_frames + 1):
            ret, frame = self.cap.read()
            
            if not ret:
                # Try next file if folder source
                if self.source_type == "video_folder":
                    if self._open_next_file():
                        return self.read()  # Recursive call
                return False, None
            
            self.frame_count += 1
        
        # Resize if needed
        if self.resize_width and frame is not None:
            h, w = frame.shape[:2]
            aspect_ratio = h / w
            new_height = int(self.resize_width * aspect_ratio)
            frame = cv2.resize(frame, (self.resize_width, new_height))
        
        return True, frame
    
    def get_fps(self) -> float:
        """Get FPS"""
        return self.fps
    
    def get_frame_count(self) -> int:
        """Get current frame count"""
        return self.frame_count
    
    def get_total_frames(self) -> int:
        """Get total frames (approximate for RTSP)"""
        return self.total_frames
    
    def get_timestamp(self) -> float:
        """Get current timestamp in seconds"""
        if self.fps > 0:
            return self.frame_count / self.fps
        return 0.0
    
    def is_opened(self) -> bool:
        """Check if stream is open"""
        return self.cap is not None and self.cap.isOpened()
    
    def release(self):
        """Release resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __del__(self):
        """Cleanup"""
        self.release()

