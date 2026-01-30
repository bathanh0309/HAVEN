"""
Unified video source handler (local files + RTSP).
"""
import cv2
from typing import Optional, Tuple
import numpy as np
from pathlib import Path


class VideoSource:
    """
    Unified interface for video files and RTSP streams.
    """
    
    def __init__(self, source: str, source_type: str = "video", 
                 username: str = None, password: str = None,
                 resize_width: int = None):
        """
        Initialize video source.
        
        Args:
            source: Video file path or RTSP URL
            source_type: "video" or "rtsp"
            username: RTSP username (optional)
            password: RTSP password (optional)
            resize_width: Resize frames to this width (optional)
        """
        self.source = source
        self.source_type = source_type
        self.resize_width = resize_width
        
        # Build connection string
        if source_type == "rtsp" and username and password:
            # Insert credentials into RTSP URL
            # rtsp://192.168.1.100:554/stream1
            # -> rtsp://user:pass@192.168.1.100:554/stream1
            url_parts = source.replace("rtsp://", "").split("/")
            host = url_parts[0]
            path = "/".join(url_parts[1:])
            self.connection_string = f"rtsp://{username}:{password}@{host}/{path}"
        else:
            self.connection_string = source
        
        self.cap = None
        self.fps = None
        self.frame_count = 0
        
    def open(self) -> bool:
        """
        Open video source.
        
        Returns:
            True if successful
        """
        self.cap = cv2.VideoCapture(self.connection_string)
        
        if not self.cap.isOpened():
            return False
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.source_type == "video":
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        return True
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame.
        
        Returns:
            (success, frame) tuple
        """
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        if not ret:
            return False, None
        
        # Resize if requested
        if self.resize_width and frame is not None:
            h, w = frame.shape[:2]
            new_h = int(h * (self.resize_width / w))
            frame = cv2.resize(frame, (self.resize_width, new_h))
        
        return True, frame
    
    def release(self):
        """Release video source."""
        if self.cap:
            self.cap.release()
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


if __name__ == "__main__":
    # Test with video file
    source = VideoSource("/path/to/test/video.mp4", "video", resize_width=640)
    if source.open():
        ret, frame = source.read()
        if ret:
            print(f"Successfully read frame: {frame.shape}")
        source.release()
