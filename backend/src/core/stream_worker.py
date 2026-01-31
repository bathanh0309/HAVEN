"""
StreamWorker - Per-camera capture + inference thread
Handles one video source (MP4 or RTSP), runs inference once per frame,
stores latest packet for multiple consumers (non-destructive reads).
"""

import cv2
import base64
import threading
import time
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Configuration for a single camera source."""
    camera_id: str
    source: str  # File path or RTSP URL
    source_type: str = "video_file"  # "video_file" or "rtsp"
    loop_video: bool = True  # Restart MP4 when finished
    ai_enabled: bool = True
    resize_width: Optional[int] = 640
    jpeg_quality: int = 80
    target_fps: int = 15  # Target FPS for playback (controls speed)


@dataclass
class FramePacket:
    """Container for processed frame + metadata."""
    frame_base64: str
    detections: list
    metadata: Dict[str, Any]
    timestamp: float


class StreamWorker:
    """
    Per-camera worker thread.
    
    Pipeline:
    1. Capture frame from source (VideoCapture)
    2. Optional resize
    3. Run AI inference ONCE (shared across all clients)
    4. JPEG encode + base64
    5. Store as last_packet (non-destructive, thread-safe)
    
    Multiple WebSocket clients read from last_packet without consuming it.
    """
    
    def __init__(self, config: CameraConfig, ai_engine=None):
        self.config = config
        self.ai_engine = ai_engine
        
        # Thread control
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        # Latest packet storage (non-destructive reads)
        self._last_packet: Optional[FramePacket] = None
        self._packet_lock = threading.Lock()
        
        # VideoCapture instance
        self._cap: Optional[cv2.VideoCapture] = None
        
        # Stats
        self._start_time = time.time()
        self._frame_count = 0
        self._dropped_frames = 0
        self._last_fps_time = time.time()
        self._fps = 0.0
        self._connected = False
        
    def start(self):
        """Start the worker thread."""
        if self._thread and self._thread.is_alive():
            logger.warning(f"[{self.config.camera_id}] Worker already running")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        logger.info(f"[{self.config.camera_id}] Worker started: {self.config.source}")
    
    def stop(self):
        """Stop the worker thread gracefully."""
        if not self._thread or not self._thread.is_alive():
            return
        
        logger.info(f"[{self.config.camera_id}] Stopping worker...")
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        
        if self._cap:
            self._cap.release()
            self._cap = None
        
        self._connected = False
        logger.info(f"[{self.config.camera_id}] Worker stopped")
    
    def get_latest_packet(self) -> Optional[FramePacket]:
        """
        Get latest frame packet (non-destructive read).
        Multiple clients can call this simultaneously.
        """
        with self._packet_lock:
            return self._last_packet
    
    def get_status(self) -> Dict[str, Any]:
        """Get current worker statistics."""
        uptime = time.time() - self._start_time
        return {
            "camera_id": self.config.camera_id,
            "source": self.config.source,
            "source_type": self.config.source_type,
            "connected": self._connected,
            "fps": round(self._fps, 2),
            "dropped_frames": self._dropped_frames,
            "uptime_seconds": round(uptime, 2),
            "ai_enabled": self.config.ai_enabled,
        }
    
    def _worker_loop(self):
        """Main capture + inference loop (runs in dedicated thread)."""
        # Open video source
        if not self._open_capture():
            return
        
        fps_counter = 0
        fps_timer = time.time()
        
        # Calculate frame interval for target FPS
        frame_interval = 1.0 / self.config.target_fps if self.config.target_fps > 0 else 0
        last_frame_time = 0
        
        while not self._stop_event.is_set():
            try:
                # FPS limiting - control playback speed
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                last_frame_time = time.time()
                
                # Read frame
                ret, frame = self._cap.read()
                
                # Handle end of video file
                if not ret:
                    if self.config.source_type == "video_file" and self.config.loop_video:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        self._connected = False
                        break
                
                self._connected = True
                self._frame_count += 1
                
                # Optional resize
                if self.config.resize_width:
                    frame = self._resize_frame(frame)
                
                # Run AI inference (ONCE per frame, shared by all clients)
                detections = []
                if self.config.ai_enabled and self.ai_engine:
                    try:
                        frame, detections = self.ai_engine.process_frame(frame)
                    except Exception as e:
                        logger.error(f"[{self.config.camera_id}] Inference failed: {e}")
                        # Fallback: use raw frame
                
                # JPEG encode
                frame_jpeg = self._encode_jpeg(frame)
                
                # Build metadata
                height, width = frame.shape[:2]
                metadata = {
                    "camera_id": self.config.camera_id,
                    "timestamp": time.time(),
                    "fps": self._fps,
                    "stream_width": width,
                    "stream_height": height,
                    "frame_count": self._frame_count,
                }
                
                # Store packet (atomic, non-destructive)
                packet = FramePacket(
                    frame_base64=frame_jpeg,
                    detections=detections,
                    metadata=metadata,
                    timestamp=time.time(),
                )
                
                with self._packet_lock:
                    self._last_packet = packet
                
                # Update FPS stats
                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    self._fps = fps_counter / (time.time() - fps_timer)
                    fps_counter = 0
                    fps_timer = time.time()
                
            except Exception as e:
                logger.error(f"[{self.config.camera_id}] Worker loop error: {e}", exc_info=True)
                self._dropped_frames += 1
                time.sleep(0.1)  # Avoid tight loop on persistent errors
        
        # Cleanup
        if self._cap:
            self._cap.release()
            self._cap = None
        
        logger.info(f"[{self.config.camera_id}] Worker loop exited")
    
    def _open_capture(self) -> bool:
        """Open VideoCapture for the configured source."""
        try:
            self._cap = cv2.VideoCapture(self.config.source)
            
            if not self._cap.isOpened():
                logger.error(f"[{self.config.camera_id}] Could not open: {self.config.source}")
                return False
            
            # Get source properties
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(
                f"[{self.config.camera_id}] Opened: {width}x{height} @ {fps:.1f}fps"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"[{self.config.camera_id}] Failed to open capture: {e}")
            return False
    
    def _resize_frame(self, frame) -> Any:
        """Resize frame while maintaining aspect ratio."""
        if not self.config.resize_width:
            return frame
        
        height, width = frame.shape[:2]
        if width <= self.config.resize_width:
            return frame
        
        aspect_ratio = height / width
        new_width = self.config.resize_width
        new_height = int(new_width * aspect_ratio)
        
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    def _encode_jpeg(self, frame) -> str:
        """Encode frame to JPEG and return base64 string."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        return base64.b64encode(buffer).decode('utf-8')

