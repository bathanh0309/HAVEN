
import cv2
import threading
import time
import queue
import logging
import numpy as np
from typing import Optional, Literal, Tuple, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from ..config.camera_config import CameraSettings

logger = logging.getLogger(__name__)

@dataclass
class StreamStats:
    """Statistics for monitoring stream health"""
    connected: bool = False
    current_stream: Literal["HD", "SD"] = "SD"
    fps: float = 0.0
    frame_count: int = 0
    dropped_frames: int = 0
    last_frame_time: Optional[datetime] = None
    reconnect_attempts: int = 0
    uptime_start: float = field(default_factory=time.time)
    error_message: Optional[str] = None

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.uptime_start

class StreamManager:
class StreamManager:
    """
    Quản lý luồng video nâng cao với kiến trúc Non-Blocking.
    
    Tính năng:
    - Thread riêng biệt để đọc camera (tách biệt I/O).
    - Chiến lược 'Latest Frame': Luôn giữ frame mới nhất, bỏ frame cũ để giảm độ trễ.
    - Chuyển đổi luồng an toàn (Thread-safe).
    - Tự động kết nối lại (Auto-reconnect).
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config: CameraSettings):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: CameraSettings):
        if self._initialized:
            return
        
        self.config = config
        self.stats = StreamStats(current_stream=config.DEFAULT_STREAM)
        
        # Internal state
        self.capture: Optional[cv2.VideoCapture] = None
        self.running = False
        self.capture_thread: Optional[threading.Thread] = None
        self.switch_lock = threading.Lock()
        
        # Output Queue: Holds the absolute latest frame tuple (frame, metadata)
        # Consumers (API/WS) pull from here.
        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        
        self._initialized = True
        logger.info(f"StreamManager Initialized | Camera: {config.CAMERA_NAME}")

    def start(self):
        """Bắt đầu worker thread đọc camera."""
        if self.running: 
            return
        
        self.running = True
        self.capture_thread = threading.Thread(
            target=self._worker_loop, 
            name="CamWorker", 
            daemon=True
        )
        self.capture_thread.start()
        logger.info("Capture Worker Started")

    def stop(self):
        """Dừng worker và giải phóng tài nguyên một cách an toàn."""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        self._release_capture()
        logger.info("Capture Worker Stopped")

    def switch_stream(self, stream_type: Literal["HD", "SD"]) -> bool:
    def switch_stream(self, stream_type: Literal["HD", "SD"]) -> bool:
        """
        Yêu cầu chuyển đổi loại luồng (HD/SD).
        Việc chuyển đổi thực tế sẽ diễn ra trong vòng lặp worker để đảm bảo an toàn thread.
        """
        if stream_type not in ["HD", "SD"]: return False
        
        with self.switch_lock:
            if self.stats.current_stream == stream_type: 
                return True
                
            logger.info(f"Switching Stream: {self.stats.current_stream} -> {stream_type}")
            self.stats.current_stream = stream_type
            
            # Force release to trigger reconnect in worker loop
            self._release_capture() 
            return True

    def get_latest_frame(self) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    def get_latest_frame(self) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Lấy frame mới nhất từ hàng đợi (Non-blocking).
        Trả về: (frame_bgr, metadata) hoặc None nếu không có frame mới.
        """
        try:
            # Get latest, but put it back for other consumers? 
            # Ideally for broadcasting we might need a different pattern (e.g. Pub/Sub)
            # But for now, we assume if queue is empty we return None.
            
            # Actually, to support multiple consumers (WS + MJPEG), 
            # we shouldn't 'consume' the frame here destructively if we want to share.
            # However, `queue` creates a copy reference. 
            # A better approach for multiple readers is keeping 'self.last_valid_frame'.
            
            # Hybrid approach: Queue for signaling new frames, Variable for storage.
            if not self.frame_queue.empty():
                return self.frame_queue.get_nowait()
            return None
        except:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Lấy thông tin thống kê trạng thái hiện tại của stream."""
        return {
            "connected": self.stats.connected,
            "stream": self.stats.current_stream,
            "fps": round(self.stats.fps, 1),
            "dropped": self.stats.dropped_frames,
            "uptime": f"{self.stats.uptime_seconds:.0f}s"
        }

    # =========================================
    # INTERNAL WORKER LOGIC
    # =========================================

    def _worker_loop(self):
    def _worker_loop(self):
        """
        Vòng lặp chính của worker thread:
        1. Duy trì kết nối Camera (Auto-reconnect).
        2. Đọc Frame từ RTSP.
        3. Giới hạn tốc độ đọc (FPS Limit).
        4. Resize ảnh nếu cần (để tối ưu hiệu năng).
        5. Đẩy frame vào hàng đợi (Queue) cho consumer.
        """
        reconnect_delay = 1
        fps_counter = 0
        last_fps_time = time.time()
        last_read_time = 0                   # Init to 0 guarantees first read
        target_interval = 1.0 / self.config.FPS_LIMIT if self.config.FPS_LIMIT > 0 else 0

        while self.running:
            # 1. Connection Management
            if self.capture is None or not self.capture.isOpened():
                self.stats.connected = False
                if not self._connect():
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 30) # Exponential backoff cap 30s
                    continue
                reconnect_delay = 1 # Reset on success

            # 2. Frame Rate Limiting (Sleep if too fast)
            now = time.time()
            if target_interval > 0:
                elapsed = now - last_read_time
                if elapsed < target_interval:
                    time.sleep(max(0.001, target_interval - elapsed))
                    continue
            
            # 3. Read Frame
            try:
                ret, frame = self.capture.read()
                last_read_time = time.time() # Update read time immediately

                if not ret:
                    logger.warning("Frame read failed. Reconnecting...")
                    self._release_capture()
                    continue

                self.stats.connected = True
                self.stats.last_frame_time = datetime.now()
                self.stats.frame_count += 1
                fps_counter += 1

                # 4. Preprocessing (Resize for Performance)
                # Resize logic moved to consumer or done here to save MEM?
                # Done here saves memory for queue.
                if self.config.RESIZE_WIDTH > 0:
                    h, w = frame.shape[:2]
                    scale = self.config.RESIZE_WIDTH / w
                    frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)

                # 5. Push to Queue (Drop old if full)
                metadata = {
                    "ts": time.time(),
                    "fps": self.stats.fps,
                    "w": frame.shape[1],
                    "h": frame.shape[0]
                }
                
                # Update FPS every second
                if time.time() - last_fps_time >= 1.0:
                    self.stats.fps = fps_counter
                    fps_counter = 0
                    last_fps_time = time.time()

                try:
                    self.frame_queue.put_nowait((frame, metadata))
                except queue.Full:
                    # Drop the old frame to make room for new (Latest Frame Strategy)
                    try:
                        self.frame_queue.get_nowait()
                        self.stats.dropped_frames += 1
                    except queue.Empty:
                        pass
                    self.frame_queue.put((frame, metadata))

            except Exception as e:
                logger.error(f"Worker Error: {e}")
                self._release_capture()
                time.sleep(1)

    def _connect(self) -> bool:
        """Thực hiện kết nối thực tế tới Camera qua OpenCV."""
        url = self.config.get_stream_url(self.stats.current_stream)
        logger.info(f"Connecting to {self.stats.current_stream}...")
        try:
            # Linux/Windows compatibility flags
            if threading.current_thread() is threading.main_thread():
                # Should not happen, but safe check
                pass
                
            self.capture = cv2.VideoCapture(url)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Crucial for low latency
            
            if self.capture.isOpened():
                logger.info("Connected!")
                return True
            return False
        except Exception as e:
            logger.error(f"Connection Failed: {e}")
            return False

    def _release_capture(self):
        """Giải phóng object VideoCapture."""
        if self.capture:
            self.capture.release()
            self.capture = None
            
# Helper for Dependency Injection
_global_mgr = None
def get_stream_manager(config: CameraSettings = None) -> StreamManager:
    """Trả về singleton instance của StreamManager."""
    global _global_mgr
    if _global_mgr is None and config:
        _global_mgr = StreamManager(config)
    return _global_mgr
