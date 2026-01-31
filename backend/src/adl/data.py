"""
HAVEN ADL - Data Structures
===========================
M hnh d liu cho h thng ADL.
"""

from typing import List, Dict, Deque, Optional
from collections import deque
import time
from pydantic import BaseModel

class Keypoint(BaseModel):
    id: int
    x: float
    y: float
    conf: float
    name: str = ""

class FrameData(BaseModel):
    """D liu ca mt track ID ti mt thi im (frame)"""
    timestamp: float
    bbox: List[float] # [x1, y1, x2, y2]
    keypoints: List[List[float]] # [[x,y,conf], ...]
    
    # Computed Features
    posture: str = "UNKNOWN" # STANDING, SITTING, LAYING
    torso_angle: float = 0.0
    aspect_ratio: float = 0.0
    velocity: float = 0.0
    
    # Context
    in_bed_zone: bool = False
    in_chair_zone: bool = False
    has_phone: bool = False
    hand_up: bool = False

class TrackHistory:
    """Lch s di chuyn v trng thi ca mt ngi (Track ID)"""
    def __init__(self, track_id: int, maxlen: int = 150): # 5s @ 30fps
        self.track_id = track_id
        self.buffer: Deque[FrameData] = deque(maxlen=maxlen)
        self.events: List[str] = []
        self.cooldowns: Dict[str, float] = {}
        
        # State Machine
        self.current_state: str = "UNKNOWN"
        self.state_start_time: float = time.time()
    
    def add_frame(self, frame_data: FrameData):
        self.buffer.append(frame_data)
        
    def get_last_frame(self) -> Optional[FrameData]:
        return self.buffer[-1] if self.buffer else None
        
    def get_average_features(self, window: int = 10):
        """Ly gi tr trung bnh ca features trong window frame gn nht"""
        if len(self.buffer) < window: return None
        
        avg_angle = 0
        avg_velocity = 0
        count = 0
        
        # Iterate backwards
        for i in range(1, window + 1):
            frame = self.buffer[-i]
            avg_angle += frame.torso_angle
            avg_velocity += frame.velocity
            count += 1
            
        return {
            "torso_angle": avg_angle / count,
            "velocity": avg_velocity / count
        }

