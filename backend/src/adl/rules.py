"""
HAVEN ADL - Rule Engine
=======================
State Machine và Logic phát hiện sự kiện (Fall Down, Bed Exit).
"""

import time
import logging
from typing import List, Dict, Optional
from .data import TrackHistory, FrameData

logger = logging.getLogger(__name__)

class RuleEngine:
    def __init__(self):
        # Configuration
        self.FALL_TIME_WINDOW = 1.0 # seconds (Transition time)
        self.LAYING_CONFIRM_TIME = 2.0 # seconds
        self.COOLDOWN_FALL = 10.0
        
    def process_track(self, track: TrackHistory, frame_data: FrameData) -> List[str]:
        """
        Cập nhật state của track và trả về danh sách sự kiện phát hiện (nếu có).
        """
        events = []
        new_state = frame_data.posture
        
        # 1. Update Buffer check
        # Lấy state cũ
        prev_state = track.current_state
        
        # Update state duration
        if new_state == prev_state:
            track.state_duration = time.time() - track.state_start_time
        else:
            # State change
            # logger.debug(f"Track {track.track_id}: {prev_state} -> {new_state}")
            track.current_state = new_state
            track.state_start_time = time.time()
            track.state_duration = 0
            
        # 2. Add to buffer
        track.add_frame(frame_data)
        
        # 3. Check Rules
        
        # --- RULE: FALL DOWN ---
        # Logic: Transition STANDING/WALKING -> LAYING in short time
        if self._check_fall_down(track):
             if self._check_cooldown(track, "fall_down"):
                 events.append("FALL_DOWN")
                 self._set_cooldown(track, "fall_down", self.COOLDOWN_FALL)
        
        return events

    def _check_fall_down(self, track: TrackHistory) -> bool:
        """
        Kiểm tra sự kiện ngã.
        Điều kiện:
        1. Hiện tại là LAYING và đã duy trì đủ lâu (confirm time).
        2. Trước đó (trong khoảng window) là STANDING.
        """
        # Condition 1: Stable LAYING
        if track.current_state != "LAYING": return False
        if track.state_duration < 1.0: return False # Chưa đủ stable (dạng instant)
        
        # Condition 2: Check history for Standing
        # Look back 1-2 seconds
        # Tìm điểm bắt đầu LAYING
        # Buffer: [Stand, Stand, ... , Laying, Laying, Laying]
        # Check transition speed
        
        found_standing = False
        transition_found = False
        
        # Iterate backwards
        current_ts = float(time.time())
        limit_ts = current_ts - 3.0 # Look back 3s
        
        # Đếm frame LAYING
        laying_frames = 0
        
        for frame in reversed(track.buffer):
            if frame.timestamp < limit_ts: break
            
            if frame.posture == "LAYING":
                laying_frames += 1
            elif frame.posture == "STANDING" or frame.posture == "WALKING":
                found_standing = True
                # Check transition time?
                # Simplest: Just existence of Standing shortly before Laying
        
        if found_standing and laying_frames > 5: # Some stable frames
             return True
             
        return False

    def _check_cooldown(self, track: TrackHistory, event_name: str) -> bool:
        last_time = track.cooldowns.get(event_name, 0)
        return (time.time() - last_time) > self.COOLDOWN_FALL

    def _set_cooldown(self, track: TrackHistory, event_name: str, duration: float):
        track.cooldowns[event_name] = time.time()
