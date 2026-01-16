"""
Safety-specific checks for fall and stroke-like detection
More sophisticated than basic rule-based
"""
import numpy as np
from typing import Optional, Dict, Any
from collections import deque
import time

from .actions import ADLAction, ADLLabel, KeypointIndex


class SafetyChecker:
    """
    Advanced safety checks for critical events
    Tracks temporal patterns to reduce false positives
    """
    
    def __init__(self, 
                 fall_history_size: int = 30,  # 30 frames @ 15fps = 2 seconds
                 immobility_threshold_sec: float = 300):  # 5 minutes
        
        self.fall_history = deque(maxlen=fall_history_size)
        self.immobility_threshold = immobility_threshold_sec
        self.last_movement_time = time.time()
        self.last_position = None
        
    def check_fall(self, keypoints: np.ndarray, bbox: tuple) -> Optional[Dict[str, Any]]:
        """
        Multi-criteria fall detection
        
        Criteria:
        1. Body orientation (horizontal)
        2. Sudden position change (velocity)
        3. Sustained horizontal position (not just bending)
        
        Returns:
            {
                "is_fall": bool,
                "confidence": float,
                "reason": str
            }
        """
        # Extract key points
        nose = keypoints[KeypointIndex.NOSE]
        l_hip = keypoints[KeypointIndex.LEFT_HIP]
        r_hip = keypoints[KeypointIndex.RIGHT_HIP]
        l_ankle = keypoints[KeypointIndex.LEFT_ANKLE]
        r_ankle = keypoints[KeypointIndex.RIGHT_ANKLE]
        
        # Check 1: Body horizontal (head near ground level with hips)
        if nose[2] > 0.3 and (l_hip[2] > 0.3 or r_hip[2] > 0.3):
            hip_y = (l_hip[1] + r_hip[1]) / 2 if l_hip[2] > 0.3 and r_hip[2] > 0.3 else (l_hip[1] if l_hip[2] > 0.3 else r_hip[1])
            
            # Head should be near or below hips
            if nose[1] > hip_y - 50:  # Within 50 pixels
                # Check 2: Verify not just crouching (ankles should be near head level)
                if (l_ankle[2] > 0.3 or r_ankle[2] > 0.3):
                    ankle_y = (l_ankle[1] + r_ankle[1]) / 2 if l_ankle[2] > 0.3 and r_ankle[2] > 0.3 else (l_ankle[1] if l_ankle[2] > 0.3 else r_ankle[1])
                    
                    if abs(nose[1] - ankle_y) < 200:  # Body fully horizontal
                        # Store in history
                        self.fall_history.append(True)
                        
                        # Require sustained detection (at least 50% of history)
                        if len(self.fall_history) >= 15 and sum(self.fall_history) / len(self.fall_history) > 0.5:
                            return {
                                "is_fall": True,
                                "confidence": 0.9,
                                "reason": "Sustained horizontal body position detected"
                            }
        
        self.fall_history.append(False)
        return {"is_fall": False, "confidence": 0.0, "reason": ""}
    
    def check_stroke_like(self, keypoints: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect abnormal posture suggesting possible stroke
        
        Criteria:
        1. Significant body asymmetry (one side collapsed)
        2. Unusual limb positions
        3. Not simply sitting/resting
        
        Returns:
            {
                "is_suspicious": bool,
                "confidence": float,
                "reason": str
            }
        """
        # Extract bilateral keypoints
        l_shoulder = keypoints[KeypointIndex.LEFT_SHOULDER]
        r_shoulder = keypoints[KeypointIndex.RIGHT_SHOULDER]
        l_elbow = keypoints[KeypointIndex.LEFT_ELBOW]
        r_elbow = keypoints[KeypointIndex.RIGHT_ELBOW]
        l_wrist = keypoints[KeypointIndex.LEFT_WRIST]
        r_wrist = keypoints[KeypointIndex.RIGHT_WRIST]
        l_hip = keypoints[KeypointIndex.LEFT_HIP]
        r_hip = keypoints[KeypointIndex.RIGHT_HIP]
        
        # Check shoulder asymmetry (one side significantly lower)
        if l_shoulder[2] > 0.3 and r_shoulder[2] > 0.3:
            shoulder_diff = abs(l_shoulder[1] - r_shoulder[1])
            shoulder_avg_width = abs(l_shoulder[0] - r_shoulder[0])
            
            # Asymmetry ratio
            if shoulder_avg_width > 0:
                asymmetry_ratio = shoulder_diff / shoulder_avg_width
                
                if asymmetry_ratio > 0.4:  # 40% vertical asymmetry
                    return {
                        "is_suspicious": True,
                        "confidence": 0.7,
                        "reason": f"Shoulder asymmetry detected (ratio: {asymmetry_ratio:.2f})"
                    }
        
        # Check arm asymmetry (one arm hanging, other raised)
        # TODO: Implement more sophisticated checks
        
        return {"is_suspicious": False, "confidence": 0.0, "reason": ""}
    
    def check_immobility(self, keypoints: np.ndarray, current_time: float) -> Optional[Dict[str, Any]]:
        """
        Detect prolonged immobility (e.g., sleeping too long)
        
        Returns:
            {
                "is_immobile": bool,
                "duration_seconds": float,
                "reason": str
            }
        """
        # Calculate center of mass
        valid_kpts = keypoints[keypoints[:, 2] > 0.3]
        if len(valid_kpts) == 0:
            return None
        
        center = np.mean(valid_kpts[:, :2], axis=0)
        
        # Check movement
        if self.last_position is not None:
            movement = np.linalg.norm(center - self.last_position)
            
            if movement > 30:  # Significant movement (pixels)
                self.last_movement_time = current_time
        
        self.last_position = center
        
        # Check immobility duration
        immobile_duration = current_time - self.last_movement_time
        
        if immobile_duration > self.immobility_threshold:
            return {
                "is_immobile": True,
                "duration_seconds": immobile_duration,
                "reason": f"No movement detected for {immobile_duration/60:.1f} minutes"
            }
        
        return {"is_immobile": False, "duration_seconds": immobile_duration, "reason": ""}
    
    def reset(self):
        """Reset checker state"""
        self.fall_history.clear()
        self.last_movement_time = time.time()
        self.last_position = None
