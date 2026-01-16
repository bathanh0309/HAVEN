"""
Rule-Based ADL Recognition (Phase A - Baseline)
Uses heuristics based on pose keypoint geometry
"""
import numpy as np
from typing import Optional, Tuple
import time

from .actions import ADLAction, ADLLabel, KeypointIndex, ACTIVITY_SEVERITY_MAP


class RuleBasedRecognizer:
    """
    Heuristic-based ADL recognition from pose keypoints
    Fast baseline for real-time inference
    """
    
    def __init__(self, conf_threshold: float = 0.3):
        self.conf_threshold = conf_threshold
        self.last_recognition_time = 0
        
    def recognize(self, keypoints: np.ndarray, bbox: tuple, frame: np.ndarray = None) -> Optional[ADLAction]:
        """
        Recognize activity using rule-based heuristics
        
        Args:
            keypoints: (17, 3) numpy array of (x, y, confidence)
            bbox: (x, y, w, h) bounding box
            frame: Optional frame for additional context
        
        Returns:
            ADLAction or None
        """
        # Filter low-confidence keypoints
        valid_kpts = keypoints[keypoints[:, 2] > self.conf_threshold]
        if len(valid_kpts) < 5:  # Need minimum keypoints
            return None
        
        # Extract key body parts
        nose = self._get_kpt(keypoints, KeypointIndex.NOSE)
        l_shoulder = self._get_kpt(keypoints, KeypointIndex.LEFT_SHOULDER)
        r_shoulder = self._get_kpt(keypoints, KeypointIndex.RIGHT_SHOULDER)
        l_wrist = self._get_kpt(keypoints, KeypointIndex.LEFT_WRIST)
        r_wrist = self._get_kpt(keypoints, KeypointIndex.RIGHT_WRIST)
        l_hip = self._get_kpt(keypoints, KeypointIndex.LEFT_HIP)
        r_hip = self._get_kpt(keypoints, KeypointIndex.RIGHT_HIP)
        l_ankle = self._get_kpt(keypoints, KeypointIndex.LEFT_ANKLE)
        r_ankle = self._get_kpt(keypoints, KeypointIndex.RIGHT_ANKLE)
        
        # Calculate body metrics
        body_angle = self._calculate_body_angle(nose, l_hip, r_hip, l_ankle, r_ankle)
        hand_head_dist = self._hand_to_head_distance(nose, l_wrist, r_wrist)
        wrist_y_avg = (l_wrist[1] + r_wrist[1]) / 2 if l_wrist is not None and r_wrist is not None else None
        shoulder_y_avg = (l_shoulder[1] + r_shoulder[1]) / 2 if l_shoulder is not None and r_shoulder is not None else None
        
        # Rule cascade (priority order)
        label, confidence = self._apply_rules(
            body_angle=body_angle,
            hand_head_dist=hand_head_dist,
            wrist_y=wrist_y_avg,
            shoulder_y=shoulder_y_avg,
            nose=nose,
            l_hip=l_hip,
            r_hip=r_hip
        )
        
        if label is None:
            return None
        
        # Create ADLAction
        action = ADLAction(
            label=label,
            confidence=confidence,
            severity=ACTIVITY_SEVERITY_MAP[label],
            bbox=bbox,
            keypoints=keypoints.tolist(),
            timestamp=time.time()
        )
        
        return action
    
    def reset(self):
        """Reset state"""
        self.last_recognition_time = 0
    
    # ==================== Helper Methods ====================
    
    def _get_kpt(self, keypoints: np.ndarray, idx: int) -> Optional[np.ndarray]:
        """Get keypoint if confidence > threshold"""
        kpt = keypoints[idx]
        return kpt[:2] if kpt[2] > self.conf_threshold else None
    
    def _calculate_body_angle(self, nose, l_hip, r_hip, l_ankle, r_ankle) -> Optional[float]:
        """
        Calculate body tilt angle (degrees from vertical)
        Returns None if insufficient keypoints
        """
        if nose is None or (l_hip is None and r_hip is None):
            return None
        
        # Use average hip position
        hip = l_hip if r_hip is None else (r_hip if l_hip is None else (l_hip + r_hip) / 2)
        
        # Vector from hip to nose
        vec = nose - hip
        angle = np.degrees(np.arctan2(vec[0], -vec[1]))  # Angle from vertical
        
        return abs(angle)
    
    def _hand_to_head_distance(self, nose, l_wrist, r_wrist) -> Optional[float]:
        """
        Calculate minimum distance from hands to head (normalized)
        """
        if nose is None:
            return None
        
        distances = []
        if l_wrist is not None:
            distances.append(np.linalg.norm(nose - l_wrist))
        if r_wrist is not None:
            distances.append(np.linalg.norm(nose - r_wrist))
        
        return min(distances) if distances else None
    
    def _apply_rules(self, body_angle, hand_head_dist, wrist_y, shoulder_y, nose, l_hip, r_hip) -> Tuple[Optional[ADLLabel], float]:
        """
        Apply heuristic rules in priority order
        
        Returns:
            (label, confidence) or (None, 0)
        """
        
        # RULE 1: Fall Detection (highest priority)
        # Body angle > 60° from vertical OR head below hips
        if body_angle is not None and body_angle > 60:
            return ADLLabel.FALL, 0.9
        
        if nose is not None and (l_hip is not None or r_hip is not None):
            hip_y = l_hip[1] if r_hip is None else (r_hip[1] if l_hip is None else (l_hip[1] + r_hip[1]) / 2)
            if nose[1] > hip_y + 50:  # Head significantly below hips (pixels)
                return ADLLabel.FALL, 0.85
        
        # RULE 2: Stroke-like / Abnormal Posture
        # One side significantly lower than other (asymmetry)
        # TODO: Implement asymmetry detection (Phase A.2)
        
        # RULE 3: Sleeping
        # Body horizontal (angle > 70°) AND stable position
        if body_angle is not None and body_angle > 70:
            return ADLLabel.SLEEPING, 0.8
        
        # RULE 4: Eating/Drinking
        # Hand near mouth (distance < threshold)
        if hand_head_dist is not None and hand_head_dist < 80:  # pixels
            # Distinguish eating vs drinking by hand height relative to nose
            # For now, default to eating
            return ADLLabel.EATING, 0.7
        
        # RULE 5: Phone / Online Learning
        # Hands in front of chest, below head
        if wrist_y is not None and shoulder_y is not None:
            if shoulder_y < wrist_y < shoulder_y + 150:  # Hands in chest region
                return ADLLabel.PHONE, 0.65
        
        # RULE 6: Reading
        # Hands below shoulders, body upright
        if body_angle is not None and body_angle < 30:  # Upright
            if wrist_y is not None and shoulder_y is not None:
                if wrist_y > shoulder_y + 100:  # Hands lower (holding book/tablet)
                    return ADLLabel.READING, 0.6
        
        # No rule matched
        return None, 0.0
