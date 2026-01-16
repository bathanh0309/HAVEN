"""
ADL Action Definitions and Constants
"""
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


class ADLLabel(str, Enum):
    """ADL activity labels"""
    EATING = "eating"
    DRINKING = "drinking"
    READING = "reading"
    SLEEPING = "sleeping"
    PHONE = "phone"  # Playing phone / online learning
    FALL = "fall"
    STROKE_LIKE = "stroke_like"  # Abnormal posture
    UNKNOWN = "unknown"


class SeverityLevel(str, Enum):
    """Alert severity levels"""
    LOW = "low"          # Normal activities
    MEDIUM = "medium"    # Unusual but not dangerous
    HIGH = "high"        # Potentially dangerous
    CRITICAL = "critical"  # Immediate attention required


@dataclass
class ADLAction:
    """
    Represents a detected ADL action
    """
    label: ADLLabel
    confidence: float
    severity: SeverityLevel
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    keypoints: List[Tuple[float, float, float]]  # [(x, y, conf), ...] 17 COCO keypoints
    timestamp: float
    person_id: int = None  # Track ID from tracker
    
    def to_dict(self):
        return {
            "label": self.label.value,
            "confidence": round(self.confidence, 3),
            "severity": self.severity.value,
            "bbox": {"x": self.bbox[0], "y": self.bbox[1], 
                     "w": self.bbox[2], "h": self.bbox[3]},
            "keypoints": self.keypoints,
            "timestamp": self.timestamp,
            "person_id": self.person_id
        }


# COCO Keypoint indices (YOLOv8-Pose format)
class KeypointIndex:
    """COCO 17 keypoints indices"""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


# Activity-Severity mapping
ACTIVITY_SEVERITY_MAP = {
    ADLLabel.EATING: SeverityLevel.LOW,
    ADLLabel.DRINKING: SeverityLevel.LOW,
    ADLLabel.READING: SeverityLevel.LOW,
    ADLLabel.SLEEPING: SeverityLevel.LOW,
    ADLLabel.PHONE: SeverityLevel.LOW,
    ADLLabel.FALL: SeverityLevel.CRITICAL,
    ADLLabel.STROKE_LIKE: SeverityLevel.HIGH,
    ADLLabel.UNKNOWN: SeverityLevel.MEDIUM
}
