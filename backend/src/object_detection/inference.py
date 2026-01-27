"""
Object Detection Module for HAVEN
==================================
Detect and classify household objects and items.
"""

from typing import List, Dict, Tuple
import numpy as np

class ObjectDetector:
    """
    Detect household objects and items in frames.
    Supports custom object classes with configurable confidence thresholds.
    """
    
    def __init__(
        self,
        model_path: str,
        object_classes: List[str],
        class_thresholds: Dict[str, float] = None,
        default_threshold: float = 0.3
    ):
        """
        Initialize object detector.
        
        Args:
            model_path: Path to object detection model
            object_classes: List of object class names to detect
            class_thresholds: Per-class confidence thresholds
            default_threshold: Default confidence threshold for classes not in class_thresholds
        """
        self.model_path = model_path
        self.object_classes = object_classes
        self.class_thresholds = class_thresholds or {}
        self.default_threshold = default_threshold
        
        # TODO: Load model
        self.model = None
    
    def get_threshold(self, class_name: str) -> float:
        """
        Get confidence threshold for a specific class.
        
        Args:
            class_name: Object class name
            
        Returns:
            Confidence threshold for the class
        """
        return self.class_thresholds.get(class_name, self.default_threshold)
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of detections:
            [
                {
                    "bbox": (x1, y1, x2, y2),
                    "class": "phone",
                    "confidence": 0.85,
                    "class_id": 5
                },
                ...
            ]
        """
        # TODO: Implement object detection
        detections = []
        
        # Filter by class-specific thresholds
        filtered_detections = []
        for det in detections:
            threshold = self.get_threshold(det["class"])
            if det["confidence"] >= threshold:
                filtered_detections.append(det)
        
        return filtered_detections
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process frame and return object detection results.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dict containing:
            {
                "objects": [...],  # List of detections
                "count": 5,        # Total objects detected
                "classes": ["phone", "cup", ...]  # Unique classes detected
            }
        """
        objects = self.detect_objects(frame)
        
        return {
            "objects": objects,
            "count": len(objects),
            "classes": list(set(obj["class"] for obj in objects))
        }
