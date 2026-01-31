"""
YOLO person detector wrapper.
"""
from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Tuple


class PersonDetector:
    """
    YOLO-based person detector.
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", 
                 conf_threshold: float = 0.5,
                 device: str = "cuda"):
        """
        Initialize detector.
        
        Args:
            model_path: Path to YOLO model
            conf_threshold: Confidence threshold
            device: "cuda" or "cpu"
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        
        # Person class ID in COCO
        self.person_class_id = 0
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons in frame.
        
        Args:
            frame: Input image (BGR)
        
        Returns:
            List of detections: [{
                'bbox': [x, y, w, h],
                'conf': float,
                'class_id': int
            }]
        """
        # Run inference
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            device=self.device,
            classes=[self.person_class_id],  # Only detect persons
            verbose=False
        )
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Convert from xyxy to xywh
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = x2 - x1
                h = y2 - y1
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(w), int(h)],
                    'conf': float(box.conf[0]),
                    'class_id': int(box.cls[0])
                })
        
        return detections


if __name__ == "__main__":
    # Test detector
    import cv2
    
    detector = PersonDetector("yolov8n.pt", conf_threshold=0.5)
    
    # Load test image
    frame = cv2.imread("/path/to/test/image.jpg")
    detections = detector.detect(frame)
    
    print(f"Detected {len(detections)} persons")
    for i, det in enumerate(detections):
        print(f"  Person {i}: bbox={det['bbox']}, conf={det['conf']:.2f}")

