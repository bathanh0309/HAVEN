"""
YOLO pose estimation wrapper.
"""
from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Optional, Tuple


class PoseEstimator:
    """
    YOLO-based pose estimation.
    """
    
    def __init__(self, model_path: str = "yolov8n-pose.pt", 
                 conf_threshold: float = 0.3,
                 device: str = "cuda"):
        """
        Initialize pose estimator.
        
        Args:
            model_path: Path to YOLO-Pose model
            conf_threshold: Confidence threshold
            device: "cuda" or "cpu"
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        
        # COCO keypoint names (17 keypoints)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def estimate(self, frame: np.ndarray) -> List[Dict]:
        """
        Estimate poses in frame.
        
        Args:
            frame: Input image (BGR)
        
        Returns:
            List of pose detections: [{
                'bbox': [x, y, w, h],
                'conf': float,
                'keypoints': np.ndarray (17, 3),  # [x, y, confidence]
                'visible_keypoints': int
            }]
        """
        # Run inference
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )
        
        # Parse results
        poses = []
        for result in results:
            # Get boxes and keypoints
            boxes = result.boxes
            keypoints = result.keypoints
            
            if boxes is None or keypoints is None:
                continue
                
            for i, box in enumerate(boxes):
                # Convert bbox from xyxy to xywh
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = x2 - x1
                h = y2 - y1
                
                # Get keypoints for this detection
                kpts = keypoints[i].data[0].cpu().numpy()  # Shape: (17, 3)
                
                # Count visible keypoints (confidence > 0.5)
                visible_count = np.sum(kpts[:, 2] > 0.5)
                
                poses.append({
                    'bbox': [int(x1), int(y1), int(w), int(h)],
                    'conf': float(box.conf[0]),
                    'keypoints': kpts,
                    'visible_keypoints': int(visible_count)
                })
        
        return poses
    
    def get_posture(self, keypoints: np.ndarray) -> str:
        """
        Classify posture based on keypoints.
        
        Args:
            keypoints: (17, 3) array of keypoints
        
        Returns:
            Posture label: 'standing', 'sitting', 'lying', 'unknown'
        """
        # Simple heuristic based on hip/shoulder/ankle positions
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
        # Check if keypoints are visible
        if (left_shoulder[2] < 0.5 or right_shoulder[2] < 0.5 or
            left_hip[2] < 0.5 or right_hip[2] < 0.5):
            return 'unknown'
        
        # Calculate average positions
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2
        ankle_y = (left_ankle[1] + right_ankle[1]) / 2 if left_ankle[2] > 0.5 and right_ankle[2] > 0.5 else hip_y
        
        # Vertical extent (head to feet)
        vertical_extent = ankle_y - shoulder_y
        
        # Horizontal check for lying down
        torso_width = abs(right_shoulder[0] - left_shoulder[0])
        torso_height = abs(hip_y - shoulder_y)
        
        # Lying down: torso is more horizontal than vertical
        if torso_width > torso_height * 1.5:
            return 'lying'
        
        # Standing vs sitting based on vertical extent
        if vertical_extent > torso_height * 2:
            return 'standing'
        elif vertical_extent > torso_height * 0.5:
            return 'sitting'
        else:
            return 'unknown'


if __name__ == "__main__":
    # Test pose estimator
    import cv2
    
    estimator = PoseEstimator("yolov8n-pose.pt", conf_threshold=0.3)
    
    # Load test image
    frame = cv2.imread("/path/to/test/image.jpg")
    poses = estimator.estimate(frame)
    
    print(f"Detected {len(poses)} poses")
    for i, pose in enumerate(poses):
        posture = estimator.get_posture(pose['keypoints'])
        print(f"  Person {i}: posture={posture}, visible_kpts={pose['visible_keypoints']}")

