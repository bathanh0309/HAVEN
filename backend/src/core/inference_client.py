"""
Roboflow Inference Client
"""
import cv2
import base64
import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class InferenceClient:
    def __init__(self, base_url="http://localhost:9001"):
        self.base_url = base_url
        self.detect_model = "yolov8n-640"
        self.pose_model = "yolov8n-pose-640"
        self.timeout = 2
    
    def detect_objects(self, frame_bgr, confidence=0.5) -> Dict[str, Any]:
        try:
            _, buffer = cv2.imencode('.jpg', frame_bgr)
            img_b64 = base64.b64encode(buffer).decode()
            
            resp = requests.post(
                f"{self.base_url}/infer/object_detection",
                json={
                    "model_id": self.detect_model,
                    "image": {"type": "base64", "value": img_b64},
                    "confidence": confidence
                },
                timeout=self.timeout
            )
            return resp.json()
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {"predictions": []}
    
    def detect_pose(self, frame_bgr, confidence=0.3) -> Dict[str, Any]:
        """
        Gửi yêu cầu phát hiện dáng người (Pose Estimation) đến server suy luận.
        Sử dụng model YOLOv8 Pose.
        """
        try:
            _, buffer = cv2.imencode('.jpg', frame_bgr)
            img_b64 = base64.b64encode(buffer).decode()
            
            resp = requests.post(
                f"{self.base_url}/infer/keypoints_detection",
                json={
                    "model_id": self.pose_model,
                    "image": {"type": "base64", "value": img_b64},
                    "confidence": confidence
                },
                timeout=self.timeout
            )
            return resp.json()
        except Exception as e:
            logger.error(f"Pose detection error: {e}")
            return {"predictions": []}
    
    def health_check(self) -> bool:
        """
        Kiểm tra kết nối đến server suy luận.
        """
        try:
            resp = requests.get(f"{self.base_url}/", timeout=1)
            return resp.status_code == 200
        except:
            return False

inference_client = InferenceClient()
