"""
HAVEN ADL - Posture Inference
=============================
Logic suy luận tư thế (Posture) và trích xuất đặc trưng (Feature Extraction) từ Keypoints.
"""

import math
import numpy as np
from typing import List, Tuple, Dict
from .data import FrameData, Keypoint

class PostureClassifier:
    """
    Phân loại tư thế: STANDING, SITTING, LAYING
    Dựa trên:
    1. Torso Angle (Góc thân người)
    2. BBox Aspect Ratio (Tỷ lệ khung hình)
    3. Relative Keypoint Positions
    """
    
    # Keypoint Indices (COCO Format)
    NOSE = 0
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    
    def __init__(self):
        pass

    def process(self, frame_data: FrameData) -> FrameData:
        """
        Xử lý chính: Tính features và xác định posture
        """
        # 1. Feature Extraction
        frame_data.aspect_ratio = self._calc_aspect_ratio(frame_data.bbox)
        frame_data.torso_angle = self._calc_torso_angle(frame_data.keypoints)
        frame_data.hand_up = self._check_hand_up(frame_data.keypoints)
        
        # 2. Posture Classification
        frame_data.posture = self._classify_posture(
            frame_data.torso_angle, 
            frame_data.aspect_ratio
        )
        
        return frame_data

    def _calc_aspect_ratio(self, bbox: List[float]) -> float:
        """Calculate Width / Height"""
        if not bbox: return 0.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if h <= 0: return 0.0
        return w / h

    def _calc_torso_angle(self, keypoints: List[List[float]]) -> float:
        """
        Tính góc nghiêng của thân người so với phương thẳng đứng (Vertical).
        0 độ = đứng thẳng. 90 độ = nằm ngang.
        Dùng trung điểm 2 vai và trung điểm 2 hông.
        """
        if not keypoints or len(keypoints) < 13: return 0.0
        
        # Helper to get point
        def get_pt(idx):
            if idx < len(keypoints) and keypoints[idx][2] > 0.3: # Conf check
                return np.array(keypoints[idx][:2])
            return None

        l_sh = get_pt(self.LEFT_SHOULDER)
        r_sh = get_pt(self.RIGHT_SHOULDER)
        l_hip = get_pt(self.LEFT_HIP)
        r_hip = get_pt(self.RIGHT_HIP)
        
        # Cần ít nhất 1 vai và 1 hông đối diện hoặc cùng bên
        if l_sh is None and r_sh is None: return 0.0
        if l_hip is None and r_hip is None: return 0.0
        
        # Mid-shoulder
        if l_sh is not None and r_sh is not None:
            mid_sh = (l_sh + r_sh) / 2
        elif l_sh is not None: mid_sh = l_sh
        else: mid_sh = r_sh
        
        # Mid-hip
        if l_hip is not None and r_hip is not None:
            mid_hip = (l_hip + r_hip) / 2
        elif l_hip is not None: mid_hip = l_hip
        else: mid_hip = r_hip
        
        # Vector from Hip to Shoulder (Upward)
        vector = mid_sh - mid_hip
        dx, dy = vector
        
        # Angle with vertical axis (0, -1)
        # Vertical is -Y in image coords
        # atan2(dx, -dy) -> 0 if dx=0, dy=-1 (up)
        # convert to degrees
        angle_rad = math.atan2(abs(dx), abs(dy)) # Deviation from vertical
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg

    def _check_hand_up(self, keypoints: List[List[float]]) -> bool:
        """Check if any wrist is significantly above nose or eyes"""
        # Keypoints: (17, 3) [x, y, conf]
        if not keypoints or len(keypoints) < 13: return False
        
        # Helper to get point
        def get_y(idx):
            if idx < len(keypoints) and keypoints[idx][2] > 0.3:
                return keypoints[idx][1]
            return None

        y_l_wrist = get_y(9)
        y_r_wrist = get_y(10)
        y_l_shoulder = get_y(5)
        y_r_shoulder = get_y(6)
        
        # Need shoulder as reference
        threshold = 0.1 # Rel to height? Hard to say without bbox context. 
        # Image Y increases downwards. So UP means strictly smaller Y.
        
        # Check Left Arm
        left_up = False
        if y_l_wrist is not None and y_l_shoulder is not None:
            if y_l_wrist < (y_l_shoulder - 20): # 20px buffer
                left_up = True
                
        # Check Right Arm
        right_up = False
        if y_r_wrist is not None and y_r_shoulder is not None:
             if y_r_wrist < (y_r_shoulder - 20):
                 right_up = True
                 
        return left_up or right_up

    def _classify_posture(self, angle: float, ar: float) -> str:
        """
        Rule-based classification
        Angle: 0=Upright, 90=Horizontal
        AR: Width/Height
        """
        # LAYING: Góc nghiêng lớn (>60) HOẶC khung hình rất dẹt (AR > 1.2)
        if angle > 65 or ar > 1.4:
            return "LAYING"
            
        # SITTING: Góc nghiêng trung bình (20-60) hoặc AR hơi lớn
        # Heuristic: Sitting thường AR ~ 0.6-0.9, Standing AR ~ 0.3-0.5
        if ar > 0.6:
            return "SITTING"
            
        # STANDING: Góc nghiêng nhỏ (<30) VÀ AR nhỏ
        if angle < 45 and ar < 0.8:
            return "STANDING"
            
        return "UNKNOWN"
