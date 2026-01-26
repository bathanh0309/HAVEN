"""
Simple IOU-based Tracker
"""
import numpy as np
from typing import List, Dict, Any

class SimpleTracker:
    def __init__(self, max_age=30, iou_threshold=0.3):
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.ages = {}
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Cập nhật trạng thái tracker với danh sách detections mới.
        Gán ID cho các đối tượng dựa trên độ trùng khớp IOU.
        """
        tracked = []
        matched_ids = set()
        
        for det in detections:
            bbox = det.get('bbox', [0,0,0,0])
            # Assume bbox is already [x1, y1, x2, y2]
             
            
            best_iou = 0
            best_id = None
            
            for tid, track in self.tracks.items():
                if tid in matched_ids:
                    continue
                iou = self._compute_iou(bbox, track['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_id = tid
            
            if best_id is not None:
                det['track_id'] = best_id
                self.tracks[best_id] = {'bbox': bbox, 'det': det}
                self.ages[best_id] = 0
                matched_ids.add(best_id)
            else:
                det['track_id'] = self.next_id
                self.tracks[self.next_id] = {'bbox': bbox, 'det': det}
                self.ages[self.next_id] = 0
                self.next_id += 1
            
            tracked.append(det)
        
        # Age out old tracks
        to_remove = []
        for tid in self.tracks:
            if tid not in matched_ids:
                self.ages[tid] = self.ages.get(tid, 0) + 1
                if self.ages[tid] > self.max_age:
                    to_remove.append(tid)
        
        for tid in to_remove:
            del self.tracks[tid]
            del self.ages[tid]
        
        return tracked
    
    def _compute_iou(self, bbox1, bbox2) -> float:
        """
        Tính toán Intersection Over Union (IOU) giữa 2 box.
        Format: [x1, y1, x2, y2] (Top-Left, Bottom-Right)
        """
        # Unpack coordinates
        box1_x1, box1_y1, box1_x2, box1_y2 = bbox1
        box2_x1, box2_y1, box2_x2, box2_y2 = bbox2
        
        # Intersection coordinates
        xi_min = max(box1_x1, box2_x1)
        yi_min = max(box1_y1, box2_y1)
        xi_max = min(box1_x2, box2_x2)
        yi_max = min(box1_y2, box2_y2)
        
        # Check if intersection exists
        if xi_max <= xi_min or yi_max <= yi_min:
            return 0.0
        
        # Calculate areas
        inter_area = (xi_max - xi_min) * (yi_max - yi_min)
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        
        union_area = box1_area + box2_area - inter_area
        
        if union_area <= 0: return 0.0
        return inter_area / union_area

tracker = SimpleTracker()
