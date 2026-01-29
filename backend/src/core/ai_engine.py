
import logging
import time
import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple, List
from ultralytics import YOLO
from config.camera_config import CameraSettings

logger = logging.getLogger(__name__)

class AIEngine:
    """
    AI Processing Engine with ROI, Frame Skipping & Caching.
    Optimized to detect distant objects (e.g., person on bed) using ROI cropping.
    """
    
    def __init__(self, config: CameraSettings):
        self.config = config
        self.model = None
        self.model_path = config.AI_MODEL_PATH
        
        # AI Parameters
        self.skip_frames = config.AI_SKIP_FRAMES
        self.conf_thres = config.AI_CONF_THRES
        self.iou_thres = config.AI_IOU_THRES
        self.img_size = config.AI_IMG_SIZE
        self.max_det = config.AI_MAX_DET
        
        self.frame_counter = 0
        self.last_results = None
        self.last_inference_time = 0
        self.last_detections_data = [] # Store simple dict for frontend
    
    def load_model(self):
        """
        Tải model YOLO từ đường dẫn cấu hình.
        Chỉ tải nếu model chưa được khởi tạo.
        """
        if self.model: return
        try:
            logger.info(f"Loading AI Model: {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info(f"AI Model Loaded (Conf={self.conf_thres}, ROI={self.config.ROI_ENABLED})")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def _parse_results(self, results, roi_offset=(0,0)) -> List[Dict]:
        """
        Chuyển đổi kết quả từ YOLO (Boxes, Keypoints) sang định dạng danh sách dictionary dễ dùng.
        Đồng thời điều chỉnh tọa độ (offset) nếu đang dùng chế độ ROI.
        """
        detections = []
        if not results: return detections
        
        # Results is a list (batch size 1), take [0]
        res = results[0]
        
        # Boxes (N, 6) -> xyxy, conf, cls
        boxes = res.boxes
        if boxes is None: return detections
        
        rx, ry = roi_offset
        
        for box in boxes:
            try:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = res.names[cls_id]
                
                # Get Track ID if available
                track_id = int(box.id[0]) if box.id is not None else -1
                
                # Adjust for ROI offset
                x1 += rx
                x2 += rx
                y1 += ry
                y2 += ry
                
                detection = {
                    "label": f"{label} #{track_id}" if track_id > 0 else label,
                    "track_id": track_id,
                    "conf": round(conf, 2),
                    "box": [int(x1), int(y1), int(x2), int(y2)]
                }
                
                # Extract Keypoints (x, y, conf)
                if hasattr(res, 'keypoints') and res.keypoints is not None:
                     # keypoints.data is (N, 17, 3) [x, y, conf]
                     # We take the corresponding index for this box
                     # Note: boxes and keypoints are usually aligned in Ultralytics results
                     # But proper way is to iterate if we can, or match index.
                     # Actually res.keypoints has same length as boxes.
                     
                     # Simple approach: Ultralytics Results object iterates boxes. 
                     # But here we are iterating 'box' from 'res.boxes'.
                     # Let's get the index of the box or assume sequential.
                     # Better: use res.keypoints[i] if we had index.
                     pass 
                
                detections.append(detection)
            except Exception as e:
                pass
        
        # Re-iterate with index to be safe for keypoints matching
        if hasattr(res, 'keypoints') and res.keypoints is not None:
            kpts_data = res.keypoints.data.cpu().numpy() # (N, 17, 3)
            
            # Update detections with keypoints
            for i, det in enumerate(detections):
                if i < len(kpts_data):
                    kpts = kpts_data[i].copy()
                    # kpts is (17, 3) -> [[x, y, conf], ...]
                    
                    # Adjust ROI offset for x, y
                    kpts[:, 0] += rx
                    kpts[:, 1] += ry
                    
                    det['keypoints'] = kpts.tolist() # Convert to list for JSON serialization

        return detections

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Thực hiện quy trình suy luận AI:
        1. Cắt vùng ROI (nếu bật).
        2. Chạy model YOLO (nếu đến lượt frame cần chạy).
        3. Vẽ box/skeleton lên frame.
        4. Trả về frame đã vẽ và danh sách phát hiện.
        """
        if self.model is None:
            self.load_model()
            if self.model is None:
                return frame, []

        self.frame_counter += 1
        h, w = frame.shape[:2]
        
        # ==========================
        # ROI Logic
        # ==========================
        roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, w, h
        inference_frame = frame
        
        if self.config.ROI_ENABLED:
            roi_coords = self.config.get_roi_coords(w, h)
            roi_x1, roi_y1, roi_x2, roi_y2 = roi_coords
            # Validate coords
            roi_x1, roi_y1 = max(0, roi_x1), max(0, roi_y1)
            roi_x2, roi_y2 = min(w, roi_x2), min(h, roi_y2)
            
            if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                inference_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy() # Copy essential for contiguous array
            
        # ==========================
        # Inference Logic (with Skip)
        # ==========================
        should_run_ai = (self.frame_counter % self.skip_frames == 0)
        
        if should_run_ai:
            try:
                start_time = time.time()
                
                results = self.model.track(
                    inference_frame, 
                    verbose=False, 
                    persist=True,
                    conf=self.conf_thres, 
                    iou=self.iou_thres,
                    imgsz=self.img_size,
                    max_det=self.max_det,
                    tracker="bytetrack.yaml"
                )
                
                self.last_results = results
                self.last_inference_time = (time.time() - start_time) * 1000
                
                # Parse and Store Data
                self.last_detections_data = self._parse_results(results, roi_offset=(roi_x1, roi_y1))
                
            except Exception as e:
                logger.error(f"Inference Error: {e}")
        
        # ==========================
        # Drawing Logic
        # ==========================
        annotated_frame = frame.copy()

        # Strategy: Use YOLO's built-in plot() to get Pose skeletons + Boxes
        # If ROI is active, we plot on the crop, then paste it back.
        
        if self.last_results:
            try:
                # results.plot() returns the annotated image (BGR numpy array)
                # It handles boxes, labels, and skeletons automatically.
                annotated_crop = self.last_results[0].plot()
                
                # If we used ROI, paste the annotated crop back into the original frame
                if self.config.ROI_ENABLED and roi_x2 > roi_x1 and roi_y2 > roi_y1:
                    # Resize logic check (sometimes plot() might resize? mainly it follows input size)
                    # The input to model was 'inference_frame' (which is the crop)
                    # So annotated_crop should match crop dimensions.
                    
                    # Safety check dimensions
                    ac_h, ac_w = annotated_crop.shape[:2]
                    target_h = roi_y2 - roi_y1
                    target_w = roi_x2 - roi_x1
                    
                    if ac_h == target_h and ac_w == target_w:
                        annotated_frame[roi_y1:roi_y2, roi_x1:roi_x2] = annotated_crop
                    else:
                        # If sizes mismatch (rare), just fallback to resize
                         annotated_frame[roi_y1:roi_y2, roi_x1:roi_x2] = cv2.resize(annotated_crop, (target_w, target_h))
                else:
                    # Full frame mode - just use the plotted component
                    # But wait, 'inference_frame' was 'frame'. 
                    # So annotated_crop IS the full frame.
                    annotated_frame = annotated_crop

            except Exception as e:
                logger.error(f"Drawing Error: {e}")
                # Fallback to manual box drawing if plot() fails
                for det in self.last_detections_data:
                    x1, y1, x2, y2 = det['box']
                    color = (0, 255, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        # 3. Draw ROI Box (Debug) - Only if enabled
        if self.config.ROI_ENABLED and self.config.AI_DEBUG_OVERLAY:
            cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
            cv2.putText(annotated_frame, "ROI AREA", (roi_x1, roi_y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 4. Draw Debug Info (Simplified)
        if self.config.AI_DEBUG_OVERLAY:
             cv2.putText(annotated_frame, f"Inf: {self.last_inference_time:.1f}ms", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return annotated_frame, self.last_detections_data

# Singleton
_ai_instance = None
def get_ai_engine(config: CameraSettings = None) -> AIEngine:
    """Trả về instance duy nhất của AIEngine (Singleton Pattern)."""
    global _ai_instance
    global _ai_instance
    if _ai_instance is None and config:
        _ai_instance = AIEngine(config)
    return _ai_instance
