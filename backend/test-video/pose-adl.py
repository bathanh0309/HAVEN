"""
HAVEN - Pose + ADL + Zone Intrusion + Object Detection
=======================================================
Features:
- Pose detection + ADL classification (Standing, Walking, Sitting, Laying)
- Majority voting cho tu the on dinh
- Phat hien su kien: Fall, Hand Raise, Sitting Down, Standing Up
- Zone Intrusion: Canh bao khi nguoi vao vung cam
- Dangerous Object Detection: Phat hien phone, knife, gun (COCO model)
- Xuat file GIF voi phim G
"""

import cv2
import time
import math
import numpy as np
from pathlib import Path
from collections import deque, Counter
from typing import Dict, List, Tuple, Optional
from PIL import Image
from ultralytics import YOLO

# Import all configurations
from config import *

# All configurations imported from config.py

# ============================================================
# LỚP QUẢN LÝ TRẠNG THÁI THEO DÕI (TRACKING)
# ============================================================
class TrackState:
    """Lưu trữ lịch sử và trạng thái của một đối tượng đang được theo dõi"""
    
    def __init__(self, track_id: int, frame_height: int):
        self.track_id = track_id
        self.frame_height = frame_height
        
        # Lịch sử vị trí và tư thế
        self.positions: deque = deque(maxlen=POSITION_HISTORY_MAXLEN)
        self.postures: deque = deque(maxlen=POSTURE_HISTORY_MAXLEN)  
        
        # Trạng thái hiện tại
        self.current_posture = "UNKNOWN"
        self.prev_posture = "UNKNOWN"
        self.posture_start_time = time.time()
        
        # Theo dõi chuyển động
        self.last_center = None
        self.velocity = 0.0
        
        # Sự kiện
        self.events: deque = deque(maxlen=EVENT_HISTORY_MAXLEN)
        
        # Theo dõi tay giơ
        self.left_hand_raised_frames = 0
        self.right_hand_raised_frames = 0
        
    def update_position(self, bbox: List[float]):
        """Cập nhật vị trí mới và tính toán tốc độ tức thời"""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        current_time = time.time()
        
        if self.last_center is not None:
            dx = cx - self.last_center[0]
            dy = cy - self.last_center[1]
            dist = math.sqrt(dx*dx + dy*dy)
            self.velocity = dist * ASSUMED_FPS  # pixels/second
        
        self.last_center = (cx, cy)
        self.positions.append((current_time, cx, cy))
        
    def get_movement_in_window(self, window_sec: float = 1.0) -> float:
        """Tính tổng quãng đường di chuyển trong một khoảng thời gian"""
        if len(self.positions) < 2:
            return 0.0
            
        now = time.time()
        total_dist = 0.0
        prev = None
        
        for t, x, y in self.positions:
            if now - t > window_sec:
                continue
            if prev is not None:
                dx = x - prev[0]
                dy = y - prev[1]
                total_dist += math.sqrt(dx*dx + dy*dy)
            prev = (x, y)
            
        return total_dist
    
    def add_posture(self, posture: str):
        """ THÊM TƯ THẾ VÀ THỰC HIỆN MAJORITY VOTING"""
        self.postures.append(posture)
        
        if len(self.postures) >= POSTURE_VOTING_MIN_FRAMES:
            # Bỏ phiếu - lấy tư thế xuất hiện nhiều nhất
            counter = Counter(self.postures)
            voted_posture = counter.most_common(1)[0][0]
            
            # Phát hiện chuyển đổi tư thế
            if voted_posture != self.current_posture:
                self.prev_posture = self.current_posture
                self.current_posture = voted_posture
                self.posture_start_time = time.time()
                
                # PHÁT HIỆN SỰ KIỆN CHUYỂN ĐỔI
                if self.prev_posture == "STANDING" and voted_posture == "SITTING":
                    self.add_event(EVENT_NAMES["sitting_down"])
                elif self.prev_posture == "SITTING" and voted_posture == "STANDING":
                    self.add_event(EVENT_NAMES["standing_up"])
                elif voted_posture == "LAYING" and self.prev_posture not in ["LAYING", "UNKNOWN"]:
                    self.add_event(EVENT_NAMES["fall_down"])
        else:
            # Chưa đủ dữ liệu voting
            self.current_posture = posture
    
    def add_event(self, event: str):
        """Thêm sự kiện mới"""
        timestamp = time.strftime("%H:%M:%S")
        event_text = f"{timestamp} - {event}"
        self.events.append(event_text)
        print(f"[ID:{self.track_id}] EVENT: {event}")
    
    def check_hand_raise(self, keypoints: np.ndarray) -> Optional[str]:
        """ PHÁT HIỆN GIƠ TAY"""
        def get_point(idx):
            if keypoints[idx][2] > KEYPOINT_CONF_THRESHOLD:
                return keypoints[idx][:2]
            return None
        
        l_wrist = get_point(KP_LEFT_WRIST)
        r_wrist = get_point(KP_RIGHT_WRIST)
        l_shoulder = get_point(KP_LEFT_SHOULDER)
        r_shoulder = get_point(KP_RIGHT_SHOULDER)
        nose = get_point(KP_NOSE)
        
        if nose is None:
            return None
        
        # Kiểm tra tay trái
        if l_wrist is not None and l_shoulder is not None:
            if l_wrist[1] < nose[1]:  # Y nhỏ hơn = cao hơn
                self.left_hand_raised_frames += 1
                if self.left_hand_raised_frames == HAND_RAISE_FRAMES_THRESHOLD:
                    return EVENT_NAMES["left_hand_raised"]
            else:
                self.left_hand_raised_frames = 0
        
        # Kiểm tra tay phải
        if r_wrist is not None and r_shoulder is not None:
            if r_wrist[1] < nose[1]:
                self.right_hand_raised_frames += 1
                if self.right_hand_raised_frames == HAND_RAISE_FRAMES_THRESHOLD:
                    return EVENT_NAMES["right_hand_raised"]
            else:
                self.right_hand_raised_frames = 0
        
        return None

# ============================================================
# TRACKER DỰA TRÊN IOU
# ============================================================
class SimpleTracker:
    """Bộ theo dõi đối tượng dựa trên độ trùng khớp Bounding Box (IOU)"""
    
    def __init__(self, iou_threshold=TRACKER_IOU_THRESHOLD, max_age=TRACKER_MAX_AGE, frame_height=720):
        self.tracks: Dict[int, dict] = {}
        self.states: Dict[int, TrackState] = {}
        self.next_id = 1
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.ages: Dict[int, int] = {}
        self.frame_height = frame_height
        
    def update(self, detections: List[dict]) -> List[dict]:
        """Gán ID cho các phát hiện mới dựa trên lịch sử"""
        # Sắp xếp detections theo vị trí x (trái → phải) để ID hiển thị theo thứ tự
        detections_sorted = sorted(detections, key=lambda d: d['bbox'][0])
        
        matched_ids = set()
        results = []
        
        for det in detections_sorted:
            bbox = det['bbox']
            best_iou = 0
            best_id = None
            
            # So sánh với các track hiện có
            for tid, track in self.tracks.items():
                if tid in matched_ids:
                    continue
                iou = self._compute_iou(bbox, track['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_id = tid
            
            if best_id is not None:
                det['track_id'] = best_id
                self.tracks[best_id] = {'bbox': bbox}
                self.ages[best_id] = 0
                matched_ids.add(best_id)
            else:
                # Tạo ID mới
                det['track_id'] = self.next_id
                self.tracks[self.next_id] = {'bbox': bbox}
                self.states[self.next_id] = TrackState(self.next_id, self.frame_height)
                self.ages[self.next_id] = 0
                self.next_id += 1
            
            # Cập nhật trạng thái
            tid = det['track_id']
            if tid in self.states:
                self.states[tid].update_position(bbox)
            
            results.append(det)
        
        # Xóa các track cũ
        to_remove = []
        for tid in self.tracks:
            if tid not in matched_ids:
                self.ages[tid] = self.ages.get(tid, 0) + 1
                if self.ages[tid] > self.max_age:
                    to_remove.append(tid)
        
        for tid in to_remove:
            del self.tracks[tid]
            if tid in self.states:
                del self.states[tid]
            if tid in self.ages:
                del self.ages[tid]
        
        return results
    
    def _compute_iou(self, box1, box2):
        """Tính toán Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0

# ============================================================
# PHÂN LOẠI TƯ THẾ (POSTURE CLASSIFICATION) - CÓ GÓC ĐẦU GỐI
# ============================================================
def classify_posture(keypoints: np.ndarray, bbox: List[float], 
                     track_state: TrackState, frame_height: int) -> str:
    """
     CÓ GÓC ĐẦU GỐI - PHÂN LOẠI CHÍNH XÁC HƠN
    """
    if keypoints is None or len(keypoints) < 17:
        return "UNKNOWN"
    
    def get_point(idx):
        if keypoints[idx][2] > KEYPOINT_CONF_THRESHOLD:
            return keypoints[idx][:2]
        return None
    
    # Lấy các điểm quan trọng
    l_sh = get_point(KP_LEFT_SHOULDER)
    r_sh = get_point(KP_RIGHT_SHOULDER)
    l_hip = get_point(KP_LEFT_HIP)
    r_hip = get_point(KP_RIGHT_HIP)
    l_knee = get_point(KP_LEFT_KNEE)
    r_knee = get_point(KP_RIGHT_KNEE)
    l_ankle = get_point(KP_LEFT_ANKLE)
    r_ankle = get_point(KP_RIGHT_ANKLE)
    
    # 1. Tính góc thân người (Torso Angle)
    torso_angle = 0
    if (l_sh is not None or r_sh is not None) and (l_hip is not None or r_hip is not None):
        mid_sh = (l_sh + r_sh) / 2 if l_sh is not None and r_sh is not None else (l_sh if l_sh is not None else r_sh)
        mid_hip = (l_hip + r_hip) / 2 if l_hip is not None and r_hip is not None else (l_hip if l_hip is not None else r_hip)
        
        dx = abs(mid_sh[0] - mid_hip[0])
        dy = abs(mid_sh[1] - mid_hip[1])
        torso_angle = math.degrees(math.atan2(dx, dy)) if dy > 0 else 0
    
    # 2.  TÍNH GÓC ĐẦU GỐI (Knee Angle) - QUAN TRỌNG!
    knee_angles = []
    
    # Góc đầu gối trái
    if l_hip is not None and l_knee is not None and l_ankle is not None:
        v1 = l_hip - l_knee
        v2 = l_ankle - l_knee
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Tránh lỗi arccos
        angle = math.degrees(math.acos(cos_angle))
        knee_angles.append(angle)
    
    # Góc đầu gối phải
    if r_hip is not None and r_knee is not None and r_ankle is not None:
        v1 = r_hip - r_knee
        v2 = r_ankle - r_knee
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.degrees(math.acos(cos_angle))
        knee_angles.append(angle)
    
    avg_knee_angle = np.mean(knee_angles) if knee_angles else DEFAULT_KNEE_ANGLE
    
    # 3. Tính aspect ratio và movement
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    aspect_ratio = w / h if h > 0 else 0
    movement = track_state.get_movement_in_window(1.0)
    
    #  CHUẨN HÓA NGƯỠNG THEO FRAME HEIGHT
    movement_threshold = frame_height * MOVEMENT_THRESHOLD_RATIO
    
    # --- LOGIC PHÂN LOẠI ---
    # NẰM: Góc thân lớn hoặc bbox ngang
    if torso_angle > TORSO_ANGLE_LAYING_THRESHOLD or aspect_ratio > ASPECT_RATIO_LAYING_THRESHOLD:
        return "LAYING"
    
    # NGỒI: Góc đầu gối nhỏ (gập lại) + ít di chuyển
    if avg_knee_angle < KNEE_ANGLE_SITTING_THRESHOLD and movement < movement_threshold:
        return "SITTING"
    
    # ĐI BỘ: Di chuyển nhiều
    if movement > movement_threshold * MOVEMENT_WALKING_MULTIPLIER:
        return "WALKING"
    
    # ĐỨNG: Mặc định
    return "STANDING"

# ============================================================
# VẼ SKELETON - MÀU SẮC CHÍNH XÁC
# ============================================================
def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, conf_threshold: float = KEYPOINT_CONF_THRESHOLD):
    """Vẽ bộ khung xương với màu sắc chính xác cho từng bộ phận"""
    if keypoints is None or len(keypoints) < 17:
        return
    
    # Vẽ các đường nối xương
    for (start_idx, end_idx, part) in SKELETON_CONNECTIONS:
        if keypoints[start_idx][2] > conf_threshold and keypoints[end_idx][2] > conf_threshold:
            start_pt = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
            end_pt = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
            color = SKELETON_COLORS.get(part, (255, 255, 255))
            cv2.line(frame, start_pt, end_pt, color, SKELETON_LINE_THICKNESS)
    
    # Vẽ các khớp điểm
    for i, kp in enumerate(keypoints):
        if kp[2] > conf_threshold:
            pt = (int(kp[0]), int(kp[1]))
            
            # Gán màu theo vị trí keypoint
            if i <= 4:  # Đầu
                color = SKELETON_COLORS["head"]
            elif i in [5, 6, 11, 12]:  # Vai và hông (Thân)
                color = SKELETON_COLORS["torso"]
            elif i in [7, 8]:  # Cùi chỏ (Tay trên)
                color = SKELETON_COLORS["upper_arm"]
            elif i in [9, 10]:  # Cổ tay (Bàn tay)
                color = SKELETON_COLORS["lower_arm"]
            elif i in [13, 14]:  # Đầu gối (Chân trên)
                color = SKELETON_COLORS["upper_leg"]
            else:  # Mắt cá (Bàn chân)
                color = SKELETON_COLORS["lower_leg"]
            
            cv2.circle(frame, pt, SKELETON_KEYPOINT_RADIUS, color, -1)

# ============================================================
# VẼ DETECTION VỚI EVENTS
# ============================================================
def draw_detection(frame: np.ndarray, bbox: List[float], track_id: int, 
                   posture: str, events: deque):
    """Vẽ hộp giới hạn, nhãn ADL và sự kiện"""
    x1, y1, x2, y2 = map(int, bbox)
    
    color = POSTURE_COLORS.get(posture, (255, 255, 255))
    
    # Vẽ BBox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BBOX_THICKNESS)
    
    # Nhãn chính
    label = f"ID:{track_id} {posture}"
    (tw, th), _ = cv2.getTextSize(label, FONT_FACE, FONT_SCALE_LABEL, FONT_THICKNESS)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 5), FONT_FACE, FONT_SCALE_LABEL, (0, 0, 0), FONT_THICKNESS)
    
    #  HIỂN THỊ SỰ KIỆN GẦN NHẤT
    if events:
        latest_event = list(events)[-1]
        event_parts = latest_event.split(" - ", 1)
        if len(event_parts) == 2:
            display_text = event_parts[1]  # Chỉ lấy phần tên event
        else:
            display_text = latest_event
        
        cv2.putText(frame, display_text, (x1, y2 + 20), 
                   FONT_FACE, FONT_SCALE_EVENT, (0, 0, 255), FONT_THICKNESS)


# ============================================================
# ZONE INTRUSION MANAGER
# ============================================================
class ZoneManager:
    """Quan ly cac vung cam va kiem tra xam nhap"""
    
    def __init__(self, zones: list, frame_width: int, frame_height: int):
        self.zones = []
        for z in zones:
            # Convert percentage to pixel coordinates
            x1 = int(z["coords"][0] * frame_width)
            y1 = int(z["coords"][1] * frame_height)
            x2 = int(z["coords"][2] * frame_width)
            y2 = int(z["coords"][3] * frame_height)
            self.zones.append({
                "name": z["name"],
                "coords": (x1, y1, x2, y2)
            })
    
    def check_intrusion(self, person_bbox) -> Optional[str]:
        """Kiem tra person co trong vung cam khong"""
        px1, py1, px2, py2 = person_bbox
        person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
        
        for zone in self.zones:
            zx1, zy1, zx2, zy2 = zone["coords"]
            if zx1 <= person_center[0] <= zx2 and zy1 <= person_center[1] <= zy2:
                return zone["name"]
        return None
    
    def draw_zones(self, frame):
        """Ve cac vung cam len frame"""
        for zone in self.zones:
            x1, y1, x2, y2 = zone["coords"]
            # Ve rectangle voi border do
            cv2.rectangle(frame, (x1, y1), (x2, y2), ZONE_COLOR, 2)
            # Ve overlay ban trong suot
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), ZONE_COLOR, -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            # Ve text
            cv2.putText(frame, f"ZONE: {zone['name']}", (x1 + 5, y1 + 20), 
                       FONT_FACE, 0.5, ZONE_COLOR, 1)


# ============================================================
# DANGEROUS OBJECT DETECTOR
# ============================================================
class DangerousObjectDetector:
    """Phat hien vat nguy hiem tu COCO model"""
    
    def __init__(self, model_path: str, dangerous_classes: dict, conf_thres: float = 0.3):
        print(f"[INFO] Loading object detection model: {model_path}")
        self.model = YOLO(model_path)
        self.dangerous_classes = dangerous_classes
        self.conf_thres = conf_thres
        self.class_names = list(dangerous_classes.keys())
    
    def detect(self, frame) -> list:
        """Detect dangerous objects in frame"""
        results = self.model(frame, verbose=False, conf=self.conf_thres)[0]
        dangerous_found = []
        
        if results.boxes is not None:
            for box, cls, conf in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.cls.cpu().numpy(),
                results.boxes.conf.cpu().numpy()
            ):
                class_name = self.model.names[int(cls)]
                if class_name in self.class_names:
                    dangerous_found.append({
                        "class": class_name,
                        "bbox": box.tolist(),
                        "conf": float(conf)
                    })
        
        return dangerous_found
    
    def draw_detections(self, frame, detections: list):
        """Ve cac object nguy hiem len frame"""
        for obj in detections:
            x1, y1, x2, y2 = map(int, obj['bbox'])
            # Ve bbox do dam
            cv2.rectangle(frame, (x1, y1), (x2, y2), DANGEROUS_OBJECT_COLOR, 3)
            # Ve label
            label = f"DANGER: {obj['class']} {obj['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, FONT_FACE, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), DANGEROUS_OBJECT_COLOR, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), FONT_FACE, 0.6, (255, 255, 255), 2)


# ============================================================
# LUONG CHINH (MAIN LOOP)
# ============================================================
def main():
    print("=" * 60)
    print("HAVEN - Pose + ADL + Zone + Object Detection")
    print("=" * 60)
    print(f"Pose Model: {MODEL_PATH}")
    print(f"Video: {VIDEO_PATH}")
    print(f"Zone Intrusion: {'ON' if ZONE_ENABLED else 'OFF'}")
    print(f"Object Detection: {'ON' if OBJECT_DETECTION_ENABLED else 'OFF'}")
    print("=" * 60)
    
    # Khoi tao pose model
    print("[INFO] Loading pose model...")
    model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERROR] Khong the mo video")
        return
    
    # Lay thong tin video
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"[INFO] Video resolution: {frame_width}x{frame_height}")
    
    # Khoi tao tracker
    tracker = SimpleTracker(iou_threshold=TRACKER_IOU_THRESHOLD, frame_height=frame_height)
    
    # Khoi tao zone manager
    zone_manager = None
    if ZONE_ENABLED:
        zone_manager = ZoneManager(FORBIDDEN_ZONES, frame_width, frame_height)
        print(f"[INFO] Zone Intrusion enabled: {len(FORBIDDEN_ZONES)} zones")
    
    # Khoi tao object detector
    object_detector = None
    if OBJECT_DETECTION_ENABLED:
        object_detector = DangerousObjectDetector(
            OBJECT_MODEL_PATH, 
            DANGEROUS_OBJECT_CLASSES, 
            OBJECT_CONF_THRES
        )
        print(f"[INFO] Object Detection enabled: {list(DANGEROUS_OBJECT_CLASSES.keys())}")
    
    # Cua so hien thi
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
    
    # Trang thai
    paused = False
    loop_mode = True
    frame_count = 0
    start_time = time.time()
    
    # GHI GIF
    recording_gif = False
    gif_frames = []
    
    print("DIEU KHIEN:")
    print(f"  {CONTROLS_TEXT['quit']}")
    print(f"  {CONTROLS_TEXT['pause']}")
    print(f"  {CONTROLS_TEXT['loop']}")
    print(f"  {CONTROLS_TEXT['gif']}")
    print("=" * 60)
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if loop_mode:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            frame_count += 1
            
            # 1. Chay YOLO Pose
            results = model(frame, verbose=False, conf=CONF_THRES)[0]
            
            # 2. Phan tich ket qua pose
            detections = []
            keypoints_list = []
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                kpts_data = results.keypoints.data.cpu().numpy() if results.keypoints is not None else None
                
                for i, box in enumerate(boxes):
                    detections.append({'bbox': box.tolist(), 'conf': float(confs[i])})
                    keypoints_list.append(kpts_data[i] if kpts_data is not None and i < len(kpts_data) else None)
            
            # 3. Cap nhat Tracker
            tracked = tracker.update(detections)
            
            # 4. Ve zone cam (truoc khi ve nguoi)
            if zone_manager is not None:
                zone_manager.draw_zones(frame)
            
            # 5. Xu ly ADL cho tung nguoi
            for i, det in enumerate(tracked):
                tid = det['track_id']
                bbox = det['bbox']
                kpts = keypoints_list[i] if i < len(keypoints_list) else None
                state = tracker.states.get(tid)
                if state is None:
                    continue
                
                # PHAN LOAI TU THE VOI MAJORITY VOTING
                posture = classify_posture(kpts, bbox, state, frame_height)
                state.add_posture(posture)
                
                # PHAT HIEN SU KIEN GIO TAY
                if kpts is not None:
                    hand_event = state.check_hand_raise(kpts)
                    if hand_event:
                        state.add_event(hand_event)
                
                # KIEM TRA XAM NHAP VUNG CAM
                if zone_manager is not None:
                    zone_name = zone_manager.check_intrusion(bbox)
                    if zone_name:
                        # Chi them event 1 lan
                        event_text = f"ZONE: {zone_name}"
                        if not state.events or event_text not in list(state.events)[-1]:
                            state.add_event(event_text)
                
                # Ve Skeleton va BBox
                if kpts is not None:
                    draw_skeleton(frame, kpts)
                draw_detection(frame, bbox, tid, state.current_posture, state.events)
            
            # 6. PHAT HIEN VAT NGUY HIEM
            dangerous_objects = []
            if object_detector is not None:
                dangerous_objects = object_detector.detect(frame)
                object_detector.draw_detections(frame, dangerous_objects)
            
            # 7. Hien thi thong tin he thong
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), 
                       FONT_FACE, FONT_SCALE_INFO, (0, 255, 0), FONT_THICKNESS)
            cv2.putText(frame, f"Persons: {len(tracked)}", (10, 60), 
                       FONT_FACE, FONT_SCALE_INFO, (0, 255, 0), FONT_THICKNESS)
            if dangerous_objects:
                cv2.putText(frame, f"DANGER: {len(dangerous_objects)} objects", (10, 90), 
                           FONT_FACE, FONT_SCALE_INFO, (0, 0, 255), FONT_THICKNESS)
            
            # 6.  GHI GIF
            if recording_gif:
                # Chuyển BGR → RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize để giảm dung lượng
                h, w = rgb_frame.shape[:2]
                new_w = int(w * GIF_RESIZE_RATIO)
                new_h = int(h * GIF_RESIZE_RATIO)
                small_frame = cv2.resize(rgb_frame, (new_w, new_h))
                gif_frames.append(Image.fromarray(small_frame))
                
                # Hiển thị đang ghi - Chấm đỏ nhấp nháy
                if (frame_count // 10) % 2 == 0:
                    cv2.circle(frame, (frame_width - 30, 30), 10, (0, 0, 255), -1)
                cv2.putText(frame, "REC GIF", (frame_width - 120, 35), 
                           FONT_FACE, FONT_SCALE_LABEL, (0, 0, 255), FONT_THICKNESS)
            
            cv2.imshow(WINDOW_NAME, frame)
        
        # Xử lý phím bấm
        key = cv2.waitKey(1 if not paused else 100) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord(' '):
            paused = not paused
            print(f"{'[PAUSED] DA TAM DUNG' if paused else '[PLAY] DA TIEP TUC'}")
        elif key == ord('l') or key == ord('L'):
            loop_mode = not loop_mode
            print(f"[INFO] Che do lap: {'BAT' if loop_mode else 'TAT'}")
        elif key == ord('g') or key == ord('G'):
            if not recording_gif:
                print("[INFO] BAT DAU GHI GIF...")
                recording_gif = True
                gif_frames = []
            else:
                print("[INFO] DUNG GHI. DANG LUU FILE GIF...")
                recording_gif = False
                
                if len(gif_frames) > 0:
                    # Tao ten file output tu dong tang dan (verX)
                    output_filename = get_next_output_path(base_name="pose-adl")
                    
                    # Luu GIF
                    gif_frames[0].save(
                        output_filename,
                        save_all=True,
                        append_images=gif_frames[1:],
                        optimize=GIF_OPTIMIZE,
                        duration=GIF_DURATION_MS,
                        loop=GIF_LOOP
                    )
                    print(f"[INFO] DA LUU: {output_filename} ({len(gif_frames)} frames)")
                else:
                    print("[WARN] KHONG CO FRAME NAO DUOC GHI.")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n" + "=" * 60)
    print("ĐÃ ĐÓNG ỨNG DỤNG")
    print("=" * 60)

if __name__ == "__main__":
    main()
    