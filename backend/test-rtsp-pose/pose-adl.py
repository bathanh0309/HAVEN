"""
HAVEN - RTSP Camera Pose + ADL Test
====================================
Test ADL detection voi RTSP camera stream.
Dung chung config tu shared_config.py

Features:
- Real-time pose detection tu RTSP camera
- Majority voting cho tu the on dinh
- Phat hien su kien: Fall, Hand Raise, Sitting Down, Standing Up
- Mau sac skeleton chinh xac
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
            self.velocity = dist * ASSUMED_FPS
        
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
        """THÊM TƯ THẾ VÀ THỰC HIỆN MAJORITY VOTING"""
        self.postures.append(posture)
        
        if len(self.postures) >= POSTURE_VOTING_MIN_FRAMES:
            counter = Counter(self.postures)
            voted_posture = counter.most_common(1)[0][0]
            
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
            self.current_posture = posture
    
    def add_event(self, event: str):
        """Thêm sự kiện mới"""
        timestamp = time.strftime("%H:%M:%S")
        event_text = f"{timestamp} - {event}"
        self.events.append(event_text)
        print(f"[ID:{self.track_id}] EVENT: {event}")
    
    def check_hand_raise(self, keypoints: np.ndarray) -> Optional[str]:
        """PHÁT HIỆN GIƠ TAY"""
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
            if l_wrist[1] < nose[1]:
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
                det['track_id'] = self.next_id
                self.tracks[self.next_id] = {'bbox': bbox}
                self.states[self.next_id] = TrackState(self.next_id, self.frame_height)
                self.ages[self.next_id] = 0
                self.next_id += 1
            
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
# PHÂN LOẠI TƯ THẾ (POSTURE CLASSIFICATION)
# ============================================================
def classify_posture(keypoints: np.ndarray, bbox: List[float], 
                     track_state: TrackState, frame_height: int) -> str:
    """Phân loại tư thế với góc đầu gối"""
    if keypoints is None or len(keypoints) < 17:
        return "UNKNOWN"
    
    def get_point(idx):
        if keypoints[idx][2] > KEYPOINT_CONF_THRESHOLD:
            return keypoints[idx][:2]
        return None
    
    l_sh = get_point(KP_LEFT_SHOULDER)
    r_sh = get_point(KP_RIGHT_SHOULDER)
    l_hip = get_point(KP_LEFT_HIP)
    r_hip = get_point(KP_RIGHT_HIP)
    l_knee = get_point(KP_LEFT_KNEE)
    r_knee = get_point(KP_RIGHT_KNEE)
    l_ankle = get_point(KP_LEFT_ANKLE)
    r_ankle = get_point(KP_RIGHT_ANKLE)
    
    # 1. Tính góc thân người
    torso_angle = 0
    if (l_sh is not None or r_sh is not None) and (l_hip is not None or r_hip is not None):
        mid_sh = (l_sh + r_sh) / 2 if l_sh is not None and r_sh is not None else (l_sh if l_sh is not None else r_sh)
        mid_hip = (l_hip + r_hip) / 2 if l_hip is not None and r_hip is not None else (l_hip if l_hip is not None else r_hip)
        
        dx = abs(mid_sh[0] - mid_hip[0])
        dy = abs(mid_sh[1] - mid_hip[1])
        torso_angle = math.degrees(math.atan2(dx, dy)) if dy > 0 else 0
    
    # 2. Tính góc đầu gối
    knee_angles = []
    
    if l_hip is not None and l_knee is not None and l_ankle is not None:
        v1 = l_hip - l_knee
        v2 = l_ankle - l_knee
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.degrees(math.acos(cos_angle))
        knee_angles.append(angle)
    
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
    movement_threshold = frame_height * MOVEMENT_THRESHOLD_RATIO
    
    # LOGIC PHÂN LOẠI
    if torso_angle > TORSO_ANGLE_LAYING_THRESHOLD or aspect_ratio > ASPECT_RATIO_LAYING_THRESHOLD:
        return "LAYING"
    
    if avg_knee_angle < KNEE_ANGLE_SITTING_THRESHOLD and movement < movement_threshold:
        return "SITTING"
    
    if movement > movement_threshold * MOVEMENT_WALKING_MULTIPLIER:
        return "WALKING"
    
    return "STANDING"


# ============================================================
# VẼ SKELETON
# ============================================================
def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, conf_threshold: float = KEYPOINT_CONF_THRESHOLD):
    """Vẽ bộ khung xương với màu sắc chính xác cho từng bộ phận"""
    if keypoints is None or len(keypoints) < 17:
        return
    
    for (start_idx, end_idx, part) in SKELETON_CONNECTIONS:
        if keypoints[start_idx][2] > conf_threshold and keypoints[end_idx][2] > conf_threshold:
            start_pt = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
            end_pt = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
            color = SKELETON_COLORS.get(part, (255, 255, 255))
            cv2.line(frame, start_pt, end_pt, color, SKELETON_LINE_THICKNESS)
    
    for i, kp in enumerate(keypoints):
        if kp[2] > conf_threshold:
            pt = (int(kp[0]), int(kp[1]))
            
            if i <= 4:
                color = SKELETON_COLORS["head"]
            elif i in [5, 6, 11, 12]:
                color = SKELETON_COLORS["torso"]
            elif i in [7, 8]:
                color = SKELETON_COLORS["upper_arm"]
            elif i in [9, 10]:
                color = SKELETON_COLORS["lower_arm"]
            elif i in [13, 14]:
                color = SKELETON_COLORS["upper_leg"]
            else:
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
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BBOX_THICKNESS)
    
    label = f"ID:{track_id} {posture}"
    (tw, th), _ = cv2.getTextSize(label, FONT_FACE, FONT_SCALE_LABEL, FONT_THICKNESS)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 5), FONT_FACE, FONT_SCALE_LABEL, (0, 0, 0), FONT_THICKNESS)
    
    if events:
        latest_event = list(events)[-1]
        event_parts = latest_event.split(" - ", 1)
        if len(event_parts) == 2:
            display_text = event_parts[1]
        else:
            display_text = latest_event
        
        cv2.putText(frame, display_text, (x1, y2 + 20), 
                   FONT_FACE, FONT_SCALE_EVENT, (0, 0, 255), FONT_THICKNESS)


# ============================================================
# LUỒNG CHÍNH (MAIN LOOP) - RTSP CAMERA
# ============================================================
def main():
    print("=" * 60)
    print("HAVEN - RTSP Camera Pose + ADL Test")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    
    # Lấy RTSP URL
    stream_type = RTSP_CONFIG.get("default_stream", "SD")
    rtsp_url = get_rtsp_url(stream_type)
    print(f"RTSP URL: rtsp://***:***@{RTSP_CONFIG['camera_ip']}:{RTSP_CONFIG['camera_port']}/...")
    print(f"Stream: {stream_type}")
    print("=" * 60)
    
    # Khởi tạo model
    model = YOLO(MODEL_PATH)
    
    # Kết nối camera
    print("[INFO] Dang ket noi camera...")
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print("[ERROR] Loi: Khong the ket noi camera RTSP")
        print("   Vui lòng kiểm tra:")
        print("   - Camera đã bật và kết nối mạng")
        print("   - IP, port, username, password trong file .env")
        return
    
    print("[OK] Da ket noi camera thanh cong!")
    
    # Lấy thông tin stream
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"[INFO] Resolution: {frame_width}x{frame_height}")
    
    # Khởi tạo tracker
    tracker = SimpleTracker(iou_threshold=TRACKER_IOU_THRESHOLD, frame_height=frame_height)
    
    # Cửa sổ hiển thị
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
    
    # Trạng thái
    frame_count = 0
    start_time = time.time()
    reconnect_attempts = 0
    max_reconnect = 5
    
    # GHI GIF
    recording_gif = False
    gif_frames = []
    
    print("\nĐIỀU KHIỂN:")
    print(f"  {CONTROLS_TEXT['quit']}")
    print(f"  {CONTROLS_TEXT['gif']}")
    print("  H = Chuyển HD stream")
    print("  S = Chuyển SD stream")
    print("=" * 60)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            reconnect_attempts += 1
            print(f"[WARN] Mat ket noi camera... ({reconnect_attempts}/{max_reconnect})")
            
            if reconnect_attempts >= max_reconnect:
                print("[ERROR] Khong the ket noi lai camera. Thoat.")
                break
            
            time.sleep(2)
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
            continue
        
        reconnect_attempts = 0
        frame_count += 1
        
        # 1. Chạy YOLO
        results = model(frame, verbose=False, conf=CONF_THRES)[0]
        
        # 2. Phân tích kết quả
        detections = []
        keypoints_list = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            kpts_data = results.keypoints.data.cpu().numpy() if results.keypoints is not None else None
            
            for i, box in enumerate(boxes):
                detections.append({'bbox': box.tolist(), 'conf': float(confs[i])})
                keypoints_list.append(kpts_data[i] if kpts_data is not None and i < len(kpts_data) else None)
        
        # 3. Cập nhật Tracker
        tracked = tracker.update(detections)
        
        # 4. Xử lý ADL cho từng người
        for i, det in enumerate(tracked):
            tid = det['track_id']
            bbox = det['bbox']
            kpts = keypoints_list[i] if i < len(keypoints_list) else None
            state = tracker.states.get(tid)
            if state is None:
                continue
            
            posture = classify_posture(kpts, bbox, state, frame_height)
            state.add_posture(posture)
            
            if kpts is not None:
                hand_event = state.check_hand_raise(kpts)
                if hand_event:
                    state.add_event(hand_event)
            
            if kpts is not None:
                draw_skeleton(frame, kpts)
            draw_detection(frame, bbox, tid, state.current_posture, state.events)
        
        # 5. Hiển thị thông tin
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), 
                   FONT_FACE, FONT_SCALE_INFO, (0, 255, 0), FONT_THICKNESS)
        cv2.putText(frame, f"Persons: {len(tracked)}", (10, 60), 
                   FONT_FACE, FONT_SCALE_INFO, (0, 255, 0), FONT_THICKNESS)
        cv2.putText(frame, f"Stream: {stream_type}", (10, 90),
                   FONT_FACE, FONT_SCALE_INFO, (0, 255, 0), FONT_THICKNESS)
        
        # 6. GHI GIF
        if recording_gif:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb_frame.shape[:2]
            new_w = int(w * GIF_RESIZE_RATIO)
            new_h = int(h * GIF_RESIZE_RATIO)
            small_frame = cv2.resize(rgb_frame, (new_w, new_h))
            gif_frames.append(Image.fromarray(small_frame))
            
            if (frame_count // 10) % 2 == 0:
                cv2.circle(frame, (frame_width - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC GIF", (frame_width - 120, 35), 
                       FONT_FACE, FONT_SCALE_LABEL, (0, 0, 255), FONT_THICKNESS)
        
        cv2.imshow(WINDOW_NAME, frame)
        
        # Xử lý phím bấm
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('h') or key == ord('H'):
            stream_type = "HD"
            rtsp_url = get_rtsp_url("HD")
            print("[INFO] Chuyen sang HD stream...")
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
        elif key == ord('s') or key == ord('S'):
            stream_type = "SD"
            rtsp_url = get_rtsp_url("SD")
            print("[INFO] Chuyen sang SD stream...")
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
        elif key == ord('g') or key == ord('G'):
            if not recording_gif:
                print("[INFO] BAT DAU GHI GIF...")
                recording_gif = True
                gif_frames = []
            else:
                print("[INFO] DUNG GHI. DANG LUU FILE GIF...")
                recording_gif = False
                
                if len(gif_frames) > 0:
                    # Tạo tên file output tự động tăng dần (verX)
                    output_filename = get_next_output_path(base_name="pose-adl")
                    
                    gif_frames[0].save(
                        output_filename,
                        save_all=True,
                        append_images=gif_frames[1:],
                        optimize=GIF_OPTIMIZE,
                        duration=GIF_DURATION_MS,
                        loop=GIF_LOOP
                    )
                    print(f"[OK] DA LUU: {output_filename} ({len(gif_frames)} frames)")
                else:
                    print("[WARN] KHONG CO FRAME NAO DUOC GHI.")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n" + "=" * 60)
    print("ĐÃ ĐÓNG ỨNG DỤNG")
    print("=" * 60)


if __name__ == "__main__":
    main()
