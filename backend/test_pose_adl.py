"""
HAVEN - Pose + ADL Combined Test
=================================
Hiển thị đồng thời:
- Skeleton Pose với màu sắc quy định
- ADL State (Standing, Walking, Sitting, Laying)
- Events (Fall Down, Hand Up, etc.)
"""

import cv2
import time
import os
import sys
import math
import numpy as np
from pathlib import Path
from collections import deque
from dotenv import load_dotenv
from typing import Dict, List, Tuple
from PIL import Image # Dùng để lưu file GIF

# Thêm thư mục gốc vào path để có thể import các module khác
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO

# Tải cấu hình từ file .env
load_dotenv()
MODEL_PATH = os.getenv("AI_MODEL_PATH", "models/cpu_laptop/yolo11s-pose.pt")
VIDEO_PATH = os.getenv("TEST_VIDEO_PATH", "data/video/walking.mp4")
CONF_THRES = float(os.getenv("AI_CONF_THRES", "0.25"))

# ============================================================
# CHỈ SỐ KEYPOINT (ĐỊNH DẠNG COCO - 17 điểm)
# ============================================================
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

# MÀU SẮC SKELETON (BGR) theo quy ước yêu cầu
SKELETON_COLORS = {
    "head": (0, 0, 255),        # Đầu: Đỏ (Red)
    "torso": (180, 105, 255),   # Thân: Hồng (Pink)
    "arm": (0, 255, 0),         # Tay: Xanh lá (Green)
    "hand": (42, 42, 165),      # Bàn tay: Nâu (Brown)
    "leg": (0, 165, 255),       # Đùi/Chân: Cam (Orange)
    "foot": (0, 255, 255),      # Bàn chân: Vàng (Yellow)
}

# CÁC KẾT NỐI XƯƠNG ứng với từng bộ phận
SKELETON_CONNECTIONS = [
    # Đầu (Red)
    (KP_NOSE, KP_LEFT_EYE, "head"),
    (KP_NOSE, KP_RIGHT_EYE, "head"),
    (KP_LEFT_EYE, KP_LEFT_EAR, "head"),
    (KP_RIGHT_EYE, KP_RIGHT_EAR, "head"),
    
    # Thân (Pink)
    (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, "torso"),
    (KP_LEFT_SHOULDER, KP_LEFT_HIP, "torso"),
    (KP_RIGHT_SHOULDER, KP_RIGHT_HIP, "torso"),
    (KP_LEFT_HIP, KP_RIGHT_HIP, "torso"),
    
    # Tay (Green)
    (KP_LEFT_SHOULDER, KP_LEFT_ELBOW, "arm"),
    (KP_LEFT_ELBOW, KP_LEFT_WRIST, "arm"),
    (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW, "arm"),
    (KP_RIGHT_ELBOW, KP_RIGHT_WRIST, "arm"),
    
    # Chân (Orange)
    (KP_LEFT_HIP, KP_LEFT_KNEE, "leg"),
    (KP_LEFT_KNEE, KP_LEFT_ANKLE, "leg"),
    (KP_RIGHT_HIP, KP_RIGHT_KNEE, "leg"),
    (KP_RIGHT_KNEE, KP_RIGHT_ANKLE, "leg"),
]

# ============================================================
# LỚP QUẢN LÝ TRẠNG THÁI THEO DÕI (TRACKING)
# ============================================================
class TrackState:
    """Lưu trữ lịch sử và trạng thái của một đối tượng đang được theo dõi"""
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.positions: deque = deque(maxlen=30)  # Lưu lịch sử vị trí trong 1 giây (@30fps)
        self.postures: deque = deque(maxlen=15)   # Để thực hiện bỏ phiếu (majority voting)
        self.current_posture = "UNKNOWN"
        self.posture_start_time = time.time()
        self.last_center = None
        self.velocity = 0.0                       # Tốc độ tức thời
        self.events: List[str] = []
        
    def update_position(self, bbox: List[float]):
        """Cập nhật vị trí mới và tính toán tốc độ tức thời"""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        current_time = time.time()
        
        if self.last_center is not None:
            dx = cx - self.last_center[0]
            dy = cy - self.last_center[1]
            dist = math.sqrt(dx*dx + dy*dy)
            # Giả sử 30fps -> dt ~ 0.033s. Đơn vị: pixels per second
            self.velocity = dist * 30  
        
        self.last_center = (cx, cy)
        self.positions.append((current_time, cx, cy))
        
    def get_movement_in_window(self, window_sec: float = 1.0) -> float:
        """Tính tổng quãng đường di chuyển trong một khoảng thời gian (window)"""
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

# ============================================================
# TRACKER DỰA TRÊN IOU ĐƠN GIẢN
# ============================================================
class SimpleTracker:
    """Bộ theo dõi đối tượng dựa trên độ trùng khớp Bounding Box (IOU)"""
    def __init__(self, iou_threshold=0.3, max_age=30):
        self.tracks: Dict[int, dict] = {}
        self.states: Dict[int, TrackState] = {}
        self.next_id = 1
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.ages: Dict[int, int] = {}
        
    def update(self, detections: List[dict]) -> List[dict]:
        """Gán ID cho các phát hiện mới dựa trên lịch sử"""
        matched_ids = set()
        results = []
        
        for det in detections:
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
                # Tạo ID mới nếu không khớp
                det['track_id'] = self.next_id
                self.tracks[self.next_id] = {'bbox': bbox}
                self.states[self.next_id] = TrackState(self.next_id)
                self.ages[self.next_id] = 0
                self.next_id += 1
            
            # Cập nhật trạng thái (vị trí, tốc độ)
            tid = det['track_id']
            if tid in self.states:
                self.states[tid].update_position(bbox)
            
            results.append(det)
        
        # Xóa các track đã cũ không còn xuất hiện
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
        """Tính toán Intersection over Union giữa 2 hộp giới hạn"""
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
def classify_posture(keypoints: np.ndarray, bbox: List[float], track_state: TrackState) -> str:
    """
    Phân loại tư thế (Standing, Walking, Sitting, Laying) dựa trên
    keypoints, tỷ lệ khung hình và tốc độ di chuyển.
    """
    if keypoints is None or len(keypoints) < 17:
        return "UNKNOWN"
    
    # 1. Tính Tỷ lệ khung hình (Aspect Ratio)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    aspect_ratio = w / h if h > 0 else 0
    
    # 2. Tính Góc Thân người (Torso Angle)
    def get_point(idx):
        if keypoints[idx][2] > 0.3: # Chỉ lấy điểm có độ tin cậy > 30%
            return keypoints[idx][:2]
        return None
    
    l_sh = get_point(KP_LEFT_SHOULDER)
    r_sh = get_point(KP_RIGHT_SHOULDER)
    l_hip = get_point(KP_LEFT_HIP)
    r_hip = get_point(KP_RIGHT_HIP)
    
    torso_angle = 0
    if (l_sh is not None or r_sh is not None) and (l_hip is not None or r_hip is not None):
        if l_sh is not None and r_sh is not None:
            mid_sh = (l_sh + r_sh) / 2
        elif l_sh is not None: mid_sh = l_sh
        else: mid_sh = r_sh
            
        if l_hip is not None and r_hip is not None:
            mid_hip = (l_hip + r_hip) / 2
        elif l_hip is not None: mid_hip = l_hip
        else: mid_hip = r_hip
        
        dx = abs(mid_sh[0] - mid_hip[0])
        dy = abs(mid_sh[1] - mid_hip[1])
        torso_angle = math.degrees(math.atan2(dx, dy)) if dy > 0 else 0
    
    # 3. Lấy dữ liệu di chuyển
    movement = track_state.get_movement_in_window(1.0)
    velocity = track_state.velocity
    
    # --- Logic Phân Loại ---
    # NẰM (LAYING): Thân người nghiêng góc lớn (>60 độ) hoặc bbox rất "dẹt"
    if torso_angle > 60 or aspect_ratio > 1.3:
        return "LAYING"
    
    # NGỒI (SITTING): AR hơi lớn, di chuyển ít
    if aspect_ratio > 0.65 and movement < 50:
        return "SITTING"
    
    # ĐI BỘ (WALKING) vs ĐỨNG (STANDING): Dựa trên tổng di chuyển trong 1s
    if movement > 80 or velocity > 100:
        return "WALKING"
    
    return "STANDING"

# ============================================================
# CÁC HÀM VẼ GIAO DIỆN (DRAWING)
# ============================================================
def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, conf_threshold: float = 0.3):
    """Vẽ bộ khung xương với màu sắc quy định cho từng bộ phận"""
    if keypoints is None or len(keypoints) < 17:
        return
    
    # Vẽ các đường nối xương
    for (start_idx, end_idx, part) in SKELETON_CONNECTIONS:
        if keypoints[start_idx][2] > conf_threshold and keypoints[end_idx][2] > conf_threshold:
            start_pt = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
            end_pt = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
            color = SKELETON_COLORS.get(part, (255, 255, 255))
            cv2.line(frame, start_pt, end_pt, color, 2)
    
    # Vẽ các khớp điểm
    for i, kp in enumerate(keypoints):
        if kp[2] > conf_threshold:
            pt = (int(kp[0]), int(kp[1]))
            if i <= 4: color = SKELETON_COLORS["head"]
            elif i <= 6: color = SKELETON_COLORS["torso"]
            elif i <= 10: color = SKELETON_COLORS["arm"]
            elif i <= 12: color = SKELETON_COLORS["torso"]
            elif i <= 14: color = SKELETON_COLORS["leg"]
            else: color = SKELETON_COLORS["foot"]
            cv2.circle(frame, pt, 4, color, -1)

def draw_detection(frame: np.ndarray, bbox: List[float], track_id: int, 
                   posture: str, velocity: float, movement: float):
    """Vẽ hộp giới hạn và nhãn ADL tương ứng với từng tư thế"""
    x1, y1, x2, y2 = map(int, bbox)
    
    colors = {
        "STANDING": (0, 255, 0),      # Xanh lá
        "WALKING": (255, 255, 0),     # Xanh lơ (Cyan)
        "SITTING": (0, 255, 255),     # Vàng
        "LAYING": (0, 0, 255),        # Đỏ
        "UNKNOWN": (128, 128, 128),   # Xám
    }
    color = colors.get(posture, (255, 255, 255))
    
    # Vẽ BBox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Vẽ nền cho nhãn văn bản
    label = f"ID:{track_id} {posture}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Vẽ thông tin di chuyển phía dưới
    vel_text = f"Move:{movement:.0f}px"
    cv2.putText(frame, vel_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# ============================================================
# LUỒNG CHÍNH (MAIN LOOP)
# ============================================================
def main():
    print("=" * 60)
    print("HAVEN - Pose + ADL Combined Test")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Video: {VIDEO_PATH}")
    print("=" * 60)
    
    # Khởi tạo model và camera
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Lỗi: Không thể mở tập tin video")
        return
    
    # Khởi tạo bộ theo dõi
    tracker = SimpleTracker(iou_threshold=0.3)
    
    # Cửa sổ hiển thị
    cv2.namedWindow("HAVEN Pose + ADL", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HAVEN Pose + ADL", 1280, 720)
    
    paused = False
    loop_mode = True
    frame_count = 0
    start_time = time.time()
    
    # Trạng thái ghi GIF
    recording_gif = False
    gif_frames = []
    
    print("Điều khiển: Q=Thoát, Space=Tạm dừng, L=Bật/Tắt lặp, G=Bật/Tắt ghi GIF")
    print("=" * 60)
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if loop_mode:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else: break
            
            frame_count += 1
            
            # --- LOGIC XỬ LÝ (GIỮ NGUYÊN) ---
            # 1. Chạy YOLO suy luận (Inference)
            results = model(frame, verbose=False, conf=CONF_THRES)[0]
            
            # 2. Phân tích kết quả phát hiện
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
            
            # 4. Xử lý logic ADL cho từng đối tượng
            for i, det in enumerate(tracked):
                tid = det['track_id']
                bbox = det['bbox']
                kpts = keypoints_list[i] if i < len(keypoints_list) else None
                state = tracker.states.get(tid)
                if state is None: continue
                
                # Phân loại tư thế
                posture = classify_posture(kpts, bbox, state)
                state.current_posture = posture
                movement = state.get_movement_in_window(1.0)
                
                # Vẽ Skeleton và BBox
                if kpts is not None: draw_skeleton(frame, kpts)
                draw_detection(frame, bbox, tid, posture, state.velocity, movement)
            
            # 5. Hiển thị thông tin hệ thống (FPS, số người)
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Persons: {len(tracked)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 6. Logic Ghi GIF
            if recording_gif:
                # Chuyển BGR (OpenCV) -> RGB (PIL)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize để giảm dung lượng file GIF (tùy chọn, ví dụ giảm 50%)
                h, w = frame.shape[:2]
                small_frame = cv2.resize(rgb_frame, (w//2, h//2))
                gif_frames.append(Image.fromarray(small_frame))
                
                # Hiển thị thông báo đang ghi - Nhấp nháy đỏ
                if (frame_count // 10) % 2 == 0:
                    cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)
                cv2.putText(frame, "REC GIF", (w - 100, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("HAVEN Pose + ADL", frame)
        
        # Xử lý phím bấm
        key = cv2.waitKey(1 if not paused else 100) & 0xFF
        if key == ord('q') or key == ord('Q'): break
        elif key == ord(' '): 
            paused = not paused
            print(f"{'ĐÃ TẠM DỪNG' if paused else 'ĐÃ TIẾP TỤC'}")
        elif key == ord('l') or key == ord('L'):
            loop_mode = not loop_mode
            print(f"Chế độ lặp: {'BẬT' if loop_mode else 'TẮT'}")
        elif key == ord('g') or key == ord('G'):
            if not recording_gif:
                print(">>> BẮT ĐẦU GHI GIF...")
                recording_gif = True
                gif_frames = []
            else:
                print(">>> DỪNG GHI. ĐANG LƯU FILE GIF...")
                recording_gif = False
                if len(gif_frames) > 0:
                    output_filename = f"adl_output_{int(time.time())}.gif"
                    # Lưu GIF: duration = thời gian mỗi frame (ms)
                    # 1000ms / fps ~= duration per frame. Giả sử 15fps -> ~66ms
                    gif_frames[0].save(
                        output_filename,
                        save_all=True,
                        append_images=gif_frames[1:],
                        optimize=True,
                        duration=66,
                        loop=0
                    )
                    print(f">>> ĐÃ LƯU: {output_filename} ({len(gif_frames)} frames)")
                else:
                    print(">>> KHÔNG CÓ FRAME NÀO ĐƯỢC GHI.")

    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
