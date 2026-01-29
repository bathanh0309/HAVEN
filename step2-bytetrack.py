"""
HAVEN BƯỚC 2: Single Camera Tracking với ByteTrack + Pose Filter
=================================================================
- Detection: YOLO-Pose với multi-filtering
- Tracking: Custom tracker (Hungarian Algorithm)
- Output: Track IDs duy nhất cho mỗi người

Tối ưu cho: 2 người di chuyển, có thể bị occlusion
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# ==================== CONFIG ====================
VIDEO_PATH = r"D:\HAVEN\backend\data\multi-camera\1.mp4"
MODEL_PATH = r"D:\HAVEN\backend\models\yolo11n-pose.pt"
DISPLAY_WIDTH = 1280

# Detection Filtering
CONF_THRESHOLD = 0.8       # High confidence để giảm false positive
MIN_BOX_AREA = 3000        # Loại bỏ box quá nhỏ
MAX_BOX_AREA = 200000      # Loại bỏ box quá lớn
MIN_ASPECT_RATIO = 0.3     # Tỷ lệ h/w tối thiểu
MAX_ASPECT_RATIO = 5.0     # Tỷ lệ h/w tối đa

# Keypoint Filtering
MIN_KEYPOINTS = 8          # Tối thiểu 8/17 keypoints visible
MIN_KEYPOINT_CONF = 0.4    # Confidence của mỗi keypoint

# Keypoint indices (COCO format)
HEAD_KPT = [0, 1, 2, 3, 4]       # nose, eyes, ears
UPPER_KPT = [5, 6, 11, 12]       # shoulders, hips
LOWER_KPT = [13, 14, 15, 16]     # knees, ankles

# Tracking
MAX_DISTANCE = 200         # Max distance để match track (pixels)
MAX_AGE = 200              # Giữ track khi mất (frames)
NEW_TRACK_QUALITY = 0.8    # Quality tối thiểu để tạo track mới


# ==================== TRACKER ====================
class PersonTracker:
    """Simple tracker sử dụng Hungarian Algorithm"""
    
    def __init__(self, max_distance=200, max_age=200):
        self.max_distance = max_distance
        self.max_age = max_age
        self.tracks = {}          # track_id -> {center, box, age, lost, quality}
        self.next_id = 1
        self.history = defaultdict(list)  # track_id -> list of centers
        self.colors = {}
        self._frame_num = 0
    
    def get_color(self, tid):
        """Tạo màu riêng cho mỗi track ID"""
        if tid not in self.colors:
            np.random.seed(tid * 100)
            self.colors[tid] = tuple(np.random.randint(50, 255, 3).tolist())
        return self.colors[tid]
    
    def update(self, detections):
        """
        Update tracker với detections mới
        detections: list of (x, y, w, h, quality, kpt_info)
        Returns: list of (x, y, w, h, track_id)
        """
        self._frame_num += 1
        results = []
        
        # Không có detection → tăng lost count
        if len(detections) == 0:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > self.max_age:
                    del self.tracks[tid]
            return results
        
        det_centers = np.array([[(d[0] + d[2]/2), (d[1] + d[3]/2)] for d in detections])
        
        # Chưa có tracks → tạo mới tất cả
        if len(self.tracks) == 0:
            for x, y, w, h, quality, kpt_info in detections:
                center = (x + w/2, y + h/2)
                self.tracks[self.next_id] = {
                    'center': center,
                    'box': (x, y, w, h),
                    'age': 1,
                    'lost': 0,
                    'quality': quality
                }
                results.append((x, y, w, h, self.next_id))
                self.history[self.next_id].append(center)
                self.next_id += 1
            return results
        
        # Matching với Hungarian Algorithm
        track_ids = list(self.tracks.keys())
        track_centers = np.array([self.tracks[tid]['center'] for tid in track_ids])
        cost_matrix = cdist(det_centers, track_centers)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matched_dets = set()
        matched_tracks = set()
        
        # Process matched pairs
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < self.max_distance:
                x, y, w, h, quality, kpt_info = detections[row]
                tid = track_ids[col]
                center = (x + w/2, y + h/2)
                
                self.tracks[tid].update({
                    'center': center,
                    'box': (x, y, w, h),
                    'age': self.tracks[tid]['age'] + 1,
                    'lost': 0,
                    'quality': quality
                })
                
                results.append((x, y, w, h, tid))
                self.history[tid].append(center)
                if len(self.history[tid]) > 50:
                    self.history[tid].pop(0)
                
                matched_dets.add(row)
                matched_tracks.add(col)
        
        # Unmatched detections → tạo track mới (nếu quality đủ cao)
        for i, (x, y, w, h, quality, kpt_info) in enumerate(detections):
            if i not in matched_dets and quality > NEW_TRACK_QUALITY:
                center = (x + w/2, y + h/2)
                self.tracks[self.next_id] = {
                    'center': center,
                    'box': (x, y, w, h),
                    'age': 1,
                    'lost': 0,
                    'quality': quality
                }
                results.append((x, y, w, h, self.next_id))
                self.history[self.next_id].append(center)
                print(f"🆕 NEW ID: {self.next_id} @ frame {self._frame_num} | Quality: {quality:.3f}")
                self.next_id += 1
        
        # Unmatched tracks → tăng lost count
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > self.max_age:
                    del self.tracks[tid]
        
        return results


# ==================== FILTERING ====================
def calculate_quality(box, kpts, kpt_confs):
    """Tính quality score cho detection (0-1)"""
    x, y, w, h = box
    score = 0.0
    
    # 40%: Keypoint visibility
    visible_kpts = np.sum(kpt_confs > MIN_KEYPOINT_CONF)
    score += min(visible_kpts / 17.0, 1.0) * 0.4
    
    # 30%: Body part distribution (head, upper, lower)
    head_vis = sum(kpt_confs[i] > MIN_KEYPOINT_CONF for i in HEAD_KPT)
    upper_vis = sum(kpt_confs[i] > MIN_KEYPOINT_CONF for i in UPPER_KPT)
    lower_vis = sum(kpt_confs[i] > MIN_KEYPOINT_CONF for i in LOWER_KPT)
    
    dist_score = 0
    if head_vis >= 1: dist_score += 0.33
    if upper_vis >= 2: dist_score += 0.33
    if lower_vis >= 1: dist_score += 0.34
    score += dist_score * 0.3
    
    # 20%: Aspect ratio
    ar = h / (w + 1e-6)
    if 1.5 <= ar <= 3.0:
        ar_score = 1.0
    elif 1.0 <= ar <= 4.0:
        ar_score = 0.7
    else:
        ar_score = 0.3
    score += ar_score * 0.2
    
    # 10%: Size
    area = w * h
    if MIN_BOX_AREA <= area <= MAX_BOX_AREA:
        size_score = 1.0
    elif MIN_BOX_AREA * 0.5 < area < MAX_BOX_AREA * 2:
        size_score = 0.5
    else:
        size_score = 0.0
    score += size_score * 0.1
    
    return score


def filter_detections(results):
    """
    Lọc detections hợp lệ từ YOLO results
    Returns: list of (x, y, w, h, quality, kpt_info)
    """
    valid = []
    
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return valid
    
    boxes = results[0].boxes.xywh.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    kpts = results[0].keypoints.data if results[0].keypoints is not None else None
    
    for i, (x, y, w, h) in enumerate(boxes):
        # Filter: class = person
        if classes[i] != 0:
            continue
        
        # Filter: confidence
        if confs[i] < CONF_THRESHOLD:
            continue
        
        # Filter: box size
        area = w * h
        if not (MIN_BOX_AREA <= area <= MAX_BOX_AREA):
            continue
        
        # Filter: aspect ratio
        ar = h / (w + 1e-6)
        if not (MIN_ASPECT_RATIO <= ar <= MAX_ASPECT_RATIO):
            continue
        
        # Filter: keypoints
        if kpts is not None and i < len(kpts):
            k = kpts[i].cpu().numpy()
            kpt_confs = k[:, 2]
            visible = np.sum(kpt_confs > MIN_KEYPOINT_CONF)
            
            if visible < MIN_KEYPOINTS:
                continue
            
            upper_vis = sum(kpt_confs[j] > MIN_KEYPOINT_CONF for j in UPPER_KPT)
            if upper_vis < 1:
                continue
            
            quality = calculate_quality((x, y, w, h), k, kpt_confs)
            
            if quality >= 0.6:
                kpt_info = f"{int(visible)}/17"
                valid.append((x, y, w, h, quality, kpt_info))
    
    return valid


# ==================== MAIN ====================
def main():
    print("=" * 60)
    print("HAVEN - BƯỚC 2: Single Camera Tracking")
    print("=" * 60)
    print(f"\n📌 Config:")
    print(f"   Detection: conf>{CONF_THRESHOLD}, keypoints>={MIN_KEYPOINTS}")
    print(f"   Tracking: max_dist={MAX_DISTANCE}px, max_age={MAX_AGE} frames")
    print(f"   New track: quality>{NEW_TRACK_QUALITY}\n")
    
    model = YOLO(MODEL_PATH)
    tracker = PersonTracker(max_distance=MAX_DISTANCE, max_age=MAX_AGE)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video: {width}×{height} @ {fps:.1f}fps\n")
    print("▶️  Running... (Press 'q' to quit)\n")
    
    scale = DISPLAY_WIDTH / width
    frame_count = 0
    start_time = time.time()
    unique_ids = set()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        display = cv2.resize(frame, (DISPLAY_WIDTH, int(height * scale)))
        
        # Detection + Filtering
        results = model(display, verbose=False, conf=0.3)
        detections = filter_detections(results)
        
        # Tracking
        tracked = tracker.update(detections)
        
        # Draw
        for x, y, w, h, tid in tracked:
            unique_ids.add(tid)
            color = tracker.get_color(tid)
            
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
            cv2.putText(display, f"ID:{tid}", (x1, y1 - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # Trajectory
            if len(tracker.history[tid]) > 1:
                pts = np.array(tracker.history[tid], dtype=np.int32)
                cv2.polylines(display, [pts], False, color, 2)
        
        # Stats (mỗi 30 frames)
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(f"Frame {frame_count}: {len(tracked)} tracked | "
                  f"Unique IDs: {len(unique_ids)} | FPS: {frame_count/elapsed:.1f}")
        
        # Overlay
        ids_color = (0, 255, 0) if len(unique_ids) == 2 else (0, 165, 255)
        cv2.putText(display, f"Tracked: {len(tracked)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, f"Unique IDs: {len(unique_ids)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, ids_color, 2)
        
        cv2.imshow('HAVEN - ByteTrack', display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Final results
    elapsed = time.time() - start_time
    
    print(f"\n{'=' * 60}")
    print(f"✅ RESULTS")
    print(f"{'=' * 60}")
    print(f"📊 Frames: {frame_count}")
    print(f"👥 Unique persons: {len(unique_ids)}")
    print(f"📋 Track IDs: {sorted(unique_ids)}")
    print(f"⏱️  Time: {elapsed:.1f}s | FPS: {frame_count/elapsed:.1f}")
    
    if len(unique_ids) == 2:
        print(f"\n🎉 PERFECT! Đạt đúng 2 IDs!")
    elif len(unique_ids) < 2:
        print(f"\n⚠️  Ít hơn 2 IDs - Có thể giảm NEW_TRACK_QUALITY")
    else:
        print(f"\n⚠️  Nhiều hơn 2 IDs - Có thể tăng NEW_TRACK_QUALITY hoặc MAX_AGE")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
