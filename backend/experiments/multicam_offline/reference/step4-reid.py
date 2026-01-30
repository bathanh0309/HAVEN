"""
HAVEN Step 4: Sequential ReID với Cam1 làm Master
============================================
Chiến lược:
- CAM1: Gán ID mới cho mỗi người (Master) - KHỞI TẠO ID
- CAM2: CHỈ match với IDs từ Cam1, KHÔNG tạo ID mới (trừ khi chưa từng thấy)
- CAM3: CHỈ match với IDs từ Cam1+Cam2, KHÔNG tạo ID mới
"""
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import cdist, cosine
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import time

# CONFIG
CAMERAS = [
    ("cam1", r"D:\HAVEN\backend\data\multi-camera\1.mp4"),
    ("cam2", r"D:\HAVEN\backend\data\multi-camera\2.mp4"),
    ("cam3", r"D:\HAVEN\backend\data\multi-camera\3.mp4"),
]
MODEL_PATH = r"D:\HAVEN\backend\models\yolo11n-pose.pt"
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# Filter params
CONF_THRESHOLD = 0.8
MIN_BOX_AREA = 3000
MAX_BOX_AREA = 200000
MIN_KEYPOINTS = 8
MIN_KEYPOINT_CONF = 0.4
MAX_DISTANCE = 200
MAX_AGE = 200
NEW_TRACK_QUALITY = 0.8

HEAD_KPT = [0, 1, 2, 3, 4]
UPPER_KPT = [5, 6, 11, 12]
LOWER_KPT = [13, 14, 15, 16]

# ReID
REID_THRESHOLD = 0.65

# Global colors
COLORS = {
    1: (0, 255, 0),      # Green
    2: (255, 0, 0),      # Blue
    3: (0, 165, 255),    # Orange
    4: (255, 0, 255),    # Magenta
    5: (255, 255, 0),    # Cyan
}


class ColorHistogramReID:
    """Simple ReID using color histogram"""
    
    def extract(self, img_crop):
        if img_crop is None or img_crop.size == 0:
            return None
        if img_crop.shape[0] < 20 or img_crop.shape[1] < 10:
            return None
        
        try:
            hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
            h = img_crop.shape[0]
            
            # 3 parts: head, body, legs
            parts = [hsv[:h//3, :], hsv[h//3:2*h//3, :], hsv[2*h//3:, :]]
            
            features = []
            for part in parts:
                if part.size == 0:
                    features.extend([0] * 32)
                    continue
                h_hist = cv2.calcHist([part], [0], None, [16], [0, 180])
                s_hist = cv2.calcHist([part], [1], None, [16], [0, 256])
                h_hist = cv2.normalize(h_hist, h_hist).flatten()
                s_hist = cv2.normalize(s_hist, s_hist).flatten()
                features.extend(h_hist)
                features.extend(s_hist)
            
            return np.array(features)
        except:
            return None


class MasterIDDatabase:
    """
    Database với Cam1 làm Master
    - Cam1: Tạo ID mới
    - Cam2/3: CHỈ match, KHÔNG tạo mới
    """
    
    def __init__(self):
        self.reid = ColorHistogramReID()
        self.persons = {}  # global_id -> {features: [], first_cam: str}
        self.next_id = 1
        self.cam1_completed = False
    
    def get_color(self, gid):
        return COLORS.get(gid, (100, 100, 100))
    
    def register_new(self, person_crop, cam_id):
        """CHỈ Cam1 được gọi - tạo ID mới"""
        features = self.reid.extract(person_crop)
        if features is None:
            return None
        
        new_id = self.next_id
        self.next_id += 1
        self.persons[new_id] = {
            'features': [features],
            'first_cam': cam_id,
            'cameras': {cam_id}
        }
        
        print(f"   🆕 [id:{new_id}] New Person Registered in {cam_id.upper()}")
        return new_id
    
    def match_only(self, person_crop, cam_id):
        """Cam2/3 chỉ match với IDs có sẵn"""
        features = self.reid.extract(person_crop)
        if features is None:
            return None
        
        best_id = None
        best_sim = 0
        
        for gid, info in self.persons.items():
            sims = [1 - cosine(features, f) for f in info['features'][-10:]]
            if sims:
                avg_sim = np.mean(sims)
                if avg_sim > best_sim and avg_sim > REID_THRESHOLD:
                    best_id = gid
                    best_sim = avg_sim
        
        if best_id:
            # Update features
            self.persons[best_id]['features'].append(features)
            if len(self.persons[best_id]['features']) > 30:
                self.persons[best_id]['features'].pop(0)
            
            # Track cameras
            if cam_id not in self.persons[best_id]['cameras']:
                self.persons[best_id]['cameras'].add(cam_id)
                prev_cams = sorted([c for c in self.persons[best_id]['cameras'] if c != cam_id])
                print(f"   🔗 [id:{best_id}] MATCHED in {cam_id.upper()} (sim={best_sim:.2f}) - Previously seen in: {prev_cams}")
            
            return best_id
        
        return None
    
    def mark_cam1_done(self):
        """Đánh dấu Cam1 hoàn thành"""
        self.cam1_completed = True
        print(f"\n✅ CAM1 ANALYZED - Identified {len(self.persons)} distinct people.")
        print(f"   Global IDs in database: {sorted(self.persons.keys())}")
    
    def summary(self):
        print(f"\n{'='*60}")
        print(f"📋 FINAL GLOBAL ID TRAJECTORY")
        print(f"{'='*60}")
        print(f"Total Unique Persons: {len(self.persons)}")
        for gid, info in sorted(self.persons.items()):
            cams = " → ".join(sorted(info['cameras']))
            print(f"  Person-{gid}: {cams}")


class Tracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.history = defaultdict(list)
    
    def reset(self):
        self.tracks.clear()
        self.history.clear()
        self.next_id = 1
    
    def update(self, detections):
        if not detections:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > MAX_AGE:
                    del self.tracks[tid]
            return []
        
        det_centers = np.array([[(d[0] + d[2]/2), (d[1] + d[3]/2)] for d in detections])
        
        if not self.tracks:
            results = []
            for x, y, w, h, q, _ in detections:
                c = (x + w/2, y + h/2)
                self.tracks[self.next_id] = {'center': c, 'box': (x,y,w,h), 'age': 1, 'lost': 0}
                self.history[self.next_id].append(c)
                results.append((x, y, w, h, self.next_id))
                self.next_id += 1
            return results
        
        track_ids = list(self.tracks.keys())
        track_centers = np.array([self.tracks[t]['center'] for t in track_ids])
        cost = cdist(det_centers, track_centers)
        rows, cols = linear_sum_assignment(cost)
        
        matched_d, matched_t = set(), set()
        results = []
        
        for r, c in zip(rows, cols):
            if cost[r, c] < MAX_DISTANCE:
                x, y, w, h, q, _ = detections[r]
                tid = track_ids[c]
                center = (x + w/2, y + h/2)
                self.tracks[tid].update({'center': center, 'box': (x,y,w,h), 'age': self.tracks[tid]['age']+1, 'lost': 0})
                self.history[tid].append(center)
                if len(self.history[tid]) > 50:
                    self.history[tid].pop(0)
                results.append((x, y, w, h, tid))
                matched_d.add(r)
                matched_t.add(c)
        
        for i, (x, y, w, h, q, _) in enumerate(detections):
            if i not in matched_d and q > NEW_TRACK_QUALITY:
                c = (x + w/2, y + h/2)
                self.tracks[self.next_id] = {'center': c, 'box': (x,y,w,h), 'age': 1, 'lost': 0}
                self.history[self.next_id].append(c)
                results.append((x, y, w, h, self.next_id))
                self.next_id += 1
        
        for i, tid in enumerate(track_ids):
            if i not in matched_t:
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > MAX_AGE:
                    del self.tracks[tid]
        
        return results


def calc_quality(box, kpts, confs):
    x, y, w, h = box
    s = min(np.sum(confs > MIN_KEYPOINT_CONF) / 17.0, 1.0) * 0.4
    h_vis = sum(confs[i] > MIN_KEYPOINT_CONF for i in HEAD_KPT)
    u_vis = sum(confs[i] > MIN_KEYPOINT_CONF for i in UPPER_KPT)
    l_vis = sum(confs[i] > MIN_KEYPOINT_CONF for i in LOWER_KPT)
    s += (0.33 if h_vis >= 1 else 0) + (0.33 if u_vis >= 2 else 0) + (0.34 if l_vis >= 1 else 0) * 0.3
    ar = h / (w + 1e-6)
    s += (1.0 if 1.5 <= ar <= 3.0 else 0.7 if 1.0 <= ar <= 4.0 else 0.3) * 0.2
    area = w * h
    s += (1.0 if MIN_BOX_AREA <= area <= MAX_BOX_AREA else 0.5) * 0.1
    return s


def filter_detections(results):
    dets = []
    if not results[0].boxes or len(results[0].boxes) == 0:
        return dets
    
    boxes = results[0].boxes.xywh.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    kpts = results[0].keypoints.data if results[0].keypoints is not None else None
    
    for i, (x, y, w, h) in enumerate(boxes):
        if classes[i] != 0 or confs[i] < CONF_THRESHOLD:
            continue
        if not (MIN_BOX_AREA <= w*h <= MAX_BOX_AREA):
            continue
        if not (0.3 <= h/(w+1e-6) <= 5.0):
            continue
        
        if kpts is not None and i < len(kpts):
            k = kpts[i].cpu().numpy()
            kc = k[:, 2]
            if np.sum(kc > MIN_KEYPOINT_CONF) < MIN_KEYPOINTS:
                continue
            if sum(kc[j] > MIN_KEYPOINT_CONF for j in UPPER_KPT) < 1:
                continue
            q = calc_quality((x,y,w,h), k, kc)
            if q >= 0.6:
                dets.append((x, y, w, h, q, ""))
    return dets


def process_camera(cam_id, video_path, model, tracker, db):
    """Process one camera"""
    
    print(f"\n{'='*60}")
    print(f"📹 Processing {cam_id.upper()}")
    print(f"{'='*60}")
    
    is_cam1 = (cam_id == "cam1")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"   Frames: {total_frames}")
    
    tracker.reset()
    local_to_global = {}
    seen_global_ids = set()
    unmatched_count = 0
    
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        disp = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        
        res = model(disp, verbose=False, conf=0.3)
        dets = filter_detections(res)
        tracked = tracker.update(dets)
        
        for x, y, w, h, local_id in tracked:
            x1, y1 = max(0, int(x - w/2)), max(0, int(y - h/2))
            x2, y2 = min(DISPLAY_WIDTH, int(x + w/2)), min(DISPLAY_HEIGHT, int(y + h/2))
            person_crop = disp[y1:y2, x1:x2]
            
            # ReID matching (every 5 frames)
            if local_id not in local_to_global or frame_num % 5 == 0:
                if is_cam1:
                    # Cam1: Register new ID if not exists
                    if local_id not in local_to_global:
                         # Check if this person already got a global ID in this camera session
                         # If not, register new global ID
                        gid = db.register_new(person_crop, cam_id)
                        if gid:
                            local_to_global[local_id] = gid
                            seen_global_ids.add(gid)
                else:
                    # Cam2/3: Match against DB
                    gid = db.match_only(person_crop, cam_id)
                    if gid:
                        local_to_global[local_id] = gid
                        seen_global_ids.add(gid)
                    else:
                        unmatched_count += 1
            
            # Draw
            gid = local_to_global.get(local_id)
            if gid:
                color = db.get_color(gid)
                cv2.rectangle(disp, (x1, y1), (x2, y2), color, 3)
                
                # Big Global ID
                cv2.putText(disp, f"P-ID:{gid}", (x1, y1-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
                
                # Small debug info
                if is_cam1:
                    info_text = "Tracking (Master)"
                else:
                    info_text = f"Matched (from {db.persons[gid]['first_cam']})"
                
                cv2.putText(disp, info_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                
                if len(tracker.history[local_id]) > 1:
                    pts = np.array(tracker.history[local_id], dtype=np.int32)
                    cv2.polylines(disp, [pts], False, color, 2)
            else:
                # No match (usually briefly before detection or if heavily occluded in Slave cams)
                if not is_cam1:
                     cv2.rectangle(disp, (x1, y1), (x2, y2), (100, 100, 100), 2)
        
        # Info Panel
        mode = "MASTER (Assigning IDs)" if is_cam1 else "SLAVE (Matching Only)"
        color_mode = (0, 255, 255) if is_cam1 else (255, 100, 0)
        
        cv2.putText(disp, f"{cam_id.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_mode, 2)
        cv2.putText(disp, mode, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.putText(disp, f"Frame: {frame_num}/{total_frames}", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # List detected IDs
        id_str = "None"
        if seen_global_ids:
            id_str = ", ".join([f"P-{gid}" for gid in sorted(seen_global_ids)])
        
        cv2.putText(disp, f"Visible: {id_str}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Progress
        progress = frame_num / total_frames
        cv2.rectangle(disp, (10, DISPLAY_HEIGHT-20), (10 + int((DISPLAY_WIDTH-20) * progress), DISPLAY_HEIGHT-5), (0,255,0), -1)
        cv2.rectangle(disp, (10, DISPLAY_HEIGHT-20), (DISPLAY_WIDTH-10, DISPLAY_HEIGHT-5), (255,255,255), 1)
        
        cv2.imshow('HAVEN - Sequential Master ReID', disp)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 27:
            cap.release()
            return False
    
    cap.release()
    
    print(f"\n   ✅ {cam_id.upper()} completed!")
    print(f"   � Global IDs seen: {sorted(seen_global_ids)}")
    if not is_cam1:
        print(f"   ⚠️  Unmatched detections: {unmatched_count}")
    
    if is_cam1:
        db.mark_cam1_done()
    
    return True


def main():
    print("="*60)
    print("HAVEN Step 4 | Master-Slave Cross-Camera ReID")
    print("="*60)
    print("\n🎯 Strategy:")
    print("   CAM1 (MASTER): Assign new Global IDs for everyone seen.")
    print("   CAM2 (SLAVE):  Match against Cam1 IDs. Do NOT create new IDs.")
    print("   CAM3 (SLAVE):  Match against Cam1 IDs. Do NOT create new IDs.")
    print("\n⌨️  Controls: 'q'=skip camera, ESC=exit")
    
    model = YOLO(MODEL_PATH)
    db = MasterIDDatabase()
    tracker = Tracker()
    
    for cam_id, video_path in CAMERAS:
        success = process_camera(cam_id, video_path, model, tracker, db)
        if not success:
            break
    
    cv2.destroyAllWindows()
    db.summary()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()