"""
HAVEN Step 5: Global Matching with Mutual Exclusion (Group Matching)
=====================================================================
Giải pháp cho ánh sáng chói/bóng tối:
- Giả định: Có tối đa 2 người quan trọng (ID 1 và ID 2)
- Sử dụng Group Matching (Hungarian Algorithm) cho mỗi frame
- Ràng buộc: Trong 1 frame, nếu 2 người xuất hiện, họ phải có ID khác nhau
- Temporal Smoothing: Voting ID trong cửa sổ 30 frames để tránh flicker
"""

import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import cdist, cosine
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, Counter
import pickle

# CONFIG
CAMERAS = [
    ("cam1", r"D:\HAVEN\backend\data\multi-camera\1.mp4"),
    ("cam2", r"D:\HAVEN\backend\data\multi-camera\2.mp4"),
    ("cam3", r"D:\HAVEN\backend\data\multi-camera\3.mp4"),
]
MODEL_PATH = r"D:\HAVEN\backend\models\yolo11n-pose.pt"
OUTPUT_FILE = "global_tracks.pkl"

# Detection params
CONF_THRESHOLD = 0.5
MIN_BOX_AREA = 1000  # Giảm để bắt tốt hơn
MAX_BOX_AREA = 400000

# Tracker params
MAX_DISTANCE = 300
MAX_AGE = 150

# Global IDs predefined colors
# ID 1: Áo trắng (Green)
# ID 2: Áo đen (Blue)
COLORS = {
    1: (0, 255, 0),    # Green - White shirt
    2: (255, 0, 0),    # Blue - Black shirt
}

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]


class FeatureExtractor:
    """Extract features robust for matching"""
    
    def normalize_lighting(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def extract(self, img_crop):
        if img_crop is None or img_crop.size == 0:
            return None
            
        try:
            crop = cv2.resize(img_crop, (64, 128))
            crop = self.normalize_lighting(crop)
            
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            
            # 1. Lightness (quan trọng phân biệt trắng/đen)
            l_mean = np.mean(lab[:, :, 0])
            
            # 2. Histogram features
            # L-channel histogram (16 bins)
            l_hist = cv2.calcHist([lab], [0], None, [16], [0, 256])
            l_hist = cv2.normalize(l_hist, l_hist).flatten()
            
            # Hue histogram (Màu sắc) (16 bins)
            h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            h_hist = cv2.normalize(h_hist, h_hist).flatten()
            
            # Combine
            features = np.concatenate([l_hist * 1.5, h_hist]) # Weight L-channel more
            
            return features, l_mean
            
        except Exception as e:
            return None


class GlobalManager:
    """Quản lý Global IDs với logic Group Matching"""
    
    def __init__(self):
        self.extractor = FeatureExtractor()
        
        # Lưu trữ profile cho 2 ID chính
        # ID 1: Expected Lightness HIGH (White shirt)
        # ID 2: Expected Lightness LOW (Black shirt)
        self.profiles = {
            1: {'features': [], 'l_history': [], 'type': 'WHITE'},
            2: {'features': [], 'l_history': [], 'type': 'BLACK'}
        }
        
        # Buffer để smoothing ID
        self.track_votes = defaultdict(lambda: list()) # local_id -> [global_id_votes]
        self.id_lock = {} # local_id -> final_global_id
        
    def init_profiles_from_cam1(self, detections, frame):
        """Khởi tạo profile từ Cam1 (giả sử Cam1 nhìn rõ nhất ở đầu)"""
        # detections: [(x,y,w,h,crop), ...]
        # Sort by lightness -> Sáng nhất là ID 1, Tối nhất là ID 2
        
        candidates = []
        for det in detections:
            x, y, w, h, crop = det
            res = self.extractor.extract(crop)
            if res:
                feat, l_mean = res
                candidates.append({
                    'feat': feat, 'l_mean': l_mean, 'crop': crop
                })
        
        if len(candidates) >= 2:
            # Sort by lightness descending
            candidates.sort(key=lambda x: x['l_mean'], reverse=True)
            
            # Assign ID 1 (White) to brightest
            self.profiles[1]['features'].append(candidates[0]['feat'])
            self.profiles[1]['l_history'].append(candidates[0]['l_mean'])
            print(f"   Initialized ID 1 (WHITE): L={candidates[0]['l_mean']:.1f}")
            
            # Assign ID 2 (Black) to darkest (or second brightest if only 2)
            # Find detection with lowest lightness
            darkest = candidates[-1]
            self.profiles[2]['features'].append(darkest['feat'])
            self.profiles[2]['l_history'].append(darkest['l_mean'])
            print(f"   Initialized ID 2 (BLACK): L={darkest['l_mean']:.1f}")
            return True
            
        return False

    def update_profile(self, gid, feat, l_mean):
        self.profiles[gid]['features'].append(feat)
        self.profiles[gid]['l_history'].append(l_mean)
        if len(self.profiles[gid]['features']) > 50:
            self.profiles[gid]['features'].pop(0)
            self.profiles[gid]['l_history'].pop(0)

    def compute_cost(self, feat, l_mean, gid):
        """Tính cost (distance) giữa feature mới và profile gid"""
        profile = self.profiles[gid]
        if not profile['features']:
            return 1.0
            
        # 1. Feature Distance (Cosine) - Lấy min distance với recent history
        dists = [cosine(feat, f) for f in profile['features'][-20:]]
        feat_dist = np.min(dists) if dists else 1.0
        
        # 2. Lightness Distance
        # Normalize L distance (0-255) to 0-1 range
        # ID 1 should be high L, ID 2 should be low L
        # Tuy nhiên ánh sáng thay đổi, nên so sánh với average recent history
        avg_l = np.mean(profile['l_history'][-20:])
        l_dist = abs(l_mean - avg_l) / 255.0
        
        # 3. Type Constraint (White vs Black)
        # ID 1 (White): Penalty nếu quá tối (< 50)
        # ID 2 (Black): Penalty nếu quá sáng (> 200)
        type_penalty = 0.0
        if gid == 1 and l_mean < 60: type_penalty = 0.5
        if gid == 2 and l_mean > 190: type_penalty = 0.5
        
        # Combined Cost
        total_cost = 0.6 * feat_dist + 0.4 * l_dist + type_penalty
        return total_cost

    def match_group(self, detections):
        """
        Group Matching using Hungarian Algorithm
        detections: list of {'crop': img, 'local_id': int}
        returns: {local_id: global_id}
        """
        if not detections:
            return {}
            
        # Nếu chưa init profile, cố gắng init nếu đủ 2 người
        if not self.profiles[1]['features']:
            crops = [(0,0,0,0, d['crop']) for d in detections]
            if self.init_profiles_from_cam1(crops, None):
                pass # Continue to match
            else:
                return {} # Chưa đủ dữ liệu để init
        
        # Extract features for current detections
        current_feats = []
        valid_indices = []
        
        for i, det in enumerate(detections):
            res = self.extractor.extract(det['crop'])
            if res:
                current_feats.append({'feat': res[0], 'l_mean': res[1], 'local_id': det['local_id']})
                valid_indices.append(i)
        
        if not current_feats:
            return {}
            
        # Cost matrix: Rows=Detections, Cols=GlobalIDs (1, 2)
        # Để cho phép miss detection, ta có thể match với "None" (dummy ID) nhưng ở đây giả định 2 người
        # Nếu chỉ có 1 detection, ta xem nó gần ID nào nhất
        
        gids = [1, 2]
        cost_matrix = np.zeros((len(current_feats), len(gids)))
        
        for r, det_info in enumerate(current_feats):
            for c, gid in enumerate(gids):
                cost_matrix[r, c] = self.compute_cost(det_info['feat'], det_info['l_mean'], gid)
                
        # Hungarian Algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        assignments = {}
        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            local_id = current_feats[r]['local_id']
            global_id = gids[c]
            
            # Threshold to accept match
            if cost < 0.6: 
                # Add vote
                self.track_votes[local_id].append(global_id)
                if len(self.track_votes[local_id]) > 30:
                    self.track_votes[local_id].pop(0)
                
                # Decision smoothing: Majority vote
                votes = self.track_votes[local_id]
                most_common = Counter(votes).most_common(1)[0][0]
                
                assignments[local_id] = most_common
                
                # Update profile (online learning)
                # Chỉ update nếu detection confident (cost thấp) để tránh drift
                if cost < 0.3:
                    self.update_profile(most_common, current_feats[r]['feat'], current_feats[r]['l_mean'])
            else:
                # Nếu cost cao, có thể giữ ID cũ nếu có
                if local_id in self.id_lock:
                     assignments[local_id] = self.id_lock[local_id]
        
        # Update locks
        for lid, gid in assignments.items():
            self.id_lock[lid] = gid
            
        return assignments


class Tracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
    
    def update(self, detections):
        # Euclidean tracker
        if not detections: return []
        
        det_centers = np.array([[d[0], d[1]] for d in detections])
        
        if not self.tracks:
            res = []
            for x,y,w,h,kpts in detections:
                self.tracks[self.next_id] = {'center': (x,y), 'age': 0}
                res.append((x,y,w,h,self.next_id, kpts))
                self.next_id += 1
            return res
            
        track_ids = list(self.tracks.keys())
        track_centers = np.array([self.tracks[t]['center'] for t in track_ids])
        
        cost = cdist(det_centers, track_centers)
        rows, cols = linear_sum_assignment(cost)
        
        matched_d = set()
        res = []
        
        for r, c in zip(rows, cols):
            if cost[r, c] < 200:
                tid = track_ids[c]
                x,y,w,h,kpts = detections[r]
                self.tracks[tid]['center'] = (x,y)
                self.tracks[tid]['age'] = 0
                res.append((x,y,w,h,tid,kpts))
                matched_d.add(r)
        
        for i in range(len(detections)):
            if i not in matched_d:
                x,y,w,h,kpts = detections[i]
                self.tracks[self.next_id] = {'center': (x,y), 'age': 0}
                res.append((x,y,w,h,self.next_id,kpts))
                self.next_id += 1
                
        return res

def filter_detections(results):
    dets = []
    if not results[0].boxes: return dets
    
    boxes = results[0].boxes.xywh.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    kpts_data = results[0].keypoints.data if results[0].keypoints is not None else None
    
    for i, (x,y,w,h) in enumerate(boxes):
        if classes[i] == 0 and confs[i] > 0.4 and w*h > 1500:
            kpts = kpts_data[i].cpu().numpy() if kpts_data is not None else None
            dets.append((x,y,w,h,kpts))
    return dets

def process_all_cameras():
    print("🚀 Running Global Group Matching...")
    model = YOLO(MODEL_PATH)
    manager = GlobalManager()
    
    all_data = {} # cam_id -> frame_idx -> list
    person_cameras = defaultdict(set)
    
    # Process sequentially but maintaining global state
    for cam_id, video_path in CAMERAS:
        print(f"📹 Processing {cam_id}...")
        cap = cv2.VideoCapture(video_path)
        tracker = Tracker()
        frame_data = {}
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            
            # Detect & Track
            results = model(frame, verbose=False, conf=0.4)
            dets = filter_detections(results)
            tracked = tracker.update(dets)
            
            # Prepare batch for matching
            batch_for_matching = []
            batch_result_mapping = {} # index -> (x,y,w,h,local_id,kpts)
            
            for i, (x,y,w,h,lid,kpts) in enumerate(tracked):
                x1,y1 = max(0,int(x-w/2)), max(0,int(y-h/2))
                x2,y2 = min(frame.shape[1],int(x+w/2)), min(frame.shape[0],int(y+h/2))
                crop = frame[y1:y2, x1:x2]
                
                batch_for_matching.append({'crop': crop, 'local_id': lid})
                batch_result_mapping[lid] = (x,y,w,h,kpts)
            
            # Group Matching
            assignments = manager.match_group(batch_for_matching)
            
            # Store results
            frame_res = []
            for lid, gid in assignments.items():
                x,y,w,h,kpts = batch_result_mapping[lid]
                kpts_list = kpts.tolist() if kpts is not None else None
                frame_res.append((x,y,w,h,gid,kpts_list))
                person_cameras[gid].add(cam_id)
                
            frame_data[frame_idx] = frame_res
            
            if frame_idx % 100 == 0:
                print(f"   Frame {frame_idx}")
                
        all_data[cam_id] = frame_data
        cap.release()
        
    # Save
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump({
            'all_data': all_data,
            'num_persons': 2,
            'person_cameras': {gid: list(cams) for gid, cams in person_cameras.items()},
            'skeleton': SKELETON
        }, f)
    print("✅ Done.")

if __name__ == "__main__":
    process_all_cameras()
