"""
HAVEN Step 3: Multi-Camera Hub với Custom Filter
Mỗi camera có tracker riêng + filter quality
"""
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import threading
from queue import Queue
import time

# CONFIG
CAMERAS = {
    "cam1": r"D:\HAVEN\backend\data\multi-camera\1.mp4",
    "cam2": r"D:\HAVEN\backend\data\multi-camera\2.mp4",
    "cam3": r"D:\HAVEN\backend\data\multi-camera\3.mp4",
}
MODEL_PATH = r"D:\HAVEN\backend\models\yolo11n-pose.pt"

# Kích thước hiển thị CỐ ĐỊNH cho mỗi camera
DISPLAY_WIDTH = 480   # Chiều rộng mỗi camera
DISPLAY_HEIGHT = 360  # Chiều cao mỗi camera

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


class Tracker:
    def __init__(self, cam_id):
        self.cam_id = cam_id
        self.tracks = {}
        self.next_id = 1
        self.history = defaultdict(list)
        self.colors = {}
    
    def get_color(self, tid):
        if tid not in self.colors:
            # Dùng số đơn giản thay vì hash() để tránh số âm
            cam_num = int(self.cam_id[-1]) if self.cam_id[-1].isdigit() else 1
            seed = abs(tid * 100 + cam_num * 1000) % (2**31)
            np.random.seed(seed)
            self.colors[tid] = tuple(np.random.randint(50, 255, 3).tolist())
        return self.colors[tid]
    
    def update(self, detections):
        if not detections:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > MAX_AGE:
                    del self.tracks[tid]
            return []
        
        det_centers = np.array([[(d[0] + d[2]/2), (d[1] + d[3]/2)] for d in detections])
        
        if not self.tracks:
            for x, y, w, h, q, _ in detections:
                c = (x + w/2, y + h/2)
                self.tracks[self.next_id] = {'center': c, 'box': (x,y,w,h), 'age': 1, 'lost': 0, 'quality': q}
                self.history[self.next_id].append(c)
                self.next_id += 1
            return [(x, y, w, h, i+1) for i, (x, y, w, h, _, _) in enumerate(detections)]
        
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
                self.tracks[tid].update({'center': center, 'box': (x,y,w,h), 'age': self.tracks[tid]['age']+1, 'lost': 0, 'quality': q})
                self.history[tid].append(center)
                if len(self.history[tid]) > 30:
                    self.history[tid].pop(0)
                results.append((x, y, w, h, tid))
                matched_d.add(r)
                matched_t.add(c)
        
        for i, (x, y, w, h, q, _) in enumerate(detections):
            if i not in matched_d and q > NEW_TRACK_QUALITY:
                c = (x + w/2, y + h/2)
                self.tracks[self.next_id] = {'center': c, 'box': (x,y,w,h), 'age': 1, 'lost': 0, 'quality': q}
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
    s += (1.0 if MIN_BOX_AREA <= area <= MAX_BOX_AREA else 0.5 if MIN_BOX_AREA*0.5 < area < MAX_BOX_AREA*2 else 0.0) * 0.1
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
                dets.append((x, y, w, h, q, f"{int(np.sum(kc > MIN_KEYPOINT_CONF))}/17"))
    return dets


class CameraWorker:
    def __init__(self, cam_id, video_path):
        self.cam_id = cam_id
        self.video_path = video_path
        self.model = None
        self.tracker = Tracker(cam_id)
        self.cap = None
        self.running = False
        self.thread = None
        self.frame_queue = Queue(maxsize=2)
        self.frame_count = 0
        self.unique_ids = set()
        self.fps = 0
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"✅ [{self.cam_id}] Started")
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
    
    def _run(self):
        self.model = YOLO(MODEL_PATH)
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            return
        
        fps_timer = time.time()
        fps_counter = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.tracker.history.clear()
                continue
            
            self.frame_count += 1
            fps_counter += 1
            
            # Resize về kích thước CỐ ĐỊNH (tất cả cameras cùng size)
            disp = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            res = self.model(disp, verbose=False, conf=0.3)
            dets = filter_detections(res)
            tracked = self.tracker.update(dets)
            
            for x, y, w, h, tid in tracked:
                self.unique_ids.add(tid)
                color = self.tracker.get_color(tid)
                x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
                cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
                cv2.putText(disp, f"ID:{tid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                if len(self.tracker.history[tid]) > 1:
                    pts = np.array(self.tracker.history[tid], dtype=np.int32)
                    cv2.polylines(disp, [pts], False, color, 2)
            
            cv2.putText(disp, self.cam_id.upper(), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(disp, f"Track: {len(tracked)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(disp, f"IDs: {len(self.unique_ids)}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            
            if time.time() - fps_timer >= 1.0:
                self.fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()
            
            cv2.putText(disp, f"FPS: {self.fps:.1f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
            self.frame_queue.put(disp)
    
    def get_latest_frame(self):
        try:
            return self.frame_queue.get_nowait()
        except:
            return None


def main():
    print("HAVEN Step 3 | Multi-Camera + Custom Filter")
    
    workers = {cam_id: CameraWorker(cam_id, path) for cam_id, path in CAMERAS.items()}
    
    for w in workers.values():
        w.start()
    
    time.sleep(3)
    print("▶️  Running (press 'q' to quit)\n")
    
    last_frames = {cam_id: None for cam_id in CAMERAS.keys()}
    
    try:
        while True:
            # Hiển thị từng camera riêng biệt
            for cam_id in ["cam1", "cam2", "cam3"]:
                new_frame = workers[cam_id].get_latest_frame()
                
                if new_frame is not None:
                    last_frames[cam_id] = new_frame.copy()
                    cv2.imshow(f'HAVEN - {cam_id.upper()}', new_frame)
                elif last_frames[cam_id] is not None:
                    cv2.imshow(f'HAVEN - {cam_id.upper()}', last_frames[cam_id])
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if workers["cam1"].frame_count > 0 and workers["cam1"].frame_count % 150 == 0:
                print("\n📊 Stats:")
                for cam_id, w in workers.items():
                    print(f"  {cam_id}: {w.frame_count}f | IDs: {sorted(w.unique_ids)} | {w.fps:.1f}fps")
    
    finally:
        for w in workers.values():
            w.stop()
        cv2.destroyAllWindows()
        
        print("\n✅ Final:")
        for cam_id, w in workers.items():
            print(f"  {cam_id}: {w.frame_count}f | IDs: {sorted(w.unique_ids)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()