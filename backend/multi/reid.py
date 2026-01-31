"""
Color Histogram ReID - Simple & Effective
Based on step4-reid.py
"""
import cv2
import numpy as np
from scipy.spatial.distance import cosine


class ColorHistogramReID:
    """Simple ReID using color histogram (HSV)."""
    
    def extract(self, img_crop):
        """Extract features from person crop."""
        if img_crop is None or img_crop.size == 0:
            return None
        if img_crop.shape[0] < 20 or img_crop.shape[1] < 10:
            return None
        
        try:
            hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
            h = img_crop.shape[0]
            
            # 3 parts: head, body, legs
            parts = [
                hsv[:h//3, :],           # Head
                hsv[h//3:2*h//3, :],     # Body
                hsv[2*h//3:, :]          # Legs
            ]
            
            features = []
            for part in parts:
                if part.size == 0:
                    features.extend([0] * 32)
                    continue
                
                # H & S histograms
                h_hist = cv2.calcHist([part], [0], None, [16], [0, 180])
                s_hist = cv2.calcHist([part], [1], None, [16], [0, 256])
                h_hist = cv2.normalize(h_hist, h_hist).flatten()
                s_hist = cv2.normalize(s_hist, s_hist).flatten()
                
                features.extend(h_hist)
                features.extend(s_hist)
            
            return np.array(features)
        except:
            return None


class MasterSlaveReIDDB:
    """Master-Slave ReID Database."""
    
    def __init__(self, reid_threshold=0.55):
        self.reid = ColorHistogramReID()
        self.persons = {}  # global_id -> {features: [], first_cam: str, cameras: set}
        self.next_id = 1
        self.reid_threshold = reid_threshold
    
    def register_new(self, person_crop, cam_id):
        """Register new person (ONLY Cam1)."""
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
        
        print(f"    G-ID {new_id} created in {cam_id.upper()}")
        return new_id
    
    def match_only(self, person_crop, cam_id):
        """Match against existing IDs (Cam2/3/4)."""
        features = self.reid.extract(person_crop)
        if features is None:
            return None
        
        best_id = None
        best_sim = 0
        
        for gid, info in self.persons.items():
            # Compare with last 10 features
            sims = [1 - cosine(features, f) for f in info['features'][-10:]]
            if sims:
                avg_sim = np.mean(sims)
                if avg_sim > best_sim and avg_sim > self.reid_threshold:
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
                print(f"    G-ID {best_id} matched in {cam_id.upper()} (sim={best_sim:.2f})")
            
            return best_id
        
        return None
    
    def summary(self):
        """Print summary."""
        print(f"\n{'='*60}")
        print(f" GLOBAL ID SUMMARY")
        print(f"{'='*60}")
        print(f"Total Persons: {len(self.persons)}")
        for gid, info in sorted(self.persons.items()):
            cams = "  ".join(sorted(info['cameras']))
            print(f"  Person-{gid}: {cams}")

