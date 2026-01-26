"""
HAVEN ADL - Video Test Logic
============================
Script kiểm thử logic ADL (Action Detection) trên video file.
"""

import cv2
import time
import os
import sys
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict

# Add backend to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from backend.src.core.tracker import SimpleTracker
from backend.src.adl.data import FrameData, TrackHistory
from backend.src.adl.inference import PostureClassifier
from backend.src.adl.rules import RuleEngine
from ultralytics import YOLO

# Load Config
load_dotenv()
MODEL_PATH = os.getenv("AI_MODEL_PATH", "models/yolo11n-pose.pt")
VIDEO_PATH = os.getenv("TEST_VIDEO_PATH", "data/video/walking.mp4")

def main():
    print("="*60)
    print("HAVEN ADL - Test Automation")
    print("="*60)
    
    # Init Components
    tracker = SimpleTracker(iou_threshold=0.3)
    posture_classifier = PostureClassifier()
    rule_engine = RuleEngine()
    
    # Load Model
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Open Video
    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error opening video.")
        return

    # Track Histories
    tracks: Dict[int, TrackHistory] = {}
    
    # UI Config
    cv2.namedWindow("HAVEN ADL", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HAVEN ADL", 1280, 720)
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Video finished. Restarting...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # Reset tracks? Maybe keep history? 
                # Better to clear for clean loop test
                tracker = SimpleTracker() 
                tracks = {}
                continue
            
            # 1. AI Inference
            results = model(frame, verbose=False, conf=0.25)[0]
            
            # 2. Parse Detections
            detections = []
            if results.boxes:
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                # Keypoints (N, 17, 3)
                if results.keypoints:
                    kpts_data = results.keypoints.data.cpu().numpy()
                else:
                    kpts_data = [None] * len(boxes)
                
                for i, box in enumerate(boxes):
                    det = {
                        "bbox": box.tolist(),
                        "conf": confs[i],
                        "keypoints": kpts_data[i].tolist() if kpts_data[i] is not None else []
                    }
                    detections.append(det)
            
            # 3. Tracking
            tracked_dets = tracker.update(detections)
            
            # 4. ADL Processing
            current_time = time.time()
            
            for det in tracked_dets:
                tid = det['track_id']
                
                # Init history if new
                if tid not in tracks:
                    tracks[tid] = TrackHistory(tid)
                
                track = tracks[tid]
                
                # Create FrameData
                frame_data = FrameData(
                    timestamp=current_time,
                    bbox=det['bbox'],
                    keypoints=det['keypoints']
                )
                
                # A. Feature Extraction & Classification
                frame_data = posture_classifier.process(frame_data)
                
                # B. Rule Engine
                events = rule_engine.process_track(track, frame_data)
                
                # --- VISUALIZATION ---
                x1, y1, x2, y2 = map(int, det['bbox'])
                
                # Color based on Posture
                color = (0, 255, 0) # Green (Standing)
                if frame_data.posture == "SITTING": color = (0, 255, 255) # Yellow
                elif frame_data.posture == "LAYING": color = (0, 0, 255) # Red
                
                # Draw Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw Info
                label = f"ID:{tid} {frame_data.posture} ({track.state_duration:.1f}s)"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw Events
                if events:
                    event_text = f"EVENT: {', '.join(events)}"
                    cv2.putText(frame, event_text, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                    print(f"!!! DETECTED EVENT: {events} for ID {tid}")

            cv2.imshow("HAVEN ADL", frame)
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord(' '): paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
