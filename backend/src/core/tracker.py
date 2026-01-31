"""
Unified tracker wrapper (ByteTrack/BoT-SORT via Ultralytics).
"""
from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Optional


class Tracker:
    """
    Unified tracking interface using Ultralytics built-in trackers.
    """
    
    def __init__(self, tracker_type: str = "bytetrack", persist: bool = True):
        """
        Initialize tracker.
        
        Args:
            tracker_type: "bytetrack" or "botsort"
            persist: Whether to persist tracks across frames
        """
        self.tracker_type = tracker_type
        self.persist = persist
        self.tracker_config = f"{tracker_type}.yaml"
        
        # Track history storage
        self.track_history = {}
    
    def update(self, detections: List[Dict], frame: np.ndarray, 
               model: YOLO) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections from detector
            frame: Current frame (needed for some trackers)
            model: YOLO model instance (for track method)
        
        Returns:
            List of tracked objects: [{
                'track_id': int,
                'bbox': [x, y, w, h],
                'conf': float,
                'class_id': int
            }]
        """
        # Convert detections to YOLO format for tracking
        # Note: Ultralytics tracking works directly with prediction results
        # This is a simplified wrapper - in practice, you'd use model.track()
        
        results = model.track(
            frame,
            persist=self.persist,
            tracker=self.tracker_config,
            verbose=False
        )
        
        tracks = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                # Check if track ID exists
                if box.id is None:
                    continue
                
                track_id = int(box.id[0])
                
                # Convert bbox from xyxy to xywh
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = x2 - x1
                h = y2 - y1
                
                track = {
                    'track_id': track_id,
                    'bbox': [int(x1), int(y1), int(w), int(h)],
                    'conf': float(box.conf[0]),
                    'class_id': int(box.cls[0])
                }
                
                tracks.append(track)
                
                # Update track history
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                self.track_history[track_id].append(track)
        
        return tracks
    
    def get_track_history(self, track_id: int, max_frames: int = 30) -> List[Dict]:
        """
        Get history for a specific track.
        
        Args:
            track_id: Track ID
            max_frames: Maximum number of historical frames to return
        
        Returns:
            List of track dictionaries (most recent first)
        """
        if track_id not in self.track_history:
            return []
        
        history = self.track_history[track_id]
        return history[-max_frames:]
    
    def reset(self):
        """Reset tracker state."""
        self.track_history = {}


if __name__ == "__main__":
    # Test tracker
    import cv2
    from .detector import PersonDetector
    
    # Initialize
    model = YOLO("yolov8n.pt")
    tracker = Tracker(tracker_type="bytetrack", persist=True)
    
    # Load test video
    cap = cv2.VideoCapture("/path/to/test/video.mp4")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Track directly using model
        tracks = tracker.update([], frame, model)
        
        print(f"Frame: {len(tracks)} tracks")
        for track in tracks:
            print(f"  Track {track['track_id']}: bbox={track['bbox']}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()

