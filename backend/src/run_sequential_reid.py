"""
HAVEN Multi-Camera Sequential ReID Runner

Processes cameras SEQUENTIALLY (cam1 -> cam2 -> cam3 -> cam4).
- Cam1 (MASTER): Creates all initial Global IDs
- Cam2 (MASTER): Matches against cam1's gallery, can create new IDs
- Cam3+ (SLAVE): Only matches, cannot create new IDs

Usage:
    python run_sequential_reid.py --config configs/multicam.yaml
    
Author: HAVEN Team
Version: 2.0
"""
import sys
import cv2
import yaml
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add paths
SCRIPT_DIR = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from ultralytics import YOLO
from src.io.video_stream import VideoStream
from src.global_id.manager import GlobalIDManager, MatchReason


class SequentialReIDRunner:
    """
    Sequential multi-camera ReID runner.
    
    Processes cameras ONE BY ONE:
    1. Cam1 first (creates Global IDs)
    2. Cam2 next (matches + creates)
    3. Cam3 (matches only)
    4. Cam4 (matches only)
    """
    
    def __init__(self, config_path: str):
        """Initialize runner."""
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print("=" * 60)
        print("HAVEN Sequential ReID (Master-Slave)")
        print("=" * 60)
        
        # Display settings
        self.display_w = self.config['display']['width']
        self.display_h = self.config['display']['height']
        
        # Output directory
        self.output_dir = Path(self.config['output']['dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize GlobalIDManager
        self._init_global_id_manager()
        
        # Load models
        self._init_models()
        
        # Camera configs
        self.data_root = Path(self.config['data']['data_root'])
        self.cameras = self.config['data']['cameras']
        
        print(f"Cameras: {len(self.cameras)}")
        print(f"Master cameras: {self.config['master_cameras']['ids']}")
        print("=" * 60)
    
    def _init_global_id_manager(self):
        """Initialize GlobalIDManager."""
        master_ids = self.config['master_cameras']['ids']
        reid_cfg = self.config['reid']
        
        camera_graph = self.config.get('camera_graph', {}).get('transitions', {})
        camera_graph = {int(k): v for k, v in camera_graph.items()}
        
        # Get config values with defaults
        quality_cfg = reid_cfg.get('quality', {})
        memory_cfg = reid_cfg.get('memory', {})
        thresholds_cfg = reid_cfg.get('thresholds', {})
        
        self.global_id_manager = GlobalIDManager(
            master_camera_ids=master_ids,
            accept_threshold=thresholds_cfg.get('accept', 0.75),
            reject_threshold=thresholds_cfg.get('reject', 0.50),
            camera_graph=camera_graph,
            max_prototypes=memory_cfg.get('max_prototypes', 10),
            ema_alpha=memory_cfg.get('ema_alpha', 0.3),
            min_bbox_size=quality_cfg.get('min_bbox_size', 80),
            min_track_frames=quality_cfg.get('min_tracklet_frames', quality_cfg.get('min_track_frames', 5))
        )
    
    def _init_models(self):
        """Initialize YOLO models."""
        yolo_cfg = self.config['yolo']
        models_dir = BACKEND_DIR / "models"
        
        pose_path = models_dir / yolo_cfg['pose_model']
        if pose_path.exists():
            self.pose_model = YOLO(str(pose_path))
        else:
            self.pose_model = YOLO(yolo_cfg['pose_model'])
        print(f"Pose model: {yolo_cfg['pose_model']}")
        
        self.yolo_conf = yolo_cfg['conf_threshold']
    
    def _extract_embedding(self, frame: np.ndarray, bbox: Tuple) -> np.ndarray:
        """Extract color histogram embedding."""
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(768)
        
        # Split into 3 parts
        h = crop.shape[0]
        parts = [crop[:h//3], crop[h//3:2*h//3], crop[2*h//3:]]
        
        embedding = []
        for part in parts:
            if part.size == 0:
                embedding.extend([0] * 256)
                continue
            for c in range(3):
                hist = cv2.calcHist([part], [c], None, [64], [0, 256])
                hist = hist.flatten()
                hist = hist / (np.sum(hist) + 1e-8)
                embedding.extend(hist[:64])
        
        embedding = np.array(embedding[:768])
        if len(embedding) < 768:
            embedding = np.pad(embedding, (0, 768 - len(embedding)))
        
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _draw_detection(self, frame, bbox, global_id, track_id, reason, kpts=None):
        """Draw detection overlay."""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Color based on status
        if reason == MatchReason.MASTER_NEW:
            color = (0, 255, 0)  # Green - new
        elif reason == MatchReason.MATCHED:
            color = (255, 255, 0)  # Cyan - matched
        else:
            color = (0, 165, 255)  # Orange - temp
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Label
        if global_id > 0:
            label = f"G{global_id}"
        else:
            label = f"T{track_id}"
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Skeleton
        if kpts is not None:
            self._draw_skeleton(frame, kpts)
    
    def _draw_skeleton(self, frame, keypoints):
        """Draw pose skeleton."""
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        for i, j in skeleton:
            if i < len(keypoints) and j < len(keypoints):
                x1, y1, c1 = keypoints[i]
                x2, y2, c2 = keypoints[j]
                if c1 > 0.3 and c2 > 0.3:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        
        for kp in keypoints:
            x, y, c = kp
            if c > 0.3:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
    
    def process_camera(self, cam_config: dict, cam_index: int) -> bool:
        """
        Process one camera completely.
        
        Returns:
            True to continue, False to quit
        """
        cam_id = cam_config['id']
        cam_name = cam_config.get('name', f'cam{cam_id}')
        source_type = cam_config.get('source_type', 'video_folder')
        source_path = cam_config.get('path', f'cam{cam_id}')
        pattern = cam_config.get('pattern', '*.mp4')
        
        print(f"\n{'='*60}")
        print(f"CAMERA {cam_id}: {source_path}")
        print(f"{'='*60}")
        
        # Open video stream with proper parameters
        try:
            stream = VideoStream(
                camera_id=cam_id,
                camera_name=cam_name,
                source_type=source_type,
                source_path=source_path,
                data_root=str(self.data_root),
                pattern=pattern
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            return True
        
        fps = stream.get_fps()
        total_frames = stream.get_total_frames()
        print(f"  Frames: {total_frames}, FPS: {fps:.1f}")
        
        # Check if master
        is_master = self.global_id_manager.is_master_camera(cam_id)
        if is_master:
            print(f"  Mode: MASTER (can create new IDs)")
        else:
            print(f"  Mode: SLAVE (match only)")
        
        # Track states
        track_history = {}  # track_id -> num_frames
        global_ids_seen = set()
        
        # Video writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.output_dir / f"cam{cam_id}_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (self.display_w, self.display_h))
        
        # Recording state
        recording = False
        rec_writer = None
        
        cv2.namedWindow(f"HAVEN Sequential - CAM{cam_id}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"HAVEN Sequential - CAM{cam_id}", self.display_w, self.display_h)
        
        frame_idx = 0
        paused = False
        
        while True:
            if not paused:
                ret, frame = stream.read()
                if not ret:
                    break
                
                frame_idx += 1
                orig_h, orig_w = frame.shape[:2]
                
                # YOLO Pose with tracking
                results = self.pose_model.track(frame, persist=True, verbose=False, conf=self.yolo_conf)
                
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    keypoints_data = results[0].keypoints
                    
                    for i, track_id in enumerate(track_ids):
                        bbox = tuple(boxes[i])
                        
                        # Count frames for this track
                        track_history[track_id] = track_history.get(track_id, 0) + 1
                        num_frames = track_history[track_id]
                        
                        kpts = None
                        if keypoints_data is not None:
                            kpts = keypoints_data.data[i].cpu().numpy()
                        
                        # Extract embedding
                        embedding = self._extract_embedding(frame, bbox)
                        bbox_size = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
                        
                        # Assign global ID
                        global_id, reason, score = self.global_id_manager.assign_global_id(
                            camera_id=cam_id,
                            local_track_id=track_id,
                            embedding=embedding,
                            quality=1.0,
                            timestamp=stream.get_timestamp(),
                            frame_idx=frame_idx,
                            num_frames=num_frames,
                            bbox_size=int(bbox_size)
                        )
                        
                        if global_id > 0:
                            global_ids_seen.add(global_id)
                        
                        # Log new IDs and matches
                        if reason == MatchReason.MASTER_NEW:
                            print(f"   Frame {frame_idx}: Track {track_id}  G{global_id} (NEW)")
                        elif reason == MatchReason.MATCHED and num_frames == 1:
                            print(f"   Frame {frame_idx}: Track {track_id}  G{global_id} (MATCH: {score:.2f})")
                        
                        # Draw
                        self._draw_detection(frame, bbox, global_id, track_id, reason, kpts)
                
                # Resize for display
                display = cv2.resize(frame, (self.display_w, self.display_h))
                
                # UI overlay
                mode_text = "MASTER" if is_master else "SLAVE"
                cv2.putText(display, f"CAM{cam_id} ({mode_text})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display, f"Frame: {frame_idx}/{total_frames}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(display, f"Global IDs: {sorted(global_ids_seen)}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Recording indicator
                if recording:
                    cv2.circle(display, (self.display_w - 30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(display, "REC", (self.display_w - 70, 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    if rec_writer:
                        rec_writer.write(display)
                
                # Write to output
                writer.write(display)
                
                # Show
                cv2.imshow(f"HAVEN Sequential - CAM{cam_id}", display)
            
            # Key handling
            wait_time = max(1, int(1000 / fps)) if not paused else 50
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord('q'):
                stream.release()
                writer.release()
                if rec_writer:
                    rec_writer.release()
                cv2.destroyAllWindows()
                return False  # Quit completely
            elif key == ord(' '):
                paused = not paused
                print("  PAUSED" if paused else "  RESUMED")
            elif key == ord('n'):
                break  # Next camera
            elif key == ord('g'):
                if not recording:
                    recording = True
                    rec_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    rec_path = self.output_dir / f"rec_cam{cam_id}_{rec_ts}.mp4"
                    rec_writer = cv2.VideoWriter(str(rec_path), fourcc, fps, (self.display_w, self.display_h))
                    print(f"   Recording started: {rec_path}")
                else:
                    recording = False
                    if rec_writer:
                        rec_writer.release()
                        rec_writer = None
                    print(f"   Recording stopped")
        
        # Cleanup
        stream.release()
        writer.release()
        if rec_writer:
            rec_writer.release()
        cv2.destroyWindow(f"HAVEN Sequential - CAM{cam_id}")
        
        print(f"  Finished. Global IDs seen: {sorted(global_ids_seen)}")
        print(f"  Output: {out_path}")
        
        return True
    
    def run(self):
        """Run sequential processing."""
        print("\n Starting sequential processing...")
        print("Controls: SPACE=Pause, N=Next Camera, Q=Quit, G=Record MP4\n")
        
        for i, cam_config in enumerate(self.cameras):
            should_continue = self.process_camera(cam_config, i)
            if not should_continue:
                break
            
            # Transition screen
            if i < len(self.cameras) - 1:
                next_cam = self.cameras[i + 1]
                transition = np.zeros((self.display_h, self.display_w, 3), dtype=np.uint8)
                cv2.putText(transition, f"Next: Camera {next_cam['id']}",
                           (self.display_w//2 - 150, self.display_h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.imshow("HAVEN Sequential", transition)
                cv2.waitKey(1500)
                cv2.destroyWindow("HAVEN Sequential")
        
        # Summary
        self.global_id_manager.print_summary()
        print("\n Complete!")


def main():
    parser = argparse.ArgumentParser(description="HAVEN Sequential ReID")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multicam.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = BACKEND_DIR.parent / config_path
    
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)
    
    runner = SequentialReIDRunner(str(config_path))
    
    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n Stopped by user")


if __name__ == "__main__":
    main()

