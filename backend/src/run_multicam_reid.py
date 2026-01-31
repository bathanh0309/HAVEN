"""
HAVEN Multi-Camera ReID Runner

Entrypoint for running multi-camera person re-identification
with dual-master logic, ADL detection, and pose estimation.

Features:
- Dual-master camera logic
- Fixed 2x2 Mosaic View (Cam1 TL, Cam2 TR, Cam3 BL, Cam4 BR)
- YOLO pose detection
- Real-time visualization
- Video output

Usage:
    python run_multicam_reid.py --config configs/multicam.yaml
    
Author: HAVEN Team
Version: 2.1 (Fixed Mosaic)
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


class MultiCameraReIDRunner:
    """
    Multi-camera ReID system runner.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize runner.
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print("=" * 60)
        print("HAVEN Multi-Camera ReID System")
        print("=" * 60)
        
        # Display settings (must be set before _init_output)
        self.display_w = self.config['display']['width']
        self.display_h = self.config['display']['height']
        self.show_mosaic = self.config['display'].get('mosaic', True)
        
        # Initialize components
        self._init_cameras()
        self._init_global_id_manager()
        self._init_models()
        self._init_output()
        
        print("=" * 60)
    
    def _init_cameras(self):
        """Initialize camera streams."""
        data_root = Path(self.config['data']['data_root'])
        cameras = self.config['data']['cameras']
        
        self.streams: Dict[int, VideoStream] = {}
        
        for cam_cfg in cameras:
            if not cam_cfg.get('enabled', True):
                continue
                
            cam_id = cam_cfg['id']
            cam_name = cam_cfg['name']
            source_type = cam_cfg['source_type']
            source_path = cam_cfg['path']
            pattern = cam_cfg.get('pattern', '*.mp4')
            fps_override = cam_cfg.get('fps_override')
            resize_width = cam_cfg.get('resize_width')
            skip_frames = cam_cfg.get('skip_frames', 0)
            
            try:
                stream = VideoStream(
                    camera_id=cam_id,
                    camera_name=cam_name,
                    source_type=source_type,
                    source_path=source_path,
                    data_root=str(data_root),
                    pattern=pattern,
                    fps_override=fps_override,
                    resize_width=resize_width,
                    skip_frames=skip_frames
                )
                self.streams[cam_id] = stream
                print(f"  Camera {cam_id} ({cam_name}): {source_type} - {source_path}")
            except Exception as e:
                print(f"  Camera {cam_id} ({cam_name}): FAILED - {e}")
        
        print(f"Total cameras: {len(self.streams)}")
    
    def _init_global_id_manager(self):
        """Initialize GlobalIDManager with dual-master logic."""
        master_ids = self.config['master_cameras']['ids']
        reid_cfg = self.config['reid']
        
        camera_graph = self.config.get('camera_graph', {})
        # Convert string keys to int if present, but config usually has int keys
        if isinstance(camera_graph, dict) and 'transitions' in camera_graph:
             # Handle complex graph format from new config
             # Simplified: just use adjacency if needed, or ignore for now as map is complex
             # For now, pass empty or basic graph
             camera_graph_simple = {} 
        else:
             camera_graph_simple = {int(k): v for k, v in camera_graph.items()}

        self.global_id_manager = GlobalIDManager(
            master_camera_ids=master_ids,
            accept_threshold=reid_cfg['thresholds']['accept'],
            reject_threshold=reid_cfg['thresholds']['reject'],
            camera_graph=camera_graph_simple,
            max_prototypes=reid_cfg.get('memory', {}).get('max_prototypes', 5), # Handle config variations
            ema_alpha=reid_cfg.get('memory', {}).get('ema_alpha', 0.3),
            min_bbox_size=reid_cfg['quality']['min_bbox_size'],
            # Handle config structure difference (some have min_track_frames in quality, some top level)
            min_track_frames=reid_cfg['quality'].get('min_tracklet_frames', 5) 
        )
        
        print(f"Master cameras: {master_ids}")
    
    def _init_models(self):
        """Initialize YOLO models."""
        # Handle different config structures for YOLO
        if 'yolo' in self.config:
            yolo_cfg = self.config['yolo']
            model_name = yolo_cfg.get('model', 'yolov11n-pose.pt')
            conf = yolo_cfg.get('conf_threshold', 0.5)
        elif 'detection' in self.config and 'yolo' in self.config['detection']:
             yolo_cfg = self.config['detection']['yolo']
             model_name = Path(yolo_cfg['model_path']).name
             conf = yolo_cfg['conf_threshold']
        else:
             model_name = 'yolov11n-pose.pt'
             conf = 0.5
             
        models_dir = BACKEND_DIR / "models"
        pose_path = models_dir / model_name
        
        if pose_path.exists():
            self.pose_model = YOLO(str(pose_path))
        else:
            # Fallback
            self.pose_model = YOLO(model_name)
            
        print(f"Pose model: {model_name}")
        self.yolo_conf = conf
    
    def _init_output(self):
        """Initialize output directory and writers."""
        out_cfg = self.config['output']
        
        # Handle config variations
        if 'dir' in out_cfg:
             out_dir = out_cfg['dir']
             enabled = out_cfg.get('enabled', True)
        else:
             out_dir = out_cfg.get('out_dir', 'outputs')
             enabled = True # Default true
            
        if enabled:
            self.output_dir = Path(out_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Video writers
            self.video_writers: Dict[int, cv2.VideoWriter] = {}
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30 # Default
            
            for cam_id in self.streams.keys():
                out_path = self.output_dir / f"cam{cam_id}_{timestamp}.mp4"
                writer = cv2.VideoWriter(
                    str(out_path), fourcc, fps,
                    (self.display_w, self.display_h)
                )
                self.video_writers[cam_id] = writer
            
            # Mosaic writer
            if out_cfg.get('save_mosaic', True) or out_cfg.get('mosaic_output', True):
                mosaic_path = self.output_dir / f"mosaic_{timestamp}.mp4"
                # Fixed 2x2 size
                mosaic_size = (self.display_w * 2, self.display_h * 2)
                self.mosaic_writer = cv2.VideoWriter(
                    str(mosaic_path), fourcc, fps, mosaic_size
                )
            else:
                self.mosaic_writer = None
            
            print(f"Output: {self.output_dir}")
        else:
            self.output_dir = None
            self.video_writers = {}
            self.mosaic_writer = None
    
    def _extract_embedding(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract embedding (Color Histogram)."""
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(768)
        
        # Split into 3 parts (top, middle, bottom)
        h = crop.shape[0]
        parts = [
            crop[:h//3],
            crop[h//3:2*h//3],
            crop[2*h//3:]
        ]
        
        # Compute color histogram for each part
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
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _draw_detection(self, frame, bbox, global_id, local_track_id, reason, keypoints=None):
        """Draw detection overlay."""
        x1, y1, x2, y2 = map(int, bbox)
        
        if reason == MatchReason.MASTER_NEW:
            color = (0, 255, 0)
        elif reason == MatchReason.MATCHED:
            color = (255, 255, 0)
        else:
            color = (0, 165, 255)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"G{global_id}" if global_id > 0 else f"T{local_track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        if keypoints is not None and self.config['display'].get('show_skeleton', True):
            self._draw_skeleton(frame, keypoints)
    
    def _draw_skeleton(self, frame, keypoints):
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
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
    
    def _create_mosaic(self, frames: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Create fixed 2x2 mosaic:
        TL: Cam1  | TR: Cam2
        BL: Cam3  | BR: Cam4
        """
        w, h = self.display_w, self.display_h
        black_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Resize inputs or use black if missing
        f1 = cv2.resize(frames.get(1, black_frame), (w, h)) if 1 in frames else black_frame
        f2 = cv2.resize(frames.get(2, black_frame), (w, h)) if 2 in frames else black_frame
        f3 = cv2.resize(frames.get(3, black_frame), (w, h)) if 3 in frames else black_frame
        f4 = cv2.resize(frames.get(4, black_frame), (w, h)) if 4 in frames else black_frame
        
        # Add labels if black frame (camera missing/ended)
        if 1 not in frames: cv2.putText(f1, "CAM1: NO SIGNAL", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        if 2 not in frames: cv2.putText(f2, "CAM2: NO SIGNAL", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        if 3 not in frames: cv2.putText(f3, "CAM3: NO SIGNAL", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        if 4 not in frames: cv2.putText(f4, "CAM4: NO SIGNAL", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        
        # Concatenate
        top = np.hstack([f1, f2])
        bottom = np.hstack([f3, f4])
        mosaic = np.vstack([top, bottom])
        
        return mosaic
    
    def run(self):
        """Run multi-camera processing."""
        print("\n🚀 Starting multi-camera processing...")
        print("Controls: Q=Quit, SPACE=Pause, G=Record MP4\n")
        
        cv2.namedWindow("HAVEN Multi-Camera ReID", cv2.WINDOW_NORMAL)
        
        paused = False
        frame_count = 0
        
        while True:
            if not paused:
                frames: Dict[int, np.ndarray] = {}
                any_alive = False
                
                # Read from all cameras
                for cam_id, stream in self.streams.items():
                    ret, frame = stream.read()
                    if ret and frame is not None:
                        any_alive = True
                        frames[cam_id] = frame.copy()
                
                if not any_alive and len(self.streams) > 0:
                    print("All streams finished.")
                    break
                
                frame_count += 1
                
                # Process each camera
                for cam_id, frame in frames.items():
                    is_master = self.global_id_manager.is_master_camera(cam_id)
                    
                    # YOLO Pose detection with tracking
                    results = self.pose_model.track(frame, persist=True, verbose=False, conf=self.yolo_conf)
                    
                    if results[0].boxes is not None and results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                        keypoints_data = results[0].keypoints
                        
                        for i, track_id in enumerate(track_ids):
                            bbox = tuple(boxes[i])
                            kpts = keypoints_data.data[i].cpu().numpy() if keypoints_data is not None else None
                            
                            # Extract embedding
                            embedding = self._extract_embedding(frame, bbox)
                            bbox_size = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
                            
                            # Assign global ID
                            global_id, reason, score = self.global_id_manager.assign_global_id(
                                camera_id=cam_id,
                                local_track_id=track_id,
                                embedding=embedding,
                                quality=1.0,
                                timestamp=0,
                                frame_idx=frame_count,
                                num_frames=10, 
                                bbox_size=int(bbox_size)
                            )
                            
                            # Log
                            if reason == MatchReason.MASTER_NEW:
                                print(f"🆕 [cam{cam_id}] Track {track_id} → Global ID {global_id} (NEW)")
                            elif reason == MatchReason.MATCHED:
                                print(f"✅ [cam{cam_id}] Track {track_id} → Global ID {global_id} (MATCH: {score:.2f})")
                            
                            # Draw
                            self._draw_detection(frame, bbox, global_id, track_id, reason, kpts)
                    
                    # Camera label
                    master_text = "MASTER" if is_master else "SLAVE"
                    cv2.putText(frame, f"CAM{cam_id} ({master_text})", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    # Write individual video
                    if cam_id in self.video_writers:
                        out_frame = cv2.resize(frame, (self.display_w, self.display_h))
                        self.video_writers[cam_id].write(out_frame)
                    
                    frames[cam_id] = frame
                
                # Create and show mosaic (using fixed 2x2 logic)
                if self.show_mosaic:
                    mosaic = self._create_mosaic(frames)
                    cv2.imshow("HAVEN Multi-Camera ReID", mosaic)
                    
                    if self.mosaic_writer:
                        self.mosaic_writer.write(mosaic)
                elif frames:
                    first_cam = sorted(frames.keys())[0]
                    cv2.imshow("HAVEN Multi-Camera ReID", frames[first_cam])
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print("PAUSED" if paused else "RESUMED")
        
        self._cleanup()
    
    def _cleanup(self):
        """Cleanup resources."""
        for writer in self.video_writers.values():
            writer.release()
        if self.mosaic_writer:
            self.mosaic_writer.release()
        for stream in self.streams.values():
            stream.release()
        cv2.destroyAllWindows()
        self.global_id_manager.print_summary()
        print("\n✅ Complete!")


def main():
    parser = argparse.ArgumentParser(description="HAVEN Multi-Camera ReID")
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
    
    runner = MultiCameraReIDRunner(str(config_path))
    
    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n⚠️ Stopped by user")
        runner._cleanup()


if __name__ == "__main__":
    main()
