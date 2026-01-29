"""
HAVEN Step 6: Global Tracking Visualization với Skeleton
=========================================================
Hiển thị tuần tự: Cam1 → Cam2 → Cam3
Vẽ skeleton từ keypoints đã lưu ở Step 5
"""

import cv2
import pickle
import numpy as np

DATA_FILE = "global_tracks.pkl"
VIDEOS = [
    ("cam1", r"D:\HAVEN\backend\data\multi-camera\1.mp4"),
    ("cam2", r"D:\HAVEN\backend\data\multi-camera\2.mp4"),
    ("cam3", r"D:\HAVEN\backend\data\multi-camera\3.mp4"),
]

# Colors for each Global ID
COLORS = {
    1: (0, 255, 0),      # Green
    2: (255, 0, 0),      # Blue  
    3: (0, 165, 255),    # Orange
    4: (255, 0, 255),    # Magenta
    5: (255, 255, 0),    # Cyan
}

# Default skeleton if not in data
DEFAULT_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

DISPLAY_W = 960
DISPLAY_H = 720
MIN_KPT_CONF = 0.4


def get_color(gid):
    return COLORS.get(gid, (100, 100, 100))


def draw_skeleton(frame, keypoints, color, skeleton):
    """Vẽ skeleton từ keypoints"""
    if keypoints is None:
        return
    
    kpts = np.array(keypoints)  # (17, 3)
    
    # Draw keypoints
    for i, (x, y, conf) in enumerate(kpts):
        if conf > MIN_KPT_CONF:
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)
            cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 255), 1)
    
    # Draw skeleton lines
    for start_idx, end_idx in skeleton:
        if start_idx < len(kpts) and end_idx < len(kpts):
            x1, y1, c1 = kpts[start_idx]
            x2, y2, c2 = kpts[end_idx]
            if c1 > MIN_KPT_CONF and c2 > MIN_KPT_CONF:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)


def load_data():
    try:
        with open(DATA_FILE, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Could not find {DATA_FILE}. Please run Step 5 first.")
        return None


def play_camera(cam_id, video_path, cam_data, person_cameras, cam_index, total_cams, skeleton):
    """Play one camera video với Global ID và skeleton"""
    
    print(f"\n▶️  Playing {cam_id.upper()}...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"   ❌ Cannot open {video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    scale_x = DISPLAY_W / orig_w
    scale_y = DISPLAY_H / orig_h
    
    frame_idx = 0
    paused = False
    seen_gids = set()
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
            ret, frame = cap.read()
            if not ret:
                break
        
        display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        
        # Draw detections for this frame
        frame_detections = cam_data.get(frame_idx, [])
        
        for detection in frame_detections:
            if len(detection) == 6:
                x, y, w, h, gid, kpts = detection
            else:
                x, y, w, h, gid = detection
                kpts = None
            
            seen_gids.add(gid)
            
            # Scale coordinates
            x_scaled = x * scale_x
            y_scaled = y * scale_y
            w_scaled = w * scale_x
            h_scaled = h * scale_y
            
            x1 = int(x_scaled - w_scaled/2)
            y1 = int(y_scaled - h_scaled/2)
            x2 = int(x_scaled + w_scaled/2)
            y2 = int(y_scaled + h_scaled/2)
            
            color = get_color(gid)
            
            # Draw bounding box
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            # Draw skeleton if keypoints available
            if kpts is not None:
                # Scale keypoints
                scaled_kpts = []
                for kx, ky, kc in kpts:
                    scaled_kpts.append([kx * scale_x, ky * scale_y, kc])
                draw_skeleton(display, scaled_kpts, color, skeleton)
            
            # Draw Global ID label
            label = f"G-ID: {gid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(display, (x1, y1-th-15), (x1+tw+10, y1), color, -1)
            cv2.putText(display, label, (x1+5, y1-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # === INFO PANEL ===
        cv2.rectangle(display, (0, 0), (DISPLAY_W, 90), (30, 30, 30), -1)
        
        mode = "MASTER - Assigning New IDs" if cam_id == "cam1" else "SLAVE - Matching IDs from Cam1"
        mode_color = (0, 255, 255) if cam_id == "cam1" else (255, 150, 0)
        
        cv2.putText(display, f"{cam_id.upper()}", (15, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, mode_color, 3)
        cv2.putText(display, mode, (130, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.putText(display, f"Frame: {frame_idx}/{total_frames}", (15, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(display, f"Camera {cam_index + 1}/{total_cams}", (DISPLAY_W - 170, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        
        # Pose indicator
        cv2.putText(display, "POSE + SKELETON", (DISPLAY_W - 200, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Legend
        legend_x = DISPLAY_W - 200
        legend_y = 110
        cv2.rectangle(display, (legend_x - 10, legend_y - 20), 
                     (DISPLAY_W - 10, legend_y + 80), (30, 30, 30), -1)
        cv2.putText(display, "Global IDs:", (legend_x, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        for gid, cams in person_cameras.items():
            color = get_color(gid)
            y_pos = legend_y + 25 + (gid - 1) * 25
            cv2.circle(display, (legend_x + 10, y_pos - 5), 8, color, -1)
            status = "✓" if gid in seen_gids else ""
            cv2.putText(display, f"P{gid}: {' -> '.join(cams)} {status}", 
                       (legend_x + 25, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # Progress bar
        bar_y = DISPLAY_H - 25
        progress = frame_idx / total_frames
        cv2.rectangle(display, (10, bar_y), (DISPLAY_W - 10, bar_y + 15), (50, 50, 50), -1)
        cv2.rectangle(display, (10, bar_y), (10 + int((DISPLAY_W - 20) * progress), bar_y + 15), 
                     mode_color, -1)
        
        status_text = "PAUSED" if paused else "Playing..."
        cv2.putText(display, status_text, (15, DISPLAY_H - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        cv2.putText(display, "Q=Quit | SPACE=Pause | N=Next | A/D=Seek", 
                   (DISPLAY_W - 380, DISPLAY_H - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        
        cv2.imshow("HAVEN - Sequential Global Tracking with Pose", display)
        
        wait_time = int(1000 / fps) if not paused else 50
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord('q'):
            cap.release()
            return False
        elif key == ord(' '):
            paused = not paused
        elif key == ord('n'):
            break
        elif key == ord('a'):
            frame_idx = max(1, frame_idx - 30)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        elif key == ord('d'):
            frame_idx = min(total_frames, frame_idx + 30)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
    
    cap.release()
    print(f"   ✅ {cam_id.upper()} finished. Global IDs seen: {sorted(seen_gids)}")
    return True


def main():
    print("="*60)
    print("HAVEN Step 6 | Sequential Global Tracking with Pose")
    print("="*60)
    print("\n🎯 Features:")
    print("   - Sequential playback: Cam1 → Cam2 → Cam3")
    print("   - Global IDs preserved across cameras")
    print("   - Skeleton visualization from pose detection")
    
    data = load_data()
    if not data:
        return
    
    all_data = data['all_data']
    num_persons = data['num_persons']
    person_cameras = data['person_cameras']
    skeleton = data.get('skeleton', DEFAULT_SKELETON)
    
    print(f"\n📊 Loaded {num_persons} persons:")
    for gid, cams in person_cameras.items():
        color_name = {1: "Green", 2: "Blue", 3: "Orange", 4: "Magenta", 5: "Cyan"}.get(gid, "Gray")
        print(f"   G-ID {gid} ({color_name}): {' → '.join(cams)}")
    
    print("\n⌨️  Controls:")
    print("   SPACE = Pause/Resume")
    print("   N     = Skip to next camera")
    print("   A/D   = Rewind/Forward 30 frames")
    print("   Q     = Quit")
    
    cv2.namedWindow("HAVEN - Sequential Global Tracking with Pose", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HAVEN - Sequential Global Tracking with Pose", DISPLAY_W, DISPLAY_H)
    
    for i, (cam_id, video_path) in enumerate(VIDEOS):
        cam_data = all_data.get(cam_id, {})
        
        should_continue = play_camera(
            cam_id, video_path, cam_data, 
            person_cameras, i, len(VIDEOS), skeleton
        )
        
        if not should_continue:
            break
        
        if i < len(VIDEOS) - 1:
            print(f"\n⏳ Next: {VIDEOS[i+1][0].upper()}...")
            
            transition = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
            cv2.putText(transition, f"Next: {VIDEOS[i+1][0].upper()}", 
                       (DISPLAY_W//2 - 120, DISPLAY_H//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            cv2.putText(transition, "Global IDs & Poses will be matched...", 
                       (DISPLAY_W//2 - 200, DISPLAY_H//2 + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            cv2.imshow("HAVEN - Sequential Global Tracking with Pose", transition)
            cv2.waitKey(1500)
    
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("📋 FINAL SUMMARY")
    print("="*60)
    print(f"Total Unique Persons: {num_persons}")
    for gid, cams in person_cameras.items():
        print(f"   Person {gid}: {' → '.join(cams)}")
    print("\n✅ Playback complete!")


if __name__ == "__main__":
    main()
