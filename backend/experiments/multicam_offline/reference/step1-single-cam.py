"""
BƯỚC 1: Single Camera Baseline
Test YOLO detection trên 1 video file trước khi làm multi-camera
"""

import cv2
from ultralytics import YOLO
import time

# CONFIG
VIDEO_PATH = r"D:\HAVEN\backend\data\video\phute.mp4"
MODEL_PATH = r"D:\HAVEN\backend\models\yolo11n-pose.pt"  # Hoặc yolov11n-pose.pt
DISPLAY_WIDTH = 1280

def test_single_camera():
    """Test YOLO trên 1 camera đơn giản"""
    
    print(" Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    
    print(f" Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(" Không mở được video!")
        return
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f" Video: {width}x{height} @ {fps:.1f}fps")
    
    frame_count = 0
    start_time = time.time()
    
    print("\n▶  Đang chạy... (Nhấn 'q' để thoát)\n")
    
    while True:
        ret, frame = cap.read()
        
        # Loop video
        if not ret:
            print(" Video kết thúc, loop lại...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_count += 1
        
        # Resize để hiển thị
        scale = DISPLAY_WIDTH / width
        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, int(height * scale)))
        
        # YOLO inference
        results = model(display_frame, verbose=False)
        
        # Vẽ kết quả
        annotated = results[0].plot()
        
        # Đếm số người
        num_people = len(results[0].boxes)
        
        # Hiển thị FPS
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed
            print(f"Frame {frame_count}: {num_people} người | FPS: {current_fps:.1f}")
        
        # Vẽ info lên frame
        cv2.putText(annotated, f"Nguoi: {num_people}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, f"Frame: {frame_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Hiển thị
        cv2.imshow('HAVEN - Single Camera Test', annotated)
        
        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed
    
    print(f"\n Kết thúc!")
    print(f" Tổng frames: {frame_count}")
    print(f"  Thời gian: {elapsed:.1f}s")
    print(f" FPS trung bình: {avg_fps:.1f}")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("="*60)
    print("HAVEN - BƯỚC 1: Single Camera Baseline")
    print("="*60)
    
    try:
        test_single_camera()
    except Exception as e:
        print(f"\n Lỗi: {e}")
        import traceback
        traceback.print_exc()
