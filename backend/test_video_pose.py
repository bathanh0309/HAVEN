"""
HAVEN - Video Pose Detection Test
==================================
Script test pose detection trên video sử dụng YOLO11-pose.
Hiển thị kết quả trực tiếp trên cửa sổ OpenCV (không cần frontend).

Phím tắt:
- Q: Thoát
- Space: Tạm dừng/Tiếp tục
- R: Restart video từ đầu
- L: Bật/Tắt chế độ Loop

Cách chạy: python backend/test_video_pose.py
Hoặc: double-click video.bat
"""

import cv2
import time
import os
from pathlib import Path
from dotenv import load_dotenv

# Tải cấu hình từ .env
load_dotenv()

# Lấy đường dẫn model từ .env hoặc dùng mặc định
MODEL_PATH = os.getenv("AI_MODEL_PATH", "models/yolo11n-pose.pt")
CONF_THRES = float(os.getenv("AI_CONF_THRES", "0.25"))
VIDEO_PATH = os.getenv("TEST_VIDEO_PATH", "data/video/walking.mp4")

def main():
    """
    Hàm chính: Chạy pose detection trên video và hiển thị kết quả.
    Hỗ trợ loop, restart, pause.
    """
    print("=" * 60)
    print("HAVEN - Video Pose Detection Test")
    print("=" * 60)
    
    # Kiểm tra file video
    if not Path(VIDEO_PATH).exists():
        print(f"[ERROR] Không tìm thấy video: {VIDEO_PATH}")
        print("Vui lòng đặt video vào thư mục data/video/")
        input("Nhấn Enter để thoát...")
        return
    
    # Kiểm tra file model
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] Không tìm thấy model: {MODEL_PATH}")
        print("Vui lòng chạy webAI.bat trước để tải model.")
        input("Nhấn Enter để thoát...")
        return
    
    print(f"[INFO] Model: {MODEL_PATH}")
    print(f"[INFO] Video: {VIDEO_PATH}")
    print(f"[INFO] Confidence: {CONF_THRES}")
    print("=" * 60)
    
    # Import YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Chưa cài ultralytics!")
        print("Chạy: pip install ultralytics")
        input("Nhấn Enter để thoát...")
        return
    
    # Tải model
    print("[INFO] Đang tải model YOLO11-pose...")
    model = YOLO(MODEL_PATH)
    print("[INFO] Model đã sẵn sàng!")
    
    # Biến điều khiển
    paused = False
    loop_mode = True  # Mặc định bật loop
    running = True
    
    # Tạo cửa sổ
    window_name = "HAVEN Pose Test | Q=Quit, Space=Pause, R=Restart, L=Loop"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    while running:
        # Mở video
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"[ERROR] Không thể mở video: {VIDEO_PATH}")
            break
        
        # Lấy thông tin video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[INFO] Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        print(f"[INFO] Loop mode: {'ON' if loop_mode else 'OFF'}")
        print("=" * 60)
        print("[INFO] Phím tắt: Q=Thoát, Space=Pause, R=Restart, L=Toggle Loop")
        print("=" * 60)
        
        frame_count = 0
        start_time = time.time()
        restart_requested = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    # Video kết thúc
                    if loop_mode:
                        print("\n[INFO] Video kết thúc - Đang restart (Loop mode ON)...")
                        break  # Thoát inner loop để restart
                    else:
                        print("\n[INFO] Video đã kết thúc!")
                        running = False
                        break
                
                frame_count += 1
                
                # Chạy pose detection
                results = model(frame, conf=CONF_THRES, verbose=False)
                
                # Vẽ kết quả
                annotated_frame = results[0].plot()
                
                # Tính FPS
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Hiển thị thông tin
                info_text = f"Frame: {frame_count}/{total_frames} | FPS: {current_fps:.1f} | Detections: {len(results[0].boxes)}"
                cv2.putText(annotated_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Hiển thị trạng thái loop
                loop_text = f"Loop: {'ON' if loop_mode else 'OFF'}"
                cv2.putText(annotated_frame, loop_text, (width - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Progress bar
                progress = frame_count / total_frames
                bar_width = 400
                bar_height = 10
                bar_x = 10
                bar_y = height - 30
                cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
                cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
                
                # Hiển thị frame
                cv2.imshow(window_name, annotated_frame)
            
            # Xử lý phím (waitKey phải có để cửa sổ hoạt động)
            key = cv2.waitKey(1 if not paused else 100) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\n[INFO] Đã thoát bởi người dùng.")
                running = False
                break
            elif key == ord(' '):
                paused = not paused
                status = "TẠM DỪNG" if paused else "TIẾP TỤC"
                print(f"[INFO] {status}")
            elif key == ord('r') or key == ord('R'):
                print("[INFO] Restart video...")
                restart_requested = True
                break
            elif key == ord('l') or key == ord('L'):
                loop_mode = not loop_mode
                status = "ON" if loop_mode else "OFF"
                print(f"[INFO] Loop mode: {status}")
        
        # Dọn dẹp video
        cap.release()
        
        # Nếu không restart và không loop thì thoát
        if not restart_requested and not loop_mode:
            break
    
    # Dọn dẹp
    cv2.destroyAllWindows()
    
    # Thống kê cuối
    print("=" * 60)
    print("KẾT QUẢ TEST")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Video: {VIDEO_PATH}")
    print("=" * 60)
    
    input("Nhấn Enter để thoát...")

if __name__ == "__main__":
    main()
