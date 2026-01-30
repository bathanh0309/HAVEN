# 🎥 Multi-Camera ReID Configuration Guide

## 📁 File Structure

```
HAVEN/
├── multi.bat                           # ← Script chạy chính (double-click để chạy!)
└── backend/
    ├── multi_camera_config.yaml        # ← File cấu hình tổng hợp
    ├── run_multi_camera.py             # Script Python chính
    └── data/
        └── multi-camera/
            ├── 1.mp4                   # Camera 1 video
            ├── 2.mp4                   # Camera 2 video
            └── 3.mp4                   # Camera 3 video
```

---

## 🚀 Quick Start

### Cách 1: Double-click (Đơn giản nhất)
```
1. Mở Windows Explorer
2. Tìm file: D:\HAVEN\multi.bat
3. Double-click để chạy!
```

### Cách 2: Command Line
```bash
cd D:\HAVEN
multi.bat
```

### Cách 3: Python Trực Tiếp
```bash
cd D:\HAVEN
.venv\Scripts\activate
python backend\run_multi_camera.py
```

---

## ⚙️ Configuration File: `backend/multi_camera_config.yaml`

### 1️⃣ Camera Sources

Mỗi camera có thể dùng **video file** hoặc **RTSP stream**.

#### Dùng Video File (Default):
```yaml
cameras:
  - id: "cam1"
    name: "Camera 1"
    enabled: true
    source_type: "video"                           # ← Chọn "video"
    video_path: "D:/HAVEN/backend/data/multi-camera/1.mp4"  # ← Đường dẫn video
```

#### Dùng RTSP Camera:
```yaml
cameras:
  - id: "cam1"
    name: "Living Room"
    enabled: true
    source_type: "rtsp"                            # ← Đổi thành "rtsp"
    rtsp_url: "rtsp://192.168.1.101:554/stream1"   # ← RTSP URL
    rtsp_username: "admin"                         # ← Username
    rtsp_password: "password123"                   # ← Password
```

#### Tắt Camera:
```yaml
cameras:
  - id: "cam3"
    enabled: false    # ← Đặt false để skip camera này
```

---

### 2️⃣ Spatiotemporal Constraints (Camera Graph)

Định nghĩa thời gian di chuyển giữa các cameras:

```yaml
camera_graph:
  cam1:
    cam2:
      min_time: 5    # Tối thiểu 5 giây từ cam1 → cam2
      max_time: 30   # Tối đa 30 giây
    cam3:
      min_time: 10   # Tối thiểu 10 giây từ cam1 → cam3
      max_time: 60
```

**Ý nghĩa:**
- Nếu người xuất hiện ở cam2 chỉ **2 giây** sau khi rời cam1 → **Không thể** là cùng người (vận tốc vô lý)
- Hệ thống sẽ tự động **reject** matches violate physics

**Cách đo:**
1. Đi bộ từ vị trí cam1 đến cam2 với tốc độ bình thường
2. Đo thời gian = T giây
3. Đặt `min_time = T - 5`, `max_time = T + 15`

---

### 3️⃣ Detection & Tracking Thresholds

#### YOLO Detection:
```yaml
inference:
  yolo:
    model: "yolov11n.pt"           # Model trong backend/models/
    conf_threshold: 0.5            # ↑ Tăng = ít false positives, ↓ Giảm = nhiều detections
    iou_threshold: 0.45            # NMS threshold
    device: "cuda"                 # "cuda" hoặc "cpu"
```

#### Tracking:
```yaml
tracking:
  tracker_type: "bytetrack"        # "bytetrack" hoặc "botsort"
  min_tracklet_frames: 5           # Tối thiểu 5 frames mới tạo tracklet
```

---

### 4️⃣ ReID Thresholds (QUAN TRỌNG!)

#### Open-Set Two-Threshold:
```yaml
reid:
  thresholds:
    accept: 0.75                   # T_high: > 0.75 = Confident match
    reject: 0.50                   # T_low: < 0.50 = Definitely new person
    margin: 0.15                   # Margin to 2nd-best candidate
```

**Giải thích:**
- **Similarity > 0.75** → Match với ID cũ ✅
- **Similarity < 0.50** → Tạo ID mới ✅
- **0.50 - 0.75** → Vùng **uncertain**, dùng thêm evidence (quality, face, gait)

#### Signal-Specific Thresholds:
```yaml
reid:
  thresholds:
    face_similarity: 0.6           # Face matching (chưa dùng, TODO)
    gait_similarity: 0.7           # Gait matching (chưa dùng, TODO)
    appearance_similarity: 0.5     # Appearance matching
```

#### Quality Gating:
```yaml
reid:
  quality:
    min_tracklet_frames: 5         # Tối thiểu 5 frames
    min_quality_score: 0.5         # Overall quality > 0.5
    min_bbox_size: 80              # Bbox tối thiểu 80x80 pixels
```

**Tăng thresholds nếu:**
- Quá nhiều false matches (khác người mà bị gộp chung ID)
  → Tăng `accept` từ 0.75 lên 0.80

**Giảm thresholds nếu:**
- Quá nhiều ID switches (cùng người nhưng bị tách ID)
  → Giảm `accept` từ 0.75 xuống 0.70

---

### 5️⃣ Visualization

```yaml
visualization:
  enabled: true                    # Hiển thị video
  show_bbox: true                  # Hiện bounding box
  show_track_id: true              # Hiện local track ID
  show_global_id: true             # Hiện global ID (ReID)
```

---

### 6️⃣ Output Settings

#### Database:
```yaml
output:
  database:
    enabled: true
    path: "D:/HAVEN/backend/haven_reid.db"
```

#### Video Recording:
```yaml
output:
  video:
    enabled: false                 # Đặt true để lưu video output
    output_dir: "D:/HAVEN/backend/outputs/multi-camera"
```

#### Logging:
```yaml
output:
  logging:
    enabled: true
    level: "INFO"                  # DEBUG, INFO, WARNING, ERROR
    log_file: "D:/HAVEN/backend/logs/multi_camera.log"
```

---

## 🎛️ Common Tuning Scenarios

### Scenario 1: Quá Nhiều False Matches
**Triệu chứng:** Khác người nhưng bị gán cùng global ID

**Giải pháp:**
```yaml
reid:
  thresholds:
    accept: 0.80        # Tăng từ 0.75 → 0.80 (stricter)
    margin: 0.20        # Tăng từ 0.15 → 0.20
  quality:
    min_quality_score: 0.6  # Tăng từ 0.5 → 0.6
```

### Scenario 2: Quá Nhiều ID Switches
**Triệu chứng:** Cùng người nhưng bị tách ra nhiều global ID

**Giải pháp:**
```yaml
reid:
  thresholds:
    accept: 0.70        # Giảm từ 0.75 → 0.70 (looser)
    reject: 0.45        # Giảm từ 0.50 → 0.45
  reuse_window: 10800   # Tăng từ 7200 → 10800 (3 hours)
```

### Scenario 3: Chậm, Cần Tăng FPS
**Giải pháp:**
```yaml
cameras:
  - resize_width: 480        # Giảm từ 640 → 480
    skip_frames: 1           # Process every 2nd frame

inference:
  yolo:
    device: "cuda"           # Dùng GPU (nếu có)
    img_size: 480            # Giảm từ 640 → 480

tracking:
  min_tracklet_frames: 3     # Giảm từ 5 → 3
```

### Scenario 4: Videos Khác Timing
**Triệu chứng:** 3 videos không đồng bộ thời gian

**Lưu ý:** Hệ thống dùng **relative time** từ khi start script, không phải timestamp trong video.

**Nếu muốn đồng bộ:**
1. Cắt video để cùng start time
2. Hoặc dùng RTSP (real-time streams)

---

## 📊 Output & Metrics

### Console Output:
```
============================================================
HAVEN Multi-Camera ReID System
============================================================
✅ Database: D:/HAVEN/backend/haven_reid.db
✅ Camera graph: 3 cameras
✅ Global ID Manager initialized
✅ cam1 (Camera 1): D:/HAVEN/backend/data/multi-camera/1.mp4
✅ cam2 (Camera 2): D:/HAVEN/backend/data/multi-camera/2.mp4
✅ cam3 (Camera 3): D:/HAVEN/backend/data/multi-camera/3.mp4
✅ YOLO model: yolov11n.pt
============================================================

🚀 Starting multi-camera processing...
Press 'Q' to quit

  [cam1] Track 1 → Global ID 1 (high: new_identity)
  [cam1] Track 2 → Global ID 2 (high: new_identity)
  [cam2] Track 1 → Global ID 1 (medium: gait_0.72)    ← Matched!
  [cam3] Track 3 → Global ID 3 (high: new_identity)
```

### Database:
- Tables: `identities`, `observations`, `embeddings`
- Query example:
  ```sql
  SELECT global_id, COUNT(*) as observations
  FROM observations
  GROUP BY global_id
  ORDER BY observations DESC;
  ```

---

## 🐛 Troubleshooting

### Error: "Failed to open source"
```
✅ cam1: D:/HAVEN/backend/data/multi-camera/1.mp4
❌ cam2: Failed to open source
```

**Giải pháp:**
1. Kiểm tra đường dẫn video có đúng không
2. Kiểm tra file video có tồn tại không
3. Nếu dùng RTSP, kiểm tra network connection

### Error: "Config file not found"
```
ERROR: Config file not found!
Expected: backend\multi_camera_config.yaml
```

**Giải pháp:**
- Chắc chắn file `multi_camera_config.yaml` nằm trong `backend/`

### Video Chạy Quá Nhanh
**Giải pháp:**
```yaml
performance:
  max_fps: 10    # Giảm từ 30 → 10
```

### Không Thấy Window Hiển Thị
**Giải pháp:**
```yaml
visualization:
  enabled: true    # Đảm bảo = true
```

---

## 📝 Next Steps

1. **✅ Chạy với videos hiện tại** để test
2. **📊 Đánh giá metrics**: ID switches, false matches
3. **🎛️ Tune thresholds** theo kết quả
4. **🎥 Thêm RTSP cameras** khi ready
5. **🧠 Integrate feature extractors** (Face, Gait, Appearance)

---

## 📞 Quick Reference

| Setting | File | Line |
|---------|------|------|
| Video paths | `multi_camera_config.yaml` | 13-40 |
| RTSP URLs | `multi_camera_config.yaml` | 20-22 |
| Camera graph | `multi_camera_config.yaml` | 46-73 |
| ReID thresholds | `multi_camera_config.yaml` | 115-130 |
| Quality gates | `multi_camera_config.yaml` | 132-138 |

---

**Happy Tracking! 🎯**
