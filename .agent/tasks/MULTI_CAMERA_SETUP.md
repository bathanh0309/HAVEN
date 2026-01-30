# ✅ Multi-Camera ReID System - Setup Complete!

## 📦 Đã Tạo

### 1. Configuration File
- **File**: `backend/multi_camera_config.yaml`
- **Mô tả**: File cấu hình tổng hợp, chứa:
  - ✅ 3 camera sources (video paths + RTSP URLs)
  - ✅ Camera graph (spatiotemporal constraints)
  - ✅ YOLO detection thresholds
  - ✅ ReID thresholds (accept/reject/quality)
  - ✅ Tracking settings
  - ✅ Visualization & output settings

### 2. Runner Script
- **File**: `backend/run_multi_camera.py`
- **Mô tả**: Python script chạy multi-camera ReID với:
  - ✅ Load 3 videos hoặc RTSP streams
  - ✅ ByteTrack tracking per camera
  - ✅ Tracklet aggregation
  - ✅ Global ID assignment
  - ✅ Spatiotemporal filtering
  - ✅ Real-time visualization (3 cameras side-by-side)

### 3. Batch Script
- **File**: `multi.bat` (thư mục gốc HAVEN/)
- **Mô tả**: Double-click để chạy!

### 4. Documentation
- **File**: `MULTI_CAMERA_GUIDE.md`
- **Mô tả**: Hướng dẫn chi tiết:
  - Quick start
  - Config tuning
  - Troubleshooting
  - Common scenarios

---

## 🎯 Video Sources Configured

Default setup sử dụng **video files** trong `backend/data/multi-camera/`:

| Camera | Source | Path |
|--------|--------|------|
| cam1 | Video | `D:/HAVEN/backend/data/multi-camera/1.mp4` |
| cam2 | Video | `D:/HAVEN/backend/data/multi-camera/2.mp4` |
| cam3 | Video | `D:/HAVEN/backend/data/multi-camera/3.mp4` |

**Có sẵn các videos phụ:**
- `phu1.mp4`, `phu2.mp4`, `phu3.mp4`, `phu4.mp4`

---

## 🔧 Camera Graph (Default)

```
cam1 ←→ cam2: 5-30 seconds
cam1 ←→ cam3: 10-60 seconds
cam2 ←→ cam3: 15-90 seconds
```

**Bạn cần điều chỉnh theo layout thực tế!**

---

## 🎛️ Key Thresholds (Quan Trọng!)

### Detection
```yaml
conf_threshold: 0.5      # YOLO detection confidence
```

### ReID Two-Threshold
```yaml
accept: 0.75             # T_high: Confident match
reject: 0.50             # T_low: Definitely new
```

### Quality Gate
```yaml
min_tracklet_frames: 5   # Minimum frames per tracklet
min_bbox_size: 80        # Minimum bbox size (pixels)
```

---

## 🚀 How to Run

### Method 1: Double-Click (Easiest)
```
1. Mở Windows Explorer
2. Navigate to: D:\HAVEN\
3. Double-click: multi.bat
```

### Method 2: Command Line
```bash
cd D:\HAVEN
multi.bat
```

### Method 3: Python Direct
```bash
cd D:\HAVEN
.venv\Scripts\activate
python backend\run_multi_camera.py
```

---

## 🎥 Expected Output

### Console:
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
  [cam2] Track 1 → Global ID 1 (medium: appearance_0.72)  ← Match!
```

### Window Display:
- 3 cameras hiển thị side-by-side
- Mỗi người có:
  - Green bbox (bounding box)
  - Yellow "T{id}" (local track ID)
  - Cyan "G{id}" (global ID - khi được assign)

---

## 📊 Performance Tracking

### Metrics (End of Session):
```
=== ReID Performance ===
Accuracy: 90.5%
ID Switch Rate: 3.2%
False Match Rate: 1.8%
New IDs Created: 5
IDs Reused: 12

Signal Usage:
  Face: 0              (TODO: Not implemented yet)
  Gait: 0              (TODO: Not implemented yet)
  Appearance: 12

Spatiotemporal Rejections: 8
```

### Database Stats:
```sql
-- Query database
sqlite3 backend/haven_reid.db

SELECT global_id, COUNT(*) as observations
FROM observations
GROUP BY global_id
ORDER BY observations DESC;
```

---

## 🔄 Switching to RTSP Cameras

When ready to use real cameras, edit `backend/multi_camera_config.yaml`:

```yaml
cameras:
  - id: "cam1"
    name: "Living Room"
    enabled: true
    source_type: "rtsp"                          # ← Change from "video"
    rtsp_url: "rtsp://192.168.1.101:554/stream1" # ← Your camera URL
    rtsp_username: "admin"                       # ← Your username
    rtsp_password: "your_password"               # ← Your password
```

---

## ⚡ Quick Tuning Guide

### Too Many False Matches? (Khác người nhưng cùng ID)
```yaml
reid:
  thresholds:
    accept: 0.80    # ↑ Increase (stricter)
```

### Too Many ID Switches? (Cùng người nhưng khác ID)
```yaml
reid:
  thresholds:
    accept: 0.70    # ↓ Decrease (looser)
```

### Too Slow?
```yaml
cameras:
  - resize_width: 480    # ↓ Smaller
inference:
  yolo:
    device: "cuda"       # Use GPU
```

---

## 📝 Next Steps

1. **✅ Test ngay**: Chạy `multi.bat` để test với 3 videos
2. **📊 Đánh giá**: Xem metrics, ID switches, false matches
3. **🎛️ Tune**: Điều chỉnh thresholds trong `multi_camera_config.yaml`
4. **🎥 RTSP**: Khi ready, switch to real cameras
5. **🧠 Features**: Integrate Face/Gait extractors (Phase 2)

---

## 🆘 Need Help?

Đọc chi tiết tại: **`MULTI_CAMERA_GUIDE.md`**

**Common Issues:**
- Config not found → Check path `backend/multi_camera_config.yaml`
- Video not opening → Check video paths are correct
- Too slow → Use GPU or reduce resolution

---

**Status**: ✅ Ready to run! Just execute `multi.bat`! 🚀
