# ✅ Updated: 4-Camera Configuration

## 🎥 Video Sources Changed

### Old Configuration (3 cameras):
```yaml
cam1: D:/HAVEN/backend/data/multi-camera/1.mp4
cam2: D:/HAVEN/backend/data/multi-camera/2.mp4
cam3: D:/HAVEN/backend/data/multi-camera/3.mp4
```

### New Configuration (4 cameras):
```yaml
cam1: D:/HAVEN/backend/data/multi-camera/phu1.mp4
cam2: D:/HAVEN/backend/data/multi-camera/phu2.mp4
cam3: D:/HAVEN/backend/data/multi-camera/phu3.mp4
cam4: D:/HAVEN/backend/data/multi-camera/phu4.mp4  ← NEW!
```

---

## 📊 Video Info

| Camera | File | Size |
|--------|------|------|
| cam1 | phu1.mp4 | 1.6 MB |
| cam2 | phu2.mp4 | 3.1 MB |
| cam3 | phu3.mp4 | 4.7 MB |
| cam4 | phu4.mp4 | 4.7 MB |

---

## 🗺️ Camera Graph Updated

```
cam1 ↔ cam2: 5-30s
cam1 ↔ cam3: 10-60s
cam1 ↔ cam4: 15-90s

cam2 ↔ cam3: 15-90s
cam2 ↔ cam4: 10-60s

cam3 ↔ cam4: 5-30s
```

**Notes:**
- cam3 ↔ cam4 có thời gian transition nhanh nhất (5-30s) → Gần nhau
- cam1 ↔ cam4 có thời gian transition chậm nhất (15-90s) → Xa nhau

---

## 🚀 How to Run

### Cách 1: Double-click
```
D:\HAVEN\multi.bat
```

### Cách 2: Command Line
```bash
cd D:\HAVEN
.\multi.bat
```

---

## 🖼️ Expected Display

```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  Camera 1   │  Camera 2   │  Camera 3   │  Camera 4   │
│  (phu1.mp4) │  (phu2.mp4) │  (phu3.mp4) │  (phu4.mp4) │
│             │             │             │             │
│  [Person]   │  [Person]   │  [Person]   │  [Person]   │
│  T1 → G1    │  T2 → G1    │  T3 → G2    │  T4 → G1    │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

**4 videos hiển thị cùng lúc, side-by-side!**

---

## ⚙️ Configuration File

All settings in: `backend/multi_camera_config.yaml`

**To modify:**
1. Open `backend/multi_camera_config.yaml`
2. Edit camera paths, thresholds, etc.
3. Save and run `multi.bat`

---

## 📝 What Changed

### Files Modified:
- ✅ `backend/multi_camera_config.yaml`
  - Changed 3 video paths from `1/2/3.mp4` → `phu1/phu2/phu3.mp4`
  - Added cam4 with `phu4.mp4`
  - Added cam4 to camera graph

### Files Unchanged:
- ✅ `backend/run_multi_camera.py` (no change needed, auto-detects cameras)
- ✅ `multi.bat` (no change needed)

---

## ✨ Features

All ReID features work with 4 cameras:
- ✅ ByteTrack local tracking per camera
- ✅ Tracklet aggregation
- ✅ Global ID assignment across 4 cameras
- ✅ Spatiotemporal constraints (camera graph)
- ✅ Open-set recognition
- ✅ Real-time visualization

---

**Ready to run!** Just execute `multi.bat` 🚀
