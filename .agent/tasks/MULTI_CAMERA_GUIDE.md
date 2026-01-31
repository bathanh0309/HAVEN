#  Multi-Camera ReID Configuration Guide

##  File Structure

```
HAVEN/
 multi.bat                           #  Script chy chnh (double-click  chy!)
 backend/
     multi_camera_config.yaml        #  File cu hnh tng hp
     run_multi_camera.py             # Script Python chnh
     data/
         multi-camera/
             1.mp4                   # Camera 1 video
             2.mp4                   # Camera 2 video
             3.mp4                   # Camera 3 video
```

---

##  Quick Start

### Cch 1: Double-click (n gin nht)
```
1. M Windows Explorer
2. Tm file: D:\HAVEN\multi.bat
3. Double-click  chy!
```

### Cch 2: Command Line
```bash
cd D:\HAVEN
multi.bat
```

### Cch 3: Python Trc Tip
```bash
cd D:\HAVEN
.venv\Scripts\activate
python backend\run_multi_camera.py
```

---

##  Configuration File: `backend/multi_camera_config.yaml`

### 1 Camera Sources

Mi camera c th dng **video file** hoc **RTSP stream**.

#### Dng Video File (Default):
```yaml
cameras:
  - id: "cam1"
    name: "Camera 1"
    enabled: true
    source_type: "video"                           #  Chn "video"
    video_path: "D:/HAVEN/backend/data/multi-camera/1.mp4"  #  ng dn video
```

#### Dng RTSP Camera:
```yaml
cameras:
  - id: "cam1"
    name: "Living Room"
    enabled: true
    source_type: "rtsp"                            #  i thnh "rtsp"
    rtsp_url: "rtsp://192.168.1.101:554/stream1"   #  RTSP URL
    rtsp_username: "admin"                         #  Username
    rtsp_password: "password123"                   #  Password
```

#### Tt Camera:
```yaml
cameras:
  - id: "cam3"
    enabled: false    #  t false  skip camera ny
```

---

### 2 Spatiotemporal Constraints (Camera Graph)

nh ngha thi gian di chuyn gia cc cameras:

```yaml
camera_graph:
  cam1:
    cam2:
      min_time: 5    # Ti thiu 5 giy t cam1  cam2
      max_time: 30   # Ti a 30 giy
    cam3:
      min_time: 10   # Ti thiu 10 giy t cam1  cam3
      max_time: 60
```

** ngha:**
- Nu ngi xut hin  cam2 ch **2 giy** sau khi ri cam1  **Khng th** l cng ngi (vn tc v l)
- H thng s t ng **reject** matches violate physics

**Cch o:**
1. i b t v tr cam1 n cam2 vi tc  bnh thng
2. o thi gian = T giy
3. t `min_time = T - 5`, `max_time = T + 15`

---

### 3 Detection & Tracking Thresholds

#### YOLO Detection:
```yaml
inference:
  yolo:
    model: "yolov11n.pt"           # Model trong backend/models/
    conf_threshold: 0.5            #  Tng = t false positives,  Gim = nhiu detections
    iou_threshold: 0.45            # NMS threshold
    device: "cuda"                 # "cuda" hoc "cpu"
```

#### Tracking:
```yaml
tracking:
  tracker_type: "bytetrack"        # "bytetrack" hoc "botsort"
  min_tracklet_frames: 5           # Ti thiu 5 frames mi to tracklet
```

---

### 4 ReID Thresholds (QUAN TRNG!)

#### Open-Set Two-Threshold:
```yaml
reid:
  thresholds:
    accept: 0.75                   # T_high: > 0.75 = Confident match
    reject: 0.50                   # T_low: < 0.50 = Definitely new person
    margin: 0.15                   # Margin to 2nd-best candidate
```

**Gii thch:**
- **Similarity > 0.75**  Match vi ID c 
- **Similarity < 0.50**  To ID mi 
- **0.50 - 0.75**  Vng **uncertain**, dng thm evidence (quality, face, gait)

#### Signal-Specific Thresholds:
```yaml
reid:
  thresholds:
    face_similarity: 0.6           # Face matching (cha dng, TODO)
    gait_similarity: 0.7           # Gait matching (cha dng, TODO)
    appearance_similarity: 0.5     # Appearance matching
```

#### Quality Gating:
```yaml
reid:
  quality:
    min_tracklet_frames: 5         # Ti thiu 5 frames
    min_quality_score: 0.5         # Overall quality > 0.5
    min_bbox_size: 80              # Bbox ti thiu 80x80 pixels
```

**Tng thresholds nu:**
- Qu nhiu false matches (khc ngi m b gp chung ID)
   Tng `accept` t 0.75 ln 0.80

**Gim thresholds nu:**
- Qu nhiu ID switches (cng ngi nhng b tch ID)
   Gim `accept` t 0.75 xung 0.70

---

### 5 Visualization

```yaml
visualization:
  enabled: true                    # Hin th video
  show_bbox: true                  # Hin bounding box
  show_track_id: true              # Hin local track ID
  show_global_id: true             # Hin global ID (ReID)
```

---

### 6 Output Settings

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
    enabled: false                 # t true  lu video output
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

##  Common Tuning Scenarios

### Scenario 1: Qu Nhiu False Matches
**Triu chng:** Khc ngi nhng b gn cng global ID

**Gii php:**
```yaml
reid:
  thresholds:
    accept: 0.80        # Tng t 0.75  0.80 (stricter)
    margin: 0.20        # Tng t 0.15  0.20
  quality:
    min_quality_score: 0.6  # Tng t 0.5  0.6
```

### Scenario 2: Qu Nhiu ID Switches
**Triu chng:** Cng ngi nhng b tch ra nhiu global ID

**Gii php:**
```yaml
reid:
  thresholds:
    accept: 0.70        # Gim t 0.75  0.70 (looser)
    reject: 0.45        # Gim t 0.50  0.45
  reuse_window: 10800   # Tng t 7200  10800 (3 hours)
```

### Scenario 3: Chm, Cn Tng FPS
**Gii php:**
```yaml
cameras:
  - resize_width: 480        # Gim t 640  480
    skip_frames: 1           # Process every 2nd frame

inference:
  yolo:
    device: "cuda"           # Dng GPU (nu c)
    img_size: 480            # Gim t 640  480

tracking:
  min_tracklet_frames: 3     # Gim t 5  3
```

### Scenario 4: Videos Khc Timing
**Triu chng:** 3 videos khng ng b thi gian

**Lu :** H thng dng **relative time** t khi start script, khng phi timestamp trong video.

**Nu mun ng b:**
1. Ct video  cng start time
2. Hoc dng RTSP (real-time streams)

---

##  Output & Metrics

### Console Output:
```
============================================================
HAVEN Multi-Camera ReID System
============================================================
 Database: D:/HAVEN/backend/haven_reid.db
 Camera graph: 3 cameras
 Global ID Manager initialized
 cam1 (Camera 1): D:/HAVEN/backend/data/multi-camera/1.mp4
 cam2 (Camera 2): D:/HAVEN/backend/data/multi-camera/2.mp4
 cam3 (Camera 3): D:/HAVEN/backend/data/multi-camera/3.mp4
 YOLO model: yolov11n.pt
============================================================

 Starting multi-camera processing...
Press 'Q' to quit

  [cam1] Track 1  Global ID 1 (high: new_identity)
  [cam1] Track 2  Global ID 2 (high: new_identity)
  [cam2] Track 1  Global ID 1 (medium: gait_0.72)     Matched!
  [cam3] Track 3  Global ID 3 (high: new_identity)
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

##  Troubleshooting

### Error: "Failed to open source"
```
 cam1: D:/HAVEN/backend/data/multi-camera/1.mp4
 cam2: Failed to open source
```

**Gii php:**
1. Kim tra ng dn video c ng khng
2. Kim tra file video c tn ti khng
3. Nu dng RTSP, kim tra network connection

### Error: "Config file not found"
```
ERROR: Config file not found!
Expected: backend\multi_camera_config.yaml
```

**Gii php:**
- Chc chn file `multi_camera_config.yaml` nm trong `backend/`

### Video Chy Qu Nhanh
**Gii php:**
```yaml
performance:
  max_fps: 10    # Gim t 30  10
```

### Khng Thy Window Hin Th
**Gii php:**
```yaml
visualization:
  enabled: true    # m bo = true
```

---

##  Next Steps

1. ** Chy vi videos hin ti**  test
2. ** nh gi metrics**: ID switches, false matches
3. ** Tune thresholds** theo kt qu
4. ** Thm RTSP cameras** khi ready
5. ** Integrate feature extractors** (Face, Gait, Appearance)

---

##  Quick Reference

| Setting | File | Line |
|---------|------|------|
| Video paths | `multi_camera_config.yaml` | 13-40 |
| RTSP URLs | `multi_camera_config.yaml` | 20-22 |
| Camera graph | `multi_camera_config.yaml` | 46-73 |
| ReID thresholds | `multi_camera_config.yaml` | 115-130 |
| Quality gates | `multi_camera_config.yaml` | 132-138 |

---

**Happy Tracking! **

