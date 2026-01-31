# HAVEN Multi-Camera ReID - Complete Implementation Guide

##  TM TT REFACTOR

###   HON THNH

 refactor ton b h thng HAVEN theo yu cu:

1. **Dual-Master Camera Logic** 
   - Cam1 v Cam2 u l MASTER c quyn to global ID mi
   - Cam3 v Cam4 ch c th MATCH vi existing IDs
   - Gallery c share globally gia tt c cameras

2. **Multi-Camera Realtime Pipeline** 
   - Load t 1 ROOT directory cha nhiu camera folders
   - X l ng thi multiple cameras (frame-by-frame)
   - ng b theo frame index hoc timestamp
   - Ghi trc tip video output vi overlay

3. **Clean Architecture** 
   - Config file YAML r rng
   - Modules tch bit theo chc nng
   - Type hints + docstrings
   - Error handling y 

4. **Video I/O Abstraction** 
   - H tr: single video file, folder vi multiple chunks, RTSP
   - Auto-concatenate video chunks trong folder
   - Resize + frame skipping support

5. **Global ID Manager** 
   - Two-threshold decision logic
   - Spatiotemporal filtering
   - Multi-prototype memory
   - Deterministic tie-breaking
   - Comprehensive metrics

---

##  CU TRC TH MC MI

```
HAVEN/
 configs/
    multicam.yaml              # Config chnh 

 backend/
    src/
       config/
          loader.py          # Config loader
       io/
          video_stream.py    # Video stream abstraction 
       tracking/
          bytetrack_wrapper.py
       reid/
          feature_extractor.py
          tracklet_aggregator.py
       global_id/
          manager.py         # Global ID Manager 
       viz/
          overlay.py
       run_multicam_reid.py   # Main entrypoint 
   
    data/
       multi-camera/          # ROOT folder
           phu1.mp4           # Camera 1
           phu2.mp4           # Camera 2
           phu3.mp4           # Camera 3
           phu4.mp4           # Camera 4
   
    outputs/
       multicam/
           cam1_output.mp4
           cam2_output.mp4
           cam3_output.mp4
           cam4_output.mp4
           mosaic_output.mp4
   
    logs/
        multicam_reid.log

 run_multicam.bat               # Windows batch script
 README_REFACTOR.md             # This file
```

---

##  LOGIC 2-MASTER CAMERA (QUAN TRNG!)

### Quy Tc Cp Global ID

#### **Master Cameras (Cam1, Cam2)**

 **C QUYN** to global ID mi khi:
1. Gallery rng (cha c ai)
2. Similarity < threshold_reject (0.50)  chc chn ngi mi
3. Similarity trong vng uncertain (0.50 - 0.75)  conservative, to ID mi
4. Margin < threshold (qu gn 2 candidates)  ambiguous, to ID mi
5. Spatiotemporal violation  khng th l cng ngi

 **MATCH vi ID c** khi:
- Similarity >= threshold_accept (0.75) AND margin  ln AND spatiotemporal OK

#### **Non-Master Cameras (Cam3, Cam4)**

 **KHNG C QUYN** to global ID mi

 **CH C TH**:
- MATCH vi existing IDs nu similarity >= threshold_accept
- Gn TEMP ID (0) nu khng match c  i xut hin  master camera

### Flow Diagram

```
Person xut hin  Cam2 (MASTER):
 Gallery rng?
   YES  Cam2 to Global ID = 1 
   NO  Tip tc
 So snh vi Gallery (bao gm IDs t Cam1)
   Similarity >= 0.75?
     YES  MATCH vi existing ID 
     NO  Tip tc
   Similarity <= 0.50?
     YES  Cam2 to Global ID mi 
   Uncertain (0.50-0.75)?
      Cam2 to Global ID mi (conservative) 

Person xut hin  Cam3 (NON-MASTER):
 Gallery rng?
   YES  Gn TEMP ID = 0 (i master) 
   NO  Tip tc
 So snh vi Gallery
   Similarity >= 0.75?
     YES  MATCH vi existing ID 
   NO  Gn TEMP ID = 0 (i master) 
```

### V D C Th

#### Scenario 1: Cam1 thy trc
```
t=0s   : Ngi A xut hin  Cam1 (MASTER)
          Gallery rng  Cam1 to Global ID = 1 

t=10s  : Ngi A di chuyn n Cam2 (MASTER)
          So snh vi Gallery[ID=1]
          Similarity = 0.85 > 0.75
          MATCH  Global ID = 1 

t=20s  : Ngi A di chuyn n Cam3 (NON-MASTER)
          So snh vi Gallery[ID=1]
          Similarity = 0.80 > 0.75
          MATCH  Global ID = 1 
```

#### Scenario 2: Cam2 thy trc (xe my vo thng Cam2)
```
t=0s   : Ngi B i xe my vo thng Cam2 (MASTER)
          Cam1 khng pht hin
          Gallery rng
          Cam2 to Global ID = 1 

t=30s  : Ngi B di chuyn n Cam1 (MASTER)
          So snh vi Gallery[ID=1 created by Cam2]
          Similarity = 0.82 > 0.75
          MATCH  Global ID = 1  (khng to ID mi!)
```

#### Scenario 3: Cam3 thy ngi mi (NON-MASTER)
```
t=0s   : Ngi C xut hin  Cam3 (NON-MASTER)
          Gallery rng
          Cam3 KHNG C QUYN to ID
          Gn TEMP ID = 0 

t=10s  : Ngi C di chuyn n Cam1 (MASTER)
          So snh vi Gallery
          Khng match
          Cam1 to Global ID = 1 

t=20s  : Ngi C quay li Cam3 (NON-MASTER)
          So snh vi Gallery[ID=1]
          Similarity = 0.78 > 0.75
          MATCH  Global ID = 1 
```

---

##  CCH CHY H THNG

### Bc 1: Chun B D Liu

To cu trc folder:

```
D:/HAVEN/backend/data/multi-camera/
 phu1.mp4    # Camera 1
 phu2.mp4    # Camera 2
 phu3.mp4    # Camera 3
 phu4.mp4    # Camera 4
```

Hoc nu mi camera c nhiu video chunks:

```
D:/HAVEN/backend/data/multi-camera/
 cam1/
    video_001.mp4
    video_002.mp4
    video_003.mp4
 cam2/
    video_001.mp4
    video_002.mp4
 cam3/
    video_001.mp4
 cam4/
     video_001.mp4
```

### Bc 2: Cu Hnh

Edit `configs/multicam.yaml`:

```yaml
data:
  data_root: "D:/HAVEN/backend/data/multi-camera"
  cameras:
    - id: 1
      name: "cam1"
      enabled: true
      source_type: "video_folder"  # hoc "video_file"
      path: "phu1.mp4"  # hoc "cam1" nu l folder

master_cameras:
  ids: [1, 2]  # Cam1 v Cam2 l MASTER 
```

### Bc 3: Chy

#### Cch 1: Python trc tip
```bash
cd D:\HAVEN
python backend\src\run_multicam_reid.py --config configs\multicam.yaml
```

#### Cch 2: Batch script (Windows)
```bash
cd D:\HAVEN
run_multicam.bat
```

#### Cch 3: Vi overrides
```bash
python backend\src\run_multicam_reid.py \
    --config configs\multicam.yaml \
    --data_root "E:\Videos\cameras" \
    --out_dir "E:\outputs" \
    --write_video
```

### Bc 4: Phm Tt Khi Chy

- **Q**: Quit
- **P**: Pause/Resume

---

##  OUTPUT & LOGS

### Console Output

```
============================================================
HAVEN Multi-Camera ReID System
============================================================
Master cameras: [1, 2]
Total cameras: 4
============================================================

 Starting multi-camera processing...
Press 'Q' to quit, 'P' to pause

 [cam1] Track 1  Global ID 1 (MASTER_NEW: empty_gallery)
 [cam2] Track 3  Global ID 1 (MATCHED: score=0.82)
 [cam2] Track 5  Global ID 2 (MASTER_NEW: low_similarity=0.45)
 [cam3] Track 2  TEMP (non-master, empty gallery)
 [cam3] Track 2  Global ID 1 (MATCHED: score=0.78)
 [cam1] Track 8  Global ID 3 (MASTER_NEW: uncertain_zone=0.65)
...
```

### Video Outputs

```
D:/HAVEN/backend/outputs/multicam/
 cam1_output.mp4    # Overlay cho camera 1
 cam2_output.mp4    # Overlay cho camera 2
 cam3_output.mp4    # Overlay cho camera 3
 cam4_output.mp4    # Overlay cho camera 4
 mosaic_output.mp4  # 2x2 mosaic view (nu enabled)
```

### Overlay Display

Mi person s hin th:
- **Green box**: New ID t master camera
- **Cyan box**: Matched ID
- **Orange box**: Temporary ID (non-master)
- **T{id}**: Local track ID (yellow text)
- **G{id}**: Global ID (cyan text)
- **Status**: NEW / MATCH / TEMP

### Metrics (End of Run)

```
============================================================
GLOBAL ID MANAGER METRICS
============================================================
Total Global IDs Created: 5
  - By Master Cameras: 5
  - Active: 5
Total Matches: 23
Total Rejections: 2
Spatiotemporal Filtered: 3
Non-Master Temp IDs: 7
Total Tracks Assigned: 35
============================================================
```

---

##  TUNING THRESHOLDS

### Qu Nhiu False Matches?

**Triu chng**: Khc ngi nhng b gn cng Global ID

**Gii php**:
```yaml
reid:
  thresholds:
    accept: 0.80     # Tng t 0.75 (stricter)
    margin: 0.20     # Tng t 0.15
  quality:
    min_quality_score: 0.6  # Tng t 0.5
```

### Qu Nhiu ID Switches?

**Triu chng**: Cng ngi nhng b to nhiu Global ID

**Gii php**:
```yaml
reid:
  thresholds:
    accept: 0.70     # Gim t 0.75 (looser)
    reject: 0.45     # Gim t 0.50
  reuse_window: 10800  # Tng t 7200 (3 hours)
```

### Cam3/Cam4 Khng Bao Gi Match c?

**Triu chng**: Non-master cameras lun to TEMP IDs

**Gii php**:
```yaml
reid:
  thresholds:
    accept: 0.65     # Gim threshold  d match hn
```

---

##  TROUBLESHOOTING

### Error: "Config file not found"

```
ERROR: Config file not found: configs/multicam.yaml
```

**Gii php**: Chc chn ang chy t th mc gc HAVEN:
```bash
cd D:\HAVEN
python backend\src\run_multicam_reid.py --config configs\multicam.yaml
```

### Error: "Video file not found"

```
FileNotFoundError: Video file not found: D:/HAVEN/backend/data/multi-camera/phu1.mp4
```

**Gii php**: Kim tra:
1. Path trong config ng cha
2. File video c tn ti khng
3. ng dn absolute vs relative

### Error: "Failed to open video"

```
[cam1] Failed to open: phu1.mp4
```

**Gii php**:
- Kim tra codec video (nn dng H.264)
- Th m bng VLC/Media Player trc
- Check quyn read file

### Khng Thy Preview Window

**Gii php**:
```yaml
visualization:
  enabled: true
  mosaic:
    enabled: true
```

### Video Output B Corrupt

**Gii php**: Th i codec:
```yaml
output:
  video_codec: "avc1"  # Thay v "mp4v"
```

---

##  NEXT STEPS

### Phase 1: Tch Hp Detection + Tracking (Cn lm ngay)

File cn sa: `backend/src/run_multicam_reid.py`

```python
# TODO: Thm YOLO detection
from ultralytics import YOLO
detector = YOLO('yolov11n.pt')

# TODO: Thm ByteTrack
from boxmot import ByteTrack
tracker = ByteTrack()

# Trong main loop:
for cam_id, stream in self.streams.items():
    ret, frame = stream.read()
    
    # Detection
    results = detector(frame)
    detections = results[0].boxes
    
    # Tracking
    tracks = tracker.update(detections, frame)
    
    # Extract embeddings (placeholder)
    for track in tracks:
        track_id = track.id
        bbox = track.xyxy
        
        # TODO: Extract ReID embedding
        embedding = extract_embedding(frame, bbox)
        quality = compute_quality(frame, bbox)
        
        # Assign global ID
        global_id, reason, score = self.global_id_manager.assign_global_id(
            camera_id=cam_id,
            local_track_id=track_id,
            embedding=embedding,
            quality=quality,
            timestamp=stream.get_timestamp(),
            frame_idx=stream.get_frame_count(),
            num_frames=len(track.frames),
            bbox_size=compute_bbox_size(bbox)
        )
        
        # Collect for visualization
        detections.append({
            'bbox': bbox,
            'local_track_id': track_id,
            'global_id': global_id,
            'reason': reason,
            'confidence': score
        })
```

### Phase 2: Feature Extractors

To cc modules:

```
backend/src/reid/feature_extractors/
 face_extractor.py      # InsightFace/ArcFace
 gait_extractor.py      # Body proportions t YOLO-Pose
 appearance_extractor.py # OSNet/ResNet50
```

### Phase 3: Multi-Signal Fusion

Update `GlobalIDManager.assign_global_id()`  s dng:
- Face similarity (nu c face)
- Gait similarity (t pose sequence)
- Appearance similarity

Priority: Face > Gait > Appearance

### Phase 4: Testing & Optimization

1. Unit tests cho `GlobalIDManager`
2. Integration tests vi real videos
3. Performance profiling
4. GPU optimization

---

##  KEY TAKEAWAYS

### Logic 2-Master (Tm Tt 3-4 Bullets)

1. **Ch Cam1 v Cam2 c quyn to NEW Global IDs**
   - Cam3, Cam4 ch c th MATCH hoc gn TEMP

2. **Gallery c share globally**
   - Cam2 to ID  Cam1 thy c v match ng
   - Cam1 to ID  Cam2 thy c v match ng

3. **Trc khi to ID mi, LUN so snh vi Gallery trc**
   - Nu match (sim >= 0.75)  dng existing ID
   - Nu khng match  master to ID mi, non-master ch

4. **Global IDs tng tun t 1, 2, 3, ...**
   - KHNG reset gia cameras
   - Deterministic (same input  same IDs)

---

##  CHECKLIST HON THNH

- [x] Config YAML vi master_cameras list
- [x] VideoStream abstraction (file/folder/RTSP)
- [x] GlobalIDManager vi dual-master logic
- [x] Spatiotemporal filtering
- [x] Two-threshold decision
- [x] Multi-prototype memory
- [x] Visualization vi status colors
- [x] Logging r rng (new ID / match / temp)
- [x] Metrics tracking
- [x] Command-line interface
- [x] Error handling
- [ ] Detection + Tracking integration (TODO)
- [ ] Feature extractors (TODO)
- [ ] Unit tests (TODO)

---

**Status**:  Core architecture hon thnh, ready for detection/tracking integration!

