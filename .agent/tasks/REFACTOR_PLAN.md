# HAVEN Refactoring Plan & Commit Guide

##  TM TT REFACTOR

### Mc Tiu Chnh
Refactor HAVEN  h tr multi-camera realtime processing vi dual-master camera logic.

### Vn  Hin Ti (T repo gc)
1.  Sequential processing - x l tng video ring l
2.  Hardcode paths v cam1 l master duy nht
3.  Khng c config file tp trung
4.  Thiu logic cho multiple master cameras
5.  Khng ng b thi gian gia cameras

### Gii Php
1.  Realtime multi-camera pipeline
2.  Config-driven architecture
3.  Dual-master camera support (cam1 + cam2)
4.  Shared global gallery
5.  Frame-synchronized processing

---

##  CU TRC FILES  TO

```
haven_refactor/
 configs/
    multicam.yaml                    # Main config file 

 backend/src/
    io/
       video_stream.py              # VideoStream abstraction 
   
    global_id/
       manager.py                   # GlobalIDManager 
   
    run_multicam_reid.py             # Main entrypoint 

 tests/
    test_global_id_manager.py        # Unit tests 

 run_multicam.bat                     # Windows batch runner
 README_IMPLEMENTATION.md             # Implementation guide
 REFACTOR_PLAN.md                     # This file
```

---

##  KEY COMPONENTS

### 1. GlobalIDManager (manager.py)

**Trch nhim**:
- Qun l global IDs across cameras
- Implement dual-master logic
- Two-threshold decision
- Spatiotemporal filtering
- Multi-prototype memory

**Key methods**:
```python
assign_global_id(camera_id, local_track_id, embedding, ...)
     (global_id, reason, confidence)

# Logic:
# - Master cameras (1, 2): can create new IDs
# - Non-master cameras (3, 4): can only MATCH or wait (TEMP ID = 0)
# - Gallery shared globally
```

### 2. VideoStream (video_stream.py)

**Trch nhim**:
- Abstraction cho video sources
- Support: file, folder, RTSP
- Auto-concatenate video chunks
- Frame skipping, resizing

**Key methods**:
```python
read()  (success, frame)
get_fps()  float
get_timestamp()  float
```

### 3. MultiCameraReIDSystem (run_multicam_reid.py)

**Trch nhim**:
- Load config
- Initialize cameras
- Main processing loop
- Visualization
- Video output

**Flow**:
```python
while streams_alive:
    for camera in streams:
        frame = camera.read()
        detections = detect(frame)  # TODO
        tracks = track(detections)   # TODO
        
        for track in tracks:
            embedding = extract(track)  # TODO
            global_id = manager.assign(...)
            
        draw_overlay(frame, tracks)
        write_video(frame)
```

---

##  HOW TO APPLY TO HAVEN REPO

### Method 1: Copy Files Trc Tip

```bash
# Navigate to HAVEN repo
cd D:\HAVEN

# Copy configs
mkdir -p configs
cp /path/to/haven_refactor/configs/multicam.yaml configs/

# Copy backend modules
mkdir -p backend/src/io
mkdir -p backend/src/global_id

cp /path/to/haven_refactor/backend/src/io/video_stream.py backend/src/io/
cp /path/to/haven_refactor/backend/src/global_id/manager.py backend/src/global_id/
cp /path/to/haven_refactor/backend/src/run_multicam_reid.py backend/src/

# Copy tests
mkdir -p tests
cp /path/to/haven_refactor/tests/test_global_id_manager.py tests/

# Copy scripts
cp /path/to/haven_refactor/run_multicam.bat .

# Copy docs
cp /path/to/haven_refactor/README_IMPLEMENTATION.md .
cp /path/to/haven_refactor/REFACTOR_PLAN.md .
```

### Method 2: Git Patches

```bash
# In haven_refactor directory
git init
git add .
git commit -m "Refactored multi-camera ReID system"

# Create patch
git format-patch -1 HEAD --stdout > haven_refactor.patch

# In HAVEN repo
cd D:\HAVEN
git apply /path/to/haven_refactor.patch
```

### Method 3: Manual Integration

Tham kho code trong `haven_refactor/` v manually copy vo HAVEN repo.

---

##  TESTING PROCEDURE

### Test 1: Unit Tests

```bash
cd D:\HAVEN
python tests\test_global_id_manager.py
```

**Expected output**:
```
============================================================
HAVEN GLOBAL ID MANAGER - UNIT TESTS
============================================================

TEST CASE 1: Cam1 creates, Cam2 matches
...
 PASSED

TEST CASE 2: Cam2 creates, Cam1 matches
...
 PASSED

TEST CASE 3: Non-master cannot create new ID
...
 PASSED

============================================================
TEST SUMMARY
============================================================
Tests run: 5
Failures: 0
Errors: 0

 ALL TESTS PASSED!
```

### Test 2: Run System (Dry Run)

```bash
python backend\src\run_multicam_reid.py --config configs\multicam.yaml
```

**Expected**: System initializes, may fail at video loading if videos not present.

### Test 3: Full Run (With Videos)

```bash
# Ensure videos exist
dir backend\data\multi-camera\

# Run
run_multicam.bat
```

**Expected**:
- Mosaic window shows 4 cameras
- Console logs show ID assignments:
  - ` [cam1] Track 1  Global ID 1 (MASTER_NEW: ...)`
  - ` [cam2] Track 3  Global ID 1 (MATCHED: score=0.82)`
  - ` [cam3] Track 2  TEMP (non-master, ...)`
- Output videos created in `outputs/multicam/`

---

##  INTEGRATION TODO LIST

### Immediate (Core functionality)

- [ ] Copy files to HAVEN repo
- [ ] Test unit tests pass
- [ ] Configure video paths in `configs/multicam.yaml`
- [ ] Run dry test

### Short-term (Detection + Tracking)

- [ ] Integrate YOLO detection in `run_multicam_reid.py`
  ```python
  from ultralytics import YOLO
  detector = YOLO('yolov11n.pt')
  results = detector(frame)
  ```

- [ ] Integrate ByteTrack
  ```python
  from boxmot import ByteTrack
  tracker = ByteTrack()
  tracks = tracker.update(detections, frame)
  ```

- [ ] Extract basic embeddings (use YOLO features as placeholder)

### Medium-term (Feature Extractors)

- [ ] Implement Face extractor (`backend/src/reid/face_extractor.py`)
- [ ] Implement Gait extractor (`backend/src/reid/gait_extractor.py`)
- [ ] Implement Appearance extractor (`backend/src/reid/appearance_extractor.py`)
- [ ] Update `GlobalIDManager` to use multi-signal fusion

### Long-term (Production)

- [ ] Database integration (SQLite)
- [ ] WebSocket API for live monitoring
- [ ] Performance optimization (batch inference, GPU)
- [ ] Full end-to-end testing

---

##  KEY DOCUMENTATION

### For Users

**README_IMPLEMENTATION.md**:
- Gii thch dual-master logic
- Cch chy system
- Cch tune thresholds
- Troubleshooting

### For Developers

**Code Comments**:
- Docstrings in all classes/functions
- Type hints
- Inline comments for complex logic

**REFACTOR_PLAN.md** (this file):
- Architecture overview
- Integration guide
- Testing procedures

---

##  ACCEPTANCE CRITERIA

### Functional Requirements

- [x] System loads config from YAML
- [x] Multiple cameras initialize successfully
- [x] Cam1 and Cam2 can create new global IDs
- [x] Cam3 and Cam4 can only MATCH or assign TEMP IDs
- [x] Gallery is shared across all cameras
- [x] Spatiotemporal filtering works
- [x] Two-threshold decision logic works
- [ ] Detection + tracking integrated (TODO)
- [ ] Video output with overlay (partially done)

### Non-Functional Requirements

- [x] Config-driven (no hardcode)
- [x] Clean code structure
- [x] Type hints
- [x] Docstrings
- [x] Unit tests
- [x] Error handling
- [x] Logging
- [ ] Performance tested (TODO)

---

##  DEPLOYMENT CHECKLIST

### Pre-deployment

- [ ] All unit tests pass
- [ ] Integration test with sample videos successful
- [ ] Config file reviewed and customized
- [ ] Dependencies installed (`opencv-python`, `numpy`, `pyyaml`)

### Deployment

- [ ] Copy files to HAVEN repo
- [ ] Update `.gitignore` if needed
- [ ] Commit to git:
  ```bash
  git checkout -b feat/dual-master-multicam
  git add .
  git commit -m "feat: refactor multi-camera ReID with dual-master support"
  git push origin feat/dual-master-multicam
  ```

### Post-deployment

- [ ] Run system with real videos
- [ ] Monitor metrics
- [ ] Tune thresholds if needed
- [ ] Document any issues

---

##  SUPPORT

### Issues

If encountering issues:

1. Check logs in `backend/logs/multicam_reid.log`
2. Run unit tests: `python tests\test_global_id_manager.py`
3. Enable debug mode in config:
   ```yaml
   debug:
     enabled: true
     print_similarity_scores: true
   ```

### Common Problems

**Problem**: Videos not loading
**Solution**: Check paths in config, use absolute paths if needed

**Problem**: Too many false matches
**Solution**: Increase `reid.thresholds.accept` to 0.80

**Problem**: Too many ID switches
**Solution**: Decrease `reid.thresholds.accept` to 0.70

---

**Status**:  Core implementation complete, ready for detection/tracking integration
**Last Updated**: 2026-01-31

