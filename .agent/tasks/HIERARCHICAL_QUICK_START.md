# HAVEN Hierarchical Security System - Quick Start Guide

##  H THNG PHN CP

### Logic Cameras

```
CAM1 (Cng) - REGISTRATION MASTER
 Bt buc phi ng k face r rng
 Red box  detect face  Green box + ID (1, 2, 3, ...)
 CH cam1 to ID mi

CAM2 (Bi xe) - VERIFICATION
 Check registry t cam1
  ng k  Green box + ID c
 Cha ng k  Red box + UNK1, UNK2, ...

CAM3 (Thang my) - VERIFICATION
 Check registry t cam1
  ng k  Green box + ID c
 Cha ng k  Red box + UNK3, UNK4, ...

CAM4 (Phng) - STRICT VERIFICATION
 Check registry t cam1
  ng k  Green box + ID c
 Cha ng k  Red box + INTRUDER! (Cnh bo xm nhp)
```

---

##  CU TRC FOLDER

```
D:/HAVEN/backend/data/multi-camera/
 cam1/              # Cng - REGISTRATION
    video1.mp4
    video2.mp4     # Nhiu videos khc nhau
    video3.mp4     # S lng ngi khc nhau

 cam2/              # Bi xe - VERIFICATION
    video1.mp4
    video2.mp4

 cam3/              # Thang my - VERIFICATION
    video1.mp4
    video2.mp4

 cam4/              # Phng - STRICT VERIFICATION
     video1.mp4
     video2.mp4
```

**Lu **: Folder cam1 cha nhng video ngi  hin th r khun mt v mi c gn ID.

---

##  CCH CHY (2 GIAI ON)

### GIAI ON 1: CHY SEQUENTIAL TRC 

**Ti sao phi chy sequential trc?**
-  kim tra tng camera ring l
- m bo videos load c
- Test face detection trn cam1
- Verify registry c build ng

#### Step 1.1: Chy Cam1 (Registration)

```bash
cd D:\HAVEN
sequential.bat cam1
```

**Expected output**:
```
[cam1] Processing video1.mp4...
[cam1] Track 1: No face detected  Red box
[cam1] Track 1: Face detected! Quality=0.85  Green box
 [cam1] Track 1  REGISTERED as ID 1
[cam1] Track 2: Face detected! Quality=0.92  Green box
 [cam1] Track 2  REGISTERED as ID 2
...
```

**Check**:
- Registry c bao nhiu ngi? (ID 1, 2, 3, ...)
- Tt c u c face quality > 0.8?
- Videos trong cam1/  x l ht cha?

#### Step 1.2: Chy Cam2 (Parking)

```bash
sequential.bat cam2
```

**Expected output**:
```
[cam2] Processing video1.mp4...
[cam2] Track 1: Matching against registry (2 persons)...
 [cam2] Track 1  VERIFIED as ID 1 (score=0.78)
[cam2] Track 2: Matching against registry...
 [cam2] Track 2  UNKNOWN (UNK1): Low similarity=0.42
[cam2] Track 3: Matching against registry...
 [cam2] Track 3  VERIFIED as ID 2 (score=0.81)
...
```

**Check**:
- C ngi c VERIFIED khng?
- C ngi b UNK khng?
- Scores c hp l khng?

#### Step 1.3: Chy Cam3 (Elevator)

```bash
sequential.bat cam3
```

**Expected output**: Tng t cam2

#### Step 1.4: Chy Cam4 (Room - Strict)

```bash
sequential.bat cam4
```

**Expected output**:
```
[cam4] Track 1: Matching against registry...
 [cam4] Track 1  VERIFIED as ID 1 (score=0.79)
[cam4] Track 2: Matching against registry...
 [cam4] Track 2  INTRUDER! (UNK1)
 SECURITY ALERT: Unregistered person in restricted area!
```

**Check**:
- Intruder alerts c c trigger khng?
- Logs c ghi li khng?

---

### GIAI ON 2: CHY NG B (SAU KHI SEQUENTIAL N)

Sau khi chy sequential xong v kim tra mi th OK:

```bash
cd D:\HAVEN
multi_hierarchical.bat
```

**Expected output**:
```
============================================================
HAVEN Hierarchical Security System
============================================================
Registration camera: 1 (Gate)
Verification cameras: 2 (Parking), 3 (Elevator)
Strict camera: 4 (Room)
============================================================

 Starting multi-camera processing...
Press 'Q' to quit

[cam1] Track 1: Face detected!  REGISTERED as ID 1
[cam2] Track 1:  VERIFIED as ID 1
[cam3] Track 2:  UNKNOWN (UNK1)
[cam4] Track 1:  VERIFIED as ID 1
[cam4] Track 3:  INTRUDER! (UNK2)
 ALERT: Intruder detected in Room!
```

---

##  OUTPUT VIDEOS

### Sequential Mode (1 camera at a time)

```
D:/HAVEN/backend/outputs/sequential/
 cam1_output.mp4    # Green boxes (ID 1, 2, 3, ...)
 cam2_output.mp4    # Green (VERIFIED) + Red (UNK)
 cam3_output.mp4    # Green (VERIFIED) + Red (UNK)
 cam4_output.mp4    # Green (VERIFIED) + Red (INTRUDER)
```

### Multi-camera Mode (all cameras together)

```
D:/HAVEN/backend/outputs/hierarchical/
 cam1_output.mp4
 cam2_output.mp4
 cam3_output.mp4
 cam4_output.mp4
 mosaic_output.mp4  # 2x2 grid
```

---

##  VISUALIZATION

### Bounding Box Colors

- **Green**: Registered person (ID 1, 2, 3, ...)
- **Red**: Unknown person (UNK1, UNK2, ...) or INTRUDER
- **Orange**: Processing (waiting for face detection on cam1)

### Text Labels

```
CAM1 (Gate):

 T1                 Local track ID
 [PROCESSING]       Waiting for face
                 
  RED box

After face detected:

 T1  ID 1       
 [REGISTERED]    
 Face: 0.85      
  GREEN box

CAM2 (Parking):

 T3  ID 1       
 [VERIFIED]      
 Score: 0.78     
  GREEN box


 T5  UNK1       
 [UNKNOWN]       
 Score: 0.42     
  RED box

CAM4 (Room):

 T2  UNK1       
 [INTRUDER!]     
 ALERT!          
  RED box + Beep sound
```

---

##  CONFIGURATION

Edit `configs/hierarchical_security.yaml`:

### Cam1 Registration Requirements

```yaml
registration:
  registration_camera: 1
  
  face_quality:
    min_sharpness: 50         # Laplacian variance
    min_brightness: 30
    max_brightness: 220
    frontal_angle_max: 30     # Degrees from frontal
    min_face_width: 100       # Pixels
    min_face_height: 100
```

### Verification Thresholds

```yaml
reid:
  thresholds:
    registered_match: 0.70    # >= 0.70 = VERIFIED
    unknown_threshold: 0.60   # < 0.60 = UNKNOWN
    margin: 0.15
```

### Camera Roles

```yaml
cameras:
  - id: 1
    role: "registration"       # Only cam1
    
  - id: 2
    role: "verification"
    
  - id: 3
    role: "verification"
    
  - id: 4
    role: "strict_verification"  # Intruder alert
```

---

##  TUNING

### Too Many UNK (Unknown) Detections?

**Problem**: Nhiu ngi  ng k nhng b nh du UNK

**Solution**: Gim threshold
```yaml
reid:
  thresholds:
    registered_match: 0.65   # Gim t 0.70
```

### Too Many False Verifications?

**Problem**: Ngi cha ng k nhng b VERIFIED nhm

**Solution**: Tng threshold
```yaml
reid:
  thresholds:
    registered_match: 0.75   # Tng t 0.70
```

### Face Detection Too Strict?

**Problem**: Cam1 kh ng k, lun Red box

**Solution**: Gim face quality requirements
```yaml
registration:
  face_quality:
    min_sharpness: 30        # Gim t 50
    frontal_angle_max: 45    # Tng t 30 degrees
```

---

##  TESTING CHECKLIST

### Before Running Sequential

- [ ] Videos c trong cc folder cam1/, cam2/, cam3/, cam4/?
- [ ] Config file `hierarchical_security.yaml`  edit?
- [ ] Data_root path ng cha?

### After Cam1 Sequential

- [ ] C t nht 1 person c REGISTERED?
- [ ] Face quality >= 0.8?
- [ ] Green boxes xut hin?
- [ ] Video output c trong outputs/sequential/?

### After Cam2/3/4 Sequential

- [ ] C VERIFIED persons?
- [ ] C UNK persons?
- [ ] Similarity scores hp l?
- [ ] Intruder alerts (cam4) c trigger?

### Before Multi-camera

- [ ] Sequential mode  chy n cho c 4 cameras?
- [ ] Registry c y  ngi?
- [ ] Thresholds  tune?

---

##  TROUBLESHOOTING

### Error: "No face detected on cam1"

```
[cam1] Track 1: No face detected yet
```

**Solution**:
- Kim tra videos cam1/ c show mt ngi khng
- Gim `min_face_confidence` trong config
- Check YOLO-Face model c load c khng

### Error: "All tracks marked as UNK on cam2"

```
[cam2] Track 1  UNK1
[cam2] Track 2  UNK2
[cam2] Track 3  UNK3
```

**Solution**:
- Chy cam1 trc  build registry!
- Check registry c empty khng
- Gim `registered_match` threshold

### Error: "Videos not loading"

```
FileNotFoundError: D:/HAVEN/backend/data/multi-camera/cam1/video1.mp4
```

**Solution**:
- Kim tra ng dn trong config
- m bo videos tn ti
- Check pattern `*.mp4` c ng khng

---

##  WORKFLOW TNG QUAN

```
1. PREPARE DATA
    Place videos in cam1/, cam2/, cam3/, cam4/
    Ensure cam1 videos show clear faces

2. CONFIGURE
    Edit configs/hierarchical_security.yaml
    Set data_root path
    Tune thresholds if needed

3. RUN SEQUENTIAL (Testing)
    Run cam1 first  Build registry
    Run cam2  Verify against registry
    Run cam3  Verify against registry
    Run cam4  Strict verification + alerts

4. CHECK OUTPUTS
    Videos in outputs/sequential/
    Check registry built correctly
    Verify metrics

5. RUN MULTI-CAMERA (Production)
    Run multi_hierarchical.bat
    All cameras process simultaneously
    Real-time mosaic view

6. MONITOR
    Watch console logs
    Check alerts (intruders)
    Review output videos
```

---

##  EXPECTED RESULTS

### Cam1 (Registration)

- **Input**: Videos vi ngi c mt r rng
- **Output**: IDs (1, 2, 3, ...) vi Green boxes
- **Registry**: Populated vi face embeddings

### Cam2 & Cam3 (Verification)

- **Input**: Videos vi ngi i qua
- **Output**: 
  - Green boxes (VERIFIED) cho registered persons
  - Red boxes (UNK1, UNK2, ...) cho unknown persons

### Cam4 (Strict)

- **Input**: Videos  restricted area
- **Output**:
  - Green boxes (VERIFIED)
  - Red boxes (INTRUDER) + Alerts

---

**Ready to test! **

**Next**: Chy `sequential.bat cam1`  bt u!

