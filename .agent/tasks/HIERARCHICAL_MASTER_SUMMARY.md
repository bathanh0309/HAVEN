# HAVEN Hierarchical Security System - Complete Solution

##  GII PHP HON CHNH

Ti  thit k li hon ton h thng HAVEN theo yu cu **Hierarchical Security** ca bn vi **Single Registration Point**.

---

##  KIN TRC H THNG

### Logic Phn Cp Cameras

```

 CAM1 (Cng) - REGISTRATION MASTER                          

  Bt buc phi ng k                                     
  Phi c face r rng (quality >= 0.8)                     
  Red box  Face detected  Green box + ID (1, 2, 3, ...)  
  CH cam1 c to ID mi                                  

                            
                             Registry (Global)
            
                                           
       
 CAM2 (Bi xe)                CAM3 (Thang my)     
 VERIFICATION                 VERIFICATION         
       
  Check registry              Check registry     
   ng k:                  ng k:        
   Green + ID c                Green + ID c      
  Cha ng k:               Cha ng k:      
   Red + UNK1,UNK2...           Red + UNK3,UNK4... 
       
                            
                            
                
                 CAM4 (Phng)          
                 STRICT VERIFICATION   
                
                  Check registry      
                   ng k:         
                   Green + ID c       
                  Cha ng k:       
                   Red + INTRUDER!     
                    Cnh bo xm nhp
                
```

---

##  KHC BIT VI DUAL-MASTER

### Dual-Master (Solution c)
-  Cam1 v Cam2 u to ID mi
-  Khng c registration point duy nht
-  Khng c face verification requirement

### Hierarchical (Solution mi) 
-  **CH Cam1 to ID mi** (strict registration)
-  **Bt buc face detection** trn Cam1
-  **Cam2, Cam3, Cam4 ch verify** (khng to ID)
-  **Unknown tracking** (UNK1, UNK2, ...)
-  **Intruder alerts** (Cam4)

---

##  FILES CREATED

### 1. Configuration
```
configs/hierarchical_security.yaml     Main config
```

**Key settings**:
```yaml
registration:
  registration_camera: 1              # Only Cam1
  
  face_quality:
    min_sharpness: 50
    min_face_width: 100
    frontal_angle_max: 30

cameras:
  - id: 1
    role: "registration"              # MASTER
  - id: 2  
    role: "verification"              # VERIFY ONLY
  - id: 3
    role: "verification"
  - id: 4
    role: "strict_verification"       # + Intruder alerts
```

### 2. Core Module
```
backend/src/global_id/hierarchical_manager.py     Main logic
```

**Class**: `HierarchicalIDManager`

**Methods**:
- `process_cam1_registration()` - Registration vi face verification
- `process_verification_camera()` - Verification cho cam2/3/4
- `match_against_registry()` - Match vi registry

### 3. Scripts
```
sequential.bat                         Test tng camera
multi_hierarchical.bat                Run all cameras
```

### 4. Documentation
```
HIERARCHICAL_QUICK_START.md           Complete guide
```

---

##  WORKFLOW (2 GIAI ON)

### GIAI ON 1: SEQUENTIAL (Testing)

**Purpose**: Test tng camera ring l

```bash
# Step 1: Process Cam1 (build registry)
sequential.bat cam1

# Expected:
# - Red boxes  Face detected  Green boxes
# - Registry populated: ID 1, 2, 3, ...

# Step 2: Process Cam2 (verify)
sequential.bat cam2

# Expected:
# - Green boxes (VERIFIED) for registered persons
# - Red boxes (UNK1, UNK2) for unknown persons

# Step 3: Process Cam3 (verify)
sequential.bat cam3

# Step 4: Process Cam4 (strict verify + alerts)
sequential.bat cam4

# Expected:
# - INTRUDER alerts for unknown persons
```

### GIAI ON 2: MULTI-CAMERA (Production)

**Purpose**: X l ng thi tt c cameras

```bash
multi_hierarchical.bat

# Expected:
# - 2x2 mosaic view
# - Real-time processing
# - Intruder alerts when detected
```

---

##  EXAMPLE FLOW

### Scenario: 3 ngi vo ta nh

```
=== T=0s: Cam1 (Gate) ===
Person A enters:
  [cam1] Track 1: No face  Red box "WAITING_FACE"
  [cam1] Track 1: Face detected (quality=0.85)  Green box
   [cam1] Track 1  REGISTERED as ID 1

Person B enters:
  [cam1] Track 2: Face detected (quality=0.92)  Green box
   [cam1] Track 2  REGISTERED as ID 2

Person C enters BUT face not clear:
  [cam1] Track 3: Face quality=0.50 (too low)  Orange box
    [cam1] Track 3  Not registered (poor quality)

=== T=30s: Cam2 (Parking) ===
Person A arrives:
  [cam2] Track 1: Matching registry (2 persons)...
   [cam2] Track 1  VERIFIED as ID 1 (score=0.78)
   Green box "ID 1 VERIFIED"

Person C arrives (not registered):
  [cam2] Track 2: Matching registry...
   [cam2] Track 2  UNKNOWN (UNK1, score=0.42)
   Red box "UNK1 UNKNOWN"

=== T=60s: Cam3 (Elevator) ===
Person A arrives:
  [cam3] Track 1: Matching registry...
   [cam3] Track 1  VERIFIED as ID 1 (score=0.81)
   Green box "ID 1 VERIFIED"

Person C arrives:
  [cam3] Track 2: Matching registry...
   [cam3] Track 2  UNKNOWN (UNK2, score=0.38)
   Red box "UNK2 UNKNOWN"

=== T=90s: Cam4 (Room - Strict) ===
Person A enters room:
  [cam4] Track 1: Matching registry...
   [cam4] Track 1  VERIFIED as ID 1 (score=0.79)
   Green box "ID 1 VERIFIED"

Person C tries to enter:
  [cam4] Track 2: Matching registry...
   [cam4] Track 2  INTRUDER! (UNK1)
   Red box "INTRUDER!" + Beep sound
   Log to security.log
   Alert notification
```

---

##  VISUALIZATION

### Color Codes

| Color | Meaning | Used On |
|-------|---------|---------|
|  Green | Registered & Verified | All cameras (ID 1, 2, 3, ...) |
|  Red | Unknown or Intruder | Cam2/3/4 (UNK1, UNK2, ...) |
|  Orange | Processing (waiting face) | Cam1 only |

### Text Labels

**Cam1 (Registration)**:
```

 T1                    Local track ID
 [WAITING_FACE]        Status
 Quality: 0.50         Face quality
   Red

 After face detected


 T1  ID 1           
 [REGISTERED]        
 Face: 0.85          
   Green
```

**Cam2/3 (Verification)**:
```

 T3  ID 1           
 [VERIFIED]          
 Score: 0.78         
   Green


 T5  UNK1           
 [UNKNOWN]           
 Score: 0.42         
   Red
```

**Cam4 (Strict)**:
```

 T2  UNK1           
 [INTRUDER!]         
  ALERT!           
   Red + Beep
```

---

##  CONFIGURATION TUNING

### Face Detection (Cam1)

**Too strict** (khng ai ng k c)?
```yaml
registration:
  face_quality:
    min_sharpness: 30        # Gim t 50
    frontal_angle_max: 45    # Tng t 30
    min_face_width: 80       # Gim t 100
```

**Too loose** (accept faces khng r)?
```yaml
registration:
  face_quality:
    min_sharpness: 70        # Tng t 50
    min_face_confidence: 0.9 # Tng t 0.8
```

### Verification Matching

**Too many UNK** (ngi  ng k b UNK)?
```yaml
reid:
  thresholds:
    registered_match: 0.65   # Gim t 0.70
```

**Too many false matches**?
```yaml
reid:
  thresholds:
    registered_match: 0.75   # Tng t 0.70
```

---

##  TESTING CHECKLIST

### Before Sequential

- [ ] Videos trong cam1/, cam2/, cam3/, cam4/?
- [ ] Cam1 videos c show face r rng?
- [ ] Config  edit data_root?

### After Cam1 Sequential

- [ ] Registry c t nht 1 person?
- [ ] Green boxes xut hin?
- [ ] Face quality >= 0.8?

### After Cam2/3/4 Sequential

- [ ] C VERIFIED persons (green)?
- [ ] C UNKNOWN persons (red UNK)?
- [ ] Intruder alerts trigger (cam4)?

### Before Multi-camera

- [ ] Sequential  chy OK cho 4 cameras?
- [ ] Registry  populated?
- [ ] Thresholds  tune?

---

##  METRICS

### Cam1 (Registration)

```
Total Tracks Processed: 10
  - Registered: 7 (ID 1-7)
  - Rejected (poor face): 3
  
Face Quality Distribution:
  - Excellent (>0.9): 4
  - Good (0.8-0.9): 3
  - Poor (<0.8): 3
```

### Cam2/3 (Verification)

```
Total Tracks Processed: 15
  - Verified: 10
  - Unknown: 5 (UNK1-UNK5)
  
Match Scores:
  - High (>0.75): 8
  - Medium (0.65-0.75): 2
  - Low (<0.65): 5
```

### Cam4 (Strict)

```
Total Tracks Processed: 8
  - Verified: 6
  - Intruders: 2 (UNK1-UNK2)
  
Security Alerts: 2
  - High priority: 2
```

---

##  ADVANTAGES

### 1. Strict Access Control 
- Ch ngi  ng k ti cng mi c vo
- Face verification bt buc
- Unknown persons c track

### 2. Hierarchical Security 
- Cam1: Entry control
- Cam2/3: Monitoring
- Cam4: Restricted area protection

### 3. Intruder Detection 
- Real-time alerts
- Red boxes for visual warning
- Logs for investigation

### 4. Flexible Configuration 
- Easy to tune thresholds
- Camera roles configurable
- Alert settings adjustable

---

##  NEXT STEPS

### Phase 1: Testing (Ngay)
1. Copy files vo HAVEN repo
2. Configure video paths
3. Run sequential.bat cam1
4. Verify registry built
5. Run sequential.bat cam2/3/4
6. Check verifications work

### Phase 2: Integration (1-2 ngy)
1. Integrate YOLO detection
2. Integrate face detection (RetinaFace/MTCNN)
3. Extract face embeddings
4. Test vi real videos

### Phase 3: Production (1 tun)
1. Fine-tune thresholds
2. Setup alerts (email/SMS)
3. Database logging
4. Performance optimization

---

##  FILES STRUCTURE

```
haven_hierarchical/
 configs/
    hierarchical_security.yaml            Config

 backend/src/
    global_id/
        hierarchical_manager.py           Core logic

 sequential.bat                            Test script
 multi_hierarchical.bat                    Run script

 HIERARCHICAL_QUICK_START.md               Guide
```

---

##  DELIVERABLES

- [x] Hierarchical ID Manager (single registration point)
- [x] Face verification requirement (Cam1)
- [x] Unknown tracking (UNK1, UNK2, ...)
- [x] Intruder alerts (Cam4)
- [x] Sequential + Multi-camera modes
- [x] Configuration file
- [x] Complete documentation
- [x] Batch scripts

---

##  READY TO USE!

**Start with**: `sequential.bat cam1`  build registry!

**Status**:  Complete & Ready for Testing

