# HAVEN Solutions Comparison

##  SO SNH 2 SOLUTIONS

Ti  to **2 solutions khc nhau** da trn 2 yu cu:

---

##  SOLUTION 1: DUAL-MASTER (Yu cu u tin)

### c im
- **Cam1 v Cam2 u l MASTER**
- C 2 u c quyn to Global ID mi
- Cam3, Cam4 ch c th MATCH
- Gallery shared globally

### Use Case
- H thng m, nhiu im vo
- Ngi c th vo t nhiu ch
- Khng c im ng k duy nht

### Logic
```
Cam1 (MASTER):
 Thy ngi mi  To ID 1
 Thy ngi t Cam2  Match ID t Cam2

Cam2 (MASTER):
 Thy ngi mi  To ID 2
 Thy ngi t Cam1  Match ID t Cam1

Cam3, Cam4 (NON-MASTER):
 Thy ngi mi  TEMP ID (i master)
 Thy ngi  ng k  Match ID
```

### Files
```
haven_refactor_complete/
 configs/multicam.yaml
 backend/src/global_id/manager.py
 run_multicam.bat
```

---

##  SOLUTION 2: HIERARCHICAL SECURITY (Yu cu mi) 

### c im
- **CH Cam1 l REGISTRATION MASTER**
- **Bt buc face detection** trn Cam1
- Cam2, Cam3, Cam4 ch VERIFY
- Unknown tracking (UNK1, UNK2, ...)
- Intruder alerts (Cam4)

### Use Case 
- **H thng kim sot an ninh ta nh**
- 1 im ng k duy nht (cng)
- Strict access control
- Pht hin xm nhp

### Logic
```
Cam1 (Gate - REGISTRATION):
 Red box  Detect face  Green box + ID 1, 2, 3...
 PHI c face r rng (quality >= 0.8)
 CH cam1 to ID mi

Cam2 (Parking - VERIFICATION):
 Check registry t Cam1
  ng k  Green box + ID c
 Cha ng k  Red box + UNK1, UNK2...

Cam3 (Elevator - VERIFICATION):
 Check registry t Cam1
  ng k  Green box + ID c
 Cha ng k  Red box + UNK3, UNK4...

Cam4 (Room - STRICT):
 Check registry t Cam1
  ng k  Green box + ID c
 Cha ng k  Red box + INTRUDER! 
```

### Files
```
haven_hierarchical/
 configs/hierarchical_security.yaml
 backend/src/global_id/hierarchical_manager.py
 sequential.bat           Test tng camera
 multi_hierarchical.bat   Run all cameras
```

---

##  BNG SO SNH CHI TIT

| Feature | Dual-Master | Hierarchical | Winner |
|---------|-------------|--------------|--------|
| **S Master Cameras** | 2 (Cam1, Cam2) | 1 (Cam1 only) | - |
| **Face Verification** |  Khng bt buc |  Bt buc trn Cam1 | Hierarchical |
| **Registration Point** | Multiple | Single (Cam1) | Hierarchical |
| **Unknown Tracking** |  Khng c |  UNK1, UNK2, ... | Hierarchical |
| **Intruder Alerts** |  Khng c |  Cam4 strict mode | Hierarchical |
| **Box Colors** | Green (matched)<br>Cyan (new)<br>Orange (temp) | Green (registered)<br>Red (unknown/intruder)<br>Orange (processing) | Hierarchical |
| **Sequential Mode** |  Khng c |  Test tng camera | Hierarchical |
| **Use Case** | Open system | Security system | - |
| **ID Format** | 1, 2, 3, ... | Registered: 1, 2, 3...<br>Unknown: UNK1, UNK2... | Hierarchical |

---

##  WHICH ONE TO USE?

### Dng DUAL-MASTER nu:
-  H thng m, khng c im kim sot duy nht
-  Ngi c th vo t nhiu ch
-  Khng cn face verification
-  Khng cn intruder detection

**Example**: Trung tm thng mi, cng vin cng cng

### Dng HIERARCHICAL nu: 
-  **H thng an ninh ta nh** (nh yu cu ca bn!)
-  **1 im ng k duy nht** (cng)
-  **Cn face verification** bt buc
-  **Cn detect intruders** (ngi khng ng k)
-  **Cn test sequential** trc khi chy ng b

**Example**: Ta nh vn phng, khu cng nghip, khu dn c

---

##  RECOMMENDED: HIERARCHICAL SECURITY

Da trn yu cu mi ca bn:
- Cam1  cng, bt buc ng k
- Face detection requirement
- Sequential testing (cam1  cam2  cam3  cam4)
- Intruder detection

 **Bn nn dng HIERARCHICAL SECURITY** (Solution 2)

---

##  DELIVERABLES

### Solution 1: Dual-Master
```
haven_refactor_complete.tar.gz (29 KB)
 Config: configs/multicam.yaml
 Manager: backend/src/global_id/manager.py
 Docs: README_IMPLEMENTATION.md
```

### Solution 2: Hierarchical Security 
```
haven_hierarchical_security.tar.gz
 Config: configs/hierarchical_security.yaml
 Manager: backend/src/global_id/hierarchical_manager.py
 Scripts: sequential.bat, multi_hierarchical.bat
 Docs: HIERARCHICAL_QUICK_START.md
```

---

##  CAN I USE BOTH?

**Yes!** C 2 solutions u hon chnh v c lp:

1. **For Production** (Security System):
   - Use **Hierarchical Security**
   - Copy `haven_hierarchical/` vo HAVEN repo

2. **For Experimentation**:
   - Keep **Dual-Master** as backup
   - Reference implementation

---

##  MIGRATION PATH

### From Dual-Master  Hierarchical

```bash
# 1. Extract hierarchical solution
tar -xzf haven_hierarchical_security.tar.gz

# 2. Copy to HAVEN repo
cp -r haven_hierarchical/* D:/HAVEN/

# 3. Update config
edit D:/HAVEN/configs/hierarchical_security.yaml

# 4. Test sequential
D:/HAVEN/sequential.bat cam1

# 5. Run multi-camera
D:/HAVEN/multi_hierarchical.bat
```

---

##  FEATURE COMPARISON

### Core Features

| Feature | Dual-Master | Hierarchical |
|---------|:-----------:|:------------:|
| Multi-camera support |  |  |
| Global ID management |  |  |
| Spatiotemporal filtering |  |  |
| Config-driven |  |  |
| Clean architecture |  |  |
| Type hints |  |  |
| Unit tests |  |  TODO |

### Security Features

| Feature | Dual-Master | Hierarchical |
|---------|:-----------:|:------------:|
| Face verification |  |  |
| Registration control |  |  |
| Unknown tracking |  |  |
| Intruder alerts |  |  |
| Sequential testing |  |  |
| Box color coding |  Limited |  Full |

### Workflow Features

| Feature | Dual-Master | Hierarchical |
|---------|:-----------:|:------------:|
| Sequential mode |  |  |
| Multi-camera mode |  |  |
| Batch scripts |  |  (2 modes) |
| Real-time preview |  |  |
| Video output |  |  |

---

##  KEY TAKEAWAYS

### Dual-Master
**Strengths**:
- Flexible (nhiu im vo)
- No strict registration
- Good for open systems

**Limitations**:
- No access control
- No unknown detection
- No intruder alerts

### Hierarchical Security 
**Strengths**:
- **Strict access control**
- **Face verification required**
- **Unknown & intruder detection**
- **Sequential testing workflow**
- **Perfect for security systems**

**Limitations**:
- Single registration point only
- Requires good face visibility at gate

---

##  RECOMMENDATION

**For Your Use Case** (Security System):

 **Use HIERARCHICAL SECURITY**

**Reasons**:
1.  Matches your workflow (cam1 gate registration)
2.  Face verification requirement
3.  Unknown tracking (UNK)
4.  Intruder detection (cam4)
5.  Sequential testing support

**Start with**:
```bash
sequential.bat cam1
```

---

**Both solutions ready to use!** 

