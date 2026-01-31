# HAVEN Multi-Camera ReID - Quick Start

##  CHY NGAY (3 BC)

### Bc 1: Copy Files

Copy ton b ni dung t `haven_refactor/` vo repo HAVEN ca bn:

```
HAVEN/
 configs/multicam.yaml               COPY
 backend/src/
    io/video_stream.py              COPY
    global_id/manager.py            COPY
    run_multicam_reid.py            COPY
 tests/test_global_id_manager.py     COPY
 run_multicam.bat                    COPY
 README_IMPLEMENTATION.md            COPY
```

### Bc 2: Configure

Edit `configs/multicam.yaml`:

```yaml
data:
  data_root: "D:/HAVEN/backend/data/multi-camera"  # ng dn n videos
  
master_cameras:
  ids: [1, 2]  # Cam1 v Cam2 l MASTER
```

### Bc 3: Run!

```bash
cd D:\HAVEN
run_multicam.bat
```

Hoc:

```bash
python backend\src\run_multicam_reid.py --config configs\multicam.yaml
```

---

##  EXPECTED OUTPUT

### Console

```
============================================================
HAVEN Multi-Camera ReID System
============================================================
Master cameras: [1, 2]
Total cameras: 4
============================================================

 Starting multi-camera processing...

 [cam1] Track 1  Global ID 1 (MASTER_NEW: empty_gallery)
 [cam2] Track 3  Global ID 1 (MATCHED: score=0.82)
 [cam2] Track 5  Global ID 2 (MASTER_NEW: low_similarity=0.45)
 [cam3] Track 2  TEMP (non-master, wait for master)
 [cam3] Track 2  Global ID 1 (MATCHED: score=0.78)
```

### Video Outputs

```
D:/HAVEN/backend/outputs/multicam/
 cam1_output.mp4
 cam2_output.mp4
 cam3_output.mp4
 cam4_output.mp4
 mosaic_output.mp4  (2x2 grid)
```

---

##  KEY POINTS

### Dual-Master Logic (3 Bullets)

1. **Cam1 v Cam2 = MASTER** c quyn to Global ID mi
   - Khi thy ngi mi (khng match)  to ID mi (1, 2, 3, ...)

2. **Cam3 v Cam4 = NON-MASTER** ch c th MATCH
   - Khi thy ngi mi  gn TEMP ID = 0, i master camera

3. **Gallery c share ton b**
   - Cam2 to ID  Cam1 thy c
   - Cam1 to ID  Cam2 thy c
   -  Khng bao gi duplicate IDs

### Flow Example

```
Scenario: Ngi i xe my vo thng Cam2

t=0s  : Cam2 (MASTER) thy person A
         So snh gallery (rng)
         To Global ID = 1 

t=10s : Cam1 (MASTER) thy person A
         So snh gallery [ID=1 by Cam2]
         Similarity = 0.82 > 0.75
         MATCH  Global ID = 1  (khng to mi!)

t=20s : Cam3 (NON-MASTER) thy person A
         So snh gallery [ID=1]
         Similarity = 0.78 > 0.75
         MATCH  Global ID = 1 

t=30s : Cam3 (NON-MASTER) thy person B (mi)
         So snh gallery [ID=1]
         Similarity = 0.40 < 0.50
         Cam3 KHNG C QUYN to ID
         Gn TEMP ID = 0 

t=40s : Cam1 (MASTER) thy person B
         So snh gallery [ID=1]
         Similarity = 0.42 < 0.50
         Cam1 to Global ID = 2 

t=50s : Cam3 (NON-MASTER) thy li person B
         So snh gallery [ID=1, ID=2]
         Similarity vi ID=2 = 0.80 > 0.75
         MATCH  Global ID = 2 
```

---

##  TUNING

### Too Many False Matches?
```yaml
reid:
  thresholds:
    accept: 0.80  # Tng t 0.75
```

### Too Many ID Switches?
```yaml
reid:
  thresholds:
    accept: 0.70  # Gim t 0.75
```

---

##  FULL DOCS

- **README_IMPLEMENTATION.md**: Chi tit implementation
- **REFACTOR_PLAN.md**: Architecture & integration guide
- **configs/multicam.yaml**: All settings with comments

---

##  FAQ

**Q: Lm sao  Cam3 cng thnh master?**
```yaml
master_cameras:
  ids: [1, 2, 3]
```

**Q: C th ch c 1 master khng?**
```yaml
master_cameras:
  ids: [1]  # Only Cam1
```

**Q: C th tt c u l master khng?**
```yaml
master_cameras:
  ids: [1, 2, 3, 4]  # All are masters
```

**Note**: Nu tt c u l master, s khng c TEMP IDs.

---

**Ready to run! **

