#  Cleanup Recommendations

## Should Keep 

### 1. `backend/shared_config.py`
**Recommendation**: **KEEP (for now)**

**L do:**
- File ny ang c dng bi cc test scripts c
- C th hu ch cho single-camera testing
- Khng conflict vi `multi_camera_config.yaml`

**Nu mun xa sau ny:**
- Chc chn khng cn chy `test-video/` hoc `test-rtsp-pose/`
- Migrate ton b sang `multi_camera_config.yaml`

---

### 2. `backend/test-video/` folder
**Recommendation**: **KEEP as reference**

**L do:**
- Cha working single-camera pose + ADL implementation
- C features hu ch:
  -  Pose classification vi majority voting
  -  ADL event detection (fall, hand raise, sitting/standing)
  -  Zone intrusion detection
  -  Dangerous object detection
  -  GIF export
- C th dng lm reference khi integrate ADL vo multi-camera system

**Scripts trong folder:**
- `pose-adl.py` (714 lines) - Complete single-camera ADL system
- `config.py` - Import t `shared_config.py`

**Use cases:**
- Quick testing vi 1 video
- ADL development/debugging
- Feature reference cho multi-camera

---

### 3. `backend/test-rtsp-pose/` folder
**Recommendation**: **KEEP as reference**

**L do:**
- Similar to test-video nhng cho RTSP
- C th dng test RTSP connection trc khi integrate vo multi-camera

---

## Folder Structure After Cleanup

```
backend/
 shared_config.py               KEEP (legacy test scripts)
 multi_camera_config.yaml       NEW (multi-camera system)
 run_multi_camera.py            NEW (main runner)

 test-video/                    KEEP (reference)
    pose-adl.py               # Single camera ADL
    config.py

 test-rtsp-pose/                KEEP (reference)
    pose-adl.py               # RTSP ADL
    config.py

 src/                           NEW (production code)
     core/
     reid/
     utils/
     storage/
```

---

## Migration Plan (Future)

### Phase 1: Current State
-  Multi-camera ReID system working (`run_multi_camera.py`)
-  Legacy test scripts available (`test-video/`, `test-rtsp-pose/`)
-  Both use different configs (no conflict)

### Phase 2: ADL Integration (Next steps)
1. Extract ADL logic from `test-video/pose-adl.py`
2. Create `src/adl/` module:
   - `posture_classifier.py`
   - `event_detector.py`
   - `zone_manager.py`
3. Integrate ADL into `run_multi_camera.py`
4. Test vi multi-camera

### Phase 3: Full Migration
1. Verify multi-camera system has all features
2. Update docs
3. **THEN**: Consider archiving/removing test folders

---

## Quick Decision

### If you want cleaner structure NOW:

**Option 1: Archive (Recommended)**
```bash
# Create archive folder
mkdir backend/archived_tests

# Move old tests
move backend/test-video backend/archived_tests/
move backend/test-rtsp-pose backend/archived_tests/
move backend/shared_config.py backend/archived_tests/
```

**Option 2: Keep as is**
```
# Do nothing - both systems coexist peacefully
# Use test-video/ for quick ADL testing
# Use run_multi_camera.py for ReID
```

---

## My Recommendation

**KEEP everything for now** because:
1.  No conflicts between old & new systems
2.  `test-video/` has valuable ADL features you'll need
3.  Good reference when integrating ADL into multi-camera
4.  Can quickly test single features without full system

**When to delete:**
-  After you've fully integrated ADL into multi-camera system
-  After you've migrated all useful features
-  When you're 100% confident with new system

---

## Summary Table

| File/Folder | Status | Action | Reason |
|-------------|--------|--------|--------|
| `shared_config.py` |  Active | **KEEP** | Used by test scripts |
| `test-video/` |  Legacy | **KEEP as reference** | ADL features needed |
| `test-rtsp-pose/` |  Legacy | **KEEP as reference** | RTSP testing |
| `multi_camera_config.yaml` |  Active | **Primary config** | New system |
| `run_multi_camera.py` |  Active | **Primary runner** | New system |

---

**Bottom line**: Gi li tt c! Khng c l do phi xa ngay by gi. Chng khng chim nhiu dung lng v rt hu ch lm reference.

