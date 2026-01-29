# HAVEN Multi-Camera Refactoring Guide

## File Changes Summary

### NEW FILES (Create These)
```
backend/src/core/stream_worker.py    # Per-camera worker with built-in inference
backend/src/core/stream_hub.py       # Multi-camera registry
```

### MODIFIED FILES
```
backend/src/api/routes.py            # Replace WebSocket logic (remove inference)
backend/src/main.py                  # Add startup: register 3 cameras
```

### DEPRECATED FILES (Can Remove After Testing)
```
backend/src/core/stream_manager.py   # Old singleton StreamManager
```

---

## Step-by-Step Refactoring Instructions

### STEP 1: Create New Core Files

#### 1.1 Create `backend/src/core/stream_worker.py`
Copy the complete StreamWorker code provided above. Key features:
- `StreamWorker` class with per-camera thread
- `CameraConfig` dataclass for settings
- `FramePacket` dataclass for processed frames
- `_worker_loop()` runs: capture → resize → inference → encode → store
- `get_latest_packet()` provides non-destructive reads

#### 1.2 Create `backend/src/core/stream_hub.py`
Copy the complete StreamHub code provided above. Key features:
- `StreamHub` singleton registry
- `register_camera()` to add cameras
- `get_worker(camera_id)` for lookups
- `start_all()` / `stop_all()` lifecycle management
- `get_stream_hub()` factory function

---

### STEP 2: Update Routes File

#### 2.1 Open `backend/src/api/routes.py`

**REMOVE these sections:**
```python
# OLD CODE TO DELETE:
# Any import of old StreamManager
from core.stream_manager import StreamManager  # DELETE THIS LINE

# Any code that does:
stream_mgr = StreamManager()  # DELETE
frame = stream_mgr.get_latest_frame()  # DELETE
annotated, detections = ai_engine.process_frame(frame)  # DELETE (inference moved to worker!)
```

**ADD these imports at the top:**
```python
import asyncio
import time
from stream_hub import get_stream_hub  # Or: from core.stream_hub import get_stream_hub
from stream_worker import FramePacket  # Or: from core.stream_worker import FramePacket
```

**REPLACE the WebSocket endpoint logic:**

Old pattern (WRONG - inference in WS loop):
```python
@router.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        frame = stream_mgr.get_latest_frame()  # ❌ Destructive read
        annotated, detections = ai_engine.process_frame(frame)  # ❌ Duplicate inference per client!
        jpeg = encode_jpeg(annotated)
        await websocket.send_json({"data": jpeg, ...})
```

New pattern (CORRECT - just read + send):
```python
@router.websocket("/ws/stream")
async def websocket_stream_legacy(websocket: WebSocket):
    await websocket.accept()
    hub = get_stream_hub()
    worker = hub.get_worker("cam1")  # Default to cam1
    
    frame_interval = 1.0 / FPS_LIMIT
    last_send_time = 0
    
    while True:
        packet = worker.get_latest_packet()  # ✓ Non-destructive read
        if packet is None:
            await asyncio.sleep(0.05)
            continue
        
        # FPS throttling
        if time.time() - last_send_time < frame_interval:
            await asyncio.sleep(0.01)
            continue
        
        last_send_time = time.time()
        
        # Send pre-processed packet (inference already done in worker!)
        await websocket.send_json({
            "type": "frame",
            "data": packet.frame_base64,
            "metadata": packet.metadata,
        })
```

**ADD new multi-camera endpoint:**
```python
@router.websocket("/ws/stream/{camera_id}")
async def websocket_stream_camera(websocket: WebSocket, camera_id: str):
    # Same logic as above, but use camera_id parameter
    # See routes.py code provided earlier
```

**ADD REST endpoint:**
```python
@router.get("/api/cameras")
async def get_cameras():
    hub = get_stream_hub()
    return {"cameras": hub.get_all_statuses()}
```

#### 2.2 Complete Routes Refactoring Checklist
- [ ] Remove all `StreamManager` imports
- [ ] Remove all `ai_engine.process_frame()` calls from WebSocket loops
- [ ] Remove `queue.get()` or destructive frame reads
- [ ] Add `get_stream_hub()` import
- [ ] Replace WebSocket logic with non-destructive `get_latest_packet()`
- [ ] Add `/ws/stream/{camera_id}` endpoint
- [ ] Add `/api/cameras` REST endpoint
- [ ] Add FPS throttling in WebSocket sender (not in worker)

---

### STEP 3: Update Main Application

#### 3.1 Open `backend/src/main.py`

**ADD imports:**
```python
from contextlib import asynccontextmanager
from stream_hub import get_stream_hub
from stream_worker import CameraConfig
```

**REPLACE old app initialization:**

Old (if you had this):
```python
app = FastAPI()
# Maybe some @app.on_event("startup") that initialized StreamManager
```

New:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    hub = get_stream_hub()
    
    # TODO: Set your AI engine here
    # hub.set_ai_engine(your_ai_engine_instance)
    
    # Register 3 cameras
    hub.register_camera(CameraConfig(
        camera_id="cam1",
        source=r"D:\HAVEN\backend\data\multi-camera\1.mp4",
        source_type="video_file",
        loop_video=True,
        ai_enabled=True,
        resize_width=640,
        jpeg_quality=80,
    ))
    hub.register_camera(CameraConfig(
        camera_id="cam2",
        source=r"D:\HAVEN\backend\data\multi-camera\2.mp4",
        source_type="video_file",
        loop_video=True,
        ai_enabled=True,
        resize_width=640,
        jpeg_quality=80,
    ))
    hub.register_camera(CameraConfig(
        camera_id="cam3",
        source=r"D:\HAVEN\backend\data\multi-camera\3.mp4",
        source_type="video_file",
        loop_video=True,
        ai_enabled=True,
        resize_width=640,
        jpeg_quality=80,
    ))
    
    hub.start_all()
    logger.info("✅ All cameras started")
    
    yield  # App runs
    
    # Shutdown
    hub.stop_all()
    logger.info("✅ All cameras stopped")

app = FastAPI(lifespan=lifespan)
```

#### 3.2 Integrate Your AI Engine

Find where you initialize your AI engine (YOLO model, etc.):
```python
# Example - adjust to your actual code:
from core.ai_engine import YOLOPoseEngine  # Your AI module

ai_engine = YOLOPoseEngine(model_path="path/to/model.pt")
hub.set_ai_engine(ai_engine)
```

Place this BEFORE `hub.start_all()` in the lifespan function.

#### 3.3 Main.py Refactoring Checklist
- [ ] Remove old StreamManager singleton initialization
- [ ] Add `lifespan` context manager
- [ ] Register 3 cameras with correct MP4 paths
- [ ] Set AI engine via `hub.set_ai_engine()`
- [ ] Call `hub.start_all()` on startup
- [ ] Call `hub.stop_all()` on shutdown

---

### STEP 4: Clean Up Old Code (After Testing)

Once everything works:

1. **Delete or archive** `backend/src/core/stream_manager.py`
2. **Remove any unused imports** referencing old StreamManager
3. **Search codebase** for:
   - `StreamManager(` → Should be zero results
   - `ai_engine.process_frame(` in routes → Should be zero results (only in worker now)

---

## Pseudocode: Core Logic Flow

### StreamWorker Thread Loop (Simplified)
```
function _worker_loop():
    cap = cv2.VideoCapture(source)
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        
        if not ret and loop_video:
            cap.set(CAP_PROP_POS_FRAMES, 0)  # Restart video
            continue
        
        # Resize
        if resize_width:
            frame = resize(frame, width=resize_width)
        
        # Inference (ONCE per frame - shared by all clients)
        if ai_enabled:
            frame, detections = ai_engine.process_frame(frame)
        else:
            detections = []
        
        # Encode
        jpeg_bytes = cv2.imencode('.jpg', frame, quality=80)
        frame_base64 = base64.encode(jpeg_bytes)
        
        # Store packet (atomic write)
        with lock:
            self.last_packet = FramePacket(
                frame_base64=frame_base64,
                detections=detections,
                metadata={camera_id, timestamp, fps, ...},
                timestamp=time.time()
            )
        
        # Update FPS stats
        update_fps_counter()
    
    cap.release()
```

### StreamHub Registry (Simplified)
```
class StreamHub:
    workers = {}  # camera_id -> StreamWorker
    
    function register_camera(config):
        worker = StreamWorker(config, ai_engine)
        workers[config.camera_id] = worker
    
    function start_all():
        for worker in workers.values():
            worker.start()  # Starts thread
    
    function get_worker(camera_id):
        return workers.get(camera_id)
    
    function stop_all():
        for worker in workers.values():
            worker.stop()  # Joins thread
```

### WebSocket Route (Simplified)
```
async function websocket_stream_camera(websocket, camera_id):
    worker = hub.get_worker(camera_id)
    
    while True:
        # Non-destructive read (no queue.get(), just read shared memory)
        packet = worker.get_latest_packet()
        
        if packet is None:
            await sleep(0.05)
            continue
        
        # FPS throttling (send at most 10fps to client)
        if time_since_last_send < (1/FPS_LIMIT):
            await sleep(small_delay)
            continue
        
        # Send pre-processed packet (inference already done!)
        await websocket.send_json({
            "type": "frame",
            "data": packet.frame_base64,  # Already JPEG + base64
            "metadata": packet.metadata   # Already includes detections
        })
```

---

## Configuration Notes

### Per-Camera Settings (CameraConfig)
```python
CameraConfig(
    camera_id="cam1",              # Unique identifier
    source="path/to/video.mp4",    # File path or RTSP URL
    source_type="video_file",      # "video_file" or "rtsp"
    loop_video=True,               # Restart MP4 when done
    ai_enabled=True,               # Run inference or not
    resize_width=640,              # Resize frames (None = no resize)
    jpeg_quality=80,               # JPEG compression (1-100)
)
```

### Global Settings (in routes.py or config file)
```python
FPS_LIMIT = 10  # Max FPS sent to WebSocket clients (worker decodes at native FPS)
```

### AI Engine Integration
Your AI engine must implement:
```python
class YourAIEngine:
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Args:
            frame: Input frame (BGR format from OpenCV)
        
        Returns:
            annotated_frame: Frame with visualizations
            detections: List of detection dicts
        """
        # Your inference code
        return annotated_frame, detections
```

---

## Testing the Refactor

### Test 1: Single Camera Verification
```bash
# Start backend
cd D:\HAVEN\backend
python -m uvicorn main:app --reload

# Check logs for:
# ✓ Registered: cam1 → D:\HAVEN\backend\data\multi-camera\1.mp4
# ✓ Started 3 camera workers
# [cam1] Worker started: ...
```

Expected logs:
```
INFO - Registered camera: cam1 → D:\HAVEN\backend\data\multi-camera\1.mp4
INFO - [cam1] Worker started: D:\HAVEN\backend\data\multi-camera\1.mp4
INFO - [cam1] Opened: 1920x1080 @ 30.0fps
```

### Test 2: REST API Check
```bash
# List all cameras
curl http://localhost:8000/api/cameras

# Expected response:
{
  "cameras": [
    {
      "camera_id": "cam1",
      "source": "D:\\HAVEN\\backend\\data\\multi-camera\\1.mp4",
      "source_type": "video_file",
      "connected": true,
      "fps": 29.7,
      "dropped_frames": 0,
      "uptime_seconds": 45.2,
      "ai_enabled": true
    },
    // ... cam2, cam3
  ],
  "total_cameras": 3
}
```

### Test 3: WebSocket Legacy Endpoint
Open browser console:
```javascript
const ws = new WebSocket("ws://localhost:8000/ws/stream");
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log("Frame from:", data.metadata.camera_id);  // Should be "cam1"
    console.log("Detections:", data.metadata.detections.length);
};
```

### Test 4: Multi-Camera WebSocket
```javascript
// Open 3 WebSocket connections (different cameras)
const ws1 = new WebSocket("ws://localhost:8000/ws/stream/cam1");
const ws2 = new WebSocket("ws://localhost:8000/ws/stream/cam2");
const ws3 = new WebSocket("ws://localhost:8000/ws/stream/cam3");

ws1.onmessage = (e) => console.log("CAM1:", JSON.parse(e.data).metadata.fps);
ws2.onmessage = (e) => console.log("CAM2:", JSON.parse(e.data).metadata.fps);
ws3.onmessage = (e) => console.log("CAM3:", JSON.parse(e.data).metadata.fps);
```

### Test 5: CPU Efficiency (Most Important!)
```bash
# Open Task Manager / htop
# Baseline: Start backend with 0 clients → Note CPU %

# Test A: Open 1 browser tab connecting to cam1
# → CPU should increase by X% (inference running)

# Test B: Open 2nd browser tab connecting to cam1
# → CPU should NOT double! Should increase only slightly (network overhead)
# This proves inference is shared!

# Test C: Open 3rd tab connecting to cam2
# → CPU increases by ~X% (new camera = new inference worker)
```

Expected CPU pattern:
- 1 client on cam1: ~15% CPU
- 2 clients on cam1: ~16% CPU (not 30%!)  ← Key test
- 1 client each on cam1, cam2, cam3: ~45% CPU (3× single camera)

### Test 6: Video Looping
Wait for video to finish (depends on your MP4 duration). Check logs:
```
INFO - [cam1] Video ended, looping...
INFO - [cam1] Opened: 1920x1080 @ 30.0fps
```
Video should restart automatically.

---

## Troubleshooting Guide

### Issue: "Camera cam1 not found"
**Cause:** Worker not registered or failed to start
**Fix:**
1. Check logs during startup for registration messages
2. Verify file paths exist: `D:\HAVEN\backend\data\multi-camera\1.mp4`
3. Check `hub.start_all()` was called

### Issue: Black frames or no detections
**Cause:** AI engine not set or inference failing
**Fix:**
1. Verify `hub.set_ai_engine(ai_engine)` was called
2. Check worker logs for inference errors
3. Test AI engine standalone: `ai_engine.process_frame(test_frame)`
4. Set `ai_enabled=False` temporarily to isolate issue

### Issue: High CPU usage with 1 client
**Cause:** Inference may be too slow, or decoding bottleneck
**Fix:**
1. Profile inference time: Add timing logs in worker
2. Reduce `resize_width` (e.g., 640 → 480)
3. Lower JPEG quality (e.g., 80 → 60)
4. Check if video codec is efficient (H.264 recommended)

### Issue: FPS drops over time
**Cause:** Memory leak or queue buildup
**Fix:**
1. Check `dropped_frames` in `/api/cameras` (should be ~0)
2. Monitor memory usage over time
3. Verify no OpenCV memory leaks (update opencv-python)
4. Check video file not corrupted

### Issue: WebSocket disconnects frequently
**Cause:** Network timeout or client-side issue
**Fix:**
1. Add WebSocket ping/pong heartbeat
2. Increase client timeout settings
3. Check network stability
4. Reduce `FPS_LIMIT` to lower bandwidth

---

## Migration Checklist

Use this to track your refactoring progress:

### Files
- [ ] Created `stream_worker.py`
- [ ] Created `stream_hub.py`
- [ ] Updated `routes.py` (removed inference from WS)
- [ ] Updated `main.py` (added lifespan + camera registration)
- [ ] Deleted/archived old `stream_manager.py`

### Code Changes
- [ ] Removed all `StreamManager` imports
- [ ] Removed `ai_engine.process_frame()` from WebSocket routes
- [ ] Added non-destructive `get_latest_packet()` reads
- [ ] Added `/ws/stream/{camera_id}` endpoint
- [ ] Added `/api/cameras` REST endpoint
- [ ] Registered 3 MP4 sources in `main.py`
- [ ] Set AI engine via `hub.set_ai_engine()`

### Testing
- [ ] Backend starts without errors
- [ ] All 3 cameras show in `/api/cameras`
- [ ] Legacy `/ws/stream` works (defaults to cam1)
- [ ] New `/ws/stream/cam2` and `/ws/stream/cam3` work
- [ ] CPU usage DOES NOT double with 2 clients on same camera
- [ ] Videos loop correctly when reaching end
- [ ] Detections appear in WebSocket payloads

### Verification
- [ ] No duplicate inference when multiple clients connect
- [ ] FPS stats reported correctly
- [ ] Logs show clean startup/shutdown
- [ ] No memory leaks after 10 minutes of streaming

---

## Next Steps (Future Enhancements)

After successful refactoring:

1. **RTSP Support:**
   ```python
   CameraConfig(
       camera_id="cam4",
       source="rtsp://username:password@ip:port/stream",
       source_type="rtsp",
       loop_video=False,  # RTSP doesn't loop
   )
   ```

2. **Dynamic Camera Management:**
   - Add `POST /api/cameras` to register new cameras at runtime
   - Add `DELETE /api/cameras/{camera_id}` to remove cameras

3. **Frontend Multi-Camera UI:**
   - Camera selector dropdown
   - Grid view showing all cameras simultaneously
   - Per-camera controls (toggle AI, change quality)

4. **Recording:**
   - Add FFmpeg writer in worker thread
   - Save clips when detections occur

5. **Advanced Features:**
   - Multi-model inference (pose + object detection)
   - Real-time alerts via WebSocket
   - Frame-by-frame playback controls

---

## Summary

This refactoring accomplishes:

✅ **Multi-camera support** (3 MP4 sources)
✅ **Shared inference** (CPU scales with cameras, not clients)
✅ **Non-destructive reads** (multiple consumers supported)
✅ **Clean architecture** (worker threads + hub registry)
✅ **Backward compatible** (legacy `/ws/stream` still works)
✅ **Production-ready** (proper startup/shutdown, error handling)

The key insight: **Move inference from WebSocket route (per-client) to worker thread (per-camera)** to eliminate wasteful duplicate processing.
