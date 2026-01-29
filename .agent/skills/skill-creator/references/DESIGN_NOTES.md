# HAVEN Multi-Camera Refactoring Design

## Executive Summary
Refactored HAVEN from single-camera singleton to multi-camera hub architecture with **shared inference per camera** to eliminate duplicate CPU waste when multiple clients connect.

---

## 1. Why Inference Moved to Worker Thread

### Problem
**Before:** Inference ran in WebSocket route
```
WebSocket Client Loop (per connection):
  frame = stream_mgr.get_latest_frame()
  annotated, detections = ai_engine.process_frame(frame)  ← DUPLICATED!
  send to client
```
- 2 browser tabs = 2x inference = 2x CPU usage
- N clients = N × inference cost

### Solution
**After:** Inference runs once per camera in worker thread
```
Camera Worker Thread (one per camera):
  frame = capture.read()
  annotated, detections = ai_engine.process_frame(frame)  ← ONCE!
  store as last_packet

WebSocket Client Loop (per connection):
  packet = hub.get_latest_packet(camera_id)  ← READ-ONLY
  send to client
```
- N clients = 1 × inference cost
- CPU scales with cameras, not clients

---

## 2. Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        StreamHub                            │
│  (Registry: cam1 → Worker1, cam2 → Worker2, cam3 → Worker3)│
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
  ┌──────────┐          ┌──────────┐          ┌──────────┐
  │ Worker1  │          │ Worker2  │          │ Worker3  │
  │  cam1    │          │  cam2    │          │  cam3    │
  └──────────┘          └──────────┘          └──────────┘
        │                     │                     │
   [Thread Loop]         [Thread Loop]         [Thread Loop]
        │                     │                     │
        ├─ cv2.VideoCapture(1.mp4, loop=True)
        ├─ Resize (optional)
        ├─ AI Inference (ONCE per frame)
        ├─ JPEG Encode
        └─ Store: self.last_packet = {
             frame_base64, metadata, detections, timestamp
           }
        │
        │ (Non-destructive read)
        ▼
  ┌────────────────────────────────────────┐
  │  Multiple WebSocket Clients            │
  │  /ws/stream/cam1  /ws/stream/cam2 ...  │
  │  (Just copy last_packet and send)      │
  └────────────────────────────────────────┘
```

### Key Design Decisions

1. **Non-Destructive Reads**
   - Old: `queue.get()` consumed frame (only 1 client worked)
   - New: `last_packet` stored in memory, protected by threading.Lock
   - Multiple clients read same packet concurrently

2. **Latest Frame Strategy Preserved**
   - Workers decode at native FPS (e.g., 30fps)
   - Always overwrite `last_packet` with newest frame
   - Clients throttled by `FPS_LIMIT` (e.g., 10fps) in WebSocket sender
   - Maintains low latency, drops frames intentionally

3. **Lifecycle**
   - Workers start on app startup, run continuously
   - MP4 sources loop automatically (`cap.set(cv2.CAP_PROP_POS_FRAMES, 0)`)
   - Workers stop gracefully on shutdown (Event flag + thread join)

---

## 3. Component Responsibilities

### `StreamWorker` (per camera)
- **Capture:** Read frames from source (MP4 file or RTSP later)
- **Resize:** Optional resize to config width
- **Inference:** Call `ai_engine.process_frame()` once per frame
- **Encode:** JPEG compression (quality configurable)
- **Store:** Update `self.last_packet` atomically (thread-safe)
- **Stats:** Track FPS, dropped frames, uptime

### `StreamHub` (singleton registry)
- **Register:** Add cameras with `camera_id` + `source_config`
- **Lookup:** `get_worker(camera_id)` returns StreamWorker instance
- **Lifecycle:** Start all workers, stop all on shutdown
- **Status:** Aggregate stats from all workers for `/api/cameras`

### WebSocket Routes
- **`/ws/stream`:** Legacy endpoint (defaults to `cam1`)
- **`/ws/stream/{camera_id}`:** New multi-camera endpoint
- **Logic:** 
  1. Lookup worker via hub
  2. Read `last_packet` (non-blocking)
  3. Apply FPS throttling
  4. Send JSON payload to client

### REST Endpoint
- **`GET /api/cameras`:** Returns list of camera statuses
  ```json
  [
    {
      "camera_id": "cam1",
      "source": "D:\\HAVEN\\backend\\data\\multi-camera\\1.mp4",
      "connected": true,
      "fps": 29.7,
      "dropped_frames": 0,
      "uptime_seconds": 123.4
    }
  ]
  ```

---

## 4. Configuration Per Camera

```python
camera_config = {
    "camera_id": "cam1",
    "source": "D:\\HAVEN\\backend\\data\\multi-camera\\1.mp4",
    "source_type": "video_file",  # or "rtsp"
    "loop_video": True,           # restart MP4 when done
    "ai_enabled": True,
    "resize_width": 640,
    "jpeg_quality": 80,
}
```

---

## 5. Backward Compatibility

- `/ws/stream` still works → auto-routes to `cam1`
- Existing frontend code unchanged (until ready to add camera selector)
- Same payload format (added `camera_id` field in metadata)

---

## 6. CPU Optimization Results

| Scenario                  | Before (inference in WS) | After (inference in worker) |
|---------------------------|--------------------------|------------------------------|
| 1 client, 1 camera        | 1× inference             | 1× inference ✓              |
| 3 clients, 1 camera       | 3× inference ❌          | 1× inference ✓              |
| 1 client, 3 cameras       | 3× inference             | 3× inference ✓              |
| 3 clients, 3 cameras      | 9× inference ❌          | 3× inference ✓              |

**Savings:** O(clients × cameras) → O(cameras)

---

## 7. Future Extensions (Not In Scope)

- RTSP sources: Change `source_type`, add reconnection logic
- Dynamic camera add/remove: REST API to register new cameras
- Recording: Add FFmpeg writer in worker thread
- Multi-model inference: Pass `model_type` config per camera
- WebRTC: Replace WebSocket with lower latency protocol

---

## 8. File Structure

```
backend/
├── src/
│   ├── core/
│   │   ├── stream_worker.py    # StreamWorker class (capture + inference loop)
│   │   └── stream_hub.py       # StreamHub registry (multi-camera manager)
│   ├── api/
│   │   └── routes.py           # WebSocket + /api/cameras endpoint
│   └── main.py                 # FastAPI app + startup (register 3 cameras)
```

Minimal files, maximum clarity.

---

## Summary
This refactor eliminates the "inference per client" anti-pattern by moving AI processing into dedicated per-camera worker threads. The hub-worker architecture scales efficiently: CPU cost grows with camera count (inevitable), not client count (wasteful). The system maintains low latency via latest-frame strategy and supports future multi-camera UIs seamlessly.
