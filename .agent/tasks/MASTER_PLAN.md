# HAVEN Master Development Plan

This document consolidates the development roadmap, architectural designs, and implementation guides for the HAVEN project. It serves as the single source of truth for the AI Agent (SKILL.md).

---

## 1. Multi-Camera Architecture Refactoring

**Objective:** Transition from a single-camera singleton architecture to a multi-camera hub system with shared inference.
**Key Benefit:** CPU usage scales with camera count (O(cameras)), not client count. Multiple clients viewing the same camera share the same inference result.

### 1.1 Architecture Design
- **StreamWorker:** Per-camera thread. Handles capture -> resize -> inference -> storage.
- **StreamHub:** Registry singleton. Manages lifecycle of workers (`start_all`, `stop_all`).
- **Non-Destructive Reads:** Workers update a thread-safe `last_packet`. Clients read this packet without consuming it.

**Data Flow:**
```
[Worker Thread] --(write)--> [last_packet] <--(read)-- [WebSocket Client 1]
                                           <--(read)-- [WebSocket Client 2]
```

### 1.2 Implementation Guide

#### Step 1: Core Files
1.  **`backend/src/core/stream_worker.py`**: Create class `StreamWorker` handling the capture loop and AI inference interaction.
2.  **`backend/src/core/stream_hub.py`**: Create class `StreamHub` to register and manage `StreamWorker` instances.

#### Step 2: API & Routes (`backend/src/api/routes.py`)
-   **Remove**: `StreamManager` usage and direct inference calls in WebSocket loops.
-   **Add**: `get_stream_hub` and `StreamWorker` interactions.
-   **Endpoints**:
    -   `GET /api/cameras`: List status of all cameras.
    -   `WS /ws/stream/{camera_id}`: Stream specific camera.
    -   `WS /ws/stream`: Legacy endpoint (defaults to cam1).

#### Step 3: Main Entry Point (`backend/src/main.py`)
-   Use `lifespan` context manager.
-   Initialize `StreamHub`.
-   Register 3 cameras (MP4 sources).
-   Set AI engine: `hub.set_ai_engine(ai_instance)`.
-   Start/Stop hub on app startup/shutdown.

### 1.3 Verification Checklist
-   [ ] **Efficiency**: Opening 2+ tabs on the same camera does NOT significantly increase CPU usage.
-   [ ] **Functionality**: Videos loop correctly. `/api/cameras` returns valid status.
-   [ ] **Legacy Support**: `/ws/stream` still works.

---

## 2. ADL (Activities of Daily Living) System

**Objective:** Detect activities like sitting, standing, lying down, and critical events (falls).

### 2.1 Pipeline Architecture
`Pose Detection` -> `Keypoint Extraction` -> `Feature Calculation` -> `Rule Engine` -> `State Machine`

### 2.2 Implementation Status
#### Phase 1: Core Pipeline (Done)
-   [x] Module architecture designed.
-   [x] `ai_engine` updated for full keypoints.
-   [x] `PostureClassifier` implemented (Standing/Sitting/Laying).
-   [x] `RuleEngine` implemented (State Machine).
-   [x] `test_adl_video.py` created for validation.

#### Phase 2: Advanced Logic (Pending)
-   [ ] **Phone Detection**: Integrate object detection for phones.
-   [ ] **Hand Up Detection**: Logic using Relative Wrist vs Shoulder position.
-   [ ] **Zone Configuration**: Load zones from JSON.
-   [ ] **Light Sensor**: Integrate simulated/real sensor input.

#### Phase 3: Integration (Pending)
-   [ ] Integrate ADL logic into `StreamWorker` (or previous `StreamManager` equivalent for RTSP).
-   [ ] Debouncing/Cooldown for alerts.
-   [ ] Export Event Logs to CSV.

#### Phase 4: Frontend Visualization (Pending)
-   [ ] Color-coded Skeleton visualization.
-   [ ] Warning/Danger Overlays.
-   [ ] Event Dashboard.

---

## 3. Configuration References

### Camera Config Structure
```python
CameraConfig(
    camera_id="cam1",
    source=r"D:\HAVEN\backend\data\multi-camera\1.mp4",
    source_type="video_file",
    loop_video=True,
    ai_enabled=True,
    resize_width=640
)
```

### Future Extensions
-   RTSP Source support.
-   Dynamic Camera Registration (POST /api/cameras).
-   Recording functionality.
