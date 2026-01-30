# HAVEN Multi-Camera Person Re-Identification System
## Architecture Design & Refactoring Plan

**Version:** 1.0  
**Date:** January 30, 2026  
**Project:** HAVEN (Python FastAPI + OpenCV + WebSocket + Ultralytics YOLO + pose/ADL)

---

## Executive Summary

This document outlines a comprehensive refactoring plan for the HAVEN multi-camera person re-identification system. The goal is to transform the current offline step-based pipeline into a production-ready, configurable system that supports both offline testing and real-time RTSP deployment with robust global person ID assignment.

**Key Improvements:**
- Unified configuration system for video/RTSP sources
- Database-backed global ID management with multi-signal fusion
- Clothing change resilience through multi-prototype embeddings
- Prevention of look-alike false matches via biometric gating
- Clean modular architecture separating core libraries from experiments

---

## 1. System Architecture

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HAVEN Multi-Camera System                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Camera 1   │  │   Camera 2   │  │   Camera 3   │     │
│  │ (Video/RTSP) │  │ (Video/RTSP) │  │ (Video/RTSP) │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │              │
│         └──────────────────┴──────────────────┘              │
│                            │                                 │
│         ┌──────────────────▼──────────────────┐             │
│         │    Source Manager (sources.yaml)     │             │
│         └──────────────────┬──────────────────┘             │
│                            │                                 │
│  ┌─────────────────────────┴──────────────────────────┐    │
│  │              Per-Camera Pipeline                    │    │
│  │  ┌─────────────────────────────────────────────┐   │    │
│  │  │  1. Frame Acquisition & Preprocessing       │   │    │
│  │  └─────────────────┬───────────────────────────┘   │    │
│  │  ┌─────────────────▼───────────────────────────┐   │    │
│  │  │  2. YOLO Detection (person class)           │   │    │
│  │  └─────────────────┬───────────────────────────┘   │    │
│  │  ┌─────────────────▼───────────────────────────┐   │    │
│  │  │  3. Local Tracking (ByteTrack/BoT-SORT)     │   │    │
│  │  │     → Generates local_track_id per camera   │   │    │
│  │  └─────────────────┬───────────────────────────┘   │    │
│  │  ┌─────────────────▼───────────────────────────┐   │    │
│  │  │  4. Pose Estimation (optional)              │   │    │
│  │  └─────────────────┬───────────────────────────┘   │    │
│  │                    │                                │    │
│  └────────────────────┼────────────────────────────────┘    │
│                       │                                      │
│         ┌─────────────▼──────────────┐                      │
│         │  Tracklet Aggregator       │                      │
│         │  (Multi-frame sampling)    │                      │
│         └─────────────┬──────────────┘                      │
│                       │                                      │
│         ┌─────────────▼──────────────────────┐              │
│         │     Global ID Manager (SQLite)      │              │
│         │  ┌────────────────────────────────┐ │              │
│         │  │  Multi-Signal ReID Engine:     │ │              │
│         │  │  • Face embeddings             │ │              │
│         │  │  • Gait/pose embeddings        │ │              │
│         │  │  • Appearance embeddings       │ │              │
│         │  │  • Multi-prototype matching    │ │              │
│         │  │  • Open-set decision           │ │              │
│         │  └────────────────────────────────┘ │              │
│         └─────────────┬──────────────────────┘              │
│                       │                                      │
│         ┌─────────────▼──────────────┐                      │
│         │   ADL Event Engine (FSM)   │                      │
│         │   • Fall detection         │                      │
│         │   • Bed exit alerts        │                      │
│         │   • Event throttling       │                      │
│         └─────────────┬──────────────┘                      │
│                       │                                      │
│         ┌─────────────▼──────────────┐                      │
│         │   WebSocket/MJPEG Output   │                      │
│         │   • Frame + metadata       │                      │
│         │   • Tracks with global_id  │                      │
│         │   • Throttled events       │                      │
│         └────────────────────────────┘                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Data Flow

```
Camera Stream → Frame Buffer → Detection → Local Tracking → Tracklet Aggregation
                                                                      ↓
                                                            [Quality Gate: blur/size/face check]
                                                                      ↓
                                                            Feature Extraction:
                                                            • Face (if visible)
                                                            • Gait (skeleton sequence)
                                                            • Appearance (last resort)
                                                                      ↓
                                                            Global ID Association:
                                                            1. Query existing identities
                                                            2. Multi-prototype similarity
                                                            3. Signal fusion & gating
                                                            4. Threshold check
                                                            5. Assign/Create global_id
                                                                      ↓
                                                            Update SQLite DB:
                                                            • Add observation
                                                            • Update embeddings
                                                            • Refresh timestamp
                                                                      ↓
                                                            ADL Analysis → Event Generation
                                                                      ↓
                                                            WebSocket Broadcast
```

---

## 2. Folder Structure

### 2.1 Production Core (`backend/src/`)

```
backend/src/
├── config/
│   ├── __init__.py
│   ├── sources.yaml              # Main config (cameras, models, thresholds)
│   └── settings.py                # Pydantic settings loader
│
├── core/
│   ├── __init__.py
│   ├── detector.py                # YOLO wrapper
│   ├── tracker.py                 # Unified ByteTrack/BoT-SORT wrapper
│   ├── pose_estimator.py          # Pose/skeleton extraction
│   └── video_source.py            # Video/RTSP stream handler
│
├── reid/
│   ├── __init__.py
│   ├── global_id_manager.py       # Main Global ID orchestrator
│   ├── feature_extractors/
│   │   ├── __init__.py
│   │   ├── face_extractor.py      # Face embedding (InsightFace/etc)
│   │   ├── gait_extractor.py      # Pose sequence → gait embedding
│   │   └── appearance_extractor.py # OSNet/CLIP appearance
│   ├── tracklet_aggregator.py     # Multi-frame sampling & quality scoring
│   ├── multi_prototype.py         # Multi-prototype memory bank
│   └── similarity.py              # Distance metrics & fusion
│
├── storage/
│   ├── __init__.py
│   ├── db_schema.py               # SQLAlchemy models
│   └── db_manager.py              # DB operations (CRUD)
│
├── adl/
│   ├── __init__.py
│   └── adl_engine.py              # FSM for fall/bed-exit/etc
│
├── api/
│   ├── __init__.py
│   ├── websocket_handler.py       # WebSocket server
│   └── mjpeg_handler.py           # Optional MJPEG fallback
│
└── utils/
    ├── __init__.py
    ├── logger.py
    └── metrics.py                 # ID-switch, false-match counters
```

### 2.2 Experiments (`backend/experiments/`)

```
backend/experiments/
├── multicam_offline/
│   ├── reference/                 # Keep old step1-step6 as reference only
│   │   ├── step1_detect.py
│   │   ├── step2_track.py
│   │   ├── ...
│   │   └── step6_global_tracking.py
│   │
│   ├── run_offline.py             # Main offline runner (uses sources.yaml)
│   ├── test_reid_quality.py       # Metrics: ID-switch, false-match rate
│   └── visualize_results.py       # Visualization tools
│
└── README.md                      # Experiment documentation
```

### 2.3 Configuration Files

```
backend/
├── config/
│   ├── sources.example.yaml       # Template with comments
│   └── sources.yaml               # Actual config (gitignored)
│
├── .env.example                   # RTSP credentials template
├── .env                           # Actual secrets (gitignored)
└── .gitignore                     # Block .env, sources.yaml, *.pkl, credentials
```

---

## 3. Configuration System

### 3.1 `sources.yaml` Structure

```yaml
# HAVEN Multi-Camera Configuration
# Version: 1.0

general:
  fps_limit: 30
  resize_width: 640          # Resize frames for inference
  log_level: INFO
  output_dir: "/mnt/user-data/outputs"

# Camera sources (list)
cameras:
  - id: cam1
    name: "Living Room"
    type: video                # video | rtsp
    source: "/path/to/video1.mp4"
    enabled: true
    
  - id: cam2
    name: "Bedroom"
    type: video
    source: "/path/to/video2.mp4"
    enabled: true
    
  - id: cam3
    name: "Kitchen"
    type: rtsp
    source: "${RTSP_CAM3_URL}"  # From .env
    username: "${RTSP_USERNAME}"
    password: "${RTSP_PASSWORD}"
    enabled: false              # Disabled for offline testing

# Inference models
inference:
  yolo:
    model: "yolov8n.pt"         # yolov8n/yolov8s/yolov8m
    conf_threshold: 0.5
    device: "cuda"              # cuda | cpu | mps
    
  pose:
    enabled: true
    model: "yolov8n-pose.pt"
    conf_threshold: 0.3
    
  tracker:
    type: "bytetrack"           # bytetrack | botsort
    persist: true
    track_high_thresh: 0.5
    track_low_thresh: 0.1

# Global Re-ID settings
reid:
  enabled: true
  
  # Feature extractors
  face:
    enabled: true
    model: "buffalo_l"          # InsightFace model
    min_face_size: 40           # pixels
    quality_threshold: 0.7
    
  gait:
    enabled: true
    sequence_length: 15         # frames for gait pattern
    min_motion: 5.0             # min movement (pixels)
    
  appearance:
    enabled: false              # Start with OFF
    model: "osnet_x1_0"         # osnet_x1_0 | clip
    
  # Association thresholds
  thresholds:
    face_similarity: 0.6        # Cosine similarity (higher = stricter)
    gait_similarity: 0.7
    appearance_similarity: 0.5
    unknown_threshold: 0.4      # Below this → create new ID
    
  # Multi-prototype settings
  multi_prototype:
    memory_size: 10             # Keep top-M embeddings per ID
    ema_alpha: 0.3              # Exponential moving average weight
    embedding_ttl: 3600         # Seconds before embeddings expire
    
  # Quality gating
  quality:
    min_bbox_size: 80           # min person height (pixels)
    max_blur_variance: 100      # Laplacian variance threshold
    min_tracklet_frames: 5      # frames before attempting ReID

# ADL (Activity of Daily Living) detection
adl:
  enabled: true
  throttle_window: 20           # seconds between same event type
  
  fall_detection:
    enabled: true
    min_ground_duration: 2.0    # seconds on ground → fall alert
    
  bed_exit:
    enabled: true
    bed_zone: [100, 100, 500, 400]  # bbox defining bed area
    
# Storage
storage:
  db_path: "/home/claude/haven_reid.db"
  save_snapshots: true
  snapshot_dir: "/mnt/user-data/outputs/snapshots"
  max_db_size_mb: 1000          # Auto-cleanup threshold

# WebSocket/API
api:
  websocket:
    enabled: true
    host: "0.0.0.0"
    port: 8765
    
  mjpeg:
    enabled: false              # Fallback only
    port: 8080
```

### 3.2 `.env` Structure

```bash
# RTSP Credentials - NEVER commit this file
RTSP_USERNAME=admin
RTSP_PASSWORD=secure_password_here
RTSP_CAM3_URL=rtsp://192.168.1.100:554/stream1

# Optional: Database encryption key
DB_ENCRYPTION_KEY=your-secret-key-here
```

### 3.3 `.gitignore` Additions

```gitignore
# Secrets & credentials
.env
*.env
!.env.example

# Config files with actual paths/secrets
backend/config/sources.yaml
!backend/config/sources.example.yaml

# Legacy pickles (deprecate)
global_tracks.pkl
*.pkl

# Database
*.db
*.sqlite
*.db-journal

# Snapshots & outputs
snapshots/
outputs/*.jpg
outputs/*.png
```

---

## 4. Global ID Management

### 4.1 Database Schema (SQLite)

```sql
-- Identities table
CREATE TABLE identities (
    global_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'active',  -- active | merged | archived
    merged_into INTEGER,            -- If merged, points to new ID
    FOREIGN KEY (merged_into) REFERENCES identities(global_id)
);

-- Observations table
CREATE TABLE observations (
    obs_id INTEGER PRIMARY KEY AUTOINCREMENT,
    global_id INTEGER NOT NULL,
    camera_id TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    bbox_x INTEGER,
    bbox_y INTEGER,
    bbox_w INTEGER,
    bbox_h INTEGER,
    posture TEXT,                   -- standing | sitting | lying | unknown
    snapshot_path TEXT,             -- Optional image path
    local_track_id INTEGER,         -- Original tracker ID
    FOREIGN KEY (global_id) REFERENCES identities(global_id)
);
CREATE INDEX idx_obs_global_id ON observations(global_id);
CREATE INDEX idx_obs_camera_time ON observations(camera_id, timestamp);

-- Embeddings table
CREATE TABLE embeddings (
    emb_id INTEGER PRIMARY KEY AUTOINCREMENT,
    global_id INTEGER NOT NULL,
    emb_type TEXT NOT NULL,         -- face | gait | appearance
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    vector BLOB NOT NULL,           -- Pickled numpy array or raw bytes
    quality_score REAL,             -- 0.0-1.0
    source_camera TEXT,
    FOREIGN KEY (global_id) REFERENCES identities(global_id)
);
CREATE INDEX idx_emb_global_id ON embeddings(global_id);
CREATE INDEX idx_emb_type ON embeddings(emb_type);

-- Optional: Metadata for tracklet quality
CREATE TABLE tracklet_quality (
    tracklet_id INTEGER PRIMARY KEY AUTOINCREMENT,
    global_id INTEGER,
    camera_id TEXT,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    avg_bbox_size REAL,
    avg_blur_score REAL,
    frame_count INTEGER,
    FOREIGN KEY (global_id) REFERENCES identities(global_id)
);
```

### 4.2 Global ID Manager Workflow

```python
class GlobalIDManager:
    """
    Main orchestrator for global person re-identification.
    
    Workflow:
    1. Receive tracklet from local tracker
    2. Aggregate features across multiple frames
    3. Query existing identities
    4. Multi-signal fusion & gating
    5. Decision: match existing ID or create new
    6. Update database
    """
    
    def __init__(self, config, db_manager, extractors):
        self.config = config
        self.db = db_manager
        self.face_extractor = extractors['face']
        self.gait_extractor = extractors['gait']
        self.appearance_extractor = extractors['appearance']
        self.multi_prototype = MultiPrototypeMemory(config)
        
    def process_tracklet(self, tracklet_data):
        """
        Main entry point for global ID assignment.
        
        Args:
            tracklet_data: {
                'camera_id': str,
                'local_track_id': int,
                'frames': List[frame_data],  # Multiple frames
                'bbox': [x, y, w, h],
                'timestamp': float
            }
        
        Returns:
            global_id: int
        """
        # Step 1: Quality check
        if not self._check_quality(tracklet_data):
            return None  # Skip poor quality tracklets
        
        # Step 2: Extract features
        features = self._extract_features(tracklet_data)
        
        # Step 3: Query existing identities
        candidates = self.db.get_all_active_identities()
        
        if not candidates:
            # First person ever seen
            return self._create_new_identity(tracklet_data, features)
        
        # Step 4: Multi-signal similarity
        similarities = self._compute_similarities(features, candidates)
        
        # Step 5: Gating & decision
        best_match = self._apply_gating_decision(similarities)
        
        if best_match is None:
            # No match above threshold → create new
            return self._create_new_identity(tracklet_data, features)
        else:
            # Match found → update existing
            return self._update_existing_identity(best_match, tracklet_data, features)
    
    def _extract_features(self, tracklet_data):
        """Extract face, gait, appearance features with quality scores."""
        features = {}
        
        # Face
        if self.config.reid.face.enabled:
            face_emb, face_quality = self.face_extractor.extract(tracklet_data)
            if face_quality > self.config.reid.face.quality_threshold:
                features['face'] = {
                    'embedding': face_emb,
                    'quality': face_quality,
                    'available': True
                }
        
        # Gait (from pose sequence)
        if self.config.reid.gait.enabled:
            gait_emb, gait_quality = self.gait_extractor.extract(tracklet_data)
            if gait_quality > 0.5:  # Sufficient motion
                features['gait'] = {
                    'embedding': gait_emb,
                    'quality': gait_quality,
                    'available': True
                }
        
        # Appearance (last resort)
        if self.config.reid.appearance.enabled:
            app_emb, app_quality = self.appearance_extractor.extract(tracklet_data)
            features['appearance'] = {
                'embedding': app_emb,
                'quality': app_quality,
                'available': True
            }
        
        return features
    
    def _compute_similarities(self, query_features, candidate_ids):
        """
        Compute similarity scores using multi-prototype matching.
        
        Returns:
            List[{
                'global_id': int,
                'face_sim': float | None,
                'gait_sim': float | None,
                'appearance_sim': float | None,
                'combined_score': float,
                'confidence': str  # high | medium | low
            }]
        """
        results = []
        
        for cand_id in candidate_ids:
            # Get all embeddings for this candidate
            stored_embeddings = self.db.get_embeddings(cand_id)
            
            # Compute per-signal similarities
            face_sim = self._compare_face(
                query_features.get('face'),
                stored_embeddings.get('face')
            )
            
            gait_sim = self._compare_gait(
                query_features.get('gait'),
                stored_embeddings.get('gait')
            )
            
            appearance_sim = self._compare_appearance(
                query_features.get('appearance'),
                stored_embeddings.get('appearance')
            )
            
            # Fusion: weighted average with availability gating
            combined, confidence = self._fuse_signals(
                face_sim, gait_sim, appearance_sim
            )
            
            results.append({
                'global_id': cand_id,
                'face_sim': face_sim,
                'gait_sim': gait_sim,
                'appearance_sim': appearance_sim,
                'combined_score': combined,
                'confidence': confidence
            })
        
        return sorted(results, key=lambda x: x['combined_score'], reverse=True)
    
    def _apply_gating_decision(self, similarities):
        """
        Apply multi-signal gating to prevent false matches.
        
        Logic:
        1. If face available AND face_sim > threshold → ACCEPT (strongest signal)
        2. If gait available AND gait_sim > threshold AND appearance agrees → ACCEPT
        3. If only appearance AND sim > HIGH_threshold → TENTATIVE (requires confirmation)
        4. Otherwise → REJECT (create new ID)
        """
        if not similarities:
            return None
        
        best = similarities[0]
        
        # Rule 1: Face is king
        if best.get('face_sim') is not None:
            if best['face_sim'] > self.config.reid.thresholds.face_similarity:
                return best['global_id']
        
        # Rule 2: Gait + appearance agreement
        if (best.get('gait_sim') is not None and 
            best['gait_sim'] > self.config.reid.thresholds.gait_similarity):
            if (best.get('appearance_sim') is None or 
                best['appearance_sim'] > 0.4):  # Not contradicting
                return best['global_id']
        
        # Rule 3: Appearance only (conservative)
        if best.get('appearance_sim') is not None:
            # Require VERY high similarity to avoid look-alike clothes
            if best['appearance_sim'] > 0.8:  # Much higher than normal
                return best['global_id']
        
        # Rule 4: Below thresholds → unknown
        if best['combined_score'] < self.config.reid.thresholds.unknown_threshold:
            return None
        
        return None  # Conservative: reject uncertain matches
```

---

## 5. Re-Identification Policy

### 5.1 Multi-Signal Fusion Strategy

```
Priority Hierarchy (most reliable → least reliable):
1. Face embedding (clothes-invariant, unique)
2. Gait/pose embedding (body proportions, walking pattern)
3. Appearance embedding (vulnerable to clothing changes)

Decision Matrix:
┌─────────────┬──────────────┬──────────────┬──────────┐
│ Face Avail  │ Gait Avail   │ App Avail    │ Decision │
├─────────────┼──────────────┼──────────────┼──────────┤
│ ✓ (high)    │ any          │ any          │ Use face │
│ ✗           │ ✓ (high)     │ ✓ (agrees)   │ Use gait │
│ ✗           │ ✗            │ ✓ (very high)│ Tentative│
│ ✗           │ ✗            │ ✓ (medium)   │ New ID   │
│ all low quality            │ New ID (open-set)        │
└─────────────┴──────────────┴──────────────┴──────────┘
```

### 5.2 Tracklet Aggregation

```python
class TrackletAggregator:
    """
    Aggregate features across multiple frames for stable embeddings.
    """
    
    def aggregate(self, track_history, target_frames=5):
        """
        Sample K frames from tracklet for stable feature extraction.
        
        Strategy:
        - Sample uniformly across 2-5 second window
        - Prefer frames with:
          * High detection confidence
          * Large bbox (closer to camera)
          * Low blur (sharp image)
          * Frontal pose (for face)
        
        Args:
            track_history: List of frame data for this track
            target_frames: Number of frames to sample (default 5)
        
        Returns:
            selected_frames: List of best K frames
        """
        # Sort by quality score
        scored = [(self._compute_quality(f), f) for f in track_history]
        scored.sort(reverse=True, key=lambda x: x[0])
        
        # Take top-K, but ensure temporal diversity
        selected = self._temporal_sampling(scored, target_frames)
        
        return selected
    
    def _compute_quality(self, frame_data):
        """
        Quality = detection_conf * bbox_size_factor * sharpness
        """
        conf = frame_data['detection_conf']
        bbox_area = frame_data['bbox'][2] * frame_data['bbox'][3]
        blur_score = self._laplacian_variance(frame_data['image'])
        
        # Normalize and combine
        quality = conf * (bbox_area / 10000) * (blur_score / 100)
        return quality
```

### 5.3 Open-Set Unknown Handling

```python
def is_unknown(similarity_score, quality_score):
    """
    Decide if tracklet belongs to unknown (new) identity.
    
    Conservative approach:
    - If quality is low → assume unknown (avoid bad match)
    - If similarity < threshold → unknown
    - If time gap implausible → unknown (e.g., person appeared
      in cam2 1 second after leaving cam1, but cams are 5 min walk apart)
    
    Returns:
        True if should create new ID, False if match existing
    """
    if quality_score < MIN_QUALITY:
        return True  # Don't match on poor quality
    
    if similarity_score < UNKNOWN_THRESHOLD:
        return True  # Not similar enough
    
    # Could add spatial-temporal constraint here
    # (see paper: spatial_temporal_ReID.pdf)
    
    return False
```

### 5.4 Clothing Change Handling

**Strategy: Multi-Prototype Memory Bank**

```python
class MultiPrototypeMemory:
    """
    Maintain multiple embeddings per identity to handle appearance changes.
    
    Structure:
    {
        global_id: {
            'face': [emb1, emb2, ...],  # Top-M face embeddings
            'gait': [emb1, emb2, ...],  # Top-M gait embeddings
            'appearance': [emb1, emb2, ...],  # Top-M appearance embeddings
            'ema': {  # Exponential moving average
                'face': current_ema,
                'gait': current_ema,
                'appearance': current_ema
            }
        }
    }
    """
    
    def add_embedding(self, global_id, emb_type, new_embedding, quality):
        """
        Add new embedding with quality-based insertion.
        
        Logic:
        1. Update EMA
        2. Add to memory bank if quality > threshold
        3. Keep only top-M embeddings (by quality)
        4. Expire old embeddings (> TTL)
        """
        # Update EMA
        current_ema = self.get_ema(global_id, emb_type)
        new_ema = (self.alpha * new_embedding + 
                   (1 - self.alpha) * current_ema)
        self.set_ema(global_id, emb_type, new_ema)
        
        # Add to memory bank
        if quality > MIN_QUALITY_FOR_STORAGE:
            memory = self.get_memory(global_id, emb_type)
            memory.append({
                'embedding': new_embedding,
                'quality': quality,
                'timestamp': time.time()
            })
            
            # Keep top-M
            memory.sort(key=lambda x: x['quality'], reverse=True)
            self.set_memory(global_id, emb_type, memory[:self.M])
        
        # Expire old
        self._expire_old_embeddings(global_id, emb_type)
    
    def match(self, query_embedding, global_id, emb_type):
        """
        Match against all prototypes for this identity.
        
        Returns:
            max_similarity: Best match among all prototypes
        """
        memory = self.get_memory(global_id, emb_type)
        ema = self.get_ema(global_id, emb_type)
        
        # Compare against all stored embeddings
        similarities = [
            cosine_similarity(query_embedding, proto['embedding'])
            for proto in memory
        ]
        
        # Also compare against EMA
        ema_sim = cosine_similarity(query_embedding, ema)
        similarities.append(ema_sim)
        
        # Return best match
        return max(similarities) if similarities else 0.0
```

**Clothing Change Example:**

```
Timeline:
t=0:   Person enters cam1 wearing RED shirt
       → global_id=1 created
       → appearance_emb1 stored (RED features)

t=300: Person exits cam1, enters closet (off-camera)
       → Changes to BLUE shirt

t=350: Person enters cam2 wearing BLUE shirt
       → appearance_emb2 extracted (BLUE features)
       → appearance_sim(emb2, emb1) = 0.3 (LOW - different clothes!)
       
       BUT:
       → face_emb2 extracted (face visible)
       → face_sim(face_emb2, face_emb1) = 0.92 (HIGH - same person!)
       
       Decision: MATCH to global_id=1 (face overrides appearance)
       
       → Update identity 1:
         - Add appearance_emb2 to memory (now has RED and BLUE)
         - Update EMA to blend both appearances
         
       Next time: Will match on either RED or BLUE clothes
```

---

## 6. Migration Plan

### Phase 1: Configuration & Infrastructure (Week 1)

**Tasks:**
1. Create configuration system
   - `backend/config/sources.example.yaml`
   - `backend/.env.example`
   - `backend/src/config/settings.py` (Pydantic loader)
   
2. Update `.gitignore`
   - Block `.env`, `sources.yaml`, `*.pkl`, credentials
   
3. Create database schema
   - `backend/src/storage/db_schema.py` (SQLAlchemy models)
   - `backend/src/storage/db_manager.py` (CRUD operations)
   - Migration script: `create_db.py`

**Deliverables:**
- ✅ Config files
- ✅ DB schema
- ✅ Settings loader with environment variable interpolation

---

### Phase 2: Core Library Unification (Week 2)

**Tasks:**
1. Unify tracker
   - Move `backend/src/core/tracker.py` to clean wrapper
   - Support both ByteTrack and BoT-SORT via config
   - Remove duplicate logic from step scripts
   
2. Create video source abstraction
   - `backend/src/core/video_source.py`
   - Support: local video files, RTSP streams
   - Handle reconnection, buffering
   
3. Detector wrapper
   - `backend/src/core/detector.py`
   - YOLO person detection
   - Configuration-driven model selection

**Deliverables:**
- ✅ Unified tracker (single source of truth)
- ✅ Video source manager
- ✅ Detector wrapper

---

### Phase 3: Global ID Manager (Week 3-4)

**Tasks:**
1. Feature extractors
   - `backend/src/reid/feature_extractors/face_extractor.py`
     * Use InsightFace (buffalo_l model)
     * Quality scoring: face size, blur, angle
   
   - `backend/src/reid/feature_extractors/gait_extractor.py`
     * Extract pose sequences (15 frames)
     * Compute gait embedding from skeleton motion
     * Quality: motion magnitude, sequence completeness
   
   - `backend/src/reid/feature_extractors/appearance_extractor.py`
     * OSNet or CLIP
     * START WITH: disabled by default (appearance is weakest signal)

2. Tracklet aggregator
   - `backend/src/reid/tracklet_aggregator.py`
   - Sample K frames from tracklet
   - Quality scoring: blur, bbox size, confidence

3. Multi-prototype memory
   - `backend/src/reid/multi_prototype.py`
   - Memory bank (top-M embeddings)
   - EMA (exponential moving average)
   - TTL expiration

4. Global ID manager
   - `backend/src/reid/global_id_manager.py`
   - Main orchestrator
   - Multi-signal fusion
   - Gating decision logic
   - DB integration

**Deliverables:**
- ✅ All feature extractors (face, gait, appearance)
- ✅ Tracklet aggregator with quality scoring
- ✅ Multi-prototype memory system
- ✅ Global ID manager with open-set handling

---

### Phase 4: Integration & Offline Runner (Week 5)

**Tasks:**
1. Create offline runner
   - `backend/experiments/multicam_offline/run_offline.py`
   - Reads `sources.yaml`
   - Processes all video sources
   - Assigns global IDs
   - Outputs results to DB + visualization
   
2. Move old scripts to reference
   - `backend/experiments/multicam_offline/reference/step*.py`
   - Keep for comparison only
   
3. Testing utilities
   - `backend/experiments/multicam_offline/test_reid_quality.py`
     * Metrics: ID-switch count, false match rate
     * Clothing change simulation
     * Look-alike test cases

**Deliverables:**
- ✅ Working offline pipeline using sources.yaml
- ✅ Test suite for ReID quality
- ✅ Metrics reporting

---

### Phase 5: ADL & WebSocket (Week 6)

**Tasks:**
1. ADL engine
   - `backend/src/adl/adl_engine.py`
   - FSM for fall detection, bed exit, etc.
   - Event throttling (20s window)
   
2. WebSocket integration
   - `backend/src/api/websocket_handler.py`
   - Send frame + metadata
   - Include global_id in track data
   - Throttled events

3. Optional: Keep MJPEG fallback
   - `backend/src/api/mjpeg_handler.py`

**Deliverables:**
- ✅ ADL engine with throttling
- ✅ WebSocket handler with global_id metadata
- ✅ Optional MJPEG fallback

---

### Phase 6: Testing & Documentation (Week 7)

**Tasks:**
1. End-to-end testing
   - Offline: 3 video sources
   - Verify global_id assignment
   - Test clothing change scenarios
   - Test look-alike prevention
   
2. Documentation
   - README for experiments
   - API documentation
   - Configuration guide
   - Deployment guide (Docker, systemd)

3. Backward compatibility
   - Optional: Script to convert `global_tracks.pkl` → SQLite
   - Deprecation notice

**Deliverables:**
- ✅ Test results with metrics
- ✅ Complete documentation
- ✅ Deployment scripts

---

## 7. Pseudocode

### 7.1 Tracklet Aggregation

```python
def aggregate_tracklet(track_history, config):
    """
    Sample K frames from track history for stable feature extraction.
    
    Input:
        track_history: [
            {
                'frame_id': int,
                'timestamp': float,
                'bbox': [x, y, w, h],
                'image_crop': np.array,
                'detection_conf': float,
                'pose_keypoints': np.array or None
            },
            ...
        ]
        config: ReIDConfig
    
    Output:
        sampled_frames: List of selected frames
    """
    MIN_FRAMES = config.reid.quality.min_tracklet_frames
    TARGET_FRAMES = 5
    
    # Step 1: Filter by minimum quality
    valid_frames = []
    for frame in track_history:
        quality = compute_frame_quality(frame)
        if quality > MIN_QUALITY_THRESHOLD:
            valid_frames.append((quality, frame))
    
    if len(valid_frames) < MIN_FRAMES:
        return None  # Insufficient quality frames
    
    # Step 2: Sort by quality
    valid_frames.sort(reverse=True, key=lambda x: x[0])
    
    # Step 3: Temporal diversity sampling
    # Don't just take top-K consecutive frames
    # Instead, sample across time window
    sampled = []
    time_window = valid_frames[-1][1]['timestamp'] - valid_frames[0][1]['timestamp']
    interval = time_window / TARGET_FRAMES
    
    for i in range(TARGET_FRAMES):
        target_time = valid_frames[0][1]['timestamp'] + i * interval
        # Find closest high-quality frame to target_time
        closest = min(
            valid_frames,
            key=lambda x: abs(x[1]['timestamp'] - target_time)
        )
        sampled.append(closest[1])
    
    return sampled


def compute_frame_quality(frame_data):
    """
    Quality score for a single frame.
    
    Factors:
    - Detection confidence
    - Bounding box size (larger = closer = better)
    - Image sharpness (Laplacian variance)
    - Face visibility (bonus if face detected)
    """
    conf = frame_data['detection_conf']
    bbox_area = frame_data['bbox'][2] * frame_data['bbox'][3]
    blur_score = laplacian_variance(frame_data['image_crop'])
    
    # Normalize components
    conf_norm = conf  # Already 0-1
    size_norm = min(bbox_area / 10000, 1.0)  # Normalize to 100x100
    blur_norm = min(blur_score / 100, 1.0)  # Higher = sharper
    
    # Weighted combination
    quality = (0.3 * conf_norm + 
               0.3 * size_norm + 
               0.4 * blur_norm)
    
    # Bonus for face visibility
    if has_face(frame_data):
        quality *= 1.2
    
    return quality
```

### 7.2 ReID Association with Gating

```python
def reid_association(query_features, db_manager, config):
    """
    Associate tracklet with existing identity or create new.
    
    Input:
        query_features: {
            'face': {'embedding': np.array, 'quality': float} or None,
            'gait': {'embedding': np.array, 'quality': float} or None,
            'appearance': {'embedding': np.array, 'quality': float} or None
        }
        db_manager: Database manager instance
        config: ReIDConfig
    
    Output:
        global_id: int (existing or new)
        confidence: str ('high' | 'medium' | 'low')
    """
    # Step 1: Get all active identities
    active_ids = db_manager.get_active_identities()
    
    if not active_ids:
        # First person ever
        new_id = db_manager.create_identity()
        return new_id, 'high'
    
    # Step 2: Compute similarities for each candidate
    candidates = []
    for cand_id in active_ids:
        sim_scores = compute_multi_signal_similarity(
            query_features,
            cand_id,
            db_manager
        )
        candidates.append({
            'global_id': cand_id,
            'scores': sim_scores
        })
    
    # Step 3: Apply gating rules
    best_match = apply_gating(candidates, query_features, config)
    
    if best_match is None:
        # No match → create new identity
        new_id = db_manager.create_identity()
        return new_id, 'low'
    
    return best_match['global_id'], best_match['confidence']


def compute_multi_signal_similarity(query_features, candidate_id, db_manager):
    """
    Compare query against all stored embeddings for candidate.
    
    Returns:
        {
            'face': max_similarity or None,
            'gait': max_similarity or None,
            'appearance': max_similarity or None,
            'combined': weighted_score
        }
    """
    scores = {}
    
    # Face comparison
    if query_features.get('face'):
        stored_face_embs = db_manager.get_embeddings(candidate_id, 'face')
        if stored_face_embs:
            face_sims = [
                cosine_similarity(query_features['face']['embedding'], stored['vector'])
                for stored in stored_face_embs
            ]
            scores['face'] = max(face_sims)  # Best match among prototypes
    
    # Gait comparison
    if query_features.get('gait'):
        stored_gait_embs = db_manager.get_embeddings(candidate_id, 'gait')
        if stored_gait_embs:
            gait_sims = [
                cosine_similarity(query_features['gait']['embedding'], stored['vector'])
                for stored in stored_gait_embs
            ]
            scores['gait'] = max(gait_sims)
    
    # Appearance comparison
    if query_features.get('appearance'):
        stored_app_embs = db_manager.get_embeddings(candidate_id, 'appearance')
        if stored_app_embs:
            app_sims = [
                cosine_similarity(query_features['appearance']['embedding'], stored['vector'])
                for stored in stored_app_embs
            ]
            scores['appearance'] = max(app_sims)
    
    # Combined score (weighted by availability and reliability)
    combined = compute_combined_score(scores, query_features)
    scores['combined'] = combined
    
    return scores


def apply_gating(candidates, query_features, config):
    """
    Multi-signal gating to prevent false matches.
    
    Priority:
    1. Face (strongest signal)
    2. Gait + appearance agreement
    3. Appearance only (conservative threshold)
    
    Returns:
        best_match: {
            'global_id': int,
            'confidence': str
        } or None
    """
    # Sort by combined score
    candidates.sort(key=lambda x: x['scores']['combined'], reverse=True)
    
    if not candidates:
        return None
    
    best = candidates[0]
    scores = best['scores']
    
    # Gate 1: Face match (highest priority)
    if scores.get('face') is not None:
        if scores['face'] > config.reid.thresholds.face_similarity:
            return {
                'global_id': best['global_id'],
                'confidence': 'high'
            }
    
    # Gate 2: Gait + appearance (biometric + context)
    if scores.get('gait') is not None:
        gait_threshold = config.reid.thresholds.gait_similarity
        
        if scores['gait'] > gait_threshold:
            # Gait match - check if appearance agrees or is neutral
            app_score = scores.get('appearance')
            
            if app_score is None or app_score > 0.4:  # Not contradicting
                return {
                    'global_id': best['global_id'],
                    'confidence': 'medium'
                }
    
    # Gate 3: Appearance only (very conservative)
    # CRITICAL: High threshold to prevent look-alike clothing false matches
    if scores.get('appearance') is not None:
        if scores['appearance'] > 0.8:  # Much higher than normal
            return {
                'global_id': best['global_id'],
                'confidence': 'low'
            }
    
    # Gate 4: Below all thresholds
    unknown_threshold = config.reid.thresholds.unknown_threshold
    if scores['combined'] < unknown_threshold:
        return None  # Open-set: unknown person
    
    # Conservative: reject uncertain matches
    return None


def compute_combined_score(scores, query_features):
    """
    Weighted fusion of available signals.
    
    Weights (if available):
    - Face: 0.6 (most reliable)
    - Gait: 0.3 (body biometrics)
    - Appearance: 0.1 (vulnerable to clothing changes)
    """
    weights = {'face': 0.6, 'gait': 0.3, 'appearance': 0.1}
    
    available_signals = [k for k in scores if k != 'combined' and scores[k] is not None]
    
    if not available_signals:
        return 0.0
    
    # Normalize weights by available signals
    total_weight = sum(weights[sig] for sig in available_signals)
    
    combined = sum(
        scores[sig] * (weights[sig] / total_weight)
        for sig in available_signals
    )
    
    return combined
```

### 7.3 Open-Set Unknown Decision

```python
def is_unknown_identity(similarity_scores, query_quality, config):
    """
    Decide if tracklet represents unknown (new) person.
    
    Conservative approach:
    - Prefer creating new ID over false match
    - Especially when only weak signals available
    
    Input:
        similarity_scores: {
            'face': float or None,
            'gait': float or None,
            'appearance': float or None,
            'combined': float
        }
        query_quality: {
            'face': float or None,
            'gait': float or None,
            'appearance': float or None
        }
        config: ReIDConfig
    
    Returns:
        is_unknown: bool (True = create new ID)
    """
    # Rule 1: Poor quality → assume unknown
    # Don't risk false match on bad data
    if all(q is None or q < 0.5 for q in query_quality.values()):
        return True
    
    # Rule 2: No strong biometric signal available
    # Only have appearance → be very conservative
    has_face = similarity_scores.get('face') is not None
    has_gait = similarity_scores.get('gait') is not None
    
    if not has_face and not has_gait:
        # Only appearance available
        app_score = similarity_scores.get('appearance', 0.0)
        if app_score < 0.8:  # Very high threshold for appearance-only
            return True
    
    # Rule 3: Combined score below unknown threshold
    if similarity_scores['combined'] < config.reid.thresholds.unknown_threshold:
        return True
    
    # Rule 4: Optional - spatial-temporal plausibility
    # (Can add: if person appeared too quickly between distant cameras → unknown)
    
    # Default: not unknown (match existing)
    return False
```

---

## 8. Testing Plan

### 8.1 Offline Testing (3 Video Sources)

**Test Setup:**
```yaml
# sources.yaml for testing
cameras:
  - id: cam1
    name: "Test Video 1"
    type: video
    source: "/path/to/test_video1.mp4"
    
  - id: cam2
    name: "Test Video 2"
    type: video
    source: "/path/to/test_video2.mp4"
    
  - id: cam3
    name: "Test Video 3"
    type: video
    source: "/path/to/test_video3.mp4"
```

**Test Cases:**

1. **Basic ID Assignment**
   - Input: 3 videos with 2 people appearing across all cameras
   - Expected: 2 global IDs created, maintained across cameras
   - Metric: ID-switch count = 0

2. **Clothing Change Simulation**
   - Input: Same person with different clothes in different video segments
   - Expected: Same global_id if face/gait available
   - Metric: ID retained with >90% confidence

3. **Look-Alike Prevention**
   - Input: Two people wearing similar white shirts
   - Expected: Two different global IDs (no false merge)
   - Metric: False match rate = 0

4. **Open-Set Unknown**
   - Input: New person appearing for first time
   - Expected: New global_id created immediately
   - Metric: 100% new ID assignment for novel people

**Metrics to Report:**
```python
{
    'total_identities_detected': int,
    'total_id_switches': int,  # Lower is better
    'false_match_count': int,  # Should be 0
    'avg_tracklet_length': float,  # frames
    'reid_accuracy': float,  # % correct associations
    'db_stats': {
        'total_embeddings_stored': int,
        'avg_embeddings_per_id': float,
        'db_size_mb': float
    }
}
```

### 8.2 Database Validation

**Post-Run Checks:**
```sql
-- Check identities created
SELECT COUNT(*) FROM identities WHERE status='active';

-- Check observations per camera
SELECT camera_id, COUNT(*) FROM observations GROUP BY camera_id;

-- Check embeddings distribution
SELECT emb_type, COUNT(*) FROM embeddings GROUP BY emb_type;

-- Check quality scores
SELECT emb_type, AVG(quality_score), MIN(quality_score), MAX(quality_score)
FROM embeddings
GROUP BY emb_type;
```

---

## 9. Key Design Decisions

### 9.1 Why SQLite (Not Pickle)?

**Advantages:**
- ✅ Structured queries (filter by camera, time, quality)
- ✅ Concurrent access (multiple processes can read)
- ✅ Transactions (atomic updates)
- ✅ Scalable (can migrate to PostgreSQL later)
- ✅ Standard tooling (SQL browsers, backup tools)

**Pickle Limitations:**
- ❌ Load entire file into memory
- ❌ No concurrent access
- ❌ No partial queries
- ❌ File corruption risk
- ❌ Hard to inspect/debug

### 9.2 Why Face > Gait > Appearance?

**Research Evidence:**
- Face embeddings: ~99% accuracy on LFW benchmark (clothes-invariant)
- Gait embeddings: ~85% accuracy (body proportions, walking style)
- Appearance: ~70-80% accuracy (vulnerable to clothing, lighting)

**Real-World Factors:**
- Top-down cameras: Often can't see face → gait becomes primary
- Indoor environments: Stable lighting → appearance more reliable
- Clothing changes: Face/gait unaffected, appearance fails

### 9.3 Why Multi-Prototype (Not Single Centroid)?

**Single Centroid Problem:**
```
Person wearing RED shirt:
- Centroid embedding = [r_features]

Person changes to BLUE shirt:
- New embedding = [b_features]
- Centroid updated = 0.5*[r_features] + 0.5*[b_features]
- Result: Centroid now represents purple (doesn't match RED or BLUE well!)
```

**Multi-Prototype Solution:**
```
Person wearing RED shirt:
- Memory = [[r_features]]

Person changes to BLUE shirt:
- Memory = [[r_features], [b_features]]
- Match against both → finds BLUE
- Result: Handles both appearances
```

### 9.4 Why Conservative Unknown Threshold?

**False Match Cost >> Missed Match Cost**

Scenario 1: False match (BAD)
```
Person A wearing white shirt → global_id=1
Person B wearing white shirt → Incorrectly matched to global_id=1
Result: Two different people merged → WRONG FOREVER
```

Scenario 2: Missed match (OK)
```
Person A wearing white shirt → global_id=1
Person A returns (face occluded, gait unclear) → Creates global_id=2
Result: Same person with 2 IDs → Can be merged later when better signal
```

**Recovery:**
- False match: Hard to undo (requires manual intervention)
- Missed match: Can merge IDs later with high-confidence signal

**Therefore: Prefer creating new ID over risky match**

---

## 10. Implementation Notes

### 10.1 Performance Optimization

**Target FPS:** 30 FPS per camera on RTX 3090

**Optimization Strategies:**

1. **Lazy ReID:**
   - Don't run ReID on every frame
   - Only when tracklet is stable (5+ frames)
   - Only when quality threshold met

2. **Batch Processing:**
   - Extract embeddings for multiple tracklets in batch
   - Use GPU batching for face/appearance models

3. **Database Indexing:**
   - Index on (global_id, timestamp)
   - Index on (camera_id, timestamp)
   - Limit query to recent embeddings (last 24 hours)

4. **Embedding Cache:**
   - Keep recent embeddings in memory
   - Reduce DB queries for active identities

### 10.2 Error Handling

**Graceful Degradation:**

1. **Camera Failure:**
   - Log error, continue with other cameras
   - Retry connection (RTSP)
   - Alert via WebSocket

2. **Model Failure:**
   - Face detector fails → Use gait + appearance
   - Pose model fails → Use appearance only
   - Never crash pipeline

3. **Database Issues:**
   - Queue writes in memory
   - Retry on lock timeout
   - Alert if DB size exceeds limit

### 10.3 Deployment Considerations

**Docker Setup:**
```dockerfile
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6

# Copy application
COPY backend /app/backend
WORKDIR /app

# Install Python packages
RUN pip install -r requirements.txt

# Run
CMD ["python", "backend/experiments/multicam_offline/run_offline.py"]
```

**Systemd Service (for production):**
```ini
[Unit]
Description=HAVEN Multi-Camera ReID
After=network.target

[Service]
Type=simple
User=haven
WorkingDirectory=/opt/haven
ExecStart=/opt/haven/venv/bin/python backend/experiments/multicam_offline/run_offline.py
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

---

## 11. Future Enhancements

### 11.1 Advanced Features (Post-MVP)

1. **Spatial-Temporal Constraints:**
   - Camera topology graph
   - Travel time estimation
   - Implausibility detection (person can't teleport)
   - Reference: spatial_temporal_ReID.pdf

2. **Active Learning:**
   - Flag low-confidence matches for human review
   - User feedback loop to improve thresholds
   - Fine-tune models on facility-specific data

3. **Re-ID Model Fine-Tuning:**
   - Collect facility-specific dataset
   - Fine-tune OSNet on actual camera views
   - Improve gait model for top-down cameras

4. **Identity Merging:**
   - Detect duplicate IDs (same person, multiple IDs)
   - Semi-automatic merge with user confirmation
   - Preserve audit trail

### 11.2 Scalability

**100+ Cameras:**
- Distributed processing (one process per camera)
- Central orchestrator (global ID manager)
- Shared PostgreSQL database
- Message queue (RabbitMQ/Kafka) for events

---

## 12. Success Criteria

### 12.1 Functional Requirements

- ✅ Offline mode: Processes 3 videos from sources.yaml
- ✅ Global IDs: Every person gets unique ID
- ✅ Persistence: IDs maintained across cameras
- ✅ Clothing change: ID stable when face/gait available
- ✅ Look-alike prevention: No false merges
- ✅ Open-set: New people get new IDs immediately
- ✅ Database: SQLite with embeddings, observations
- ✅ Configuration: Single YAML file for all settings

### 12.2 Quality Metrics

**Target Metrics:**
- ID-switch rate: < 2% (98%+ same ID across cameras)
- False match rate: 0% (no incorrect merges)
- ReID latency: < 200ms per tracklet
- DB size: < 1GB for 24-hour continuous operation
- Throughput: 30 FPS per camera

### 12.3 Code Quality

- Clean separation: core/ vs experiments/
- No hardcoded paths (all in config)
- Comprehensive logging
- Unit tests for ReID logic
- Integration tests for full pipeline

---

## Appendix A: File Checklist

### Files to Create

**Configuration:**
- [ ] `backend/config/sources.example.yaml`
- [ ] `backend/.env.example`
- [ ] `backend/src/config/settings.py`

**Core Library:**
- [ ] `backend/src/core/detector.py`
- [ ] `backend/src/core/tracker.py` (refactor existing)
- [ ] `backend/src/core/pose_estimator.py`
- [ ] `backend/src/core/video_source.py`

**ReID Module:**
- [ ] `backend/src/reid/global_id_manager.py`
- [ ] `backend/src/reid/tracklet_aggregator.py`
- [ ] `backend/src/reid/multi_prototype.py`
- [ ] `backend/src/reid/similarity.py`
- [ ] `backend/src/reid/feature_extractors/face_extractor.py`
- [ ] `backend/src/reid/feature_extractors/gait_extractor.py`
- [ ] `backend/src/reid/feature_extractors/appearance_extractor.py`

**Storage:**
- [ ] `backend/src/storage/db_schema.py`
- [ ] `backend/src/storage/db_manager.py`
- [ ] `backend/scripts/create_db.py`

**ADL:**
- [ ] `backend/src/adl/adl_engine.py`

**API:**
- [ ] `backend/src/api/websocket_handler.py`
- [ ] `backend/src/api/mjpeg_handler.py` (optional)

**Experiments:**
- [ ] `backend/experiments/multicam_offline/run_offline.py`
- [ ] `backend/experiments/multicam_offline/test_reid_quality.py`
- [ ] `backend/experiments/multicam_offline/visualize_results.py`

**Documentation:**
- [ ] `backend/experiments/README.md`
- [ ] `backend/README.md` (updated)
- [ ] `DEPLOYMENT.md`

### Files to Move

- [ ] Move `step1-step6.py` → `backend/experiments/multicam_offline/reference/`
- [ ] Move `global_tracks.pkl` handling → deprecated (add conversion script if needed)

### Files to Update

- [ ] `backend/.gitignore` (add secrets, config, db)
- [ ] `backend/requirements.txt` (add SQLAlchemy, InsightFace, etc.)
- [ ] `backend/src/core/tracker.py` (unify, remove duplicates)

---

## Appendix B: Dependencies

```txt
# requirements.txt

# Core
fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0
python-dotenv==1.0.0
pyyaml==6.0.1
pydantic==2.5.0
pydantic-settings==2.1.0

# Computer Vision
opencv-python==4.8.1.78
ultralytics==8.1.0  # YOLO
numpy==1.24.3
pillow==10.1.0

# ReID & Face
insightface==0.7.3
onnxruntime-gpu==1.16.3  # For InsightFace
torch==2.1.0
torchvision==0.16.0
torchreid==1.4.0  # OSNet

# Database
sqlalchemy==2.0.23
alembic==1.13.0  # Migrations

# Utilities
tqdm==4.66.1
loguru==0.7.2
scipy==1.11.4
scikit-learn==1.3.2

# Optional (for visualization)
matplotlib==3.8.2
seaborn==0.13.0
```

---

## Appendix C: Quick Start Guide

### Setup

```bash
# 1. Clone repo
cd HAVEN

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Create database
python backend/scripts/create_db.py

# 5. Copy config templates
cp backend/config/sources.example.yaml backend/config/sources.yaml
cp backend/.env.example backend/.env

# 6. Edit configs
nano backend/config/sources.yaml  # Add your video paths
nano backend/.env  # Add RTSP credentials if needed
```

### Run Offline Test

```bash
# Make sure sources.yaml has 3 video sources configured
python backend/experiments/multicam_offline/run_offline.py

# Check results
sqlite3 /home/claude/haven_reid.db "SELECT * FROM identities;"
```

### Run Production (WebSocket)

```bash
# Start WebSocket server
python backend/src/api/websocket_handler.py

# Connect from browser
# ws://localhost:8765
```

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-30 | Initial design document |

---

**End of Design Document**
