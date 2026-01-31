# HAVEN Refactoring: Step-by-Step Implementation Guide

## Overview

This document provides concrete, actionable steps to refactor your HAVEN multi-camera person re-identification system. Follow these steps sequentially for a clean, working system.

---

## Prerequisites

Before starting:
- [ ] Backup your current codebase
- [ ] Document current hardcoded paths in step6-global-tracking.py
- [ ] List all dependencies currently in use
- [ ] Identify current video file locations

---

## Phase 1: Configuration & Infrastructure (Days 1-3)

### Day 1: Configuration System

#### Step 1.1: Create configuration structure

```bash
# Create directories
mkdir -p backend/config
mkdir -p backend/src/config
```

#### Step 1.2: Create `sources.example.yaml`

Create `backend/config/sources.example.yaml`:

```yaml
# HAVEN Multi-Camera Configuration Template
# Copy to sources.yaml and customize for your setup

general:
  fps_limit: 30
  resize_width: 640
  log_level: INFO
  output_dir: "/mnt/user-data/outputs"

cameras:
  # Example: Local video file
  - id: cam1
    name: "Camera 1"
    type: video
    source: "/path/to/your/video1.mp4"
    enabled: true
  
  # Example: RTSP stream (disabled by default)
  - id: cam2
    name: "Camera 2 (RTSP)"
    type: rtsp
    source: "${RTSP_CAM2_URL}"
    username: "${RTSP_USERNAME}"
    password: "${RTSP_PASSWORD}"
    enabled: false

inference:
  yolo:
    model: "yolov8n.pt"
    conf_threshold: 0.5
    device: "cuda"
  
  pose:
    enabled: true
    model: "yolov8n-pose.pt"
    conf_threshold: 0.3
  
  tracker:
    type: "bytetrack"
    persist: true

reid:
  enabled: true
  
  face:
    enabled: true
    model: "buffalo_l"
    quality_threshold: 0.7
  
  gait:
    enabled: true
    sequence_length: 15
  
  appearance:
    enabled: false
  
  thresholds:
    face_similarity: 0.6
    gait_similarity: 0.7
    appearance_similarity: 0.5
    unknown_threshold: 0.4
  
  multi_prototype:
    memory_size: 10
    ema_alpha: 0.3
    embedding_ttl: 3600
  
  quality:
    min_bbox_size: 80
    max_blur_variance: 100
    min_tracklet_frames: 5

storage:
  db_path: "/home/claude/haven_reid.db"
  save_snapshots: true
  snapshot_dir: "/mnt/user-data/outputs/snapshots"

api:
  websocket:
    enabled: true
    host: "0.0.0.0"
    port: 8765
```

#### Step 1.3: Create `.env.example`

Create `backend/.env.example`:

```bash
# HAVEN Environment Variables Template
# Copy to .env and fill with your actual credentials

# RTSP Camera Credentials
RTSP_USERNAME=admin
RTSP_PASSWORD=your_password_here
RTSP_CAM2_URL=rtsp://192.168.1.100:554/stream1
RTSP_CAM3_URL=rtsp://192.168.1.101:554/stream1

# Optional: Database encryption
# DB_ENCRYPTION_KEY=your_secret_key_here
```

#### Step 1.4: Update `.gitignore`

Add to `backend/.gitignore`:

```gitignore
# Secrets & Credentials
.env
*.env
!.env.example

# Config files with actual settings
config/sources.yaml
!config/sources.example.yaml

# Legacy pickle files
global_tracks.pkl
*.pkl

# Database files
*.db
*.sqlite
*.db-journal
*.db-shm
*.db-wal

# Snapshots & temporary outputs
snapshots/
outputs/*.jpg
outputs/*.png
outputs/*.mp4

# Python cache
__pycache__/
*.pyc
*.pyo
.pytest_cache/

# IDE
.vscode/
.idea/
*.swp
```

#### Step 1.5: Create settings loader

Create `backend/src/config/settings.py`:

```python
"""
Configuration loader with environment variable interpolation.
"""
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv


class GeneralConfig(BaseModel):
    fps_limit: int = 30
    resize_width: int = 640
    log_level: str = "INFO"
    output_dir: str = "/mnt/user-data/outputs"


class CameraConfig(BaseModel):
    id: str
    name: str
    type: str  # "video" or "rtsp"
    source: str
    username: Optional[str] = None
    password: Optional[str] = None
    enabled: bool = True


class InferenceConfig(BaseModel):
    yolo: Dict[str, Any]
    pose: Dict[str, Any]
    tracker: Dict[str, Any]


class ReIDConfig(BaseModel):
    enabled: bool = True
    face: Dict[str, Any]
    gait: Dict[str, Any]
    appearance: Dict[str, Any]
    thresholds: Dict[str, float]
    multi_prototype: Dict[str, Any]
    quality: Dict[str, Any]


class StorageConfig(BaseModel):
    db_path: str
    save_snapshots: bool = True
    snapshot_dir: str


class APIConfig(BaseModel):
    websocket: Dict[str, Any]
    mjpeg: Optional[Dict[str, Any]] = None


class Settings(BaseSettings):
    """Main settings class."""
    
    general: GeneralConfig
    cameras: list[CameraConfig]
    inference: InferenceConfig
    reid: ReIDConfig
    storage: StorageConfig
    api: APIConfig
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def interpolate_env_vars(value: Any) -> Any:
    """
    Replace ${VAR_NAME} with environment variable values.
    
    Example:
        "${RTSP_USERNAME}" -> "admin"
    """
    if isinstance(value, str):
        # Find all ${VAR} patterns
        pattern = re.compile(r'\$\{([^}]+)\}')
        
        def replacer(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))
        
        return pattern.sub(replacer, value)
    
    elif isinstance(value, dict):
        return {k: interpolate_env_vars(v) for k, v in value.items()}
    
    elif isinstance(value, list):
        return [interpolate_env_vars(item) for item in value]
    
    return value


def load_config(config_path: str = "backend/config/sources.yaml") -> Settings:
    """
    Load configuration from YAML file with environment variable interpolation.
    
    Args:
        config_path: Path to sources.yaml
    
    Returns:
        Settings object
    """
    # Load environment variables
    env_path = Path("backend/.env")
    if env_path.exists():
        load_dotenv(env_path)
    
    # Load YAML
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Interpolate environment variables
    config_dict = interpolate_env_vars(config_dict)
    
    # Parse with Pydantic
    settings = Settings(**config_dict)
    
    return settings


# Example usage
if __name__ == "__main__":
    try:
        config = load_config()
        print(f"Loaded config with {len(config.cameras)} cameras")
        for cam in config.cameras:
            print(f"  - {cam.name} ({cam.type}): {'enabled' if cam.enabled else 'disabled'}")
    except Exception as e:
        print(f"Error loading config: {e}")
```

### Day 2: Database Schema

#### Step 2.1: Create database schema

Create `backend/src/storage/db_schema.py`:

```python
"""
SQLAlchemy database schema for HAVEN ReID system.
"""
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, 
    DateTime, Text, LargeBinary, ForeignKey, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


class Identity(Base):
    """
    Represents a unique person tracked across cameras.
    """
    __tablename__ = 'identities'
    
    global_id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String(20), default='active', nullable=False)  # active | merged | archived
    merged_into = Column(Integer, ForeignKey('identities.global_id'), nullable=True)
    
    # Relationships
    observations = relationship("Observation", back_populates="identity", cascade="all, delete-orphan")
    embeddings = relationship("Embedding", back_populates="identity", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Identity(global_id={self.global_id}, status={self.status})>"


class Observation(Base):
    """
    Represents a single observation (sighting) of a person.
    """
    __tablename__ = 'observations'
    
    obs_id = Column(Integer, primary_key=True, autoincrement=True)
    global_id = Column(Integer, ForeignKey('identities.global_id'), nullable=False)
    camera_id = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Bounding box
    bbox_x = Column(Integer)
    bbox_y = Column(Integer)
    bbox_w = Column(Integer)
    bbox_h = Column(Integer)
    
    # Metadata
    posture = Column(String(20))  # standing | sitting | lying | unknown
    snapshot_path = Column(String(500))
    local_track_id = Column(Integer)  # Original tracker ID from this camera
    
    # Relationship
    identity = relationship("Identity", back_populates="observations")
    
    # Indexes
    __table_args__ = (
        Index('idx_obs_global_id', 'global_id'),
        Index('idx_obs_camera_time', 'camera_id', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<Observation(obs_id={self.obs_id}, global_id={self.global_id}, camera={self.camera_id})>"


class Embedding(Base):
    """
    Represents a feature embedding (face, gait, or appearance).
    """
    __tablename__ = 'embeddings'
    
    emb_id = Column(Integer, primary_key=True, autoincrement=True)
    global_id = Column(Integer, ForeignKey('identities.global_id'), nullable=False)
    emb_type = Column(String(20), nullable=False)  # face | gait | appearance
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Embedding data (pickled numpy array)
    vector = Column(LargeBinary, nullable=False)
    
    # Quality metadata
    quality_score = Column(Float)  # 0.0 - 1.0
    source_camera = Column(String(50))
    
    # Relationship
    identity = relationship("Identity", back_populates="embeddings")
    
    # Indexes
    __table_args__ = (
        Index('idx_emb_global_id', 'global_id'),
        Index('idx_emb_type', 'emb_type'),
        Index('idx_emb_timestamp', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<Embedding(emb_id={self.emb_id}, global_id={self.global_id}, type={self.emb_type})>"


class TrackletQuality(Base):
    """
    Metadata about tracklet quality (optional, for analysis).
    """
    __tablename__ = 'tracklet_quality'
    
    tracklet_id = Column(Integer, primary_key=True, autoincrement=True)
    global_id = Column(Integer, ForeignKey('identities.global_id'))
    camera_id = Column(String(50))
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    
    # Quality metrics
    avg_bbox_size = Column(Float)
    avg_blur_score = Column(Float)
    frame_count = Column(Integer)
    
    def __repr__(self):
        return f"<TrackletQuality(tracklet_id={self.tracklet_id}, frames={self.frame_count})>"


# Database creation function
def create_database(db_path: str = "haven_reid.db"):
    """
    Create database with all tables.
    
    Args:
        db_path: Path to SQLite database file
    """
    engine = create_engine(f'sqlite:///{db_path}', echo=False)
    Base.metadata.create_all(engine)
    print(f"Database created at {db_path}")
    return engine


def get_session(db_path: str = "haven_reid.db"):
    """
    Get SQLAlchemy session.
    
    Args:
        db_path: Path to SQLite database file
    
    Returns:
        Session object
    """
    engine = create_engine(f'sqlite:///{db_path}', echo=False)
    Session = sessionmaker(bind=engine)
    return Session()


if __name__ == "__main__":
    # Test: Create database
    create_database("/home/claude/test_haven.db")
```

#### Step 2.2: Create database manager

Create `backend/src/storage/db_manager.py`:

```python
"""
Database CRUD operations for HAVEN ReID system.
"""
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
from sqlalchemy.orm import Session

from .db_schema import Identity, Observation, Embedding, TrackletQuality, get_session


class DBManager:
    """
    Manages all database operations for global ID tracking.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.session = get_session(db_path)
    
    # ===== Identity Operations =====
    
    def create_identity(self) -> int:
        """
        Create a new identity.
        
        Returns:
            global_id: New identity ID
        """
        identity = Identity(status='active')
        self.session.add(identity)
        self.session.commit()
        return identity.global_id
    
    def get_identity(self, global_id: int) -> Optional[Identity]:
        """Get identity by ID."""
        return self.session.query(Identity).filter_by(global_id=global_id).first()
    
    def get_all_active_identities(self) -> List[int]:
        """Get list of all active identity IDs."""
        identities = self.session.query(Identity).filter_by(status='active').all()
        return [identity.global_id for identity in identities]
    
    def update_last_seen(self, global_id: int):
        """Update last_seen_at timestamp for identity."""
        identity = self.get_identity(global_id)
        if identity:
            identity.last_seen_at = datetime.utcnow()
            self.session.commit()
    
    # ===== Observation Operations =====
    
    def add_observation(self, global_id: int, camera_id: str, bbox: Tuple[int, int, int, int],
                       posture: str = None, snapshot_path: str = None, local_track_id: int = None):
        """
        Add new observation for identity.
        
        Args:
            global_id: Identity ID
            camera_id: Camera identifier
            bbox: Bounding box (x, y, w, h)
            posture: Detected posture
            snapshot_path: Path to saved image
            local_track_id: Original tracker ID
        """
        obs = Observation(
            global_id=global_id,
            camera_id=camera_id,
            bbox_x=bbox[0],
            bbox_y=bbox[1],
            bbox_w=bbox[2],
            bbox_h=bbox[3],
            posture=posture,
            snapshot_path=snapshot_path,
            local_track_id=local_track_id
        )
        self.session.add(obs)
        self.session.commit()
        return obs.obs_id
    
    def get_observations(self, global_id: int, camera_id: str = None, 
                        since: datetime = None) -> List[Observation]:
        """
        Get observations for identity.
        
        Args:
            global_id: Identity ID
            camera_id: Filter by camera (optional)
            since: Only get observations after this time (optional)
        """
        query = self.session.query(Observation).filter_by(global_id=global_id)
        
        if camera_id:
            query = query.filter_by(camera_id=camera_id)
        
        if since:
            query = query.filter(Observation.timestamp >= since)
        
        return query.order_by(Observation.timestamp).all()
    
    # ===== Embedding Operations =====
    
    def add_embedding(self, global_id: int, emb_type: str, vector: np.ndarray,
                     quality_score: float, source_camera: str = None):
        """
        Add new embedding for identity.
        
        Args:
            global_id: Identity ID
            emb_type: Type of embedding ('face', 'gait', 'appearance')
            vector: Embedding vector (numpy array)
            quality_score: Quality score (0.0-1.0)
            source_camera: Camera that captured this
        """
        # Pickle numpy array for storage
        vector_blob = pickle.dumps(vector)
        
        embedding = Embedding(
            global_id=global_id,
            emb_type=emb_type,
            vector=vector_blob,
            quality_score=quality_score,
            source_camera=source_camera
        )
        self.session.add(embedding)
        self.session.commit()
        return embedding.emb_id
    
    def get_embeddings(self, global_id: int, emb_type: str = None, 
                      limit: int = None) -> List[Dict]:
        """
        Get embeddings for identity.
        
        Args:
            global_id: Identity ID
            emb_type: Filter by type (optional)
            limit: Max number to return (optional, returns most recent)
        
        Returns:
            List of dicts with keys: emb_id, vector, quality_score, timestamp
        """
        query = self.session.query(Embedding).filter_by(global_id=global_id)
        
        if emb_type:
            query = query.filter_by(emb_type=emb_type)
        
        # Order by timestamp descending (most recent first)
        query = query.order_by(Embedding.timestamp.desc())
        
        if limit:
            query = query.limit(limit)
        
        embeddings = query.all()
        
        # Unpickle vectors
        results = []
        for emb in embeddings:
            results.append({
                'emb_id': emb.emb_id,
                'vector': pickle.loads(emb.vector),
                'quality_score': emb.quality_score,
                'timestamp': emb.timestamp,
                'source_camera': emb.source_camera
            })
        
        return results
    
    def cleanup_old_embeddings(self, ttl_seconds: int = 3600):
        """
        Delete embeddings older than TTL.
        
        Args:
            ttl_seconds: Time-to-live in seconds (default 1 hour)
        """
        cutoff = datetime.utcnow() - timedelta(seconds=ttl_seconds)
        deleted = self.session.query(Embedding).filter(Embedding.timestamp < cutoff).delete()
        self.session.commit()
        return deleted
    
    # ===== Statistics =====
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        return {
            'total_identities': self.session.query(Identity).count(),
            'active_identities': self.session.query(Identity).filter_by(status='active').count(),
            'total_observations': self.session.query(Observation).count(),
            'total_embeddings': self.session.query(Embedding).count(),
            'embeddings_by_type': {
                'face': self.session.query(Embedding).filter_by(emb_type='face').count(),
                'gait': self.session.query(Embedding).filter_by(emb_type='gait').count(),
                'appearance': self.session.query(Embedding).filter_by(emb_type='appearance').count(),
            }
        }
    
    def close(self):
        """Close database session."""
        self.session.close()


if __name__ == "__main__":
    # Test database manager
    db = DBManager("/home/claude/test_haven.db")
    
    # Create identity
    global_id = db.create_identity()
    print(f"Created identity: {global_id}")
    
    # Add observation
    obs_id = db.add_observation(global_id, "cam1", (100, 100, 50, 150), posture="standing")
    print(f"Added observation: {obs_id}")
    
    # Add embedding
    fake_vector = np.random.randn(512)
    emb_id = db.add_embedding(global_id, "face", fake_vector, 0.95, "cam1")
    print(f"Added embedding: {emb_id}")
    
    # Get stats
    stats = db.get_stats()
    print(f"Database stats: {stats}")
    
    db.close()
```

#### Step 2.3: Create database initialization script

Create `backend/scripts/create_db.py`:

```python
"""
Initialize HAVEN ReID database.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.db_schema import create_database
from src.config.settings import load_config


def main():
    """Create database with schema."""
    try:
        # Load config to get db_path
        config = load_config()
        db_path = config.storage.db_path
        
        print(f"Creating database at: {db_path}")
        create_database(db_path)
        print(" Database created successfully")
        
    except FileNotFoundError:
        # Fallback if config doesn't exist yet
        print("Config not found, using default path")
        db_path = "/home/claude/haven_reid.db"
        create_database(db_path)
        print(f" Database created at: {db_path}")


if __name__ == "__main__":
    main()
```

### Day 3: Testing Configuration System

#### Step 3.1: Create your actual config

```bash
# Copy template
cp backend/config/sources.example.yaml backend/config/sources.yaml

# Edit with your actual video paths
nano backend/config/sources.yaml
```

Update the `cameras` section with your 3 video files:

```yaml
cameras:
  - id: cam1
    name: "Living Room"
    type: video
    source: "/path/to/your/actual/video1.mp4"
    enabled: true
  
  - id: cam2
    name: "Bedroom"
    type: video
    source: "/path/to/your/actual/video2.mp4"
    enabled: true
  
  - id: cam3
    name: "Kitchen"
    type: video
    source: "/path/to/your/actual/video3.mp4"
    enabled: true
```

#### Step 3.2: Test configuration loading

```bash
# Test config loader
python backend/src/config/settings.py

# Should output:
# Loaded config with 3 cameras
#   - Living Room (video): enabled
#   - Bedroom (video): enabled
#   - Kitchen (video): enabled
```

#### Step 3.3: Create database

```bash
# Initialize database
python backend/scripts/create_db.py

# Verify database created
ls -lh /home/claude/haven_reid.db

# Test database operations
python backend/src/storage/db_manager.py
```

---

## Phase 2: Core Library Unification (Days 4-6)

### Day 4: Video Source & Detector

#### Step 4.1: Create video source handler

Create `backend/src/core/video_source.py`:

```python
"""
Unified video source handler (local files + RTSP).
"""
import cv2
from typing import Optional, Tuple
import numpy as np
from pathlib import Path


class VideoSource:
    """
    Unified interface for video files and RTSP streams.
    """
    
    def __init__(self, source: str, source_type: str = "video", 
                 username: str = None, password: str = None,
                 resize_width: int = None):
        """
        Initialize video source.
        
        Args:
            source: Video file path or RTSP URL
            source_type: "video" or "rtsp"
            username: RTSP username (optional)
            password: RTSP password (optional)
            resize_width: Resize frames to this width (optional)
        """
        self.source = source
        self.source_type = source_type
        self.resize_width = resize_width
        
        # Build connection string
        if source_type == "rtsp" and username and password:
            # Insert credentials into RTSP URL
            # rtsp://192.168.1.100:554/stream1
            # -> rtsp://user:pass@192.168.1.100:554/stream1
            url_parts = source.replace("rtsp://", "").split("/")
            host = url_parts[0]
            path = "/".join(url_parts[1:])
            self.connection_string = f"rtsp://{username}:{password}@{host}/{path}"
        else:
            self.connection_string = source
        
        self.cap = None
        self.fps = None
        self.frame_count = 0
        
    def open(self) -> bool:
        """
        Open video source.
        
        Returns:
            True if successful
        """
        self.cap = cv2.VideoCapture(self.connection_string)
        
        if not self.cap.isOpened():
            return False
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.source_type == "video":
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        return True
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame.
        
        Returns:
            (success, frame) tuple
        """
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        if not ret:
            return False, None
        
        # Resize if requested
        if self.resize_width and frame is not None:
            h, w = frame.shape[:2]
            new_h = int(h * (self.resize_width / w))
            frame = cv2.resize(frame, (self.resize_width, new_h))
        
        return True, frame
    
    def release(self):
        """Release video source."""
        if self.cap:
            self.cap.release()
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


if __name__ == "__main__":
    # Test with video file
    source = VideoSource("/path/to/test/video.mp4", "video", resize_width=640)
    if source.open():
        ret, frame = source.read()
        if ret:
            print(f"Successfully read frame: {frame.shape}")
        source.release()
```

#### Step 4.2: Create detector wrapper

Create `backend/src/core/detector.py`:

```python
"""
YOLO person detector wrapper.
"""
from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Tuple


class PersonDetector:
    """
    YOLO-based person detector.
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", 
                 conf_threshold: float = 0.5,
                 device: str = "cuda"):
        """
        Initialize detector.
        
        Args:
            model_path: Path to YOLO model
            conf_threshold: Confidence threshold
            device: "cuda" or "cpu"
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        
        # Person class ID in COCO
        self.person_class_id = 0
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons in frame.
        
        Args:
            frame: Input image (BGR)
        
        Returns:
            List of detections: [{
                'bbox': [x, y, w, h],
                'conf': float,
                'class_id': int
            }]
        """
        # Run inference
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            device=self.device,
            classes=[self.person_class_id],  # Only detect persons
            verbose=False
        )
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Convert from xyxy to xywh
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = x2 - x1
                h = y2 - y1
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(w), int(h)],
                    'conf': float(box.conf[0]),
                    'class_id': int(box.cls[0])
                })
        
        return detections


if __name__ == "__main__":
    # Test detector
    import cv2
    
    detector = PersonDetector("yolov8n.pt", conf_threshold=0.5)
    
    # Load test image
    frame = cv2.imread("/path/to/test/image.jpg")
    detections = detector.detect(frame)
    
    print(f"Detected {len(detections)} persons")
    for i, det in enumerate(detections):
        print(f"  Person {i}: bbox={det['bbox']}, conf={det['conf']:.2f}")
```

[Due to length limits, I'll create the remaining files as separate deliverables. This implementation guide provides the foundation. Would you like me to continue with Phase 2 Days 5-6 and Phase 3?]

---

## Summary of Phase 1 Deliverables

 Configuration system (YAML + .env)
 Database schema (SQLite with SQLAlchemy)
 Database manager (CRUD operations)
 Settings loader with environment interpolation
 Video source handler
 Person detector wrapper

**Next Steps:**
- Phase 2: Tracker unification, pose estimation
- Phase 3: Global ID manager with multi-signal ReID
- Phase 4: Offline runner and testing
- Phase 5: ADL + WebSocket integration


