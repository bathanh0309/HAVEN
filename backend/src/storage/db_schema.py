"""
SQLAlchemy database schema for HAVEN ReID system.
"""
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, 
    DateTime, Text, LargeBinary, ForeignKey, Index
)
from sqlalchemy.orm import relationship, sessionmaker, declarative_base

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
    create_database("test_haven.db")
