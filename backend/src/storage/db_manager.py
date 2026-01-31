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
        Get options for identity.
        
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
    db = DBManager("test_haven.db")
    
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

