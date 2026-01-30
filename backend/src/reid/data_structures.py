"""
Core data structures for HAVEN ReID system.
Based on tracklet-level matching architecture.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np
import time


@dataclass
class TrackletSummary:
    """
    Aggregated representation of a person's trajectory in one camera.
    
    Built from ByteTrack output across multiple frames.
    """
    # Identity
    camera_id: str
    local_track_id: int  # ByteTrack ID
    
    # Time window
    start_time: float
    end_time: float
    frame_count: int
    
    # Spatial extent
    bboxes: List[tuple]  # [(x,y,w,h), ...]
    avg_bbox: tuple      # Average position/size
    
    # Features
    appearance_emb: Optional[np.ndarray] = None  # 512-D from OSNet
    gait_emb: Optional[np.ndarray] = None        # 128-D from pose
    face_emb: Optional[np.ndarray] = None        # 512-D from InsightFace
    
    # Quality scores
    quality_score: float = 0.0     # Overall quality [0,1]
    avg_blur: float = 0.0          # Laplacian variance
    has_face: bool = False
    has_gait: bool = False
    
    # Optional: Pose data
    pose_keypoints: Optional[List[np.ndarray]] = None
    body_proportions: Optional[np.ndarray] = None
    
    # Optional: ADL
    action_scores: Optional[Dict[str, float]] = None
    
    def is_valid(self, min_frames=5, min_quality=0.5):
        """Check if tracklet is good enough for ReID."""
        return (
            self.frame_count >= min_frames and
            self.quality_score >= min_quality and
            (self.has_face or self.has_gait or self.appearance_emb is not None)
        )


@dataclass
class GlobalIdentity:
    """
    Represents a unique person across all cameras.
    """
    global_id: int
    created_at: float
    last_seen_at: float
    status: str = 'active'  # active | merged | archived
    
    # Observations
    tracklets: List[TrackletSummary] = field(default_factory=list)
    
    # Prototype embeddings (multi-prototype memory)
    face_prototypes: List[Dict] = field(default_factory=list)
    gait_prototypes: List[Dict] = field(default_factory=list)
    appearance_prototypes: List[Dict] = field(default_factory=list)
    
    def add_tracklet(self, tracklet: TrackletSummary):
        """Add new observation."""
        self.tracklets.append(tracklet)
        self.last_seen_at = tracklet.end_time


@dataclass
class GalleryMemory:
    """
    Open-set gallery that grows dynamically.
    """
    identities: Dict[int, GlobalIdentity] = field(default_factory=dict)
    next_id: int = 1
    max_prototypes: int = 10
    
    def create_identity(self) -> int:
        """Create new identity and return global_id."""
        global_id = self.next_id
        self.next_id += 1
        self.identities[global_id] = GlobalIdentity(
            global_id=global_id,
            created_at=time.time(),
            last_seen_at=time.time()
        )
        return global_id
    
    def add_observation(self, global_id: int, tracklet: TrackletSummary):
        """Add tracklet to existing identity."""
        if global_id not in self.identities:
            raise ValueError(f"Identity {global_id} not in gallery")
        
        identity = self.identities[global_id]
        identity.add_tracklet(tracklet)
        
        # Update prototypes
        self._update_prototypes(identity, tracklet)
    
    def _update_prototypes(self, identity: GlobalIdentity, tracklet: TrackletSummary):
        """Add new embeddings to prototype memory."""
        # Face
        if tracklet.face_emb is not None and tracklet.has_face:
            identity.face_prototypes.append({
                'embedding': tracklet.face_emb,
                'quality': tracklet.quality_score,
                'timestamp': tracklet.end_time
            })
            # Keep top-K by quality
            identity.face_prototypes.sort(key=lambda x: x['quality'], reverse=True)
            identity.face_prototypes = identity.face_prototypes[:self.max_prototypes]
        
        # Gait
        if tracklet.gait_emb is not None:
            identity.gait_prototypes.append({
                'embedding': tracklet.gait_emb,
                'quality': tracklet.quality_score,
                'timestamp': tracklet.end_time
            })
            identity.gait_prototypes.sort(key=lambda x: x['quality'], reverse=True)
            identity.gait_prototypes = identity.gait_prototypes[:self.max_prototypes]
        
        # Appearance
        if tracklet.appearance_emb is not None:
            identity.appearance_prototypes.append({
                'embedding': tracklet.appearance_emb,
                'quality': tracklet.quality_score,
                'timestamp': tracklet.end_time
            })
            identity.appearance_prototypes.sort(key=lambda x: x['quality'], reverse=True)
            identity.appearance_prototypes = identity.appearance_prototypes[:self.max_prototypes]


@dataclass
class CameraGraph:
    """
    Spatio-temporal constraints between cameras.
    """
    adjacency: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    # Example:
    # {'cam1': {'cam2': {'min_time': 5, 'max_time': 30}}}
    
    def is_transition_possible(self, from_cam: str, to_cam: str, time_gap: float) -> bool:
        """Check if transition is physically plausible."""
        if from_cam == to_cam:
            return time_gap < 300  # 5 min max for same camera
        
        if to_cam not in self.adjacency.get(from_cam, {}):
            return False  # No connection
        
        constraints = self.adjacency[from_cam][to_cam]
        return constraints['min_time'] <= time_gap <= constraints['max_time']


@dataclass
class ReIDMetrics:
    """Track performance of global ID assignment."""
    
    # Accuracy metrics
    total_tracklets: int = 0
    correct_assignments: int = 0
    id_switches: int = 0          # Same person got different ID
    false_matches: int = 0         # Different people got same ID
    
    # Open-set metrics
    new_ids_created: int = 0
    ids_reused: int = 0
    uncertain_decisions: int = 0
    
    # Quality metrics
    avg_confidence: float = 0.0
    high_conf_count: int = 0       # Decisions with high confidence
    low_conf_count: int = 0        # Decisions with low confidence
    
    # Signal usage
    face_matches: int = 0
    gait_matches: int = 0
    appearance_matches: int = 0
    gait_override_count: int = 0   # Clothing change cases
    
    # Spatiotemporal
    st_rejections: int = 0         # Matches rejected due to time/space
    
    def accuracy(self):
        if self.total_tracklets == 0:
            return 0.0
        return self.correct_assignments / self.total_tracklets
    
    def id_switch_rate(self):
        if self.total_tracklets == 0:
            return 0.0
        return self.id_switches / self.total_tracklets
    
    def false_match_rate(self):
        if self.total_tracklets == 0:
            return 0.0
        return self.false_matches / self.total_tracklets
    
    def print_summary(self):
        print("=== ReID Performance ===")
        print(f"Accuracy: {self.accuracy()*100:.1f}%")
        print(f"ID Switch Rate: {self.id_switch_rate()*100:.1f}%")
        print(f"False Match Rate: {self.false_match_rate()*100:.1f}%")
        print(f"New IDs Created: {self.new_ids_created}")
        print(f"IDs Reused: {self.ids_reused}")
        print(f"\nSignal Usage:")
        print(f"  Face: {self.face_matches}")
        print(f"  Gait: {self.gait_matches}")
        print(f"  Appearance: {self.appearance_matches}")
        print(f"  Gait Override (clothing change): {self.gait_override_count}")
        print(f"\nSpatiotemporal Rejections: {self.st_rejections}")
