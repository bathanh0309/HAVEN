"""
HAVEN GlobalIDManager - Dual-Master Logic for Multi-Camera ReID

Key Features:
- Dual-master cameras can create new Global IDs
- Non-master cameras can only MATCH or assign TEMP IDs
- Shared gallery across all cameras
- Two-threshold decision logic
- Spatiotemporal filtering
- Multi-prototype memory with EMA update

Author: HAVEN Team
Version: 2.0
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import defaultdict


class MatchReason(Enum):
    """Reason for global ID assignment."""
    MATCHED = "MATCHED"              # Matched existing ID
    MASTER_NEW = "MASTER_NEW"        # Master camera created new ID
    NON_MASTER_TEMP = "NON_MASTER_TEMP"  # Non-master assigned temp
    REJECTED = "REJECTED"            # Below reject threshold
    UNCERTAIN = "UNCERTAIN"          # In uncertain zone


@dataclass
class GlobalIdentity:
    """Represents a global identity in the gallery."""
    global_id: int
    embeddings: List[np.ndarray] = field(default_factory=list)
    prototype: Optional[np.ndarray] = None
    created_by_camera: int = 0
    created_at_frame: int = 0
    last_seen_camera: int = 0
    last_seen_frame: int = 0
    match_count: int = 0
    quality_scores: List[float] = field(default_factory=list)
    
    def update_prototype(self, new_embedding: np.ndarray, alpha: float = 0.3, max_prototypes: int = 5):
        """Update prototype with EMA and maintain top-K embeddings."""
        # Add new embedding
        self.embeddings.append(new_embedding)
        
        # Keep only top-K by quality (FIFO if quality not available)
        if len(self.embeddings) > max_prototypes:
            self.embeddings = self.embeddings[-max_prototypes:]
        
        # EMA update for prototype
        if self.prototype is None:
            self.prototype = new_embedding.copy()
        else:
            self.prototype = alpha * new_embedding + (1 - alpha) * self.prototype
            # Normalize
            norm = np.linalg.norm(self.prototype)
            if norm > 0:
                self.prototype = self.prototype / norm


@dataclass
class MatchResult:
    """Result of global ID assignment."""
    global_id: int
    reason: MatchReason
    score: float
    matched_id: Optional[int] = None


class GlobalIDManager:
    """
    Manages global ID assignment for multi-camera person ReID.
    
    Dual-Master Logic:
    - Master cameras (e.g., cam1, cam2) can create NEW global IDs
    - Non-master cameras can only MATCH or assign TEMP ID = 0
    - Gallery is SHARED across all cameras
    
    Two-Threshold Decision:
    - score >= accept_threshold: MATCH (confident)
    - score < reject_threshold: NEW or TEMP (confident new person)
    - else: UNCERTAIN zone (need more evidence)
    """
    
    TEMP_ID = 0  # Temporary ID for non-master cameras
    
    def __init__(
        self,
        master_camera_ids: List[int],
        accept_threshold: float = 0.75,
        reject_threshold: float = 0.50,
        camera_graph: Optional[Dict[int, List[int]]] = None,
        max_prototypes: int = 5,
        ema_alpha: float = 0.3,
        min_bbox_size: int = 50,
        min_track_frames: int = 3
    ):
        """
        Initialize GlobalIDManager.
        
        Args:
            master_camera_ids: List of camera IDs that can create new global IDs
            accept_threshold: Above this = confident match
            reject_threshold: Below this = confident new person
            camera_graph: Dict mapping camera_id -> list of adjacent camera IDs
            max_prototypes: Max embeddings to store per identity
            ema_alpha: EMA weight for prototype update
            min_bbox_size: Minimum bbox dimension for quality
            min_track_frames: Minimum frames before creating new ID
        """
        self.master_camera_ids = set(master_camera_ids)
        self.accept_threshold = accept_threshold
        self.reject_threshold = reject_threshold
        self.camera_graph = camera_graph or {}
        self.max_prototypes = max_prototypes
        self.ema_alpha = ema_alpha
        self.min_bbox_size = min_bbox_size
        self.min_track_frames = min_track_frames
        
        # Gallery: global_id -> GlobalIdentity
        self.gallery: Dict[int, GlobalIdentity] = {}
        
        # Track mapping: (camera_id, local_track_id) -> global_id
        self.track_to_global: Dict[Tuple[int, int], int] = {}
        
        # Next global ID counter
        self._next_global_id = 1
        
        # Metrics
        self.metrics = {
            'total_ids_created': 0,
            'ids_by_master': defaultdict(int),
            'total_matches': 0,
            'total_rejections': 0,
            'spatiotemporal_filtered': 0,
            'non_master_temp': 0,
            'total_tracks': 0
        }
    
    def is_master_camera(self, camera_id: int) -> bool:
        """Check if camera is a master camera."""
        return camera_id in self.master_camera_ids
    
    def _compute_similarity(self, embedding: np.ndarray, identity: GlobalIdentity) -> float:
        """Compute cosine similarity between embedding and identity prototype."""
        if identity.prototype is None:
            return 0.0
        
        # Normalize
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        proto_norm = identity.prototype / (np.linalg.norm(identity.prototype) + 1e-8)
        
        return float(np.dot(emb_norm, proto_norm))
    
    def _filter_by_spatiotemporal(
        self, 
        camera_id: int, 
        candidates: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Filter candidates by spatiotemporal constraints.
        
        Only keep candidates that were last seen in adjacent cameras.
        """
        if not self.camera_graph or camera_id not in self.camera_graph:
            return candidates
        
        adjacent_cameras = set(self.camera_graph.get(camera_id, []))
        adjacent_cameras.add(camera_id)  # Same camera is always valid
        
        filtered = []
        for gid, score in candidates:
            identity = self.gallery.get(gid)
            if identity and identity.last_seen_camera in adjacent_cameras:
                filtered.append((gid, score))
            else:
                self.metrics['spatiotemporal_filtered'] += 1
        
        return filtered
    
    def _find_best_match(
        self, 
        embedding: np.ndarray, 
        camera_id: int
    ) -> Tuple[Optional[int], float]:
        """
        Find best matching global ID for embedding.
        
        Returns:
            (global_id, score) or (None, 0.0) if no match
        """
        if not self.gallery:
            return None, 0.0
        
        # Compute similarity to all identities
        candidates = []
        for gid, identity in self.gallery.items():
            score = self._compute_similarity(embedding, identity)
            candidates.append((gid, score))
        
        # Filter by spatiotemporal constraints
        candidates = self._filter_by_spatiotemporal(camera_id, candidates)
        
        if not candidates:
            return None, 0.0
        
        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Deterministic tie-breaking: if equal scores, pick lower ID
        best_score = candidates[0][1]
        tied = [c for c in candidates if abs(c[1] - best_score) < 1e-6]
        if len(tied) > 1:
            tied.sort(key=lambda x: x[0])  # Sort by ID ascending
        
        return tied[0]
    
    def _create_new_identity(
        self, 
        camera_id: int, 
        embedding: np.ndarray, 
        frame_idx: int
    ) -> int:
        """Create a new global identity."""
        gid = self._next_global_id
        self._next_global_id += 1
        
        identity = GlobalIdentity(
            global_id=gid,
            embeddings=[embedding.copy()],
            prototype=embedding.copy(),
            created_by_camera=camera_id,
            created_at_frame=frame_idx,
            last_seen_camera=camera_id,
            last_seen_frame=frame_idx,
            match_count=1
        )
        
        self.gallery[gid] = identity
        
        # Update metrics
        self.metrics['total_ids_created'] += 1
        self.metrics['ids_by_master'][camera_id] += 1
        
        return gid
    
    def assign_global_id(
        self,
        camera_id: int,
        local_track_id: int,
        embedding: np.ndarray,
        quality: float = 1.0,
        timestamp: float = 0.0,
        frame_idx: int = 0,
        num_frames: int = 1,
        bbox_size: int = 100
    ) -> Tuple[int, MatchReason, float]:
        """
        Assign a global ID to a local track.
        
        Args:
            camera_id: Camera ID
            local_track_id: Local track ID from tracker
            embedding: Feature embedding (normalized)
            quality: Quality score (0-1)
            timestamp: Current timestamp
            frame_idx: Current frame index
            num_frames: Number of frames this track has been seen
            bbox_size: Min(width, height) of bounding box
            
        Returns:
            (global_id, reason, score)
            - global_id: Assigned global ID (0 = TEMP)
            - reason: MatchReason enum
            - score: Match score (0-1)
        """
        self.metrics['total_tracks'] += 1
        
        # Check if already assigned
        track_key = (camera_id, local_track_id)
        if track_key in self.track_to_global:
            existing_gid = self.track_to_global[track_key]
            if existing_gid in self.gallery:
                # Update prototype
                self.gallery[existing_gid].update_prototype(
                    embedding, self.ema_alpha, self.max_prototypes
                )
                self.gallery[existing_gid].last_seen_camera = camera_id
                self.gallery[existing_gid].last_seen_frame = frame_idx
                self.gallery[existing_gid].match_count += 1
                return existing_gid, MatchReason.MATCHED, 1.0
        
        # Quality check
        if bbox_size < self.min_bbox_size:
            return self.TEMP_ID, MatchReason.NON_MASTER_TEMP, 0.0
        
        # Find best match
        best_gid, best_score = self._find_best_match(embedding, camera_id)
        
        is_master = self.is_master_camera(camera_id)
        
        # Decision logic
        if best_gid is not None and best_score >= self.accept_threshold:
            # MATCH - confident
            self.track_to_global[track_key] = best_gid
            self.gallery[best_gid].update_prototype(
                embedding, self.ema_alpha, self.max_prototypes
            )
            self.gallery[best_gid].last_seen_camera = camera_id
            self.gallery[best_gid].last_seen_frame = frame_idx
            self.gallery[best_gid].match_count += 1
            self.metrics['total_matches'] += 1
            return best_gid, MatchReason.MATCHED, best_score
        
        elif best_score < self.reject_threshold:
            # Below reject threshold - confident new person
            if is_master:
                # Master can create new ID
                if num_frames >= self.min_track_frames:
                    new_gid = self._create_new_identity(camera_id, embedding, frame_idx)
                    self.track_to_global[track_key] = new_gid
                    return new_gid, MatchReason.MASTER_NEW, best_score
                else:
                    # Wait for more frames
                    return self.TEMP_ID, MatchReason.NON_MASTER_TEMP, best_score
            else:
                # Non-master: assign TEMP
                self.metrics['non_master_temp'] += 1
                return self.TEMP_ID, MatchReason.NON_MASTER_TEMP, best_score
        
        else:
            # UNCERTAIN zone
            if is_master and num_frames >= self.min_track_frames * 2:
                # Master with enough evidence - create new
                new_gid = self._create_new_identity(camera_id, embedding, frame_idx)
                self.track_to_global[track_key] = new_gid
                return new_gid, MatchReason.MASTER_NEW, best_score
            else:
                # Wait for more evidence
                self.metrics['non_master_temp'] += 1
                return self.TEMP_ID, MatchReason.UNCERTAIN, best_score
    
    def get_global_id(self, camera_id: int, local_track_id: int) -> Optional[int]:
        """Get global ID for a track if already assigned."""
        return self.track_to_global.get((camera_id, local_track_id))
    
    def get_identity(self, global_id: int) -> Optional[GlobalIdentity]:
        """Get identity by global ID."""
        return self.gallery.get(global_id)
    
    def get_all_global_ids(self) -> List[int]:
        """Get all active global IDs."""
        return list(self.gallery.keys())
    
    def summary(self) -> Dict:
        """Get summary metrics."""
        return {
            'total_global_ids': len(self.gallery),
            'total_ids_created': self.metrics['total_ids_created'],
            'ids_by_master': dict(self.metrics['ids_by_master']),
            'total_matches': self.metrics['total_matches'],
            'total_rejections': self.metrics['total_rejections'],
            'spatiotemporal_filtered': self.metrics['spatiotemporal_filtered'],
            'non_master_temp': self.metrics['non_master_temp'],
            'total_tracks': self.metrics['total_tracks']
        }
    
    def print_summary(self):
        """Print summary to console."""
        print("\n" + "=" * 60)
        print("GLOBAL ID MANAGER METRICS")
        print("=" * 60)
        s = self.summary()
        print(f"Total Global IDs Created: {s['total_ids_created']}")
        print(f"  - By Master Cameras: {s['ids_by_master']}")
        print(f"  - Active: {s['total_global_ids']}")
        print(f"Total Matches: {s['total_matches']}")
        print(f"Total Rejections: {s['total_rejections']}")
        print(f"Spatiotemporal Filtered: {s['spatiotemporal_filtered']}")
        print(f"Non-Master Temp IDs: {s['non_master_temp']}")
        print(f"Total Tracks Assigned: {s['total_tracks']}")
        print("=" * 60)
