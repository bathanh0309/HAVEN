"""
Hierarchical ID Manager - Single Registration Point

Logic:
- Cam1 (Gate): REGISTRATION MASTER
  - Ch cam1 c ng k ID mi (1, 2, 3, ...)
  - Bt buc face r rng
  - Red box  detect face  Green box + ID
  
- Cam2, Cam3 (Parking, Elevator): VERIFICATION
  - Check global registry
  -  ng k  Green box + ID c
  - Cha ng k  Red box + UNK1, UNK2, ...
  
- Cam4 (Room): STRICT VERIFICATION
  - Check global registry
  -  ng k  Green box + ID c
  - Cha ng k  Red box + INTRUDER (cnh bo)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PersonEmbedding:
    """Single embedding of a person"""
    embedding: np.ndarray
    quality: float
    timestamp: float
    camera_id: int
    frame_idx: int
    has_face: bool = False
    face_quality: float = 0.0


@dataclass
class RegisteredPerson:
    """Registered person in the system"""
    person_id: int  # 1, 2, 3, ...
    embeddings: List[PersonEmbedding] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    registered_at_camera: int = 1  # Always cam1
    cameras_seen: Set[int] = field(default_factory=set)
    is_active: bool = True
    
    def add_embedding(self, embedding: PersonEmbedding, max_size: int = 10):
        """Add embedding with quality-based pruning"""
        self.embeddings.append(embedding)
        self.last_seen = embedding.timestamp
        self.cameras_seen.add(embedding.camera_id)
        
        # Keep top-k by quality
        if len(self.embeddings) > max_size:
            self.embeddings.sort(key=lambda x: x.quality, reverse=True)
            self.embeddings = self.embeddings[:max_size]
    
    def get_best_embedding(self) -> Optional[np.ndarray]:
        """Get highest quality embedding"""
        if not self.embeddings:
            return None
        return max(self.embeddings, key=lambda x: x.quality).embedding
    
    def get_average_embedding(self) -> Optional[np.ndarray]:
        """Get quality-weighted average embedding"""
        if not self.embeddings:
            return None
        
        weights = np.array([e.quality for e in self.embeddings])
        weights = weights / (weights.sum() + 1e-8)
        
        embeddings = np.array([e.embedding for e in self.embeddings])
        avg_emb = np.average(embeddings, weights=weights, axis=0)
        
        # Normalize
        return avg_emb / (np.linalg.norm(avg_emb) + 1e-8)


class HierarchicalIDManager:
    """
    Hierarchical ID Manager with single registration point (Cam1).
    
    Workflow:
    - Cam1: Register new IDs (1, 2, 3, ...) with face verification
    - Cam2, Cam3: Verify against registry, assign UNK if not found
    - Cam4: Verify against registry, alert INTRUDER if not found
    """
    
    def __init__(
        self,
        registration_camera_id: int = 1,
        camera_graph: Dict = None,
        registered_match_threshold: float = 0.70,
        unknown_threshold: float = 0.60,
        margin_threshold: float = 0.15,
        min_face_quality: float = 0.8,
        min_tracklet_frames: int = 5,
        min_bbox_size: int = 80,
        ema_alpha: float = 0.3,
        max_prototypes: int = 10,
    ):
        """
        Initialize Hierarchical ID Manager.
        
        Args:
            registration_camera_id: Camera ID cho registration (default: 1)
            camera_graph: Spatiotemporal constraints
            registered_match_threshold: Threshold  match vi registered ID
            unknown_threshold: Threshold  xc nh unknown
            margin_threshold: Margin between candidates
            min_face_quality: Minimum face quality cho registration (cam1)
            min_tracklet_frames: Minimum frames per tracklet
            min_bbox_size: Minimum bbox size
            ema_alpha: EMA alpha
            max_prototypes: Max embeddings per person
        """
        self.registration_camera_id = registration_camera_id
        self.camera_graph = camera_graph or {}
        self.registered_match_threshold = registered_match_threshold
        self.unknown_threshold = unknown_threshold
        self.margin_threshold = margin_threshold
        self.min_face_quality = min_face_quality
        self.min_tracklet_frames = min_tracklet_frames
        self.min_bbox_size = min_bbox_size
        self.ema_alpha = ema_alpha
        self.max_prototypes = max_prototypes
        
        # State
        self.next_person_id = 1  # 1, 2, 3, ...
        self.next_unknown_id = 1  # UNK1, UNK2, ...
        
        # Registries
        self.registry: Dict[int, RegisteredPerson] = {}  # person_id -> RegisteredPerson
        self.track_to_person: Dict[Tuple[int, int], int] = {}  # (cam_id, local_id) -> person_id
        self.track_to_unknown: Dict[Tuple[int, int], int] = {}  # (cam_id, local_id) -> unknown_id
        
        # Metrics
        self.metrics = {
            'total_registered': 0,
            'total_unknown': 0,
            'total_intruders': 0,
            'verification_success': 0,
            'verification_failed': 0,
            'spatiotemporal_filtered': 0,
        }
        
        logger.info(f"HierarchicalIDManager initialized:")
        logger.info(f"  Registration camera: {registration_camera_id}")
        logger.info(f"  Registered match threshold: {registered_match_threshold}")
        logger.info(f"  Unknown threshold: {unknown_threshold}")
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity"""
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        return float(np.dot(emb1_norm, emb2_norm))
    
    def match_against_registry(
        self,
        query_embedding: np.ndarray
    ) -> List[Tuple[int, float]]:
        """
        Match query against registered persons.
        
        Returns:
            List of (person_id, similarity) sorted by similarity descending
        """
        if not self.registry:
            return []
        
        similarities = []
        for person_id, person in self.registry.items():
            if not person.is_active:
                continue
            
            proto_emb = person.get_average_embedding()
            if proto_emb is None:
                continue
            
            sim = self.cosine_similarity(query_embedding, proto_emb)
            similarities.append((person_id, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def check_spatiotemporal(
        self,
        from_camera: int,
        to_camera: int,
        time_gap: float
    ) -> bool:
        """Check if transition is plausible"""
        if from_camera == to_camera:
            return True
        
        if from_camera not in self.camera_graph:
            return True
        
        transitions = self.camera_graph.get(from_camera, {})
        if to_camera not in transitions:
            return True
        
        constraint = transitions[to_camera]
        min_time = constraint['min_time']
        max_time = constraint['max_time']
        
        return min_time <= time_gap <= max_time
    
    def process_cam1_registration(
        self,
        local_track_id: int,
        embedding: np.ndarray,
        quality: float,
        timestamp: float,
        frame_idx: int,
        num_frames: int,
        bbox_size: float,
        has_face: bool,
        face_quality: float
    ) -> Tuple[Optional[int], str, float, str]:
        """
        Process registration at Cam1 (Gate).
        
        Returns:
            (person_id, status, confidence, box_color)
            - person_id: Registered ID (1, 2, 3, ...) or None
            - status: "REGISTERED" or "PROCESSING"
            - confidence: Match score
            - box_color: "green" or "red"
        """
        track_key = (self.registration_camera_id, local_track_id)
        
        # Check if already registered
        if track_key in self.track_to_person:
            person_id = self.track_to_person[track_key]
            logger.debug(f"[cam1] Track {local_track_id} already registered as ID {person_id}")
            return person_id, "REGISTERED", 1.0, "green"
        
        # Quality gate
        if num_frames < self.min_tracklet_frames:
            return None, "PROCESSING", 0.0, "red"
        
        if bbox_size < self.min_bbox_size:
            return None, "PROCESSING", 0.0, "red"
        
        # CRITICAL: Must have face with good quality
        if not has_face:
            logger.debug(f"[cam1] Track {local_track_id}: No face detected yet")
            return None, "WAITING_FACE", 0.0, "red"
        
        if face_quality < self.min_face_quality:
            logger.debug(
                f"[cam1] Track {local_track_id}: Face quality too low "
                f"({face_quality:.2f} < {self.min_face_quality})"
            )
            return None, "POOR_FACE_QUALITY", face_quality, "orange"
        
        # Face detected with good quality  Register!
        person_id = self.next_person_id
        self.next_person_id += 1
        
        # Create registered person
        emb_data = PersonEmbedding(
            embedding=embedding,
            quality=quality,
            timestamp=timestamp,
            camera_id=self.registration_camera_id,
            frame_idx=frame_idx,
            has_face=has_face,
            face_quality=face_quality
        )
        
        person = RegisteredPerson(
            person_id=person_id,
            embeddings=[emb_data],
            first_seen=timestamp,
            last_seen=timestamp,
            registered_at_camera=self.registration_camera_id,
            cameras_seen={self.registration_camera_id},
            is_active=True
        )
        
        self.registry[person_id] = person
        self.track_to_person[track_key] = person_id
        
        self.metrics['total_registered'] += 1
        
        logger.info(
            f" [cam1] Track {local_track_id}  REGISTERED as ID {person_id} "
            f"(face_quality={face_quality:.2f})"
        )
        
        return person_id, "REGISTERED", 1.0, "green"
    
    def process_verification_camera(
        self,
        camera_id: int,
        local_track_id: int,
        embedding: np.ndarray,
        quality: float,
        timestamp: float,
        frame_idx: int,
        num_frames: int,
        bbox_size: float,
        is_strict: bool = False
    ) -> Tuple[Optional[int], Optional[int], str, float, str]:
        """
        Process verification at Cam2, Cam3, or Cam4.
        
        Args:
            camera_id: 2, 3, or 4
            is_strict: True for Cam4 (intruder alert)
        
        Returns:
            (person_id, unknown_id, status, confidence, box_color)
            - person_id: Registered ID if matched, else None
            - unknown_id: UNK ID if not matched, else None
            - status: "VERIFIED" | "UNKNOWN" | "INTRUDER"
            - confidence: Match score
            - box_color: "green" | "red"
        """
        track_key = (camera_id, local_track_id)
        
        # Check if already assigned
        if track_key in self.track_to_person:
            person_id = self.track_to_person[track_key]
            return person_id, None, "VERIFIED", 1.0, "green"
        
        if track_key in self.track_to_unknown:
            unknown_id = self.track_to_unknown[track_key]
            status = "INTRUDER" if is_strict else "UNKNOWN"
            return None, unknown_id, status, 0.0, "red"
        
        # Quality gate
        if num_frames < self.min_tracklet_frames:
            return None, None, "PROCESSING", 0.0, "orange"
        
        if bbox_size < self.min_bbox_size:
            return None, None, "PROCESSING", 0.0, "orange"
        
        # Match against registry
        matches = self.match_against_registry(embedding)
        
        if not matches:
            # No registered persons yet
            unknown_id = self.next_unknown_id
            self.next_unknown_id += 1
            self.track_to_unknown[track_key] = unknown_id
            
            status = "INTRUDER" if is_strict else "UNKNOWN"
            self.metrics['total_unknown'] += 1
            if is_strict:
                self.metrics['total_intruders'] += 1
            
            logger.warning(
                f" [cam{camera_id}] Track {local_track_id}  {status} "
                f"(UNK{unknown_id}): No registry"
            )
            
            return None, unknown_id, status, 0.0, "red"
        
        # Get best match
        best_id, best_score = matches[0]
        
        # Check if match is confident
        if best_score >= self.registered_match_threshold:
            # Check margin
            if len(matches) > 1:
                second_score = matches[1][1]
                margin = best_score - second_score
                
                if margin < self.margin_threshold:
                    # Ambiguous
                    logger.warning(
                        f"  [cam{camera_id}] Track {local_track_id}: "
                        f"Ambiguous match (margin={margin:.3f})"
                    )
                    # Still assign but with lower confidence
            
            # Verified!
            self.track_to_person[track_key] = best_id
            
            # Update registry
            if best_id in self.registry:
                emb_data = PersonEmbedding(
                    embedding=embedding,
                    quality=quality,
                    timestamp=timestamp,
                    camera_id=camera_id,
                    frame_idx=frame_idx
                )
                self.registry[best_id].add_embedding(emb_data, self.max_prototypes)
            
            self.metrics['verification_success'] += 1
            
            logger.info(
                f" [cam{camera_id}] Track {local_track_id}  VERIFIED as ID {best_id} "
                f"(score={best_score:.3f})"
            )
            
            return best_id, None, "VERIFIED", best_score, "green"
        
        else:
            # Not matched  Unknown
            unknown_id = self.next_unknown_id
            self.next_unknown_id += 1
            self.track_to_unknown[track_key] = unknown_id
            
            status = "INTRUDER" if is_strict else "UNKNOWN"
            self.metrics['total_unknown'] += 1
            self.metrics['verification_failed'] += 1
            if is_strict:
                self.metrics['total_intruders'] += 1
            
            logger.warning(
                f" [cam{camera_id}] Track {local_track_id}  {status} "
                f"(UNK{unknown_id}): Low similarity={best_score:.3f}"
            )
            
            return None, unknown_id, status, best_score, "red"
    
    def process_track(
        self,
        camera_id: int,
        local_track_id: int,
        embedding: np.ndarray,
        quality: float,
        timestamp: float,
        frame_idx: int,
        num_frames: int,
        bbox_size: float,
        has_face: bool = False,
        face_quality: float = 0.0
    ) -> Dict:
        """
        Main entry point for processing a track.
        
        Returns:
            Dict with:
            - person_id: Registered ID or None
            - unknown_id: Unknown ID or None
            - status: Status string
            - confidence: Match score
            - box_color: Color for bounding box
        """
        if camera_id == self.registration_camera_id:
            # Cam1: Registration
            person_id, status, conf, color = self.process_cam1_registration(
                local_track_id, embedding, quality, timestamp, frame_idx,
                num_frames, bbox_size, has_face, face_quality
            )
            return {
                'person_id': person_id,
                'unknown_id': None,
                'status': status,
                'confidence': conf,
                'box_color': color
            }
        
        else:
            # Cam2, Cam3, Cam4: Verification
            is_strict = (camera_id == 4)  # Cam4 is strict
            
            person_id, unknown_id, status, conf, color = self.process_verification_camera(
                camera_id, local_track_id, embedding, quality, timestamp,
                frame_idx, num_frames, bbox_size, is_strict
            )
            
            return {
                'person_id': person_id,
                'unknown_id': unknown_id,
                'status': status,
                'confidence': conf,
                'box_color': color
            }
    
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        return {
            **self.metrics,
            'total_registry': len(self.registry),
            'active_registry': sum(1 for p in self.registry.values() if p.is_active),
        }
    
    def print_metrics(self):
        """Print metrics summary"""
        metrics = self.get_metrics()
        logger.info("=" * 60)
        logger.info("HIERARCHICAL ID MANAGER METRICS")
        logger.info("=" * 60)
        logger.info(f"Total Registered (Cam1): {metrics['total_registered']}")
        logger.info(f"  - Active in Registry: {metrics['active_registry']}")
        logger.info(f"Total Unknown (UNK): {metrics['total_unknown']}")
        logger.info(f"Total Intruders (Cam4): {metrics['total_intruders']}")
        logger.info(f"Verification Success: {metrics['verification_success']}")
        logger.info(f"Verification Failed: {metrics['verification_failed']}")
        logger.info(f"Spatiotemporal Filtered: {metrics['spatiotemporal_filtered']}")
        logger.info("=" * 60)

