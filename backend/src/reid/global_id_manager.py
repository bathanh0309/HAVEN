"""
Global ID Manager - Main orchestrator for person re-identification.
Implements open-set tracklet-based ReID with multi-signal fusion.

Based on code.md blueprint.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import time

from ..storage.db_manager import DBManager
from .data_structures import TrackletSummary, GlobalIdentity, GalleryMemory, CameraGraph, ReIDMetrics
from .similarity import compute_identity_similarity, two_threshold_decision, apply_cooldown


class GlobalIDManager:
    """
    Main orchestrator for global person re-identification.
    
    Workflow (from code.md):
    1. Receive tracklet from local tracker
    2. Check quality gate
    3. Query existing identities
    4. Multi-signal similarity computation
    5. Apply spatiotemporal constraints
    6. Two-threshold open-set decision
    7. Update gallery
    """
    
    def __init__(self, config, db_manager: DBManager = None, camera_graph: CameraGraph = None):
        """
        Initialize Global ID Manager.
        
        Args:
            config: Settings object with reid configuration
            db_manager: Database manager instance (optional)
            camera_graph: Spatiotemporal constraints (optional)
        """
        self.config = config
        self.db = db_manager
        self.camera_graph = camera_graph or CameraGraph()
        
        # Open-set gallery (in-memory)
        self.gallery = GalleryMemory(
            max_prototypes=getattr(config.reid.multi_prototype, 'memory_size', 10)
        )
        
        # Metrics tracking
        self.metrics = ReIDMetrics()
        
        # TODO: Initialize feature extractors when implemented
        # self.face_extractor = FaceExtractor(config.reid.face)
        # self.gait_extractor = GaitExtractor(config.reid.gait)
        # self.appearance_extractor = AppearanceExtractor(config.reid.appearance)
    
    def process_tracklet(self, tracklet: TrackletSummary) -> Tuple[int, str, str]:
        """
        Main entry point for global ID assignment.
        
        Args:
            tracklet: TrackletSummary from tracklet_aggregator
        
        Returns:
            (global_id: int, confidence: str, reason: str)
        """
        self.metrics.total_tracklets += 1
        
        # Step 1: Quality gate
        if not self._check_quality(tracklet):
            # Reject low quality, don't assign ID
            return -1, 'none', 'quality_reject'
        
        # Step 2: Query existing identities (with spatiotemporal filtering)
        candidates = self._get_candidates(tracklet)
        
        if not candidates:
            # No candidates or first person ever seen
            return self._create_new_identity(tracklet)
        
        # Step 3: Compute similarities
        for cand in candidates:
            identity = self.gallery.identities[cand['global_id']]
            sim, conf, signal = compute_identity_similarity(tracklet, identity, self.config)
            
            cand['similarity'] = sim
            cand['confidence'] = conf
            cand['signal_type'] = signal
            
            # Apply cooldown boost
            if apply_cooldown(tracklet, identity, cooldown_seconds=10):
                cand['similarity'] += 0.1  # Boost recent identity
        
        # Sort by similarity
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Step 4: Two-threshold decision
        best = candidates[0]
        decision, global_id = two_threshold_decision(
            best['similarity'],
            best['confidence'],
            best['signal_type'],
            tracklet,
            candidates,
            self.config
        )
        
        if decision == 'ACCEPT':
            # Match to existing ID
            return self._update_existing_identity(global_id, tracklet, best)
        else:
            # Create new ID
            return self._create_new_identity(tracklet)
    
    def _check_quality(self, tracklet: TrackletSummary) -> bool:
        """
        Check if tracklet meets quality requirements.
        
        Args:
            tracklet: TrackletSummary to check
        
        Returns:
            True if quality is acceptable
        """
        from .tracklet_aggregator import quality_gate
        is_valid, reason = quality_gate(tracklet, self.config)
        return is_valid
    
    def _get_candidates(self, tracklet: TrackletSummary) -> List[Dict]:
        """
        Get candidate identities for matching with spatiotemporal filtering.
        
        Args:
            tracklet: Query tracklet
        
        Returns:
            List of candidate dicts with global_id
        """
        candidates = []
        
        # Get reuse window
        reuse_window = getattr(self.config.reid, 'reuse_window', 7200)  # 2 hours default
        
        for global_id, identity in self.gallery.identities.items():
            # Skip archived identities
            if identity.status != 'active':
                continue
            
            # Check reuse window (don't match to very old identities)
            absence = tracklet.start_time - identity.last_seen_at
            if absence > reuse_window:
                continue
            
            # Check spatiotemporal constraints
            if not self._is_spatiotemporal_compatible(tracklet, identity):
                self.metrics.st_rejections += 1
                continue
            
            candidates.append({'global_id': global_id})
        
        return candidates
    
    def _is_spatiotemporal_compatible(self, tracklet: TrackletSummary, 
                                      identity: GlobalIdentity) -> bool:
        """
        Check if tracklet could belong to identity based on physics.
        
        Implements logic from code.md: A person cannot teleport.
        
        Args:
            tracklet: Query tracklet
            identity: Candidate identity
        
        Returns:
            True if transition is plausible
        """
        # Get last observation of this identity
        if not identity.tracklets:
            return True  # First observation for this identity
        
        last_tracklet = identity.tracklets[-1]
        
        # Check time gap and camera transition
        time_gap = tracklet.start_time - last_tracklet.end_time
        
        if time_gap < 0:
            return False  # Time travel impossible
        
        # Use camera graph if available
        is_possible = self.camera_graph.is_transition_possible(
            last_tracklet.camera_id,
            tracklet.camera_id,
            time_gap
        )
        
        return is_possible
    
    def _create_new_identity(self, tracklet: TrackletSummary) -> Tuple[int, str, str]:
        """
        Create new identity in gallery and database.
        
        Args:
            tracklet: TrackletSummary
        
        Returns:
            (global_id: int, confidence: str, reason: str)
        """
        # Create in gallery
        global_id = self.gallery.create_identity()
        self.gallery.add_observation(global_id, tracklet)
        
        # Create in database if available
        if self.db:
            self.db.add_observation(
                global_id=global_id,
                camera_id=tracklet.camera_id,
                bbox=tuple(tracklet.avg_bbox),
                local_track_id=tracklet.local_track_id
            )
        
        # Update metrics
        self.metrics.new_ids_created += 1
        
        return global_id, 'high', 'new_identity'
    
    def _update_existing_identity(self, global_id: int, tracklet: TrackletSummary,
                                   match_info: Dict) -> Tuple[int, str, str]:
        """
        Update existing identity with new observation.
        
        Args:
            global_id: Existing identity ID
            tracklet: TrackletSummary
            match_info: Match information (similarity, signal_type, etc.)
        
        Returns:
            (global_id: int, confidence: str, reason: str)
        """
        # Update gallery
        self.gallery.add_observation(global_id, tracklet)
        
        # Update database if available
        if self.db:
            self.db.update_last_seen(global_id)
            self.db.add_observation(
                global_id=global_id,
                camera_id=tracklet.camera_id,
                bbox=tuple(tracklet.avg_bbox),
                local_track_id=tracklet.local_track_id
            )
        
        # Update metrics
        self.metrics.ids_reused += 1
        
        # Track which signal was used
        signal = match_info['signal_type']
        if 'face' in signal:
            self.metrics.face_matches += 1
        elif 'gait' in signal:
            self.metrics.gait_matches += 1
            if 'override' in signal:
                self.metrics.gait_override_count += 1
        elif 'appearance' in signal:
            self.metrics.appearance_matches += 1
        
        # Track confidence
        conf = match_info['confidence']
        if conf == 'high':
            self.metrics.high_conf_count += 1
        elif conf == 'low':
            self.metrics.low_conf_count += 1
        else:
            self.metrics.uncertain_decisions += 1
        
        reason = f"{signal}_{match_info['similarity']:.2f}"
        return global_id, conf, reason
    
    def get_metrics(self) -> ReIDMetrics:
        """Get current metrics."""
        return self.metrics
    
    def print_stats(self):
        """Print performance statistics."""
        self.metrics.print_summary()


if __name__ == "__main__":
    # Test Global ID Manager
    from ..config.settings import load_config
    from .tracklet_aggregator import build_tracklet
    
    # Load config
    config = load_config("backend/config/sources.example.yaml")
    
    # Create camera graph
    camera_graph = CameraGraph({
        'cam1': {
            'cam2': {'min_time': 5, 'max_time': 30},
            'cam3': {'min_time': 10, 'max_time': 60}
        },
        'cam2': {
            'cam1': {'min_time': 5, 'max_time': 30},
            'cam3': {'min_time': 15, 'max_time': 90}
        }
    })
    
    # Initialize manager
    manager = GlobalIDManager(config, camera_graph=camera_graph)
    
    # Test tracklet 1
    frame_data_1 = [
        {
            'frame_idx': i,
            'timestamp': i / 30.0,
            'bbox': (100, 100, 150, 300),
            'conf': 0.9
        }
        for i in range(10)
    ]
    
    tracklet1 = build_tracklet(
        track_id=1,
        frame_data=frame_data_1,
        config=config,
        camera_id='cam1'
    )
    
    global_id1, conf1, reason1 = manager.process_tracklet(tracklet1)
    print(f"Tracklet 1: global_id={global_id1}, conf={conf1}, reason={reason1}")
    
    # Test tracklet 2 (same person in cam2?)
    frame_data_2 = [
        {
            'frame_idx': i,
            'timestamp': 15.0 + i / 30.0,  # 15 sec later
            'bbox': (200, 150, 150, 300),
            'conf': 0.85
        }
        for i in range(10)
    ]
    
    tracklet2 = build_tracklet(
        track_id=2,
        frame_data=frame_data_2,
        config=config,
        camera_id='cam2'
    )
    
    global_id2, conf2, reason2 = manager.process_tracklet(tracklet2)
    print(f"Tracklet 2: global_id={global_id2}, conf={conf2}, reason={reason2}")
    
    # Print stats
    manager.print_stats()
