"""
Similarity computation and multi-signal fusion.
Implements the priority hierarchy: Face > Gait > Appearance
"""
import numpy as np
from typing import Dict, Optional, Tuple


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        emb1: First embedding vector
        emb2: Second embedding vector
    
    Returns:
        Similarity score [0, 1]
    """
    if emb1 is None or emb2 is None:
        return 0.0
    
    # Normalize
    emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
    
    # Cosine similarity
    sim = np.dot(emb1_norm, emb2_norm)
    
    # Clip to [0, 1]
    return float(np.clip(sim, 0.0, 1.0))


def compute_identity_similarity(query_tracklet, candidate_identity, config) -> Tuple[float, str, str]:
    """
    Multi-signal similarity with fusion.
    
    Implements priority hierarchy from code.md:
    1. Face (if available and good quality)
    2. Gait (if available, can override appearance for clothing changes)
    3. Appearance (last resort, conservative threshold)
    
    Args:
        query_tracklet: TrackletSummary to match
        candidate_identity: GlobalIdentity from gallery
        config: Config object with thresholds
    
    Returns:
        (similarity: float, confidence: str, signal_type: str)
        - similarity: Best match score [0, 1]
        - confidence: 'high' | 'medium' | 'low' | 'none'
        - signal_type: Which signal was used
    """
    face_sim = None
    gait_sim = None
    app_sim = None
    
    # Face comparison (multi-prototype)
    if query_tracklet.has_face and candidate_identity.face_prototypes:
        face_sims = [
            cosine_similarity(query_tracklet.face_emb, proto['embedding'])
            for proto in candidate_identity.face_prototypes
        ]
        face_sim = max(face_sims) if face_sims else None
    
    # Gait comparison (multi-prototype)
    if query_tracklet.has_gait and candidate_identity.gait_prototypes:
        gait_sims = [
            cosine_similarity(query_tracklet.gait_emb, proto['embedding'])
            for proto in candidate_identity.gait_prototypes
        ]
        gait_sim = max(gait_sims) if gait_sims else None
    
    # Appearance comparison (multi-prototype)
    if query_tracklet.appearance_emb is not None and candidate_identity.appearance_prototypes:
        app_sims = [
            cosine_similarity(query_tracklet.appearance_emb, proto['embedding'])
            for proto in candidate_identity.appearance_prototypes
        ]
        app_sim = max(app_sims) if app_sims else None
    
    # Fusion logic (priority hierarchy)
    T_face = config.reid.thresholds.get('face_similarity', 0.6)
    T_gait = config.reid.thresholds.get('gait_similarity', 0.7)
    T_app = config.reid.thresholds.get('appearance_similarity', 0.5)
    
    # Priority 1: Face (highest confidence)
    if face_sim is not None and face_sim > T_face:
        return face_sim, 'high', 'face'
    
    # Priority 2: Gait (check for clothing change)
    if gait_sim is not None and gait_sim > T_gait:
        # Check if appearance contradicts (clothing change indicator)
        if app_sim is not None and app_sim < 0.3:
            # Gait says SAME, appearance says DIFFERENT
            #  Likely clothing change
            return gait_sim, 'medium', 'gait_override'
        return gait_sim, 'medium', 'gait'
    
    # Priority 3: Appearance (conservative threshold)
    if app_sim is not None and app_sim > 0.8:
        return app_sim, 'low', 'appearance_only'
    
    # No strong signal
    if app_sim is not None:
        return app_sim, 'none', 'weak_appearance'
    
    return 0.0, 'none', 'no_signals'


def two_threshold_decision(similarity: float, confidence: str, signal_type: str,
                           tracklet, candidates, config) -> Tuple[str, Optional[int]]:
    """
    Open-set decision using two thresholds.
    
    From code.md:
    - T_high (0.75): Accept threshold (confident match)
    - T_low (0.50): Reject threshold (definitely new)
    - Gap [0.50, 0.75]: Uncertain zone (use additional evidence)
    
    Args:
        similarity: Best similarity score
        confidence: Confidence level from fusion
        signal_type: Which signal was used
        tracklet: Query tracklet
        candidates: List of candidate matches (for margin test)
        config: Config object
    
    Returns:
        (decision: str, global_id: int or None)
        - decision: 'ACCEPT' | 'CREATE_NEW'
        - global_id: ID to assign (None if creating new)
    """
    T_high = config.reid.thresholds.get('accept', 0.75)
    T_low = config.reid.thresholds.get('reject', 0.50)
    
    if not candidates:
        return 'CREATE_NEW', None
    
    best = candidates[0]
    
    # High confidence match
    if similarity > T_high:
        return 'ACCEPT', best['global_id']
    
    # Low similarity  definitely new
    if similarity < T_low:
        return 'CREATE_NEW', None
    
    # UNCERTAIN ZONE: Check margin
    if len(candidates) > 1:
        second_best = candidates[1]
        margin = best['similarity'] - second_best['similarity']
        
        if margin < 0.15:
            # Too close to second-best  ambiguous
            return 'CREATE_NEW', None
    
    # Check quality
    if tracklet.quality_score < 0.6:
        return 'CREATE_NEW', None
    
    # Check signal type (trust biometrics in uncertain zone)
    if signal_type in ['face', 'gait', 'gait_override']:
        return 'ACCEPT', best['global_id']
    
    # Conservative: reject uncertain appearance-only matches
    return 'CREATE_NEW', None


def apply_cooldown(tracklet, identity, cooldown_seconds=10) -> bool:
    """
    Prevent rapid ID switching.
    
    If we just assigned this identity < cooldown_seconds ago,
    prefer keeping the ID even if similarity is slightly lower.
    
    Args:
        tracklet: Query tracklet
        identity: Candidate identity
        cooldown_seconds: Cooldown period in seconds
    
    Returns:
        True if within cooldown period
    """
    if not identity.tracklets:
        return False
    
    last_obs = identity.tracklets[-1]
    time_since_last = tracklet.start_time - last_obs.end_time
    
    if time_since_last < cooldown_seconds:
        # Recent observation  boost this identity's score
        return True
    
    return False


if __name__ == "__main__":
    # Test similarity functions
    
    # Test cosine similarity
    emb1 = np.random.randn(512)
    emb2 = emb1 + np.random.randn(512) * 0.1  # Similar
    emb3 = np.random.randn(512)  # Different
    
    sim_similar = cosine_similarity(emb1, emb2)
    sim_different = cosine_similarity(emb1, emb3)
    
    print(f"Cosine similarity test:")
    print(f"  Similar vectors: {sim_similar:.3f}")
    print(f"  Different vectors: {sim_different:.3f}")
    
    assert sim_similar > sim_different, "Similar should be higher!"
    print(" Similarity test passed")

