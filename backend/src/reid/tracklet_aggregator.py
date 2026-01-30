"""
Tracklet aggregation - Convert ByteTrack output to TrackletSummary.
Based on multi-frame sampling for stable embeddings.
"""
import cv2
import numpy as np
from typing import List, Dict, Optional
from .data_structures import TrackletSummary


def build_tracklet(track_id: int, frame_data: List[Dict], config, 
                   camera_id: str) -> TrackletSummary:
    """
    Aggregate multiple frames → single tracklet.
    
    Key steps:
    1. Sample K high-quality frames
    2. Extract features from each
    3. Average embeddings
    4. Compute quality score
    
    Args:
        track_id: Local track ID from ByteTrack
        frame_data: List of frame observations [{
            'frame_idx': int,
            'timestamp': float,
            'bbox': (x, y, w, h),
            'conf': float,
            'image_crop': np.ndarray,
            'pose': np.ndarray (optional)
        }]
        config: Config object
        camera_id: Camera identifier
    
    Returns:
        TrackletSummary object
    """
    if not frame_data:
        raise ValueError("Empty frame_data")
    
    # Step 1: Sample K frames (uniform + quality-weighted)
    K = config.reid.get('tracklet_sample_size', 5)
    
    # Score each frame
    scored_frames = [
        (compute_frame_quality(f), f) for f in frame_data
    ]
    scored_frames.sort(reverse=True, key=lambda x: x[0])
    
    # Sample uniformly across time but prioritize high quality
    sampled = temporal_sampling(scored_frames, K)
    
    # Step 2: Extract features (placeholder - will integrate with extractors)
    # TODO: Integrate with actual feature extractors when available
    appearance_embs = []
    face_embs = []
    gait_embs = []
    
    # For now, just average bboxes and compute quality
    bboxes = [f['bbox'] for f in frame_data]
    avg_bbox = tuple(np.mean(bboxes, axis=0).astype(int))
    
    # Step 3: Aggregate embeddings (placeholder)
    avg_appearance = np.mean(appearance_embs, axis=0) if appearance_embs else None
    avg_face = np.mean(face_embs, axis=0) if face_embs else None
    avg_gait = np.mean(gait_embs, axis=0) if gait_embs else None
    
    # Step 4: Compute quality
    quality = compute_tracklet_quality(frame_data)
    
    # Build tracklet
    tracklet = TrackletSummary(
        camera_id=camera_id,
        local_track_id=track_id,
        start_time=frame_data[0]['timestamp'],
        end_time=frame_data[-1]['timestamp'],
        frame_count=len(frame_data),
        bboxes=bboxes,
        avg_bbox=avg_bbox,
        appearance_emb=avg_appearance,
        face_emb=avg_face,
        gait_emb=avg_gait,
        quality_score=quality,
        has_face=len(face_embs) > 0,
        has_gait=len(gait_embs) > 0,
        pose_keypoints=[f.get('pose') for f in frame_data if f.get('pose') is not None]
    )
    
    return tracklet


def compute_frame_quality(frame_data: Dict) -> float:
    """
    Quality = detection_conf * size_factor * sharpness_factor
    
    Args:
        frame_data: Frame observation dict
    
    Returns:
        Quality score [0, 1]
    """
    conf = frame_data['conf']
    bbox = frame_data['bbox']
    crop = frame_data.get('image_crop')
    
    # Size factor (normalize to 100x100 pixels)
    bbox_area = bbox[2] * bbox[3]
    size_factor = min(bbox_area / 10000, 1.0)
    
    # Sharpness (Laplacian variance) if crop available
    sharpness_factor = 1.0
    if crop is not None:
        from ..utils.image_utils import compute_blur_score
        blur_score = compute_blur_score(crop)
        sharpness_factor = min(blur_score / 100, 1.0)
    
    quality = conf * size_factor * sharpness_factor
    return quality


def temporal_sampling(scored_frames: List[tuple], K: int) -> List[Dict]:
    """
    Sample K frames with temporal diversity.
    
    Don't just take top-K consecutive frames.
    Instead: divide time window into K bins, take best from each bin.
    
    Args:
        scored_frames: List of (quality_score, frame_data) tuples
        K: Number of frames to sample
    
    Returns:
        List of K sampled frames
    """
    if len(scored_frames) <= K:
        return [f for _, f in scored_frames]
    
    n = len(scored_frames)
    bin_size = n // K
    
    sampled = []
    for i in range(K):
        start_idx = i * bin_size
        end_idx = start_idx + bin_size if i < K-1 else n
        
        # Get best frame in this time bin
        bin_frames = scored_frames[start_idx:end_idx]
        if bin_frames:
            best_in_bin = max(bin_frames, key=lambda x: x[0])
            sampled.append(best_in_bin[1])
    
    return sampled


def compute_tracklet_quality(frame_data: List[Dict]) -> float:
    """
    Compute overall tracklet quality score.
    
    Consider:
    - Average frame quality
    - Temporal consistency
    - Bbox stability
    
    Args:
        frame_data: List of frame observations
    
    Returns:
        Quality score [0, 1]
    """
    if not frame_data:
        return 0.0
    
    # Average frame quality
    frame_qualities = [compute_frame_quality(f) for f in frame_data]
    avg_quality = np.mean(frame_qualities)
    
    # Bbox stability (low variance = stable)
    bboxes = np.array([f['bbox'] for f in frame_data])
    bbox_std = np.std(bboxes, axis=0)
    stability_score = 1.0 / (1.0 + np.mean(bbox_std) / 100)
    
    # Combine
    overall_quality = 0.7 * avg_quality + 0.3 * stability_score
    
    return overall_quality


def quality_gate(tracklet: TrackletSummary, config) -> tuple:
    """
    Reject low-quality tracklets from ReID.
    
    Args:
        tracklet: TrackletSummary to check
        config: Config object
    
    Returns:
        (is_valid: bool, reason: str)
    """
    min_frames = config.reid.quality.get('min_tracklet_frames', 5)
    min_quality = 0.5
    min_bbox_size = config.reid.quality.get('min_bbox_size', 80)
    
    if tracklet.frame_count < min_frames:
        return False, "insufficient_frames"
    
    if tracklet.quality_score < min_quality:
        return False, "low_quality"
    
    if not tracklet.has_face and not tracklet.has_gait and tracklet.appearance_emb is None:
        return False, "no_features"
    
    # Check bbox size (too small = far from camera)
    avg_area = tracklet.avg_bbox[2] * tracklet.avg_bbox[3]
    if avg_area < min_bbox_size * min_bbox_size:
        return False, "bbox_too_small"
    
    return True, "ok"


if __name__ == "__main__":
    # Test tracklet building
    from ..config.settings import load_config
    
    config = load_config("backend/config/sources.example.yaml")
    
    # Mock frame data
    frame_data = [
        {
            'frame_idx': i,
            'timestamp': i / 30.0,
            'bbox': (100 + i, 100, 150, 300),
            'conf': 0.9,
            'image_crop': None
        }
        for i in range(15)
    ]
    
    tracklet = build_tracklet(
        track_id=1,
        frame_data=frame_data,
        config=config,
        camera_id='cam1'
    )
    
    print(f"Built tracklet:")
    print(f"  Duration: {tracklet.end_time - tracklet.start_time:.2f}s")
    print(f"  Frames: {tracklet.frame_count}")
    print(f"  Quality: {tracklet.quality_score:.2f}")
    
    is_valid, reason = quality_gate(tracklet, config)
    print(f"  Valid: {is_valid} ({reason})")
