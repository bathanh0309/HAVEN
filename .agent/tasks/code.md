ART 1: CORE CONCEPTS YOU MUST ABSORB
1.1 Tracklet vs Frame Matching (THE FOUNDATION)
What You Need to Know
Frame-level matching (WRONG approach):
Frame 1: Person detected → Extract embedding → Match to gallery
Frame 2: Person detected → Extract embedding → Match to gallery
Frame 3: Person detected → Extract embedding → Match to gallery
Problem: Noisy, unstable, prone to false matches from single bad frame
Tracklet-level matching (CORRECT approach):
Frames 1-30: Person tracked → Accumulate observations → Build tracklet
Tracklet complete → Aggregate features → Match to gallery ONCE
Benefit: Stable, filters noise, uses temporal consistency
Key Insight from Papers
From spatial_temporal_ReID.pdf: "Following person detection and tracking, a custom transformer model built on the TorchReID framework is employed to analyze and compare bounding boxes of detected persons, ensuring that the identified individuals are accurately associated with existing IDs."
From Multi-Camera_Industrial_Open-Set.pdf: "We leverage the robustness of people detectors to occlusion and provide people's trajectory, identified by pseudo-ids, through a reliable temporal tracking system. Then, under conditions of good quality images, the appearance-based method for Re-ID provides a unique re-identification ID."
What This Means for HAVEN
python# WRONG: Per-frame ReID
for frame in video:
    detections = detector(frame)
    for det in detections:
        embedding = extract_features(det)
        global_id = match_to_gallery(embedding)  # ❌ Unstable!

# RIGHT: Tracklet-based ReID
tracklets = local_tracker(video)  # ByteTrack/BoT-SORT
for tracklet in tracklets:
    if tracklet.is_stable() and tracklet.quality_sufficient():
        # Aggregate across multiple frames
        embedding = aggregate_features(tracklet.frames)  # ✅ Stable!
        global_id = match_to_gallery(embedding)
Critical Parameters

Minimum tracklet length: 5-10 frames (papers use 15 for gait)
Aggregation strategy:

Sample K frames uniformly across tracklet (K=5-7)
Prefer high-quality frames (sharp, large bbox, frontal pose)
Average embeddings OR use attention-weighted pooling


Quality gate: Only attempt ReID if tracklet meets criteria:

python   def is_tracklet_valid(tracklet):
       return (
           len(tracklet.frames) >= 5 and
           avg_bbox_area(tracklet) > MIN_SIZE and
           avg_blur_score(tracklet) > MIN_SHARPNESS and
           detection_confidence(tracklet) > 0.5
       )

1.2 Spatio-Temporal Constraints (THE SECRET SAUCE)
What You Need to Know
The Problem: Appearance alone is ambiguous. Two people in white shirts look identical.
The Solution: Use physics! A person cannot teleport.
Camera Graph Representation
python# Example: Your facility layout
camera_graph = {
    'cam1': {  # Living room
        'cam2': {'min_time': 5, 'max_time': 30},   # 5-30 sec to bedroom
        'cam3': {'min_time': 10, 'max_time': 60},  # 10-60 sec to kitchen
    },
    'cam2': {  # Bedroom
        'cam1': {'min_time': 5, 'max_time': 30},
        'cam3': {'min_time': 15, 'max_time': 90},
    },
    'cam3': {  # Kitchen
        'cam1': {'min_time': 10, 'max_time': 60},
        'cam2': {'min_time': 15, 'max_time': 90},
    }
}
Time Window Filtering
From spatial_temporal_ReID.pdf: "We create coarse spatial-temporal histograms to describe the probability of a positive image pair... We smooth the histogram using the Parzen Window method."
Simplified for HAVEN (you don't need full Parzen smoothing):
pythondef is_spatiotemporal_plausible(tracklet1, tracklet2, camera_graph):
    """
    Check if two tracklets could be the same person.
    
    Args:
        tracklet1: {camera_id: 'cam1', end_time: 100.0}
        tracklet2: {camera_id: 'cam2', start_time: 115.0}
        camera_graph: Adjacency matrix with time windows
    """
    cam1 = tracklet1['camera_id']
    cam2 = tracklet2['camera_id']
    time_gap = tracklet2['start_time'] - tracklet1['end_time']
    
    # Same camera: immediate reappearance is fine
    if cam1 == cam2:
        return time_gap < 300  # 5 minutes max
    
    # Different cameras: check graph
    if cam2 not in camera_graph.get(cam1, {}):
        return False  # No path between cameras
    
    constraints = camera_graph[cam1][cam2]
    return constraints['min_time'] <= time_gap <= constraints['max_time']
```

### Critical Insight

**This eliminates 70-90% of false matches!**

Example:
```
Person A exits cam1 at t=100
Person B appears in cam2 at t=102 (2 seconds later)
Camera graph says: cam1→cam2 requires minimum 5 seconds walk

→ Conclusion: These CANNOT be the same person
→ Even if appearance similarity = 0.95, REJECT the match
How to Build Your Camera Graph
Option 1: Manual annotation (fastest for small setup)
python# Walk the route yourself with a timer
# cam1 to cam2: Walk at normal pace = 8 seconds
# Add safety margin: min=5, max=15
Option 2: Automatic learning (from spatial_temporal_ReID.pdf)
python# Track people across cameras for 1 day
# For each identity, collect (cam_i, t_i) → (cam_j, t_j) transitions
# Build histogram of time differences
# Extract min/max from histogram (e.g., 5th and 95th percentile)
For HAVEN: Start with manual, add automatic later.

1.3 Open-Set Identity Management (THE HARD PART)
The Fundamental Problem
Closed-set (traditional ReID): "Match this person to one of 100 known identities"

Gallery is fixed and complete
Every query must match one of the 100

Open-set (real-world): "Is this person one of the N seen so far, or someone NEW?"

Gallery grows dynamically
Must decide: match existing vs create new ID

The Core Challenge
From Multi-Camera_Industrial_Open-Set.pdf: "Open-set Re-ID presents a more complex challenge as it deals with scenarios where the probe individual may not belong to any of the pre-registered identities in the gallery. The system must not only match the probe to known identities but also determine when a probe does not correspond to any known individual."
Two-Threshold Scheme (YOUR SOLUTION)
python# Traditional (WRONG): Single threshold
if similarity > 0.6:
    return matched_id
else:
    return create_new_id()
# Problem: No distinction between "confident match" and "uncertain"

# Open-Set (RIGHT): Two thresholds
T_high = 0.75  # Accept threshold (confident match)
T_low = 0.50   # Reject threshold (definitely new)
# Gap [0.50, 0.75] = Uncertain zone

if similarity > T_high:
    return matched_id  # Confident
elif similarity < T_low:
    return create_new_id()  # Definitely new
else:
    # Uncertain: Use additional evidence
    if has_strong_biometric(tracklet):  # Face or gait
        return matched_id
    else:
        return create_new_id()  # Conservative
Gallery Update Strategy
Critical decision: When to add embeddings to gallery?
From Multi-Camera_Industrial_Open-Set.pdf: "The Re-ID module is applied only when the images of individuals are clean, using a tracking system to minimize feature representation noise."
pythonclass GalleryMemory:
    def __init__(self, max_prototypes=10):
        self.identities = {}  # {global_id: [embeddings]}
        self.max_prototypes = max_prototypes
    
    def add_observation(self, global_id, embedding, quality_score):
        """
        Add embedding to gallery if quality is sufficient.
        
        Quality-based insertion:
        - Only store embeddings with quality > 0.7
        - Keep top-K embeddings (by quality)
        - Expire embeddings older than TTL
        """
        if quality_score < 0.7:
            return  # Reject low quality
        
        if global_id not in self.identities:
            self.identities[global_id] = []
        
        # Add with metadata
        self.identities[global_id].append({
            'embedding': embedding,
            'quality': quality_score,
            'timestamp': time.time()
        })
        
        # Keep only top-K by quality
        self.identities[global_id].sort(key=lambda x: x['quality'], reverse=True)
        self.identities[global_id] = self.identities[global_id][:self.max_prototypes]
    
    def match(self, query_embedding, global_id):
        """Match against all prototypes for this identity."""
        if global_id not in self.identities:
            return 0.0
        
        similarities = [
            cosine_similarity(query_embedding, proto['embedding'])
            for proto in self.identities[global_id]
        ]
        
        return max(similarities)  # Best match among prototypes
Unknown Detection Logic
pythondef is_unknown(similarities, tracklet):
    """
    Decide if tracklet represents unknown person.
    
    Args:
        similarities: [(global_id, score), ...] sorted by score
        tracklet: Tracklet data
    
    Returns:
        (is_unknown: bool, reason: str)
    """
    if not similarities:
        return True, "empty_gallery"
    
    best_id, best_score = similarities[0]
    
    # Rule 1: Very low similarity → definitely unknown
    if best_score < T_low:
        return True, f"low_similarity_{best_score:.2f}"
    
    # Rule 2: Very high similarity → definitely known
    if best_score > T_high:
        return False, f"high_similarity_{best_score:.2f}"
    
    # Rule 3: Uncertain zone → use quality and biometrics
    if tracklet.quality < 0.6:
        return True, "low_quality_uncertain"
    
    if not tracklet.has_face and not tracklet.has_gait:
        return True, "appearance_only_uncertain"
    
    # Rule 4: Check margin to second-best
    if len(similarities) > 1:
        second_score = similarities[1][1]
        margin = best_score - second_score
        if margin < 0.1:  # Too close, ambiguous
            return True, "ambiguous_margin"
    
    # Default: Conservative - prefer creating new ID
    return True, "conservative_default"
```

---

## 1.4 Clothing-Change Robustness (THE KILLER FEATURE)

### The Problem Visualized
```
Timeline:
t=0    Person wearing RED shirt enters cam1
       → Appearance embedding captures RED features
       → global_id=1 created

t=300  Person enters closet (off-camera), changes to BLUE shirt

t=350  Person wearing BLUE shirt appears in cam2
       → Appearance embedding captures BLUE features
       → Similarity(RED, BLUE) = 0.3 (LOW!)
       → Traditional ReID: Creates global_id=2 ❌ WRONG!
Multi-Signal Fusion (THE SOLUTION)
Hierarchy of reliability (from your research):

Face (99% accurate, clothes-invariant)

DeepFace, InsightFace, ArcFace
512-D embedding, cosine similarity
Problem: Top-down cameras often can't see face


Gait/Pose (85% accurate, body biometrics)

Skeleton sequence from YOLO-Pose
Body proportions: shoulder width, leg length ratio
Walking pattern: stride length, speed
Advantage: Works with top-down cameras


Appearance (70% accurate, vulnerable to clothing)

OSNet, ResNet50, CLIP
Full-body features
Problem: Fails on clothing changes



Body Proportion Features (YOUR SECRET WEAPON)
pythondef extract_body_proportions(pose_keypoints):
    """
    Extract clothing-invariant body proportions.
    
    Args:
        pose_keypoints: Array of 17 keypoints from YOLO-Pose
            [nose, left_eye, right_eye, left_ear, right_ear,
             left_shoulder, right_shoulder, left_elbow, right_elbow,
             left_wrist, right_wrist, left_hip, right_hip,
             left_knee, right_knee, left_ankle, right_ankle]
    
    Returns:
        proportions: 8-D vector of body ratios
    """
    # Shoulder width
    shoulder_width = distance(keypoints[5], keypoints[6])
    
    # Torso height
    torso_height = distance(
        midpoint(keypoints[5], keypoints[6]),  # Shoulders
        midpoint(keypoints[11], keypoints[12])  # Hips
    )
    
    # Leg length
    left_leg = distance(keypoints[11], keypoints[15])  # Hip to ankle
    right_leg = distance(keypoints[12], keypoints[16])
    avg_leg_length = (left_leg + right_leg) / 2
    
    # Compute ratios (invariant to distance from camera)
    proportions = [
        shoulder_width / torso_height,      # Shoulder-to-torso ratio
        avg_leg_length / torso_height,      # Leg-to-torso ratio
        shoulder_width / avg_leg_length,    # Shoulder-to-leg ratio
        # Add more ratios...
    ]
    
    return np.array(proportions)
Gait Embedding from Pose Sequence
pythondef extract_gait_embedding(pose_sequence):
    """
    Extract gait features from pose sequence.
    
    Args:
        pose_sequence: List of pose keypoints over 15-30 frames
    
    Returns:
        gait_embedding: 128-D vector
    """
    # Step 1: Compute body proportions for each frame
    proportions_seq = [extract_body_proportions(pose) for pose in pose_sequence]
    
    # Step 2: Compute motion features
    # - Stride length: horizontal displacement between frames
    # - Vertical oscillation: head movement up/down
    # - Arm swing: wrist movement pattern
    
    hip_positions = [pose[11] for pose in pose_sequence]  # Left hip
    stride_pattern = np.diff([p[0] for p in hip_positions])  # X displacement
    
    # Step 3: Combine static (proportions) + dynamic (motion)
    static_features = np.mean(proportions_seq, axis=0)
    dynamic_features = [
        np.mean(stride_pattern),
        np.std(stride_pattern),
        # Add more motion statistics
    ]
    
    gait_embedding = np.concatenate([static_features, dynamic_features])
    return gait_embedding
Multi-Signal Fusion Logic
pythondef compute_identity_similarity(query, candidate):
    """
    Fuse multiple signals with priority hierarchy.
    
    Priority:
    1. Face (if available and good quality)
    2. Gait (if available)
    3. Appearance (last resort)
    """
    # Extract all available signals
    face_sim = None
    gait_sim = None
    app_sim = None
    
    if query.has_face and candidate.has_face:
        face_sim = cosine_similarity(query.face_emb, candidate.face_emb)
    
    if query.has_gait and candidate.has_gait:
        gait_sim = cosine_similarity(query.gait_emb, candidate.gait_emb)
    
    if query.has_appearance and candidate.has_appearance:
        app_sim = cosine_similarity(query.app_emb, candidate.app_emb)
    
    # Decision logic
    if face_sim is not None and face_sim > 0.6:
        return face_sim, 'high', 'face'  # Confident match via face
    
    if gait_sim is not None and gait_sim > 0.7:
        # Check if appearance contradicts
        if app_sim is not None and app_sim < 0.3:
            # Gait says SAME, appearance says DIFFERENT
            # Trust gait (clothing change likely)
            return gait_sim, 'medium', 'gait_override'
        return gait_sim, 'medium', 'gait'
    
    if app_sim is not None:
        # Appearance only → be very conservative
        if app_sim > 0.8:  # Much higher threshold
            return app_sim, 'low', 'appearance_only'
    
    return 0.0, 'none', 'no_signals'
```

---

## 1.5 SAM Segmentation for Background Bias Reduction

### The Problem

Traditional ReID extracts features from **entire bounding box**:
```
[Background | Person | Background]
       ↓          ↓         ↓
  Embedding includes background features → Bias!
Why this matters:

Person in front of white wall → embedding includes "white"
Same person in front of dark curtain → embedding includes "dark"
Similarity drops even though person is identical!

The SAM Solution
From research: "Mask-guided contrastive attention model for person re-identification with expanded cross neighborhood re-ranking."
pythondef extract_person_only_features(bbox_crop, sam_model):
    """
    Use SAM to segment person and mask background.
    
    Args:
        bbox_crop: Person crop from detector
        sam_model: Segment Anything Model
    
    Returns:
        masked_crop: Image with background set to black/mean
    """
    # Step 1: Get SAM segmentation
    masks = sam_model.predict(bbox_crop)
    
    # Step 2: Select largest mask (usually the person)
    person_mask = masks[np.argmax([m.sum() for m in masks])]
    
    # Step 3: Apply mask
    masked_crop = bbox_crop.copy()
    masked_crop[~person_mask] = 0  # Or use mean color
    
    # Step 4: Extract features from masked image
    embedding = reid_model(masked_crop)
    
    return embedding
When to Use SAM (Practical Tradeoffs)
Pros:

10-15% accuracy improvement (from papers)
Eliminates background bias
Helps with cluttered environments

Cons:

SAM inference: ~200-500ms per image (CPU)
YOLO+ByteTrack: ~30ms per frame
5-10x slower!

HAVEN recommendation:
python# Option 1: Use SAM only for gallery embeddings
if adding_to_gallery:
    use_sam = True  # High quality matters
else:
    use_sam = False  # Speed matters

# Option 2: Use SAM only when background is cluttered
if detect_cluttered_background(bbox_crop):
    use_sam = True
else:
    use_sam = False  # Simple background OK without SAM

# Option 3: Disable on CPU, enable on GPU
use_sam = config.device == 'cuda' and config.sam_enabled

1.6 Action100M/VL-JEPA for ADL Recognition
The Opportunity
Traditional ADL detection:
python# Rule-based (current HAVEN approach)
if pose.vertical_position < THRESHOLD and duration > 2.0:
    event = "fall_detected"
```

**Problems**:
- Brittle thresholds
- Misses nuanced actions
- Hard to extend to new activities

### Video-Language Pre-Training (THE NEW WAY)

From **Action100M** research: Train on 100M video-text pairs:
```
Video: Person bending down to pick up object
Text: "picking up item from floor"

Video: Person lying motionless on ground
Text: "person fallen and not moving"
VL-JEPA (Video-Language Joint Embedding Predictive Architecture):

Video encoder → embedding
Text encoder → embedding
Shared space: video_emb · text_emb = similarity

Practical Application to HAVEN
pythonclass ActionRecognizer:
    def __init__(self, model='video_clip'):
        # Use CLIP or similar video-text model
        self.model = load_model(model)
        
        # Define ADL templates
        self.adl_templates = {
            'fall': [
                "person falling to the ground",
                "elderly person collapsed on floor",
                "someone lying motionless"
            ],
            'bed_exit': [
                "person getting out of bed",
                "someone standing up from bed",
                "person leaving bed area"
            ],
            'normal_activity': [
                "person walking normally",
                "someone standing upright",
                "person sitting in chair"
            ]
        }
    
    def classify_activity(self, tracklet_frames):
        """
        Classify activity using video-text similarity.
        
        Args:
            tracklet_frames: 15-30 frames of person
        
        Returns:
            {event_type: confidence_score}
        """
        # Extract video embedding
        video_emb = self.model.encode_video(tracklet_frames)
        
        # Compare to all templates
        results = {}
        for event, templates in self.adl_templates.items():
            # Average similarity to all text variants
            scores = [
                cosine_similarity(video_emb, self.model.encode_text(t))
                for t in templates
            ]
            results[event] = max(scores)
        
        return results
CPU-Friendly Alternative
Full VL-JEPA is heavy. Lightweight option:
python# Use I3D (Inflated 3D ConvNet) pre-trained on Kinetics-400
# Outputs: 400 action classes including:
# - "falling_floor", "getting_up", "lying_down", "standing_up"

def classify_with_i3d(tracklet_frames):
    """Faster than VL-JEPA, still better than rules."""
    action_logits = i3d_model(tracklet_frames)
    
    # Map Kinetics-400 classes to your ADL events
    fall_score = max(
        action_logits['falling_floor'],
        action_logits['fainting']
    )
    
    return {
        'fall': fall_score,
        'normal': action_logits['walking']
    }
```

---

# PART 2: APPLY-TO-HAVEN BLUEPRINT

## 2.1 Learning Roadmap (What to Implement When)

### Phase 1: BASELINE (Week 1-2) - Must Have

**Goal**: Get tracklet-based ReID working with open-set gallery
```
Priority 1.1: Tracklet Building (Day 1-2)
├── Integrate ByteTrack (you already have this)
├── Add tracklet aggregation: sample K frames uniformly
├── Add quality scoring: blur detection, bbox size
└── Output: TrackletSummary objects

Priority 1.2: Global ID Manager (Day 3-5)
├── Create GalleryMemory class
├── Implement two-threshold association
├── Add open-set unknown detection
└── Output: global_id assignment per tracklet

Priority 1.3: Testing (Day 6-7)
├── Test on 3 videos with known ground truth
├── Measure: ID-switch rate, false-match rate
└── Acceptance: <5% errors
```

### Phase 2: ROBUSTNESS (Week 3-4) - Should Have
```
Priority 2.1: Spatio-Temporal Constraints (Day 8-10)
├── Build camera adjacency matrix (manual)
├── Add time window filtering to association
└── Test: reject impossible transitions

Priority 2.2: Pose-Based Features (Day 11-14)
├── Enable YOLO-Pose
├── Extract body proportions (8-D vector)
├── Add gait embedding (simple: avg proportions across frames)
└── Fuse with appearance in similarity function
```

### Phase 3: ENHANCEMENTS (Week 5-6) - Nice to Have
```
Priority 3.1: SAM Masking (Day 15-17)
├── Integrate SAM (mobile_sam for speed)
├── Use only for gallery additions
└── Measure: accuracy gain vs speed cost

Priority 3.2: Action Recognition (Day 18-21)
├── Download I3D checkpoint
├── Run on tracklets for ADL scoring
└── Replace rule-based fall detection
Reading Order

First: Multi-Camera_Industrial_Open-Set.pdf (Sections 3-4)

Focus on: open-set gallery, quality gating, tracking+ReID


Second: spatial_temporal_ReID.pdf (Sections 3-4)

Focus on: camera graph, time windows, joint metric


Third: multicam_streaming_pipeline.pdf (Section 4)

Focus on: OSNet architecture, DeepSORT integration


Skip for now: SAM/Action100M papers (read in Phase 3)


2.2 Data Structures
Core Objects
pythonfrom dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np

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
    tracklets: List[TrackletSummary] = None  # All tracklets for this person
    
    # Prototype embeddings (multi-prototype memory)
    face_prototypes: List[Dict] = None      # [{emb, quality, timestamp}, ...]
    gait_prototypes: List[Dict] = None
    appearance_prototypes: List[Dict] = None
    
    def __post_init__(self):
        if self.tracklets is None:
            self.tracklets = []
        if self.face_prototypes is None:
            self.face_prototypes = []
        if self.gait_prototypes is None:
            self.gait_prototypes = []
        if self.appearance_prototypes is None:
            self.appearance_prototypes = []
    
    def add_tracklet(self, tracklet: TrackletSummary):
        """Add new observation."""
        self.tracklets.append(tracklet)
        self.last_seen_at = tracklet.end_time


@dataclass
class GalleryMemory:
    """
    Open-set gallery that grows dynamically.
    """
    identities: Dict[int, GlobalIdentity]
    next_id: int = 1
    max_prototypes: int = 10
    
    def __post_init__(self):
        if not hasattr(self, 'identities') or self.identities is None:
            self.identities = {}
    
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
        
        # Gait (similar logic)
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
    adjacency: Dict[str, Dict[str, Dict[str, float]]]
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

2.3 Association Logic (PSEUDOCODE)
Tracklet Building
pythondef build_tracklets(video_path, detector, tracker, config):
    """
    Convert video frames → tracklets.
    
    Output: List[TrackletSummary]
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Storage for active tracks
    active_tracks = {}  # {track_id: [frame_data, ...]}
    completed_tracklets = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO detection
        detections = detector(frame)
        
        # ByteTrack updates
        tracks = tracker.update(detections)
        
        # Accumulate frame data for each track
        for track in tracks:
            track_id = track.track_id
            bbox = track.bbox
            conf = track.score
            
            if track_id not in active_tracks:
                active_tracks[track_id] = []
            
            # Store frame data
            active_tracks[track_id].append({
                'frame_idx': frame_idx,
                'timestamp': frame_idx / fps,
                'bbox': bbox,
                'conf': conf,
                'image_crop': extract_crop(frame, bbox),
                'pose': None  # Will add later if pose enabled
            })
            
            # Check if track is lost
            if track.is_lost():
                # Track ended → create tracklet
                tracklet = aggregate_tracklet(
                    track_id,
                    active_tracks[track_id],
                    config
                )
                if tracklet.is_valid():
                    completed_tracklets.append(tracklet)
                
                del active_tracks[track_id]
        
        frame_idx += 1
    
    # Handle remaining active tracks
    for track_id, frames in active_tracks.items():
        tracklet = aggregate_tracklet(track_id, frames, config)
        if tracklet.is_valid():
            completed_tracklets.append(tracklet)
    
    cap.release()
    return completed_tracklets


def aggregate_tracklet(track_id, frame_data, config):
    """
    Aggregate multiple frames → single tracklet.
    
    Key steps:
    1. Sample K high-quality frames
    2. Extract features from each
    3. Average embeddings
    4. Compute quality score
    """
    # Step 1: Sample K frames (uniform + quality-weighted)
    K = config.reid.tracklet_sample_size  # e.g., 5-7
    
    # Score each frame
    scored_frames = [
        (compute_frame_quality(f), f) for f in frame_data
    ]
    scored_frames.sort(reverse=True, key=lambda x: x[0])
    
    # Sample uniformly across time but prioritize high quality
    sampled = temporal_sampling(scored_frames, K)
    
    # Step 2: Extract features
    appearance_embs = []
    face_embs = []
    gait_embs = []
    
    for frame in sampled:
        crop = frame['image_crop']
        
        # Appearance (always)
        if config.reid.appearance.enabled:
            app_emb = appearance_model(crop)
            appearance_embs.append(app_emb)
        
        # Face (if visible)
        if config.reid.face.enabled:
            face_data = face_detector(crop)
            if face_data and face_data['quality'] > 0.7:
                face_embs.append(face_data['embedding'])
        
        # Pose/Gait (if enabled)
        if config.reid.gait.enabled:
            pose = pose_estimator(crop)
            frame['pose'] = pose
    
    # Compute gait from pose sequence
    if len([f for f in sampled if f.get('pose') is not None]) > 5:
        gait_emb = extract_gait_embedding([f['pose'] for f in sampled])
        gait_embs.append(gait_emb)
    
    # Step 3: Aggregate embeddings
    avg_appearance = np.mean(appearance_embs, axis=0) if appearance_embs else None
    avg_face = np.mean(face_embs, axis=0) if face_embs else None
    avg_gait = np.mean(gait_embs, axis=0) if gait_embs else None
    
    # Step 4: Compute quality
    quality = compute_tracklet_quality(frame_data)
    
    # Build tracklet
    tracklet = TrackletSummary(
        camera_id=config.camera_id,
        local_track_id=track_id,
        start_time=frame_data[0]['timestamp'],
        end_time=frame_data[-1]['timestamp'],
        frame_count=len(frame_data),
        bboxes=[f['bbox'] for f in frame_data],
        avg_bbox=np.mean([f['bbox'] for f in frame_data], axis=0),
        appearance_emb=avg_appearance,
        face_emb=avg_face,
        gait_emb=avg_gait,
        quality_score=quality,
        has_face=len(face_embs) > 0,
        has_gait=len(gait_embs) > 0,
        pose_keypoints=[f.get('pose') for f in frame_data]
    )
    
    return tracklet


def compute_frame_quality(frame_data):
    """
    Quality = detection_conf * size_factor * sharpness_factor
    """
    conf = frame_data['conf']
    bbox = frame_data['bbox']
    crop = frame_data['image_crop']
    
    # Size factor (normalize to 100x100)
    size = (bbox[2] * bbox[3]) / 10000
    size_factor = min(size, 1.0)
    
    # Sharpness (Laplacian variance)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_factor = min(laplacian_var / 100, 1.0)
    
    quality = conf * size_factor * sharpness_factor
    return quality


def temporal_sampling(scored_frames, K):
    """
    Sample K frames with temporal diversity.
    
    Don't just take top-K consecutive frames.
    Instead: divide time window into K bins, take best from each bin.
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
        best_in_bin = max(bin_frames, key=lambda x: x[0])
        sampled.append(best_in_bin[1])
    
    return sampled
Open-Set Global Association
pythondef associate_tracklet_to_gallery(tracklet, gallery, camera_graph, config):
    """
    Main association logic: match tracklet to existing ID or create new.
    
    Args:
        tracklet: TrackletSummary
        gallery: GalleryMemory
        camera_graph: CameraGraph
        config: Config object
    
    Returns:
        global_id: int
        confidence: str ('high' | 'medium' | 'low')
        reason: str (for debugging)
    """
    # Edge case: empty gallery
    if not gallery.identities:
        global_id = gallery.create_identity()
        gallery.add_observation(global_id, tracklet)
        return global_id, 'high', 'first_identity'
    
    # Step 1: Compute similarities to all existing identities
    candidates = []
    for global_id, identity in gallery.identities.items():
        # Skip if spatio-temporal constraint violated
        if not is_spatiotemporal_compatible(tracklet, identity, camera_graph):
            continue
        
        # Multi-signal similarity
        sim, conf, signal_type = compute_identity_similarity(tracklet, identity, config)
        
        candidates.append({
            'global_id': global_id,
            'similarity': sim,
            'confidence': conf,
            'signal_type': signal_type
        })
    
    # Sort by similarity
    candidates.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Step 2: Apply open-set decision logic
    if not candidates:
        # No spatiotemporal-compatible candidates
        global_id = gallery.create_identity()
        gallery.add_observation(global_id, tracklet)
        return global_id, 'high', 'spatiotemporal_reject'
    
    best = candidates[0]
    
    # Two-threshold decision
    T_high = config.reid.thresholds.accept  # e.g., 0.75
    T_low = config.reid.thresholds.reject   # e.g., 0.50
    
    if best['similarity'] > T_high:
        # Confident match
        gallery.add_observation(best['global_id'], tracklet)
        return best['global_id'], 'high', f"{best['signal_type']}_{best['similarity']:.2f}"
    
    elif best['similarity'] < T_low:
        # Definitely unknown
        global_id = gallery.create_identity()
        gallery.add_observation(global_id, tracklet)
        return global_id, 'high', f"low_similarity_{best['similarity']:.2f}"
    
    else:
        # Uncertain zone [T_low, T_high]
        # Use additional evidence
        
        # Check margin to second-best
        if len(candidates) > 1:
            margin = best['similarity'] - candidates[1]['similarity']
            if margin < 0.15:  # Too close
                global_id = gallery.create_identity()
                gallery.add_observation(global_id, tracklet)
                return global_id, 'low', f"ambiguous_margin_{margin:.2f}"
        
        # Check quality
        if tracklet.quality_score < 0.6:
            global_id = gallery.create_identity()
            gallery.add_observation(global_id, tracklet)
            return global_id, 'low', 'low_quality_uncertain'
        
        # Check if strong biometric available
        if best['signal_type'] in ['face', 'gait', 'gait_override']:
            # Trust biometric even in uncertain zone
            gallery.add_observation(best['global_id'], tracklet)
            return best['global_id'], 'medium', f"{best['signal_type']}_override"
        
        # Conservative default: create new ID
        global_id = gallery.create_identity()
        gallery.add_observation(global_id, tracklet)
        return global_id, 'low', 'conservative_new'


def is_spatiotemporal_compatible(tracklet, identity, camera_graph):
    """
    Check if tracklet could belong to identity based on physics.
    """
    # Get last observation of this identity
    if not identity.tracklets:
        return True  # First observation for this identity
    
    last_tracklet = identity.tracklets[-1]
    
    # Check time gap and camera transition
    time_gap = tracklet.start_time - last_tracklet.end_time
    
    if time_gap < 0:
        return False  # Time travel impossible
    
    is_possible = camera_graph.is_transition_possible(
        last_tracklet.camera_id,
        tracklet.camera_id,
        time_gap
    )
    
    return is_possible


def compute_identity_similarity(tracklet, identity, config):
    """
    Multi-signal similarity with fusion.
    
    Returns:
        (similarity: float, confidence: str, signal_type: str)
    """
    face_sim = None
    gait_sim = None
    app_sim = None
    
    # Face comparison
    if tracklet.has_face and identity.face_prototypes:
        face_sims = [
            cosine_similarity(tracklet.face_emb, proto['embedding'])
            for proto in identity.face_prototypes
        ]
        face_sim = max(face_sims)
    
    # Gait comparison
    if tracklet.has_gait and identity.gait_prototypes:
        gait_sims = [
            cosine_similarity(tracklet.gait_emb, proto['embedding'])
            for proto in identity.gait_prototypes
        ]
        gait_sim = max(gait_sims)
    
    # Appearance comparison
    if tracklet.appearance_emb is not None and identity.appearance_prototypes:
        app_sims = [
            cosine_similarity(tracklet.appearance_emb, proto['embedding'])
            for proto in identity.appearance_prototypes
        ]
        app_sim = max(app_sims)
    
    # Fusion logic (priority hierarchy)
    T_face = config.reid.thresholds.face_similarity  # 0.6
    T_gait = config.reid.thresholds.gait_similarity  # 0.7
    T_app = config.reid.thresholds.appearance_similarity  # 0.5
    
    # Priority 1: Face
    if face_sim is not None and face_sim > T_face:
        return face_sim, 'high', 'face'
    
    # Priority 2: Gait (check for clothing change)
    if gait_sim is not None and gait_sim > T_gait:
        # Check if appearance contradicts (clothing change indicator)
        if app_sim is not None and app_sim < 0.3:
            # Gait says SAME, appearance says DIFFERENT
            # → Likely clothing change
            return gait_sim, 'medium', 'gait_override'
        return gait_sim, 'medium', 'gait'
    
    # Priority 3: Appearance (conservative threshold)
    if app_sim is not None and app_sim > 0.8:
        return app_sim, 'low', 'appearance_only'
    
    # No strong signal
    if app_sim is not None:
        return app_sim, 'none', 'weak_appearance'
    
    return 0.0, 'none', 'no_signals'

2.4 Anti-False-Match Strategies
Strategy 1: Two-Threshold with Margin Test
pythondef two_threshold_decision(candidates, tracklet, config):
    """
    Prevent false matches using two thresholds + margin.
    """
    if not candidates:
        return 'CREATE_NEW', None
    
    best = candidates[0]
    T_high = 0.75  # Accept
    T_low = 0.50   # Reject
    
    # High confidence match
    if best['similarity'] > T_high:
        return 'ACCEPT', best['global_id']
    
    # Low similarity → definitely new
    if best['similarity'] < T_low:
        return 'CREATE_NEW', None
    
    # UNCERTAIN ZONE: Check margin
    if len(candidates) > 1:
        second_best = candidates[1]
        margin = best['similarity'] - second_best['similarity']
        
        if margin < 0.15:
            # Too close to second-best → ambiguous
            return 'CREATE_NEW', None
    
    # Check quality
    if tracklet.quality_score < 0.6:
        return 'CREATE_NEW', None
    
    # Check signal type
    if best['signal_type'] in ['face', 'gait']:
        return 'ACCEPT', best['global_id']
    
    # Conservative: reject uncertain appearance-only matches
    return 'CREATE_NEW', None
Strategy 2: Quality Gating
pythondef quality_gate(tracklet, min_quality=0.6, min_frames=5):
    """
    Reject low-quality tracklets from ReID.
    """
    if tracklet.frame_count < min_frames:
        return False, "insufficient_frames"
    
    if tracklet.quality_score < min_quality:
        return False, "low_quality"
    
    if not tracklet.has_face and not tracklet.has_gait and tracklet.appearance_emb is None:
        return False, "no_features"
    
    # Check bbox size (too small = far from camera)
    avg_area = tracklet.avg_bbox[2] * tracklet.avg_bbox[3]
    if avg_area < 80 * 80:
        return False, "bbox_too_small"
    
    return True, "ok"
Strategy 3: Cooldown Period
pythondef apply_cooldown(tracklet, identity, cooldown_seconds=10):
    """
    Prevent rapid ID switching.
    
    If we just assigned this identity < cooldown_seconds ago,
    prefer keeping the ID even if similarity is slightly lower.
    """
    if not identity.tracklets:
        return False
    
    last_obs = identity.tracklets[-1]
    time_since_last = tracklet.start_time - last_obs.end_time
    
    if time_since_last < cooldown_seconds:
        # Recent observation → boost this identity's score
        return True
    
    return False


def associate_with_cooldown(tracklet, candidates, gallery, config):
    """
    Modified association with cooldown boost.
    """
    cooldown = config.reid.cooldown_seconds
    
    for cand in candidates:
        identity = gallery.identities[cand['global_id']]
        if apply_cooldown(tracklet, identity, cooldown):
            # Boost score by 0.1
            cand['similarity'] += 0.1
    
    # Re-sort after boost
    candidates.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Continue with normal decision logic
    # ...
```

---

# PART 3: REAL FAILURE CASES

## Case 1: Same Person Changes Clothes

### Scenario
```
t=0:   Person wearing RED shirt in cam1
       → global_id=1 created
       → Appearance: [RED features]
       → Gait: [body proportions X]

t=300: Person changes to BLUE shirt (off-camera)

t=350: Person wearing BLUE shirt in cam2
       → Appearance: [BLUE features] (similarity to RED = 0.3)
       → Gait: [body proportions X] (similarity = 0.85)
Solution
python# Multi-signal fusion with gait override
if gait_sim > 0.7 and app_sim < 0.4:
    # Gait says SAME, appearance says DIFFERENT
    # → Trust gait (clothing change)
    return gait_sim, 'gait_override'
    # Result: Keeps global_id=1 ✅
Acceptance Test
pythondef test_clothing_change():
    # Setup
    gallery = GalleryMemory()
    
    # Add person in red shirt
    tracklet1 = TrackletSummary(
        camera_id='cam1',
        appearance_emb=red_embedding,
        gait_emb=person_A_gait,
        ...
    )
    global_id = gallery.create_identity()
    gallery.add_observation(global_id, tracklet1)
    
    # Same person in blue shirt
    tracklet2 = TrackletSummary(
        camera_id='cam2',
        appearance_emb=blue_embedding,  # Different!
        gait_emb=person_A_gait,         # Same!
        ...
    )
    
    # Associate
    assigned_id, conf, reason = associate_tracklet_to_gallery(
        tracklet2, gallery, camera_graph, config
    )
    
    # Verify: Same global_id
    assert assigned_id == global_id
    assert 'gait' in reason
    print("✅ Clothing change handled correctly")
```

---

## Case 2: Different People with Similar Clothes

### Scenario
```
Person A: White shirt, jeans
Person B: White shirt, jeans (different person!)

Appearance similarity = 0.92 (very high!)
Risk: Incorrectly merge B into A's ID
Solution 1: Face/Gait Discrimination
python# Even if appearance matches, check face/gait
if app_sim > 0.9:  # Suspiciously high appearance match
    if face_sim < 0.4 or gait_sim < 0.5:
        # Appearance says SAME, biometrics say DIFFERENT
        # → Trust biometrics (different people)
        return 'CREATE_NEW'
Solution 2: Margin Test
python# If two identities have similar appearance scores
candidates = [
    {'id': 1, 'sim': 0.92, 'signal': 'appearance'},
    {'id': 2, 'sim': 0.88, 'signal': 'appearance'}
]

margin = 0.92 - 0.88  # = 0.04 (very small!)

if margin < 0.15 and signal_type == 'appearance_only':
    # Too ambiguous → create new ID
    return 'CREATE_NEW'
Solution 3: Spatiotemporal Rejection
python# Person A last seen in cam1 at t=100
# Person B appears in cam1 at t=105 (5 seconds later)
# Same camera → should be same person OR impossible

if same_camera and time_gap < 60:
    if app_sim > 0.9 but face_sim < 0.5:
        # Appearance match but different face
        # → Likely two people in same location
        return 'CREATE_NEW'
Acceptance Test
pythondef test_look_alike():
    gallery = GalleryMemory()
    
    # Add Person A (white shirt)
    tracklet_A = TrackletSummary(
        appearance_emb=white_shirt_emb,
        face_emb=face_A,
        ...
    )
    id_A = gallery.create_identity()
    gallery.add_observation(id_A, tracklet_A)
    
    # Person B (also white shirt, different face)
    tracklet_B = TrackletSummary(
        appearance_emb=white_shirt_emb,  # Similar!
        face_emb=face_B,                 # Different!
        ...
    )
    
    id_B, conf, reason = associate_tracklet_to_gallery(
        tracklet_B, gallery, camera_graph, config
    )
    
    # Verify: Different IDs
    assert id_B != id_A
    print("✅ Look-alike correctly separated")
```

---

## Case 3: Long Absence (Reappear After Hours)

### Scenario
```
t=0:     Person enters, assigned global_id=1
t=3600:  Person leaves (1 hour later)
t=10800: Person returns (3 hours later)

Question: Reuse global_id=1 or create new ID?
Decision Logic
pythondef handle_long_absence(tracklet, identity, config):
    """
    Decide whether to reuse ID after long absence.
    """
    absence_duration = tracklet.start_time - identity.last_seen_at
    
    # Short absence (< 1 hour): Always reuse
    if absence_duration < 3600:
        return 'REUSE', identity.global_id
    
    # Medium absence (1-4 hours): Reuse if strong match
    if absence_duration < 14400:
        # Require high threshold
        if similarity > 0.85 and (has_face or has_gait):
            return 'REUSE', identity.global_id
        else:
            return 'CREATE_NEW', None
    
    # Long absence (> 4 hours): Be very conservative
    if absence_duration > 14400:
        # Only reuse if face match (most reliable)
        if has_face and face_sim > 0.8:
            return 'REUSE', identity.global_id
        else:
            # Create new ID, mark old ID as archived
            identity.status = 'archived'
            return 'CREATE_NEW', None
Practical Rule for HAVEN
python# Configuration
config.reid.reuse_window = 7200  # 2 hours

# In association logic
def should_consider_identity(identity, tracklet, config):
    """Skip identities not seen recently."""
    absence = tracklet.start_time - identity.last_seen_at
    
    if absence > config.reid.reuse_window:
        # Too long → don't compare to this identity
        return False
    
    return True
Acceptance Test
pythondef test_long_absence():
    gallery = GalleryMemory()
    
    # Person at t=0
    tracklet1 = TrackletSummary(
        start_time=0,
        end_time=60,
        face_emb=person_face,
        ...
    )
    id1 = gallery.create_identity()
    gallery.add_observation(id1, tracklet1)
    
    # Same person returns at t=10800 (3 hours)
    tracklet2 = TrackletSummary(
        start_time=10800,
        face_emb=person_face,
        ...
    )
    
    # With face: Should reuse
    id2_with_face, _, _ = associate_tracklet_to_gallery(
        tracklet2, gallery, camera_graph, config
    )
    assert id2_with_face == id1
    
    # Without face: Should create new
    tracklet3 = TrackletSummary(
        start_time=10800,
        appearance_emb=person_appearance,
        has_face=False,
        ...
    )
    id3_no_face, _, _ = associate_tracklet_to_gallery(
        tracklet3, gallery, camera_graph, config
    )
    assert id3_no_face != id1
    
    print("✅ Long absence handled correctly")

PART 4: METRICS & ACCEPTANCE CRITERIA
4.1 Metrics Checklist
Primary Metrics
python@dataclass
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
Evaluation Function
pythondef evaluate_reid_system(ground_truth, predictions):
    """
    Compare predicted global IDs to ground truth.
    
    Args:
        ground_truth: {tracklet_id: true_person_id}
        predictions: {tracklet_id: predicted_global_id}
    
    Returns:
        ReIDMetrics
    """
    metrics = ReIDMetrics()
    
    # Build mapping: true_person_id → predicted_global_ids
    person_to_predicted = {}
    for tracklet_id, true_id in ground_truth.items():
        pred_id = predictions[tracklet_id]
        
        if true_id not in person_to_predicted:
            person_to_predicted[true_id] = []
        person_to_predicted[true_id].append(pred_id)
    
    # Count errors
    for true_id, pred_ids in person_to_predicted.items():
        unique_pred_ids = set(pred_ids)
        
        if len(unique_pred_ids) > 1:
            # Same person got multiple IDs → ID switches
            metrics.id_switches += len(unique_pred_ids) - 1
    
    # Check false matches
    # Build reverse: predicted_global_id → true_person_ids
    predicted_to_persons = {}
    for tracklet_id, pred_id in predictions.items():
        true_id = ground_truth[tracklet_id]
        
        if pred_id not in predicted_to_persons:
            predicted_to_persons[pred_id] = []
        predicted_to_persons[pred_id].append(true_id)
    
    for pred_id, true_ids in predicted_to_persons.items():
        unique_true_ids = set(true_ids)
        
        if len(unique_true_ids) > 1:
            # Different people got same ID → False match
            metrics.false_matches += len(unique_true_ids) - 1
    
    metrics.total_tracklets = len(ground_truth)
    metrics.correct_assignments = (
        metrics.total_tracklets - metrics.id_switches - metrics.false_matches
    )
    
    return metrics
4.2 Acceptance Criteria
Baseline (Phase 1)
yamlMust Pass:
  - ID Switch Rate: < 10%
  - False Match Rate: < 5%
  - Overall Accuracy: > 85%
  - Processing Speed: > 5 FPS (CPU mode)

Should Pass:
  - New IDs Created: Matches true count ± 2
  - Spatiotemporal Rejections: > 0 (feature working)
Enhanced (Phase 2)
yamlMust Pass:
  - ID Switch Rate: < 5%
  - False Match Rate: < 2%
  - Overall Accuracy: > 90%
  - Clothing Change Handling: > 80% correct
  - Processing Speed: > 10 FPS (GPU), > 3 FPS (CPU)

Should Pass:
  - Gait Override Count: > 0 (feature working)
  - Look-Alike Separation: 100% (no false merges)
Production (Phase 3)
yamlMust Pass:
  - ID Switch Rate: < 2%
  - False Match Rate: < 1%
  - Overall Accuracy: > 95%
  - Clothing Change Handling: > 90% correct
  - Long Absence (< 2hr): > 85% reuse correct ID
  - Processing Speed: > 15 FPS (GPU), > 5 FPS (CPU)

Should Pass:
  - All test cases pass
  - SAM improvement: > 5% accuracy gain
  - Action recognition: > 80% ADL classification

4.3 Test Suite
pythonclass ReIDTestSuite:
    """Comprehensive test suite for HAVEN ReID."""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
    
    def run_all_tests(self):
        """Run all test cases."""
        self.test_basic_association()
        self.test_clothing_change()
        self.test_look_alike_separation()
        self.test_spatiotemporal_constraints()
        self.test_long_absence()
        self.test_quality_gating()
        self.test_performance()
        
        self.print_results()
    
    def test_basic_association(self):
        """Test: Same person across cameras gets same ID."""
        print("Running: Basic Association Test...")
        
        gallery = GalleryMemory()
        
        # Person appears in cam1
        tracklet1 = create_test_tracklet(
            camera='cam1',
            person_id='A',
            appearance=person_A_emb,
            face=person_A_face
        )
        id1 = associate(tracklet1, gallery)
        
        # Same person appears in cam2
        tracklet2 = create_test_tracklet(
            camera='cam2',
            person_id='A',  # Ground truth
            appearance=person_A_emb,
            face=person_A_face
        )
        id2 = associate(tracklet2, gallery)
        
        assert id1 == id2, "Same person should get same ID"
        self.results['basic_association'] = 'PASS'
    
    def test_clothing_change(self):
        """Test: Same person with different clothes keeps ID."""
        print("Running: Clothing Change Test...")
        
        gallery = GalleryMemory()
        
        # Person in red shirt
        tracklet1 = create_test_tracklet(
            camera='cam1',
            appearance=red_shirt_emb,
            gait=person_A_gait,
            face=person_A_face
        )
        id1 = associate(tracklet1, gallery)
        
        # Same person in blue shirt
        tracklet2 = create_test_tracklet(
            camera='cam2',
            appearance=blue_shirt_emb,  # Different
            gait=person_A_gait,         # Same
            face=person_A_face          # Same
        )
        id2, conf, reason = associate(tracklet2, gallery)
        
        assert id1 == id2, "Clothing change should not break ID"
        assert 'gait' in reason or 'face' in reason, "Should use biometric"
        self.results['clothing_change'] = 'PASS'
    
    def test_look_alike_separation(self):
        """Test: Different people with similar clothes get different IDs."""
        print("Running: Look-Alike Separation Test...")
        
        gallery = GalleryMemory()
        
        # Person A (white shirt)
        tracklet_A = create_test_tracklet(
            appearance=white_shirt_emb,
            face=face_A
        )
        id_A = associate(tracklet_A, gallery)
        
        # Person B (also white shirt, different face)
        tracklet_B = create_test_tracklet(
            appearance=white_shirt_emb,  # Similar
            face=face_B                  # Different
        )
        id_B = associate(tracklet_B, gallery)
        
        assert id_A != id_B, "Different people should get different IDs"
        self.results['look_alike'] = 'PASS'
    
    def test_spatiotemporal_constraints(self):
        """Test: Impossible transitions are rejected."""
        print("Running: Spatiotemporal Test...")
        
        camera_graph = CameraGraph({
            'cam1': {'cam2': {'min_time': 10, 'max_time': 60}}
        })
        gallery = GalleryMemory()
        
        # Person in cam1 at t=0
        tracklet1 = create_test_tracklet(
            camera='cam1',
            start_time=0,
            end_time=5,
            face=person_A_face
        )
        id1 = associate(tracklet1, gallery, camera_graph)
        
        # Same person in cam2 at t=7 (too fast!)
        tracklet2 = create_test_tracklet(
            camera='cam2',
            start_time=7,
            face=person_A_face
        )
        id2, conf, reason = associate(tracklet2, gallery, camera_graph)
        
        assert id2 != id1, "Impossible transition should create new ID"
        assert 'spatiotemporal' in reason
        self.results['spatiotemporal'] = 'PASS'
    
    def print_results(self):
        """Print test summary."""
        print("\n" + "="*50)
        print("TEST RESULTS")
        print("="*50)
        for test, result in self.results.items():
            status = "✅" if result == "PASS" else "❌"
            print(f"{status} {test}: {result}")
        
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r == "PASS")
        print(f"\nPassed: {passed}/{total} ({passed/total*100:.0f}%)")

PART 5: IMPLEMENTATION CHECKLIST
Week 1-2: Baseline

 Day 1-2: Tracklet Building

 Integrate ByteTrack (if not already done)
 Implement build_tracklets() function
 Add quality scoring (blur, bbox size, confidence)
 Implement temporal sampling (K frames from tracklet)
 Test: Generate tracklets from 1 video


 Day 3-4: Feature Extraction

 Add appearance model (OSNet or ResNet50)
 Implement aggregate_tracklet() function
 Test: Extract embeddings from tracklets


 Day 5-6: Global ID Manager

 Create GalleryMemory class
 Implement associate_tracklet_to_gallery()
 Add two-threshold decision logic
 Add margin test
 Test: Run on 2 videos with 1 shared person


 Day 7: Evaluation

 Create ground truth labels for test videos
 Implement evaluate_reid_system()
 Run baseline tests
 Measure: ID-switch rate, false-match rate



Week 3-4: Robustness

 Day 8-9: Spatiotemporal

 Create CameraGraph class
 Manually define camera adjacency matrix
 Add is_spatiotemporal_compatible()
 Test: Reject impossible transitions


 Day 10-12: Pose/Gait

 Enable YOLO-Pose
 Implement extract_body_proportions()
 Implement extract_gait_embedding()
 Test: Extract gait from walking sequences


 Day 13-14: Multi-Signal Fusion

 Implement compute_identity_similarity()
 Add gait override logic for clothing changes
 Test: Clothing change scenario
 Test: Look-alike separation



Week 5-6: Enhancements

 Day 15-16: SAM Integration

 Download mobile_sam checkpoint
 Implement person segmentation
 Use SAM only for gallery additions
 Measure: Accuracy improvement


 Day 17-18: Action Recognition

 Download I3D or CLIP checkpoint
 Implement action classification
 Integrate with ADL engine
 Test: Fall detection accuracy


 Day 19-21: Polish & Testing

 Run full test suite
 Optimize performance (profiling)
 Add logging and visualization
 Document configuration options




FINAL THOUGHTS
This blueprint gives you a concrete, step-by-step path from research papers to working code. The key insights:

Tracklets > Frames: Always aggregate before matching
Spatio-temporal: Physics eliminates 70%+ false matches
Open-set: Two thresholds + margin + quality gating
Clothing change: Gait/face override appearance
Conservative: Prefer creating new ID over false match