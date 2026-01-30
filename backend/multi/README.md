# HAVEN Multi-Camera System

## 📁 Structure

```
backend/multi/
├── config.yaml      # Main configuration
├── run.py          # Main runner script
├── reid.py         # Color Histogram ReID
├── adl.py          # ADL classification
├── visualize.py    # Visualization utilities
└── README.md       # This file
```

## 🚀 Quick Start

```bash
# From project root
.\sequential.bat

# Or directly
python backend\multi\run.py
```

## ⚙️ Configuration

Edit `config.yaml`:

```yaml
# Cameras
cameras:
  - id: "cam1"
    video_path: "path/to/video1.mp4"
    enabled: true

# ReID threshold (0.0-1.0)
reid:
  threshold: 0.55      # Lower = easier match
  update_interval: 5   # Match every N frames

# ADL thresholds
adl:
  movement_threshold_ratio: 0.03  # Walking detection
```

## 🎯 Features

### 1. Master-Slave ReID
- **CAM1 (Master)**: Creates new Global IDs
- **CAM2-4 (Slave)**: Only matches existing IDs

### 2. Color Histogram Matching
- HSV histogram in 3 parts (head/body/legs)
- Cosine similarity matching
- Feature update every 5 frames

### 3. ADL Classification
- **STANDING**: Upright, no movement
- **WALKING**: Upright, moving (threshold: 0.03 * frame_height)
- **SITTING**: Bent knees (<130°)
- **LAYING**: Torso angle >50° or aspect ratio >1.2

### 4. Event Detection
- Fall detection (transition to LAYING)
- Hand raise (left/right)
- Sit down / Stand up

## ⌨️ Controls

| Key | Action |
|-----|--------|
| SPACE | Pause/Resume |
| N | Skip to next camera |
| Q | Quit |

## 📊 Output

Console shows:
- 🆕 New ID created (CAM1)
- 🔗 ID matched (CAM2-4)
- Summary at end

## 🔧 Troubleshooting

### ID not matching?
- Lower `reid.threshold` (e.g., 0.45)
- Check lighting/clothing consistency

### Walking detection wrong?
- Lower `adl.movement_threshold_ratio` (e.g., 0.02)
- Increase observation window
