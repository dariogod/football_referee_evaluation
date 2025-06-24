# Example Predictions Visualizers

This folder contains separate visualization modules for different aspects of football referee evaluation analysis.

## Files Overview

### 1. `minimap_visualizer.py`
Basic minimap visualization showing players on the pitch with team colors and roles.

**Features:**
- Displays players as colored circles based on their team/role
- Team A: White circles
- Team B: Red circles  
- Referee: Yellow circles
- Goalkeeper: Black circles

**Usage:**
```python
from minimap_visualizer import MinimapVisualizer

visualizer = MinimapVisualizer()
visualizer.visualize_multiple_frames(detections, frame_ids, output_dir)
```

### 2. `decision_critical_zone_visualizer.py`
Advanced visualization for decision critical zones using clustering and gamma distribution heatmaps.

**Features:**
- Identifies clusters of players from opposing teams
- Creates gamma distribution heatmaps around decision critical zones
- Evaluates referee positioning relative to critical zones
- Provides both cluster visualization and heatmap visualization
- **Player/referee annotations are drawn on top of heatmap layers for clear visibility**

**Usage:**
```python
from decision_critical_zone_visualizer import DecisionCriticalZoneVisualizer

visualizer = DecisionCriticalZoneVisualizer()
visualizer.visualize_multiple_frames(detections, frame_ids, output_dir, include_heatmap=True)
```

### 3. `angle_duel_visualizer.py`
Specialized visualization for player duels and angular positioning analysis.

**Features:**
- Identifies duels between opposing players (within 3m)
- Calculates angles between duel lines and referee sight lines
- Creates angular heatmaps using cosine scoring functions
- Evaluates referee positioning relative to duel orientations
- **Player/referee annotations are drawn on top of heatmap layers for clear visibility**

**Usage:**
```python
from angle_duel_visualizer import AngleDuelVisualizer

visualizer = AngleDuelVisualizer()
visualizer.visualize_multiple_frames(detections, frame_ids, output_dir, include_heatmap=True)
```

## Key Features

### Layer Ordering for Clear Visualization
All heatmap visualizations follow this layer ordering (bottom to top):
1. **Background**: Pitch image (30% opacity)
2. **Heatmap**: Gamma/Angular distribution (70% opacity) 
3. **Player annotations**: Colored circles with black edges
4. **Referee annotations**: Yellow circles with black edges (larger size)
5. **Additional elements**: Cluster centers, duel lines, etc.

This ensures that player and referee positions are clearly visible on top of the heatmap data.

### Output Files

Each visualizer generates specific output files:

**MinimapVisualizer:**
- `minimap_{frame_id:06d}.png`

**DecisionCriticalZoneVisualizer:**
- `minimap_decision_critical_zones_{frame_id:06d}.png` (cluster visualization)
- `minimap_heatmap_{frame_id:06d}.png` (gamma heatmap)

**AngleDuelVisualizer:**
- `minimap_duels_{frame_id:06d}.png` (duel visualization)
- `minimap_duel_heatmap_{frame_id:06d}.png` (angular heatmap)

## Dependencies

Required packages:
- `opencv-python` (cv2)
- `numpy`
- `matplotlib`
- `scikit-learn` (for DBSCAN clustering)
- `scipy` (for gamma distribution)

## Example Usage

```python
import json
from minimap_visualizer import MinimapVisualizer
from decision_critical_zone_visualizer import DecisionCriticalZoneVisualizer  
from angle_duel_visualizer import AngleDuelVisualizer

# Load detection data
with open("data/example/predictions/role_assignment/detections.json", "r") as f:
    detections = json.load(f)

# Define frames to analyze
interesting_frames = [212, 400, 460]
output_dir = "data/example/images_for_paper"

# Create all visualizations
minimap_viz = MinimapVisualizer()
minimap_viz.visualize_multiple_frames(detections, interesting_frames, output_dir)

critical_zone_viz = DecisionCriticalZoneVisualizer()
critical_zone_viz.visualize_multiple_frames(detections, interesting_frames, output_dir)

duel_viz = AngleDuelVisualizer()
duel_viz.visualize_multiple_frames(detections, interesting_frames, output_dir)
```

## Configuration

All visualizers support customization through their constructor parameters:
- `pitch_image_path`: Path to the pitch background image (default: "src/utils/pitch_2.png")
- Various threshold and styling parameters can be modified in the class methods

## Notes

- All coordinates are normalized to standard football pitch dimensions (105m x 68m)
- The visualizers automatically handle coordinate transformations between different coordinate systems
- Heatmap resolution is optimized for good visual quality while maintaining reasonable computation time
- The cosine scoring function in `AngleDuelVisualizer` provides smooth transitions for angular evaluation 