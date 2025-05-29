# Augmented PrtReid System

This system augments prtreid predictions by reassigning low-confidence "unclassified" predictions back to "player" based on jersey color clustering.

## Overview

The augmentation process works as follows:

1. **Load predictions** from `prtreid_output` directory
2. **Apply confidence threshold** (default: 3.564) to turn low-confidence player predictions into "unclassified"
3. **For each frame with unclassified predictions:**
   - Extract jersey colors from all player and unclassified predictions
   - Perform k-means clustering (k=2) on player jersey colors in RGB space
   - Create convex hulls around each cluster
   - Check if unclassified predictions fall within any cluster
   - Reassign matching unclassified predictions back to "player"
4. **Save augmented results** to `augmented_output` directory

## Files

- `augmented_prtreid.py` - Main script with high-level logic
- `utils/data_loader.py` - Functions for loading and saving predictions
- `utils/color_utils.py` - Jersey color extraction and clustering
- `utils/geometry_utils.py` - Convex hull and point-in-polygon checks
- `evaluate_augmented.py` - Evaluation script for augmented results
- `confusion_matrix.py` - Original evaluation script

## Usage

### Running the augmentation:

```bash
python augmented_prtreid.py
```

### Evaluating augmented results:

```bash
# Evaluate augmented results only
python evaluate_augmented.py

# Compare with original results
python evaluate_augmented.py compare
```

### Evaluating original results with confidence threshold:

```bash
python confusion_matrix.py
```

## Configuration

You can modify the configuration in `augmented_prtreid.py`:

```python
config = {
    'base_dir': 'prtreid_output',           # Input directory
    'image_base_dir': 'data/SoccerNet/SN-GSR-2025/test',  # Image directory
    'confidence_threshold': 3.564,           # Confidence threshold for player class
    'output_dir': 'augmented_output'        # Output directory
}
```

## Dependencies

- numpy
- opencv-python
- scikit-learn
- scipy
- matplotlib
- seaborn

## Directory Structure

```
prtreid_output/
├── SNGS-XXX/
│   ├── 000YYY/
│   │   └── reid_results.json
│   └── ...
└── ...

augmented_output/
├── SNGS-XXX/
│   ├── 000YYY/
│   │   └── augmented_reid_results.json
│   └── ...
└── ...
```

## Algorithm Details

### Jersey Color Extraction
- Crops upper 50% of bounding box (torso area)
- Uses k-means clustering (k=2) to separate jersey from background
- Selects cluster with higher color saturation as jersey

### Color Clustering
- Applies k-means (k=2) on RGB values of all player predictions
- Creates convex hulls around clusters in RGB space
- Falls back to bounding boxes if convex hull fails

### Reassignment Logic
- Unclassified predictions are reassigned to "player" if their jersey color falls within any player color cluster
- Predictions outside all clusters remain "unclassified" 