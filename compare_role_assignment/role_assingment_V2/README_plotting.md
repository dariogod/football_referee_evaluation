# LAB Colorspace Clustering Visualization

This feature adds visualization capabilities to the role assignment system, specifically for plotting clustering results in LAB colorspace for frame 10.

## Features

- **3D LAB Color Space Visualization**: Plot clustering results in the perceptually uniform LAB color space
- **Original Color Representation**: Each point is colored with its original RGB color for easy identification
- **Different Markers for Different Groups**:
  - **Circles (○)**: First team (left team)
  - **Squares (■)**: Second team (right team)  
  - **Crosses (×)**: Outliers/Referees
- **Frame-Specific Plotting**: Automatically generates plots for frame 10
- **High-Quality Output**: Saves plots as 300 DPI PNG files

## Usage

### Automatic Plotting
The plotting is automatically triggered when processing frame 10 with any role assigner. Simply run your normal role assignment process:

```python
# The plotting will automatically occur for frame 10
persons_with_roles = role_assigner.assign_roles(image, persons_with_color, frame_number=10)
```

### Manual Testing
Use the provided test script:

```bash
python test_frame_10_plotting.py
```

This script will:
1. Find the first available test folder with `color_assignments.json`
2. Load frame 10 data
3. Run the DBScan role assigner
4. Generate and save the LAB colorspace clustering plot
5. Display role assignment statistics

## Output

The plots are saved to a directory named `clustering_plots_{MethodName}/` with the filename format:
```
frame_010_lab_clustering.png
```

## Plot Description

- **X-axis**: a* (Green-Red axis) - negative values are more green, positive values are more red
- **Y-axis**: L* (Lightness) - 0 is black, 100 is white
- **Z-axis**: b* (Blue-Yellow axis) - negative values are more blue, positive values are more yellow
- **Colors**: Each point is colored with its original RGB jersey color
- **Legend**: Shows the different groups (Team 1, Team 2, Referees)
- **Viewing Angle**: Optimized for best visualization of the 3D data

## Technical Details

- Uses matplotlib's 3D scatter plot capabilities
- LAB color conversion via scikit-image
- Automatic legend generation
- Black edge lines for filled markers to improve visibility
- Saves plots at 300 DPI for publication quality

## Examples

When you run the system on frame 10, you'll see output like:
```
Processing frame 10...
Frame 10 has 17 persons
Saved clustering plot to: clustering_plots_DBScanRoleAssigner/frame_010_lab_clustering.png
Role assignment complete. Found 17 persons with roles.

Role assignments by color space:
  RGB: {'player_left': 9, 'player_right': 8}
  LAB: {'player_left': 8, 'referee': 1, 'player_right': 8}
  HSV: {'referee': 4, 'player_left': 7, 'player_right': 6}
```

The plot will show how the LAB clustering algorithm separates the jersey colors into teams and identifies referees as outliers. 