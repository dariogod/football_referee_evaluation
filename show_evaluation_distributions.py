import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from collections import defaultdict
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from matplotlib import cm
from scipy.interpolate import griddata
from matplotlib.image import imread
import math

# KDE parameters (same as ground truth script)
KDE_THRESHOLD = 0.05
KDE_BANDWIDTH_ADJUSTMENT = 1.5
KDE_LEVELS = 10

# Define pitch dimensions in meters
PITCH_LENGTH = 105  # meters
PITCH_WIDTH = 68    # meters

# Base path for test directories
base_path = "data/predictions/SoccerNet/SN-GSR-2025/test"
model = "yolo" # "yolo" or "dfine"

# Create output directory for plots if it doesn't exist
output_dir = os.path.join("plots", model)
os.makedirs(output_dir, exist_ok=True)

# Initialize dictionaries to store metrics
referee_distances = []
referee_angles = []
inside_rectangle_counts = defaultdict(int)

# Arrays to store referee positions
referee_positions_x = []
referee_positions_y = []
pitch_width = 0
pitch_height = 0

# Arrays to store paired distance and angle for duels (for scatter plot)
duel_distances = []
duel_angles = []

# Initialize frame counters
total_frames = 0
frames_with_referee = 0

# Find all SNGS-XXX folders in the test directory
sngs_folders = glob.glob(os.path.join(base_path, "SNGS-*"))

# Process each SNGS folder
processed_files = 0
for sngs_folder in sngs_folders:
    results_file = os.path.join(sngs_folder, f"evaluator_{model}", "evaluation_results.json")
    
    # Skip if the file doesn't exist
    if not os.path.exists(results_file):
        print(f"Skipping {sngs_folder}: evaluation_results.json not found")
        continue
    
    # Load the evaluation results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    processed_files += 1
    print(f"Processing {sngs_folder} ({len(results)} frames)")
    
    # Extract metrics from results
    for frame_id, frame_data in results.items():
        total_frames += 1
        
        if frame_data is None:
            continue
        
        frames_with_referee += 1
        
        # Track if referee is inside rectangle
        inside_rectangle_counts[frame_data["inside_rectangle"]] += 1
        
        # Extract referee positions for heatmap
        scale = None
        if "referee_position" in frame_data:
            x = frame_data["referee_position"]["x"]
            y = frame_data["referee_position"]["y"]
            x_max = frame_data["referee_position"]["x_max"]
            y_max = frame_data["referee_position"]["y_max"]
    
            scale_x = x_max / 105
            scale_y = y_max / 68
            scale = (scale_x + scale_y) / 2
            
            # Convert pixel coordinates to meter coordinates 
            # Assuming the evaluation uses image coordinates with (0,0) at top-left
            # Convert to pitch coordinates with (0,0) at bottom-left, scaled to meters
            ref_x_meters = (x / x_max) * PITCH_LENGTH
            ref_y_meters = (y / y_max) * PITCH_WIDTH 
            
            # Store position in meters
            referee_positions_x.append(ref_x_meters)
            referee_positions_y.append(ref_y_meters)
            
            # Update pitch dimensions
            pitch_width = max(pitch_width, x_max)
            pitch_height = max(pitch_height, y_max)
        
        # Extract referee angle for the closest 2 duels that are between 5 and 20 meters away
        if "duels" in frame_data and len(frame_data["duels"]) > 0 and scale is not None:
            filtered_duels = []
            for duel in frame_data["duels"]:
                # Calculate referee distance if not present
                if "referee_distance" not in duel and "player_1" in duel and "player_2" in duel:
                    # Calculate center point of the duel (in pixels)
                    duel_x = (duel["player_1"]["x"] + duel["player_2"]["x"]) / 2
                    duel_y = (duel["player_1"]["y"] + duel["player_2"]["y"]) / 2
                    
                    # Calculate distance from referee to duel center in pixels, then convert to meters
                    dx = duel_x - x  # Using original pixel coordinates
                    dy = duel_y - y
                    duel["referee_distance"] = math.sqrt(dx**2 + dy**2) / scale
                    filtered_duels.append(duel)
            
                # Store paired distance and angle for each duel that has both metrics
                if "referee_distance" in duel and "referee_angle" in duel:
                    distance_in_meters = duel["referee_distance"]
                    duel_distances.append(distance_in_meters)
                    duel_angles.append(duel["referee_angle"])
                    
            # Sort filtered duels by distance
            sorted_duels = sorted(filtered_duels, 
                                 key=lambda d: d.get("referee_distance", float('inf')))
            
            # Take only the closest 2 duels within the distance range
            closest_duels = sorted_duels[:min(len(sorted_duels), len(sorted_duels))]
            
            for duel in closest_duels:
                if "referee_angle" in duel:
                    referee_angles.append(duel["referee_angle"])
        
        # Extract distance from potential action points and convert to meters
        if scale is None:
            continue
    
        for action_point in frame_data.get("potential_action_points", []):
            if "referee_distance" in action_point:
                # Convert distance from pixels to meters
                distance_in_meters = action_point["referee_distance"] / scale
                referee_distances.append(distance_in_meters)

print(f"Processed {processed_files} evaluation files")
print(f"Total data points: {len(referee_distances)} distances, {len(referee_angles)} angles")

# Print statistics similar to ground truth script
print("\n" + "="*60)
print("EVALUATION STATISTICS")
print("="*60)
print(f"Processed files: {processed_files}")
print(f"Total frames: {total_frames}")
print(f"Frames with referee detected: {frames_with_referee}")
if total_frames > 0:
    percentage = (frames_with_referee / total_frames) * 100
    print(f"Percentage with referee detected: {percentage:.2f}%")
print(f"Distance measurements: {len(referee_distances)}")
print(f"Angle measurements: {len(referee_angles)}")
print("="*60)

# Create plots using aggregated data
plt.figure(figsize=(15, 10))

# 1. Referee Distance Distribution
plt.subplot(2, 2, 1)
plt.hist(referee_distances, bins=30, alpha=0.7, color='blue')
plt.title('Distribution of Referee Distances to Action Points')
plt.xlabel('Distance (meters)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# 2. Referee Angle Distribution
plt.subplot(2, 2, 2)
plt.hist(referee_angles, bins=30, alpha=0.7, color='green')
plt.title('Distribution of Referee Angles to Duels')
plt.xlabel('Angle (degrees)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# 3. Inside Rectangle Ratio
plt.subplot(2, 2, 3)
labels = ['Outside Rectangle', 'Inside Rectangle']
values = [inside_rectangle_counts[False], inside_rectangle_counts[True]]
plt.bar(labels, values, color=['red', 'blue'])
plt.title('Referee Position Relative to Rectangle')
plt.ylabel('Number of Frames')
plt.grid(True, alpha=0.3)

# 4. Distance vs Angle Scatter Plot (if both metrics are available)
if duel_distances and duel_angles:
    # Take the minimum of the lengths to avoid index errors
    assert len(duel_distances) == len(duel_angles)
    plt.subplot(2, 2, 4)
    plt.scatter(duel_distances, duel_angles, alpha=0.5, s=10)
    plt.title('Referee Distance vs Angle (Duels)')
    plt.xlabel('Distance (meters)')
    plt.ylabel('Angle (degrees)')
    plt.grid(True, alpha=0.3)

# Add some stats as text
total_frames = sum(inside_rectangle_counts.values())
plt.figtext(0.5, 0.01, f'Total frames: {total_frames}\n'
                       f'Average distance: {np.mean(referee_distances):.2f} meters\n'
                       f'Average angle: {np.mean(referee_angles):.2f} degrees', 
            ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.suptitle('Referee Evaluation Distributions (Aggregated)', fontsize=16)

# Save the figure
plt.savefig(os.path.join(output_dir, 'aggregated_evaluation_distributions.png'), dpi=300)
plt.close()

# Create additional detailed plots
# 1. Histogram of distances with a vertical line for the average
plt.figure(figsize=(10, 6))
plt.hist(referee_distances, bins=50, alpha=0.7, color='blue')
plt.axvline(x=np.mean(referee_distances), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(referee_distances):.2f}')
plt.axvline(x=np.median(referee_distances), color='green', linestyle='dashed', linewidth=2, label=f'Median: {np.median(referee_distances):.2f}')
plt.title('Detailed Distribution of Referee Distances to Action Points (Aggregated)')
plt.xlabel('Distance (meters)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'aggregated_detailed_distance_distribution.png'), dpi=300)
plt.close()

# 2. Histogram of angles with a vertical line for the average
plt.figure(figsize=(12, 12))  # Changed to square figure
plt.hist(referee_angles, bins=50, alpha=0.7, color='green')
plt.axvline(x=np.mean(referee_angles), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(referee_angles):.2f}')
plt.axvline(x=np.median(referee_angles), color='blue', linestyle='dashed', linewidth=2, label=f'Median: {np.median(referee_angles):.2f}')
plt.title('Detailed Distribution of Referee Angles to Duels', fontsize=22)  # Increased title size
plt.xlabel('Angle (degrees)', fontsize=20)  # Increased label size
plt.ylabel('Frequency', fontsize=20)  # Increased label size
plt.xticks([0, 15, 30, 45, 60, 75, 90], fontsize=20)  # Set specific x-axis ticks
plt.yticks(fontsize=14)  # Increased tick label size
plt.legend(fontsize=20)  # Increased legend text size
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'aggregated_detailed_angle_distribution.png'), dpi=300)
plt.close()

# 3. Create heatmap of referee positions
if referee_positions_x and referee_positions_y:
    # 3a. Create heatmap of referee positions (ORIGINAL)
    plt.figure(figsize=(14, 10))
    ax = plt.subplot(1, 1, 1)
    
    # Use the soccer pitch image as background
    if os.path.exists('src/utils/pitch_2.png'):
        pitch_img = imread('src/utils/pitch_2.png')
        # Assume the pitch image dimensions match the meter coordinates
        ax.imshow(pitch_img, extent=[0, PITCH_LENGTH, PITCH_WIDTH, 0])
    
    # Create kernel density estimate plot using original data only
    kde_plot = sns.kdeplot(
        x=referee_positions_x,
        y=referee_positions_y,
        cmap="YlOrRd",
        fill=True,
        alpha=0.7,
        levels=KDE_LEVELS,
        thresh=KDE_THRESHOLD,
        bw_adjust=KDE_BANDWIDTH_ADJUSTMENT
    )
    
    plt.xlim(0, PITCH_LENGTH)
    plt.ylim(0, PITCH_WIDTH)
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    
    plt.xlabel('X Position (meters)', fontsize=20)
    plt.ylabel('Y Position (meters)', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aggregated_referee_position_kde_original.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3b. Create heatmap of referee positions (SYMMETRICALLY ADJUSTED)
    plt.figure(figsize=(14, 10))
    ax = plt.subplot(1, 1, 1)
    
    # Create symmetrical data by adding 180-degree rotated positions
    symmetrical_x = []
    symmetrical_y = []
    
    # Add original positions
    symmetrical_x.extend(referee_positions_x)
    symmetrical_y.extend(referee_positions_y)
    
    # Add 180-degree rotated positions around the center of the pitch
    for x, y in zip(referee_positions_x, referee_positions_y):
        # Rotate 180 degrees around center (52.5, 34)
        rotated_x = PITCH_LENGTH - x
        rotated_y = PITCH_WIDTH - y
        symmetrical_x.append(rotated_x)
        symmetrical_y.append(rotated_y)
    
    # Use the soccer pitch image as background
    if os.path.exists('src/utils/pitch_2.png'):
        pitch_img = imread('src/utils/pitch_2.png')
        # Assume the pitch image dimensions match the meter coordinates
        ax.imshow(pitch_img, extent=[0, PITCH_LENGTH, PITCH_WIDTH, 0])
    
    # Create kernel density estimate plot using symmetrical data
    kde_plot = sns.kdeplot(
        x=symmetrical_x,
        y=symmetrical_y,
        cmap="YlOrRd",
        fill=True,
        alpha=0.7,
        levels=KDE_LEVELS,
        thresh=KDE_THRESHOLD,
        bw_adjust=KDE_BANDWIDTH_ADJUSTMENT
    )
    
    plt.xlim(0, PITCH_LENGTH)
    plt.ylim(0, PITCH_WIDTH)
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    
    plt.xlabel('X Position (meters)', fontsize=20)
    plt.ylabel('Y Position (meters)', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aggregated_referee_position_kde_symmetrical.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3c. Create scatter plot of raw referee positions
    plt.figure(figsize=(14, 10))
    ax = plt.subplot(1, 1, 1)
    
    # Create symmetrical data by adding 180-degree rotated positions
    symmetrical_x = []
    symmetrical_y = []
    
    # Add original positions
    symmetrical_x.extend(referee_positions_x)
    symmetrical_y.extend(referee_positions_y)
    
    # Add 180-degree rotated positions around the center of the pitch
    for x, y in zip(referee_positions_x, referee_positions_y):
        # Rotate 180 degrees around center (52.5, 34)
        rotated_x = PITCH_LENGTH - x
        rotated_y = PITCH_WIDTH - y
        symmetrical_x.append(rotated_x)
        symmetrical_y.append(rotated_y)
    
    # Use the soccer pitch image as background
    if os.path.exists('src/utils/pitch_2.png'):
        pitch_img = imread('src/utils/pitch_2.png')
        # Assume the pitch image dimensions match the meter coordinates
        ax.imshow(pitch_img, extent=[0, PITCH_LENGTH, PITCH_WIDTH, 0])
    
    # Create scatter plot of all referee positions
    plt.scatter(symmetrical_x, symmetrical_y, alpha=0.4, s=8, c='red', edgecolors='darkred', linewidth=0.2)
    
    plt.xlim(0, PITCH_LENGTH)
    plt.ylim(0, PITCH_WIDTH)
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    
    plt.title(f'Raw Referee Positions Scatter Plot - Symmetrical Distribution (n={len(symmetrical_x)}) - Evaluation Data', fontsize=16)
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    
    plt.savefig(os.path.join(output_dir, 'aggregated_referee_positions_scatter.png'), dpi=300)
    plt.close()

# NEW: Fit curve on distance distribution
# Define the distribution functions to fit
def gamma_distribution(x, shape, scale):
    return stats.gamma.pdf(x, shape, scale=scale)

# Create histogram for the distances
plt.figure(figsize=(12, 12))  # Changed to square figure
n, bins, _ = plt.hist(referee_distances, bins=40, density=True, alpha=0.6, color='blue', label='Data')

# Calculate bin centers for curve fitting
bin_centers = 0.5 * (bins[1:] + bins[:-1])

# Fit curves to the distribution
try:
    # Gamma distribution
    gamma_params, _ = curve_fit(gamma_distribution, bin_centers, n, p0=[2, 1])
    gamma_curve = gamma_distribution(bin_centers, *gamma_params)
    
    # Plot the fitted curve
    # Create a smoother curve with more points for plotting
    x_smooth = np.linspace(min(bin_centers), max(bin_centers), 1000)
    gamma_curve_smooth = gamma_distribution(x_smooth, *gamma_params)
    plt.plot(x_smooth, gamma_curve_smooth, 'r-', lw=4, label=f'Gamma Distribution (α={gamma_params[0]:.2f}, θ={gamma_params[1]:.2f})')
    
    # Calculate mean and median
    mean_distance = np.mean(referee_distances)
    median_distance = np.median(referee_distances)
    
    # Add vertical lines for mean and median
    plt.axvline(x=mean_distance, color='black', linestyle='--', alpha=0.7, label=f'Mean: {mean_distance:.2f}m')
    plt.axvline(x=median_distance, color='black', linestyle=':', alpha=0.7, label=f'Median: {median_distance:.2f}m')
    
    # Calculate goodness of fit (R-squared)
    residuals_gamma = n - gamma_curve
    ss_res_gamma = np.sum(residuals_gamma**2)
    ss_tot = np.sum((n - np.mean(n))**2)
    r_squared_gamma = 1 - (ss_res_gamma / ss_tot)
    
    plt.text(0.25, 0.95, f'Gamma R²: {r_squared_gamma:.3f}', 
             transform=plt.gca().transAxes, fontsize=20, verticalalignment='top',  # Increased text size
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
except Exception as e:
    print(f"Error fitting distance distribution: {e}")

plt.title('Referee Distance Distribution with Fitted Gamma Curve', fontsize=22)  # Increased title size
plt.xlabel('Distance (meters)', fontsize=20)  # Increased label size
plt.ylabel('Probability Density', fontsize=20)  # Increased label size
plt.grid(True, alpha=0.3)
plt.legend(fontsize=20)  # Increased legend text size
plt.xticks(fontsize=20)  # Increased tick label size
plt.yticks(fontsize=14)  # Increased tick label size
plt.savefig(os.path.join(output_dir, 'distance_distribution_fitted.png'), dpi=300)
plt.close()

print(f"All plots saved to {output_dir} directory")