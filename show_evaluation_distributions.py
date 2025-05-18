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

# Base path for test directories
base_path = "data/predictions/SoccerNet/SN-GSR-2025/test"

# Create output directory for plots if it doesn't exist
output_dir = "plots"
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

# Find all SNGS-XXX folders in the test directory
sngs_folders = glob.glob(os.path.join(base_path, "SNGS-*"))

# Process each SNGS folder
processed_files = 0
for sngs_folder in sngs_folders:
    results_file = os.path.join(sngs_folder, "evaluator_yolo", "evaluation_results.json")
    
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
        if frame_data is None:
            continue
    
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
            
            # Store position
            referee_positions_x.append(x)
            referee_positions_y.append(y)
            
            # Update pitch dimensions
            pitch_width = max(pitch_width, x_max)
            pitch_height = max(pitch_height, y_max)
        
        # Extract referee distance and angle for each duel
        for duel in frame_data.get("duels", []):
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
if referee_distances and referee_angles:
    # Take the minimum of the lengths to avoid index errors
    min_length = min(len(referee_distances), len(referee_angles))
    plt.subplot(2, 2, 4)
    plt.scatter(referee_distances[:min_length], referee_angles[:min_length], alpha=0.5, s=10)
    plt.title('Referee Distance vs Angle')
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
plt.figure(figsize=(10, 6))
plt.hist(referee_angles, bins=50, alpha=0.7, color='green')
plt.axvline(x=np.mean(referee_angles), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(referee_angles):.2f}')
plt.axvline(x=np.median(referee_angles), color='blue', linestyle='dashed', linewidth=2, label=f'Median: {np.median(referee_angles):.2f}')
plt.title('Detailed Distribution of Referee Angles to Duels (Aggregated)')
plt.xlabel('Angle (degrees)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'aggregated_detailed_angle_distribution.png'), dpi=300)
plt.close()

# 3. Create heatmap of referee positions
if referee_positions_x and referee_positions_y:
    # Alternative heatmap using seaborn's kdeplot (smoother representation)
    plt.figure(figsize=(14, 10))
    ax = plt.subplot(1, 1, 1)
    
    # Use the soccer pitch image as background
    pitch_img = imread('src/utils/pitch.png')
    ax.imshow(pitch_img, extent=[0, pitch_width, pitch_height, 0])
    
    # Create kernel density estimate plot
    sns.kdeplot(
        x=referee_positions_x,
        y=referee_positions_y,
        cmap="YlOrRd",
        fill=True,
        alpha=0.7,
        levels=5,
        thresh=0.3
    )
    
    plt.xlim(0, pitch_width)
    plt.ylim(0, pitch_height)
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    
    plt.title('Referee Position Density Map (KDE) - Aggregated', fontsize=16)
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    
    plt.savefig(os.path.join(output_dir, 'aggregated_referee_position_kde.png'), dpi=300)
    plt.close()

# NEW: Fit curve on distance distribution
# Define the distribution functions to fit
def gamma_distribution(x, shape, scale):
    return stats.gamma.pdf(x, shape, scale=scale)

# Create histogram for the distances
plt.figure(figsize=(12, 8))
n, bins, _ = plt.hist(referee_distances, bins=40, density=True, alpha=0.6, color='blue', label='Data')

# Calculate bin centers for curve fitting
bin_centers = 0.5 * (bins[1:] + bins[:-1])

# Fit curves to the distribution
try:
    # Gamma distribution
    gamma_params, _ = curve_fit(gamma_distribution, bin_centers, n, p0=[2, 1])
    gamma_curve = gamma_distribution(bin_centers, *gamma_params)
    
    # Plot the fitted curve
    plt.plot(bin_centers, gamma_curve, 'r-', lw=2, label=f'Gamma Distribution (shape={gamma_params[0]:.2f}, scale={gamma_params[1]:.2f})')
    
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
    
    plt.text(0.05, 0.95, f'Gamma R²: {r_squared_gamma:.3f}', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
except Exception as e:
    print(f"Error fitting distance distribution: {e}")

plt.title('Referee Distance Distribution with Fitted Gamma Curve')
plt.xlabel('Distance (meters)')
plt.ylabel('Probability Density')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(os.path.join(output_dir, 'distance_distribution_fitted.png'), dpi=300)
plt.close()

print(f"All plots saved to {output_dir} directory")