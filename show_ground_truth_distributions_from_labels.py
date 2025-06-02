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
from sklearn.cluster import DBSCAN
import pickle
from scipy.stats import gaussian_kde

KDE_THRESHOLD = 0.00
KDE_BANDWIDTH_ADJUSTMENT = 1.5
KDE_LEVELS = 100

fixed_dot_size = 25

# Base paths for test directories
base_paths = [
    "data/SoccerNet/SN-GSR-2025/train",
    "data/SoccerNet/SN-GSR-2025/test", 
    "data/SoccerNet/SN-GSR-2025/valid"
]

# Create output directory for plots if it doesn't exist
output_dir = os.path.join("plots", "ground_truth")
cache_dir = os.path.join("plots", "ground_truth", "cache")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

# Cache file path
cache_file = os.path.join(cache_dir, "analysis_results.pkl")

# Check if cache exists
if os.path.exists(cache_file):
    print("Loading cached analysis results...")
    with open(cache_file, 'rb') as f:
        cached_data = pickle.load(f)
    
    # Extract cached variables
    referee_distances = cached_data['referee_distances']
    referee_angles = cached_data['referee_angles']
    inside_rectangle_counts = cached_data['inside_rectangle_counts']
    referee_positions_x = cached_data['referee_positions_x']
    referee_positions_y = cached_data['referee_positions_y']
    duel_distances = cached_data['duel_distances']
    duel_angles = cached_data['duel_angles']
    split_stats = cached_data['split_stats']
    
    # Load movement data with fallback for backward compatibility
    referee_movements = cached_data.get('referee_movements', [])
    movement_positions = cached_data.get('movement_positions', [])
    
    print("Cached data loaded successfully!")
    print(f"  Distance measurements: {len(referee_distances)}")
    print(f"  Angle measurements: {len(referee_angles)}")
    
else:
    print("No cache found. Running full analysis...")
    
    # Initialize dictionaries to store metrics
    referee_distances = []
    referee_angles = []
    inside_rectangle_counts = defaultdict(int)

    # Arrays to store referee positions
    referee_positions_x = []
    referee_positions_y = []

    # Arrays to store paired distance and angle for duels (for scatter plot)
    duel_distances = []
    duel_angles = []

    # Arrays to store referee movements (new addition)
    referee_movements = []
    movement_positions = []

    # Initialize per-split counters
    split_stats = {
        "train": {"total_frames": 0, "frames_with_main_referee": 0, "processed_files": 0},
        "test": {"total_frames": 0, "frames_with_main_referee": 0, "processed_files": 0},
        "valid": {"total_frames": 0, "frames_with_main_referee": 0, "processed_files": 0}
    }

    # Define pitch dimensions in meters
    PITCH_LENGTH = 105  # meters
    PITCH_WIDTH = 68    # meters

    # Define penalty box dimensions
    PENALTY_AREA_LENGTH = 16.5  # meters from goal line
    PENALTY_AREA_WIDTH = 40.32  # meters wide
    PENALTY_AREA_Y_MIN = (PITCH_WIDTH - PENALTY_AREA_WIDTH) / 2  # 13.84 meters
    PENALTY_AREA_Y_MAX = (PITCH_WIDTH + PENALTY_AREA_WIDTH) / 2  # 54.16 meters

    def is_inside_penalty_box(x_pitch, y_pitch):
        """Check if a position (in pitch coordinates with (0,0) at bottom-left) is inside either penalty box"""
        # Left penalty box
        if (0 <= x_pitch <= PENALTY_AREA_LENGTH and 
            PENALTY_AREA_Y_MIN <= y_pitch <= PENALTY_AREA_Y_MAX):
            return True
        # Right penalty box
        if ((PITCH_LENGTH - PENALTY_AREA_LENGTH) <= x_pitch <= PITCH_LENGTH and 
            PENALTY_AREA_Y_MIN <= y_pitch <= PENALTY_AREA_Y_MAX):
            return True
        return False

    # Process each split separately
    for base_path in base_paths:
        # Determine split name from path
        split_name = base_path.split('/')[-1]  # Get 'train', 'test', or 'valid'
        
        print(f"\n=== Processing {split_name.upper()} split ===")
        
        # Find all SNGS-XXX folders in current split
        sngs_folders = glob.glob(os.path.join(base_path, "SNGS-*"))
        print(f"Found {len(sngs_folders)} folders in {base_path}")
        
        # Process each SNGS folder in this split
        for sngs_folder in sngs_folders:
            labels_file = os.path.join(sngs_folder, "Labels-GameState.json")
            
            # Skip if the file doesn't exist
            if not os.path.exists(labels_file):
                print(f"Skipping {sngs_folder}: Labels-GameState.json not found")
                continue
            
            # Load the labels data
            with open(labels_file, 'r') as f:
                data = json.load(f)
            
            split_stats[split_name]["processed_files"] += 1
            print(f"Processing {sngs_folder}")
            
            # Group annotations by image_id
            annotations_by_image = defaultdict(list)
            for annotation in data['annotations']:
                annotations_by_image[annotation['image_id']].append(annotation)
            
            # Sort images by image_id to ensure temporal order
            sorted_images = sorted(data['images'], key=lambda x: x['image_id'])
            
            # Track previous referee position for movement calculation
            prev_ref_position = None
            
            # Process each image
            for image in sorted_images:
                split_stats[split_name]["total_frames"] += 1
                if not image.get('is_labeled', False):
                    continue
                    
                image_id = image['image_id']
                if image_id not in annotations_by_image:
                    continue
                    
                # Find referee in this frame
                referee = None
                players = []
                
                for annotation in annotations_by_image[image_id]:
                    # category_id 3 is referee
                    if annotation['category_id'] == 3:
                        if 'bbox_pitch' in annotation and annotation['bbox_pitch'] is not None:
                            # Check if bbox_pitch has required fields
                            bbox = annotation['bbox_pitch']
                            if 'x_bottom_middle' in bbox and 'y_bottom_middle' in bbox:
                                referee = annotation
                    # category_id 1 is player, 2 is goalkeeper
                    elif annotation['category_id'] in [1, 2]:
                        if 'bbox_pitch' in annotation and annotation['bbox_pitch'] is not None:
                            # Check if bbox_pitch has required fields
                            bbox = annotation['bbox_pitch']
                            if 'x_bottom_middle' in bbox and 'y_bottom_middle' in bbox:
                                players.append(annotation)
                
                if referee is None:
                    continue
                    
                
                # Get referee position (already in meters, with (0,0) at center)
                ref_x = referee['bbox_pitch']['x_bottom_middle']
                ref_y = referee['bbox_pitch']['y_bottom_middle']
                
                # Convert to pitch coordinates with (0,0) at bottom-left
                # Original: x in [-52.5, 52.5], y in [-34, 34]
                # Convert to: x in [0, 105], y in [0, 68]
                ref_x_pitch = ref_x + PITCH_LENGTH / 2
                ref_y_pitch = ref_y + PITCH_WIDTH / 2
                
                # Check if referee is within pitch bounds (exclude sideline/assistant referees)
                if not (0 <= ref_x_pitch <= PITCH_LENGTH and 5 <= ref_y_pitch <= PITCH_WIDTH - 5):
                    continue
                
                split_stats[split_name]["frames_with_main_referee"] += 1

                # Store referee position
                referee_positions_x.append(ref_x_pitch)
                referee_positions_y.append(ref_y_pitch)
                
                # Calculate and store movement vector if we have a previous position
                if prev_ref_position is not None:
                    prev_x_pitch, prev_y_pitch = prev_ref_position
                    movement_dx = ref_x_pitch - prev_x_pitch
                    movement_dy = ref_y_pitch - prev_y_pitch
                    
                    # Only store significant movements (> 0.5 meters) to avoid noise
                    movement_magnitude = math.sqrt(movement_dx**2 + movement_dy**2)
                    if movement_magnitude > 0.5:
                        referee_movements.append((movement_dx, movement_dy))
                        movement_positions.append((prev_x_pitch, prev_y_pitch))
                
                # Update previous position for next iteration
                prev_ref_position = (ref_x_pitch, ref_y_pitch)
                
                # Check if referee is inside rectangle (middle third of the pitch)
                # Rectangle is from x=35 to x=70 (middle third of 105m pitch)
                inside_rectangle = 35 <= ref_x_pitch <= 70
                inside_rectangle_counts[inside_rectangle] += 1
                
                # Use DBSCAN clustering to find action points (clusters of players)
                if len(players) >= 2:
                    # Extract player positions and team info
                    player_positions = []
                    player_teams = []
                    
                    for player in players:
                        # Validate bbox_pitch exists and has required fields
                        if 'bbox_pitch' not in player or player['bbox_pitch'] is None:
                            continue
                        if 'x_bottom_middle' not in player['bbox_pitch'] or 'y_bottom_middle' not in player['bbox_pitch']:
                            continue
                        
                        player_x = player['bbox_pitch']['x_bottom_middle']
                        player_y = player['bbox_pitch']['y_bottom_middle']
                        team = player.get('attributes', {}).get('team', '')
                        
                        if team:  # Only include players with team info
                            player_positions.append([player_x, player_y])
                            player_teams.append(team)
                    
                    if len(player_positions) >= 2:
                        # Apply DBSCAN clustering with eps=3 meters (maximum distance between players in a cluster)
                        player_positions_array = np.array(player_positions)
                        clustering = DBSCAN(eps=3, min_samples=2).fit(player_positions_array)
                        
                        # Process each cluster
                        unique_clusters = set(clustering.labels_)
                        unique_clusters.discard(-1)  # Remove noise points
                        
                        for cluster_id in unique_clusters:
                            # Get players in this cluster
                            cluster_mask = clustering.labels_ == cluster_id
                            cluster_positions = player_positions_array[cluster_mask]
                            cluster_teams = [player_teams[i] for i, mask in enumerate(cluster_mask) if mask]
                            
                            # Check if cluster contains players from opposing teams
                            unique_teams = set(cluster_teams)
                            if len(unique_teams) >= 2:  # At least two different teams
                                # Calculate cluster center (action point)
                                cluster_center_x = np.mean(cluster_positions[:, 0])
                                cluster_center_y = np.mean(cluster_positions[:, 1])
                                
                                # Calculate referee distance to this action point
                                distance = math.sqrt((cluster_center_x - ref_x)**2 + (cluster_center_y - ref_y)**2)
                                referee_distances.append(distance)
                
                # Find duels (players close to each other) and calculate angles
                # Collect all valid duels first, then only process the closest one
                valid_duels = []
                
                for i in range(len(players)):
                    for j in range(i + 1, len(players)):
                        player1 = players[i]
                        player2 = players[j]
                        
                        # Check if they are from different teams
                        team1 = player1.get('attributes', {}).get('team', '')
                        team2 = player2.get('attributes', {}).get('team', '')
                        
                        if team1 == team2 or team1 == '' or team2 == '':
                            continue
                        
                        # Validate both players have proper bbox_pitch data
                        if ('bbox_pitch' not in player1 or player1['bbox_pitch'] is None or
                            'x_bottom_middle' not in player1['bbox_pitch'] or 'y_bottom_middle' not in player1['bbox_pitch']):
                            continue
                        if ('bbox_pitch' not in player2 or player2['bbox_pitch'] is None or
                            'x_bottom_middle' not in player2['bbox_pitch'] or 'y_bottom_middle' not in player2['bbox_pitch']):
                            continue
                        
                        # Get positions
                        p1_x = player1['bbox_pitch']['x_bottom_middle']
                        p1_y = player1['bbox_pitch']['y_bottom_middle']
                        p2_x = player2['bbox_pitch']['x_bottom_middle']
                        p2_y = player2['bbox_pitch']['y_bottom_middle']
                        
                        # Calculate distance between players
                        player_distance = math.sqrt((p1_x - p2_x)**2 + (p1_y - p2_y)**2)
                        
                        # Consider it a duel if players are within 1 meter of each other
                        if player_distance >= 2 and player_distance <= 3:
                            # Calculate duel center
                            duel_x = (p1_x + p2_x) / 2
                            duel_y = (p1_y + p2_y) / 2
                            
                            # Convert duel center to pitch coordinates to check penalty box
                            duel_x_pitch = duel_x + PITCH_LENGTH / 2
                            duel_y_pitch = duel_y + PITCH_WIDTH / 2
                            
                            # Calculate referee distance to duel
                            ref_to_duel_distance = math.sqrt((duel_x - ref_x)**2 + (duel_y - ref_y)**2)
                            
                            # Store duel info for later processing
                            valid_duels.append({
                                'distance_to_ref': ref_to_duel_distance,
                                'duel_x': duel_x,
                                'duel_y': duel_y,
                                'p1_x': p1_x,
                                'p1_y': p1_y,
                                'p2_x': p2_x,
                                'p2_y': p2_y
                            })
                
                # Process all valid duels
                for duel in valid_duels:
                    # Calculate angle for each duel
                    # Vector 1: player1 to player2
                    player_line_x = duel['p2_x'] - duel['p1_x']
                    player_line_y = duel['p2_y'] - duel['p1_y']
                    
                    # Vector 2: referee to duel center
                    ref_to_duel_x = duel['duel_x'] - ref_x
                    ref_to_duel_y = duel['duel_y'] - ref_y
                    
                    # Calculate angle between the two vectors
                    dot_product = player_line_x * ref_to_duel_x + player_line_y * ref_to_duel_y
                    mag1 = math.sqrt(player_line_x**2 + player_line_y**2)
                    mag2 = math.sqrt(ref_to_duel_x**2 + ref_to_duel_y**2)
                    
                    if mag1 > 0 and mag2 > 0:
                        cos_angle = dot_product / (mag1 * mag2)
                        cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
                        angle_degrees = math.degrees(math.acos(cos_angle))
                        
                        # Get the sharp angle (0-90 degrees)
                        if angle_degrees > 90:
                            angle_degrees = 180 - angle_degrees
                        
                        # Store paired distance and angle for each duel
                        duel_distances.append(duel['distance_to_ref'])
                        duel_angles.append(angle_degrees)
                        
                        # Also add to general angles list
                        referee_angles.append(angle_degrees)

    # Save results to cache
    print("Saving analysis results to cache...")
    cache_data = {
        'referee_distances': referee_distances,
        'referee_angles': referee_angles,
        'inside_rectangle_counts': dict(inside_rectangle_counts),  # Convert defaultdict to dict
        'referee_positions_x': referee_positions_x,
        'referee_positions_y': referee_positions_y,
        'duel_distances': duel_distances,
        'duel_angles': duel_angles,
        'split_stats': split_stats,
        'referee_movements': referee_movements,
        'movement_positions': movement_positions
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"Analysis results cached to {cache_file}")

# Define pitch dimensions in meters (needed for plotting even if loaded from cache)
PITCH_LENGTH = 105  # meters
PITCH_WIDTH = 68    # meters

# Define penalty box dimensions
PENALTY_AREA_LENGTH = 16.5  # meters from goal line
PENALTY_AREA_WIDTH = 40.32  # meters wide
PENALTY_AREA_Y_MIN = (PITCH_WIDTH - PENALTY_AREA_WIDTH) / 2  # 13.84 meters
PENALTY_AREA_Y_MAX = (PITCH_WIDTH + PENALTY_AREA_WIDTH) / 2  # 54.16 meters

# Print statistics per split
print("\n" + "="*60)
print("STATISTICS PER SPLIT")
print("="*60)

total_files = 0
total_frames = 0
total_frames_with_referee = 0

for split_name in ["train", "test", "valid"]:
    split_data = split_stats[split_name]
    print(f"\n{split_name.upper()} Split:")
    print(f"  Processed files: {split_data['processed_files']}")
    print(f"  Total frames: {split_data['total_frames']}")
    print(f"  Frames with main referee: {split_data['frames_with_main_referee']}")
    if split_data['total_frames'] > 0:
        percentage = (split_data['frames_with_main_referee'] / split_data['total_frames']) * 100
        print(f"  Percentage with main referee: {percentage:.2f}%")
    
    total_files += split_data['processed_files']
    total_frames += split_data['total_frames']
    total_frames_with_referee += split_data['frames_with_main_referee']

print(f"\nTOTAL ACROSS ALL SPLITS:")
print(f"  Processed files: {total_files}")
print(f"  Total frames: {total_frames}")
print(f"  Frames with main referee: {total_frames_with_referee}")
if total_frames > 0:
    total_percentage = (total_frames_with_referee / total_frames) * 100
    print(f"  Overall percentage with main referee: {total_percentage:.2f}%")

print(f"\nEVALUATION DATA POINTS:")
print(f"  Distance measurements: {len(referee_distances)}")
print(f"  Angle measurements: {len(referee_angles)}")
print("="*60)

# Find all SNGS-XXX folders in all test directories (keeping this for backward compatibility)
sngs_folders = []
for base_path in base_paths:
    current_folders = glob.glob(os.path.join(base_path, "SNGS-*"))
    sngs_folders.extend(current_folders)

print(f"\nTotal SNGS folders found: {len(sngs_folders)}")

# Process each SNGS folder
processed_files = 0
all_frames = 0
total_frames_processed = 0

# Note: The actual processing is now done above in the per-split loop
# These variables are kept for compatibility with the rest of the code
processed_files = total_files
all_frames = total_frames  
total_frames_processed = total_frames_with_referee

print(f"Processed {processed_files} label files")
print(f"Total frames processed: {total_frames_processed}")
print(f"Total frames: {all_frames}")
print(f"Total data points: {len(referee_distances)} distances, {len(referee_angles)} angles")

# Create plots using aggregated data
plt.figure(figsize=(15, 10))

# 1. Referee Distance Distribution
plt.subplot(2, 2, 1)
plt.hist(referee_distances, bins=30, alpha=0.7, color='blue')
plt.title('Distribution of Referee Distances to Players (Ground Truth)')
plt.xlabel('Distance (meters)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# 2. Referee Angle Distribution
plt.subplot(2, 2, 2)
plt.hist(referee_angles, bins=np.arange(0, 91, 15), alpha=0.7, color='green')
plt.title('Distribution of Referee Angles to Duels (Outside Box) - Ground Truth')
plt.xlabel('Angle (degrees)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# 3. Inside Rectangle Ratio
plt.subplot(2, 2, 3)
labels = ['Outside Rectangle', 'Inside Rectangle']
values = [inside_rectangle_counts[False], inside_rectangle_counts[True]]
plt.bar(labels, values, color=['red', 'blue'])
plt.title('Referee Position Relative to Rectangle (Ground Truth)')
plt.ylabel('Number of Frames')
plt.grid(True, alpha=0.3)

# 4. Distance vs Angle Scatter Plot (if both metrics are available)
if duel_distances and duel_angles:
    plt.subplot(2, 2, 4)
    plt.scatter(duel_distances, duel_angles, alpha=0.5, s=10)
    plt.title('Referee Distance vs Angle to Duels (Outside Box) - Ground Truth')
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
plt.suptitle('Referee Evaluation Distributions - Ground Truth Data', fontsize=16)

# Save the figure
plt.savefig(os.path.join(output_dir, 'aggregated_evaluation_distributions.png'), dpi=300)
plt.close()

# Create additional detailed plots
# 1. Histogram of distances with a vertical line for the average
plt.figure(figsize=(10, 6))
plt.hist(referee_distances, bins=50, alpha=0.7, color='blue')
plt.axvline(x=np.mean(referee_distances), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(referee_distances):.2f}')
plt.axvline(x=np.median(referee_distances), color='green', linestyle='dashed', linewidth=2, label=f'Median: {np.median(referee_distances):.2f}')
plt.title('Detailed Distribution of Referee Distances to Players (Ground Truth)')
plt.xlabel('Distance (meters)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'aggregated_detailed_distance_distribution.png'), dpi=300)
plt.close()

# 2. Histogram of angles with a vertical line for the average
plt.figure(figsize=(12, 12))  # Square figure
plt.hist(referee_angles, bins=np.arange(0, 91, 15), alpha=0.7, color='green', density=True)
plt.axvline(x=np.mean(referee_angles), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(referee_angles):.2f}')
plt.axvline(x=np.median(referee_angles), color='blue', linestyle='dashed', linewidth=2, label=f'Median: {np.median(referee_angles):.2f}')
plt.xlabel('Angle (degrees)', fontsize=20)
plt.ylabel('Probability Density', fontsize=20)
plt.xticks([0, 15, 30, 45, 60, 75, 90], fontsize=20)
plt.xlim(0, 90)  # Set x-axis limits to 0-90
plt.yticks(fontsize=14)
plt.legend(fontsize=20, loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'aggregated_detailed_angle_distribution.png'), dpi=300)
plt.close()

# 3. Create heatmap of referee positions (ORIGINAL)
if referee_positions_x and referee_positions_y:
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

# 4. Create heatmap of referee positions (SYMMETRICALLY ADJUSTED)
if referee_positions_x and referee_positions_y:
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
    # For scoring surface: use more levels, no threshold, and controlled bandwidth
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

# 5. Create scatter plot of raw referee positions
if referee_positions_x and referee_positions_y:
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
    
    plt.title(f'Raw Referee Positions Scatter Plot - Symmetrical Distribution (n={len(symmetrical_x)}) - Ground Truth', fontsize=16)
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    
    plt.savefig(os.path.join(output_dir, 'aggregated_referee_positions_scatter.png'), dpi=300)
    plt.close()

# Fit curve on distance distribution
def gamma_distribution(x, shape, scale):
    return stats.gamma.pdf(x, shape, scale=scale)

# Create histogram for the distances
plt.figure(figsize=(12, 12))  # Square figure
n, bins, _ = plt.hist(referee_distances, bins=40, density=True, alpha=0.6, color='blue', label='Data')

# Calculate bin centers for curve fitting
bin_centers = 0.5 * (bins[1:] + bins[:-1])

# Fit curves to the distribution
try:
    # Gamma distribution
    gamma_params, _ = curve_fit(gamma_distribution, bin_centers, n, p0=[2, 1])
    
    # Create a smoother curve with more points for plotting
    x_smooth = np.linspace(min(bin_centers), max(bin_centers), 1000)
    gamma_curve_smooth = gamma_distribution(x_smooth, *gamma_params)

    
    # Calculate mean and median
    mean_distance = np.mean(referee_distances)
    median_distance = np.median(referee_distances)
    
    # Add vertical lines for mean and median
    plt.axvline(x=mean_distance, color='black', linestyle='--', alpha=0.7, label=f'Mean: {mean_distance:.2f}m')
    plt.axvline(x=median_distance, color='black', linestyle=':', alpha=0.7, label=f'Median: {median_distance:.2f}m')
    
    # Calculate goodness of fit (R-squared)
    gamma_curve = gamma_distribution(bin_centers, *gamma_params)
    residuals_gamma = n - gamma_curve
    ss_res_gamma = np.sum(residuals_gamma**2)
    ss_tot = np.sum((n - np.mean(n))**2)
    r_squared_gamma = 1 - (ss_res_gamma / ss_tot)
    
    plt.plot(x_smooth, gamma_curve_smooth, 'r-', lw=4, 
             label=f'Gamma Distribution\n  α={gamma_params[0]:.2f}\n  θ={gamma_params[1]:.2f}\n  R²={r_squared_gamma:.3f}')
    
    # Set x-axis limits to 0-60
    plt.xlim(0, 60)
    
    
    
except Exception as e:
    print(f"Error fitting distance distribution: {e}")

plt.title('Referee Distance Distribution with Fitted Gamma Curve (Ground Truth)', fontsize=22)
plt.xlabel('Distance (meters)', fontsize=20)
plt.ylabel('Probability Density', fontsize=20)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=14)
plt.savefig(os.path.join(output_dir, 'distance_distribution_fitted.png'), dpi=300)
plt.close()

# 6. Create symmetrical KDE with vector field overlay showing referee movements
if referee_positions_x and referee_positions_y and referee_movements and movement_positions:
    plt.figure(figsize=(16, 12))
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
    
    # Create vector field from movement data
    # Create a 7×6 grid matrix (42 areas total)
    rows = 15  # 7 rows across the width of the pitch
    cols = 19  # 6 columns across the length of the pitch
    
    x_bins = np.linspace(0, PITCH_LENGTH, cols + 1)
    y_bins = np.linspace(0, PITCH_WIDTH, rows + 1)
    
    # Initialize grid for accumulating movements
    movement_grid_x = np.zeros((rows, cols))
    movement_grid_y = np.zeros((rows, cols))
    movement_counts = np.zeros((rows, cols))
    movement_magnitudes = np.zeros((rows, cols))  # Add this to track movement speeds
    
    # Process original movements
    for (start_x, start_y), (dx, dy) in zip(movement_positions, referee_movements):
        # Find which grid cell this movement starts from
        x_idx = np.digitize(start_x, x_bins) - 1
        y_idx = np.digitize(start_y, y_bins) - 1
        
        # Ensure indices are within bounds
        if 0 <= x_idx < cols and 0 <= y_idx < rows:
            movement_grid_x[y_idx, x_idx] += dx
            movement_grid_y[y_idx, x_idx] += dy
            movement_counts[y_idx, x_idx] += 1
            # Add movement magnitude to track speed
            movement_magnitudes[y_idx, x_idx] += math.sqrt(dx**2 + dy**2)
    
    # Process symmetrical movements (180-degree rotated)
    for (start_x, start_y), (dx, dy) in zip(movement_positions, referee_movements):
        # Rotate position and movement vector 180 degrees around center
        rotated_start_x = PITCH_LENGTH - start_x
        rotated_start_y = PITCH_WIDTH - start_y
        rotated_dx = -dx
        rotated_dy = -dy
        
        # Find which grid cell this rotated movement starts from
        x_idx = np.digitize(rotated_start_x, x_bins) - 1
        y_idx = np.digitize(rotated_start_y, y_bins) - 1
        
        # Ensure indices are within bounds
        if 0 <= x_idx < cols and 0 <= y_idx < rows:
            movement_grid_x[y_idx, x_idx] += rotated_dx
            movement_grid_y[y_idx, x_idx] += rotated_dy
            movement_counts[y_idx, x_idx] += 1
            # Add movement magnitude to track speed
            movement_magnitudes[y_idx, x_idx] += math.sqrt(rotated_dx**2 + rotated_dy**2)
    
    # Calculate average movement vectors and speeds for each grid cell
    avg_movement_x = np.divide(movement_grid_x, movement_counts, 
                              out=np.zeros_like(movement_grid_x), where=movement_counts!=0)
    avg_movement_y = np.divide(movement_grid_y, movement_counts, 
                              out=np.zeros_like(movement_grid_y), where=movement_counts!=0)
    avg_movement_speed = np.divide(movement_magnitudes, movement_counts,
                                  out=np.zeros_like(movement_magnitudes), where=movement_counts!=0)
    
    # Create grid centers for plotting arrows
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    y_centers = (y_bins[:-1] + y_bins[1:]) / 2
    X_centers, Y_centers = np.meshgrid(x_centers, y_centers)
    
    # Sample KDE density at each grid center to determine referee presence
    kde_data = np.vstack([symmetrical_x, symmetrical_y])
    kde = gaussian_kde(kde_data)
    
    # Sample density at each grid center
    grid_positions = np.vstack([X_centers.ravel(), Y_centers.ravel()])
    density_values = kde(grid_positions).reshape(X_centers.shape)
    
    # Define thresholds
    movement_threshold = 2  # Minimum movements needed for arrow
    presence_threshold = np.percentile(density_values, 25)  # 25th percentile of density as presence threshold
    
    # Create masks for different visualization types
    arrow_mask = movement_counts >= movement_threshold  # Areas with sufficient movement data
    dot_mask = (movement_counts < movement_threshold) & (density_values > presence_threshold)  # Areas with presence but insufficient movement
    no_data_mask = density_values <= presence_threshold  # Areas with no referee presence data
    
    # Plot arrows for areas with sufficient movement data
    if np.any(arrow_mask):
        # Normalize direction vectors to unit length
        magnitudes = np.sqrt(avg_movement_x**2 + avg_movement_y**2)
        normalized_x = np.divide(avg_movement_x, magnitudes, 
                               out=np.zeros_like(avg_movement_x), where=magnitudes!=0)
        normalized_y = np.divide(avg_movement_y, magnitudes, 
                               out=np.zeros_like(avg_movement_y), where=magnitudes!=0)
        
        # Use fixed arrow length for all vectors
        fixed_arrow_length = 2.5  # Fixed length in meters for all arrows
        
        # Create scaled vectors for quiver (direction * fixed length)
        scaled_x = normalized_x * fixed_arrow_length
        scaled_y = normalized_y * fixed_arrow_length
        
        # Plot arrows with fixed head properties and consistent scale
        # Use scale=1 so that vector magnitudes directly translate to arrow lengths
        ax.quiver(X_centers[arrow_mask], Y_centers[arrow_mask], 
                 scaled_x[arrow_mask], scaled_y[arrow_mask],
                 angles='xy', scale_units='xy', scale=1,
                 color='black', alpha=0.8, width=0.004, 
                 headwidth=3, headlength=3, headaxislength=2.5)
    
    # Plot dots for areas with referee presence but insufficient movement data (stationary areas)
    if np.any(dot_mask):
        # Use fixed dot size for all dots
        
        # Plot dots indicating stationary movement
        ax.scatter(X_centers[dot_mask], Y_centers[dot_mask], 
                  s=fixed_dot_size, c='black', alpha=0.7, 
                  marker='o', edgecolors='black', linewidths=0.5)
    
    # Plot dots for areas with no referee presence data (complete grid coverage)
    if np.any(no_data_mask):
        ax.scatter(X_centers[no_data_mask], Y_centers[no_data_mask], 
                  s=fixed_dot_size, c='black', alpha=0.7, 
                  marker='o', edgecolors='black', linewidths=0.5)
    
    plt.xlim(0, PITCH_LENGTH)
    plt.ylim(0, PITCH_WIDTH)
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    
    plt.xlabel('X Position (meters)', fontsize=16)
    plt.ylabel('Y Position (meters)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Add text box with movement statistics
    total_movements = len(referee_movements)
    avg_movement_magnitude = np.mean([math.sqrt(dx**2 + dy**2) for dx, dy in referee_movements]) if referee_movements else 0
    arrow_cells = np.sum(arrow_mask)
    dot_cells = np.sum(dot_mask)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'referee_position_kde_with_vector_field.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Vector field plot created with {total_movements} movement vectors")
    print(f"Added {arrow_cells} arrow cells and {dot_cells} stationary dot cells")

print(f"All plots saved to {output_dir} directory") 