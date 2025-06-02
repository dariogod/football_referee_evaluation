# KDE Heatmap Visualization (replacement for the OpenCV version)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.image import imread
from sklearn.cluster import DBSCAN
import os

# KDE parameters matching the ground truth analysis
KDE_THRESHOLD = 0.05
KDE_BANDWIDTH_ADJUSTMENT = 1.5
KDE_LEVELS = 10

# Pitch dimensions
PITCH_LENGTH = 105  # meters
PITCH_WIDTH = 68    # meters

def create_kde_heatmap(cluster_centers, pitch_length=105, pitch_width=68):
    """Create a KDE heatmap overlay based on cluster centers using seaborn."""
    if not cluster_centers:
        return None, None
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Load and display the soccer pitch background
    if os.path.exists('src/utils/pitch_2.png'):
        pitch_img = imread('src/utils/pitch_2.png')
        ax.imshow(pitch_img, extent=[0, pitch_length, pitch_width, 0])
    
    # Extract x and y coordinates from cluster centers
    cluster_x = [center[0] for center in cluster_centers]
    cluster_y = [center[1] for center in cluster_centers]
    
    # Create symmetrical data by adding multiple rotations/reflections to increase data density
    symmetrical_x = []
    symmetrical_y = []
    
    # Add original positions
    symmetrical_x.extend(cluster_x)
    symmetrical_y.extend(cluster_y)
    
    # Add 180-degree rotated positions around the center of the pitch
    for x, y in zip(cluster_x, cluster_y):
        rotated_x = pitch_length - x
        rotated_y = pitch_width - y
        symmetrical_x.append(rotated_x)
        symmetrical_y.append(rotated_y)
    
    # Add mirrored positions across the center line (y-axis mirror)
    for x, y in zip(cluster_x, cluster_y):
        mirrored_x = pitch_length - x
        symmetrical_x.append(mirrored_x)
        symmetrical_y.append(y)
    
    # Add mirrored positions across the middle line (x-axis mirror)
    for x, y in zip(cluster_x, cluster_y):
        mirrored_y = pitch_width - y
        symmetrical_x.append(x)
        symmetrical_y.append(mirrored_y)
    
    # Create KDE plot if we have enough data points
    if len(symmetrical_x) >= 2:
        sns.kdeplot(
            x=symmetrical_x,
            y=symmetrical_y,
            cmap="YlOrRd",
            fill=True,
            alpha=0.7,
            levels=KDE_LEVELS,
            thresh=KDE_THRESHOLD,
            bw_adjust=KDE_BANDWIDTH_ADJUSTMENT,
            ax=ax
        )
    
    # Set axis properties
    ax.set_xlim(0, pitch_length)
    ax.set_ylim(0, pitch_width)
    ax.invert_yaxis()  # Invert y-axis to match image coordinates
    ax.set_xlabel('X Position (meters)', fontsize=16)
    ax.set_ylabel('Y Position (meters)', fontsize=16)
    
    return fig, ax

# Process each interesting frame
for interesting_frame in interesting_frames:
    frame_detections = next((item for item in detections if item["frame_id"] == interesting_frame), None)
    if not frame_detections:
        continue
    
    # Extract player positions and team information
    player_positions = []
    player_teams = []
    
    for detection in frame_detections["detections"]:
        coords = detection["minimap_coordinates"]
        role = detection["role"] if "role" in detection else "UNKNOWN"
        
        # Skip if coordinates are None or if it's a referee
        if coords is None or role == "REF":
            continue
        
        x = coords["x"]
        y = coords["y"]
        x_max = coords["x_max"]
        y_max = coords["y_max"]
        
        # Normalize to pitch dimensions (in meters)
        norm_x = x / x_max * 105  # 105m is pitch width
        norm_y = y / y_max * 68   # 68m is pitch height
        
        player_positions.append([norm_x, norm_y])
        player_teams.append(1 if role == "TEAM A" else 2 if role == "TEAM B" else 0)
    
    # Convert to numpy arrays
    player_positions = np.array(player_positions)
    player_teams = np.array(player_teams)
    
    # Skip if no player positions
    if len(player_positions) == 0:
        print(f"No player positions found for frame {interesting_frame}")
        continue
    
    # Run DBSCAN to identify clusters
    clustering = DBSCAN(eps=4, min_samples=2).fit(player_positions)
    
    # Get cluster labels
    labels = clustering.labels_
    
    # Number of clusters (excluding noise points with label -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    if n_clusters == 0:
        print(f"No clusters found for frame {interesting_frame}")
        continue
    
    # Extract cluster centers
    cluster_centers = []
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_points = player_positions[cluster_mask]
        if len(cluster_points) > 0:
            center = np.mean(cluster_points, axis=0)
            cluster_centers.append(center)
    
    if not cluster_centers:
        print(f"No valid cluster centers found for frame {interesting_frame}")
        continue
    
    # Create the KDE heatmap
    fig, ax = create_kde_heatmap(cluster_centers, PITCH_LENGTH, PITCH_WIDTH)
    
    if fig is not None:
        # Save the plot
        output_path = f"{images_for_paper_path}/minimap_heatmap_kde_{interesting_frame:06d}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory
        print(f"Saved KDE heatmap for frame {interesting_frame} to {output_path}")
    else:
        print(f"Could not create heatmap for frame {interesting_frame}") 