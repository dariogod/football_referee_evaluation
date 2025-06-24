import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
import os
import json

# Load LAB colorspace data from JSON file
json_path = '../compare_role_assignment/role_assingment_V2/clustering_plots_DBScanRoleAssigner/frame_010_lab_coordinates.json'

with open(json_path, 'r') as f:
    data = json.load(f)

# Extract LAB coordinates, RGB colors, and cluster information
lab_coordinates = []
rgb_colors = []
original_clusters = []
roles = []
person_indices = []

for entry in data['coordinates']:
    lab_coord = [entry['lab_coordinates']['L'], entry['lab_coordinates']['a'], entry['lab_coordinates']['b']]
    rgb_color = [entry['rgb_coordinates']['r']/255.0, entry['rgb_coordinates']['g']/255.0, entry['rgb_coordinates']['b']/255.0]
    lab_coordinates.append(lab_coord)
    rgb_colors.append(rgb_color)
    original_clusters.append(entry['cluster'])
    roles.append(entry['role'])
    person_indices.append(entry['person_index'])

# Convert to numpy array - using all 3 LAB dimensions: [a*, b*, L*]
all_data = np.array([[coord[1], coord[2], coord[0]] for coord in lab_coordinates])  # a*, b*, L*
rgb_colors = np.array(rgb_colors)  # Convert RGB colors to numpy array

print(f"Loaded {len(all_data)} data points from LAB colorspace")
print(f"Original clusters: {set(original_clusters)}")
print(f"Roles: {set(roles)}")

# Create output directory
output_dir = 'dbscan_results'
os.makedirs(output_dir, exist_ok=True)

# Original data visualization (with original cluster labels) - 3D
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Define markers for different roles
role_markers = {'player_left': 'o', 'player_right': 's', 'referee': '^'}

for i, (cluster, role) in enumerate(zip(original_clusters, roles)):
    ax.scatter(all_data[i, 0], all_data[i, 1], all_data[i, 2],
               c=[rgb_colors[i]], 
               marker=role_markers[role], 
               s=300, alpha=1.0, edgecolors='black', linewidths=2,
               label=f'{cluster}_{role}' if i == 0 or f'{cluster}_{role}' not in [f'{original_clusters[j]}_{roles[j]}' for j in range(i)] else "")

ax.set_xlabel('a* (Green-Red)', fontsize=12)
ax.set_ylabel('b* (Blue-Yellow)', fontsize=12)
ax.set_zlabel('L* (Lightness)', fontsize=12)
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f'{output_dir}/original_data_3d.png', dpi=300, bbox_inches='tight')
plt.close()

# Test different epsilon values for DBSCAN
epsilon_values = [2, 4, 6, 8, 10, 12, 15, 18, 20, 22, 25]  # Range from 2 to 25
min_samples = 3

for i, eps in enumerate(epsilon_values):
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(all_data)
    
    # Count clusters and noise points
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    # Create 3D visualization
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate colors based on actual RGB values and cluster centers
    unique_labels = set(cluster_labels)
    
    for k in unique_labels:
        class_member_mask = (cluster_labels == k)
        xyz = all_data[class_member_mask]
        
        if k == -1:
            # For outliers, use their actual RGB colors
            colors_for_points = rgb_colors[class_member_mask]
            marker = 'x'
            size = 400
            
            # Plot each outlier with its actual color
            for i, (point, color) in enumerate(zip(xyz, colors_for_points)):
                ax.scatter(point[0], point[1], point[2], c=[color], marker=marker, s=size, alpha=1.0,
                          edgecolors='black', linewidths=4,
                          label='Outliers' if i == 0 else "")
        else:
            # For clusters, use the cluster center RGB color
            cluster_rgb_colors = rgb_colors[class_member_mask]
            cluster_center_color = np.mean(cluster_rgb_colors, axis=0)
            
            # Use different markers for different clusters
            if k == 0:
                marker = 'o'  # circles for first cluster
            elif k == 1:
                marker = 's'  # squares for second cluster
            else:
                marker = 'D'  # diamonds for additional clusters
            
            size = 400
            
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=[cluster_center_color], marker=marker, s=size, alpha=1.0,
                      edgecolors='black', linewidths=2,
                      label=f'Cluster {k}')
    
    ax.set_xlabel('a* (Green-Red)', fontsize=12)
    ax.set_ylabel('b* (Blue-Yellow)', fontsize=12)
    ax.set_zlabel('L* (Lightness)', fontsize=12)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save visualization
    filename = f'{output_dir}/dbscan_3d_eps_{eps}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ε={eps}: {n_clusters} clusters, {n_noise} noise points -> saved as {filename}")

print(f"\nAll 3D visualizations saved in '{output_dir}/' directory")
print("Data summary:")
print(f"Team A players: {original_clusters.count('team_a')} samples")
print(f"Team B players: {original_clusters.count('team_b')} samples") 
print(f"Referees: {original_clusters.count('outlier')} samples")
print(f"Total points: {len(all_data)}")
print(f"LAB coordinate ranges:")
print(f"  a* (Green-Red): {all_data[:, 0].min():.1f} to {all_data[:, 0].max():.1f}")
print(f"  b* (Blue-Yellow): {all_data[:, 1].min():.1f} to {all_data[:, 1].max():.1f}")
print(f"  L* (Lightness): {all_data[:, 2].min():.1f} to {all_data[:, 2].max():.1f}")
