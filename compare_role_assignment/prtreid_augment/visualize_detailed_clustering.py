import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils.data_loader import load_frame_data
from utils.color_utils import extract_jersey_colors, cluster_player_colors
from utils.geometry_utils import check_point_in_clusters
import cv2
from scipy.spatial import ConvexHull

def visualize_cluster_boundaries_3d(clusters, ax, alpha=0.1):
    """Visualize cluster boundaries in 3D"""
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
    
    for i, cluster in enumerate(clusters):
        color = colors[i % len(colors)]
        
        if cluster['hull'] is not None:
            # Draw convex hull
            hull_points = cluster['hull'].points[cluster['hull'].vertices]
            try:
                hull_3d = ConvexHull(hull_points)
                for simplex in hull_3d.simplices:
                    triangle = hull_points[simplex]
                    ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                                   color=color, alpha=alpha)
            except:
                # Fallback to scatter plot
                ax.scatter(hull_points[:, 0], hull_points[:, 1], hull_points[:, 2], 
                          c=color, alpha=alpha*3, s=100, marker='o')
        
        elif cluster['bbox'] is not None:
            # Draw bounding box
            bbox = cluster['bbox']
            min_vals = bbox['min']
            max_vals = bbox['max']
            
            # Create box vertices
            vertices = []
            for x in [min_vals[0], max_vals[0]]:
                for y in [min_vals[1], max_vals[1]]:
                    for z in [min_vals[2], max_vals[2]]:
                        vertices.append([x, y, z])
            vertices = np.array(vertices)
            
            # Draw box edges
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                      c=color, alpha=alpha*3, s=50, marker='s')

def analyze_individual_frame(frame_data, frame_idx):
    """Create detailed analysis for a single frame"""
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'Frame Analysis #{frame_idx+1}: {os.path.basename(frame_data["image_path"])}', fontsize=16)
    
    # Extract data
    player_colors = np.array(frame_data['player_colors'])
    clusters = frame_data['clusters']
    unclassified_data = frame_data['unclassified_data']
    
    # Separate reassigned and non-reassigned
    non_reassigned = [data for data in unclassified_data if not data['reassigned']]
    reassigned = [data for data in unclassified_data if data['reassigned']]
    
    # Plot 1: 3D scatter with clusters
    ax1 = fig.add_subplot(231, projection='3d')
    
    # Plot cluster boundaries
    visualize_cluster_boundaries_3d(clusters, ax1, alpha=0.1)
    
    # Plot player colors
    ax1.scatter(player_colors[:, 0], player_colors[:, 1], player_colors[:, 2], 
               c='blue', alpha=0.8, s=60, label='Player Colors', marker='o')
    
    # Plot cluster centers
    for i, cluster in enumerate(clusters):
        center = cluster['center']
        ax1.scatter(center[0], center[1], center[2], 
                   c='black', s=200, marker='X', label=f'Cluster {i+1} Center')
    
    # Plot unclassified
    if reassigned:
        reassigned_colors = np.array([data['color'] for data in reassigned])
        ax1.scatter(reassigned_colors[:, 0], reassigned_colors[:, 1], reassigned_colors[:, 2], 
                   c='green', alpha=1.0, s=80, label='Reassigned', marker='^')
    
    if non_reassigned:
        non_reassigned_colors = np.array([data['color'] for data in non_reassigned])
        ax1.scatter(non_reassigned_colors[:, 0], non_reassigned_colors[:, 1], non_reassigned_colors[:, 2], 
                   c='red', alpha=1.0, s=80, label='Non-Reassigned', marker='s')
    
    ax1.set_xlabel('Red')
    ax1.set_ylabel('Green')
    ax1.set_zlabel('Blue')
    ax1.set_title('3D Color Clusters')
    ax1.legend()
    
    # Plot 2: Load and show the actual image with bboxes
    ax2 = fig.add_subplot(232)
    try:
        image = cv2.imread(frame_data['image_path'])
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax2.imshow(image_rgb)
            
            # Draw bboxes for players
            for i, color in enumerate(player_colors):
                # Find corresponding prediction (this is approximate)
                # We need to match colors back to predictions
                pass  # This would require more complex matching
            
            ax2.set_title('Original Frame')
            ax2.axis('off')
        else:
            ax2.text(0.5, 0.5, 'Image not found', ha='center', va='center', transform=ax2.transAxes)
    except Exception as e:
        ax2.text(0.5, 0.5, f'Error loading image:\n{str(e)}', ha='center', va='center', transform=ax2.transAxes)
    
    # Plot 3: Color distribution histogram
    ax3 = fig.add_subplot(233)
    all_colors = list(player_colors)
    if reassigned:
        all_colors.extend([data['color'] for data in reassigned])
    if non_reassigned:
        all_colors.extend([data['color'] for data in non_reassigned])
    
    all_colors = np.array(all_colors)
    
    # RGB histograms
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        ax3.hist(all_colors[:, i], bins=20, alpha=0.5, label=f'{color.upper()} channel', color=color)
    
    ax3.set_xlabel('Color Value')
    ax3.set_ylabel('Count')
    ax3.set_title('Color Channel Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distance analysis
    ax4 = fig.add_subplot(234)
    if non_reassigned:
        distances_to_clusters = []
        for data in non_reassigned:
            color = np.array(data['color'])
            min_dist = float('inf')
            for cluster in clusters:
                cluster_center = cluster['center']
                dist = np.linalg.norm(color - cluster_center)
                min_dist = min(min_dist, dist)
            distances_to_clusters.append(min_dist)
        
        ax4.hist(distances_to_clusters, bins=10, alpha=0.7, color='red', edgecolor='black')
        ax4.set_xlabel('Distance to Nearest Cluster Center')
        ax4.set_ylabel('Count')
        ax4.set_title('Non-Reassigned: Distance to Clusters')
        ax4.grid(True, alpha=0.3)
        
        # Add mean line
        mean_dist = np.mean(distances_to_clusters)
        ax4.axvline(mean_dist, color='blue', linestyle='--', label=f'Mean: {mean_dist:.1f}')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No Non-Reassigned\nSamples', ha='center', va='center', transform=ax4.transAxes)
    
    # Plot 5: Cluster analysis table
    ax5 = fig.add_subplot(235)
    ax5.axis('off')
    
    # Create table data
    table_data = []
    table_data.append(['Metric', 'Value'])
    table_data.append(['Player Colors', f'{len(player_colors)}'])
    table_data.append(['Total Unclassified', f'{len(unclassified_data)}'])
    table_data.append(['Reassigned', f'{len(reassigned)}'])
    table_data.append(['Non-Reassigned', f'{len(non_reassigned)}'])
    table_data.append(['Clusters', f'{len(clusters)}'])
    
    if non_reassigned:
        non_reassigned_colors = np.array([data['color'] for data in non_reassigned])
        avg_red = np.mean(non_reassigned_colors[:, 0])
        avg_green = np.mean(non_reassigned_colors[:, 1])
        avg_blue = np.mean(non_reassigned_colors[:, 2])
        table_data.append(['Avg Non-Reassigned RGB', f'({avg_red:.0f}, {avg_green:.0f}, {avg_blue:.0f})'])
    
    # Create table
    table = ax5.table(cellText=table_data[1:], colLabels=table_data[0], 
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax5.set_title('Frame Statistics')
    
    # Plot 6: 2D projections
    ax6 = fig.add_subplot(236)
    
    # Project to RG plane
    ax6.scatter(player_colors[:, 0], player_colors[:, 1], c='blue', alpha=0.6, s=40, label='Players')
    
    if reassigned:
        reassigned_colors = np.array([data['color'] for data in reassigned])
        ax6.scatter(reassigned_colors[:, 0], reassigned_colors[:, 1], 
                   c='green', alpha=0.8, s=60, label='Reassigned', marker='^')
    
    if non_reassigned:
        non_reassigned_colors = np.array([data['color'] for data in non_reassigned])
        ax6.scatter(non_reassigned_colors[:, 0], non_reassigned_colors[:, 1], 
                   c='red', alpha=0.8, s=60, label='Non-Reassigned', marker='s')
    
    # Plot cluster centers
    for i, cluster in enumerate(clusters):
        center = cluster['center']
        ax6.scatter(center[0], center[1], c='black', s=150, marker='X', 
                   label=f'Cluster {i+1}' if i < 2 else "")
    
    ax6.set_xlabel('Red')
    ax6.set_ylabel('Green')
    ax6.set_title('2D Projection (Red-Green)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def analyze_and_visualize(base_dir='prtreid_output', image_base_dir='test', 
                         confidence_threshold=3.564, max_frames=5, show_individual_frames=True):
    """Complete analysis and visualization pipeline"""
    
    print("Collecting data for detailed analysis...")
    
    analysis_results = []
    sngs_dirs = [d for d in os.listdir(base_dir) if d.startswith('SNGS-')]
    print(f"Found {len(sngs_dirs)} SNGS directories")
    
    frames_processed = 0
    
    for sngs_dir in sorted(sngs_dirs):
        if frames_processed >= max_frames:
            break
            
        print(f"Processing SNGS directory: {sngs_dir}")
        sngs_path = os.path.join(base_dir, sngs_dir)
        if not os.path.isdir(sngs_path):
            continue
            
        frame_dirs = [d for d in os.listdir(sngs_path) 
                     if d.startswith('000') and os.path.isdir(os.path.join(sngs_path, d))]
        
        for frame_dir in sorted(frame_dirs):
            if frames_processed >= max_frames:
                break
                
            reid_results_path = os.path.join(sngs_path, frame_dir, 'reid_results.json')
            
            if not os.path.exists(reid_results_path):
                continue
                
            predictions, has_unclassified = load_frame_data(reid_results_path, confidence_threshold)
            
            if has_unclassified:
                frame_num = frame_dir
                image_path = os.path.join(image_base_dir, sngs_dir, 'img1', f'{frame_num}.jpg')
                
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found at {image_path}")
                    continue
                
                try:
                    from visualize_clustering import analyze_frame_clustering
                    frame_analysis = analyze_frame_clustering(predictions, image_path)
                    if frame_analysis is not None:
                        # Check if this frame has non-reassigned samples
                        non_reassigned = [data for data in frame_analysis['unclassified_data'] 
                                        if not data['reassigned']]
                        if non_reassigned:  # Only include frames with non-reassigned samples
                            analysis_results.append(frame_analysis)
                            frames_processed += 1
                            print(f"  Found frame with non-reassigned samples: {frame_dir} ({frames_processed}/{max_frames})")
                except Exception as e:
                    print(f"  Error processing frame {frame_dir}: {e}")
                    continue
    
    if not analysis_results:
        print("No frames with non-reassigned samples found!")
        return
    
    print(f"\nFound {len(analysis_results)} frames with non-reassigned samples")
    
    # Show individual frame analysis
    if show_individual_frames:
        for i, frame_data in enumerate(analysis_results):
            fig = analyze_individual_frame(frame_data, i)
            plt.show()
            
            # Ask user if they want to continue
            if i < len(analysis_results) - 1:
                response = input(f"\nShowing frame {i+1}/{len(analysis_results)}. Continue to next frame? (y/n): ")
                if response.lower() != 'y':
                    break
    
    # Create overall summary visualization
    print("\nCreating overall summary visualization...")
    from visualize_clustering import visualize_clustering_3d
    visualize_clustering_3d(analysis_results)

if __name__ == "__main__":
    print("Starting detailed clustering analysis...")
    
    config = {
        'base_dir': 'prtreid_output',
        'image_base_dir': 'test', 
        'confidence_threshold': 3.564,
        'max_frames': 10,
        'show_individual_frames': True
    }
    
    analyze_and_visualize(**config) 