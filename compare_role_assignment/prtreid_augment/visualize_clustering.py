import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.data_loader import load_frame_data
from utils.color_utils import extract_jersey_colors, cluster_player_colors
from utils.geometry_utils import check_point_in_clusters
import cv2

def analyze_frame_clustering(predictions, image_path):
    """Analyze clustering for a single frame and return detailed data"""
    # Separate players and unclassified
    players = [p for p in predictions if p.get('predicted_role') == 'player']
    unclassified = [p for p in predictions if p.get('predicted_role') == 'unclassified']
    
    if len(players) < 2 or len(unclassified) == 0:
        return None
    
    # Extract jersey colors for all predictions
    jersey_colors = extract_jersey_colors(predictions, image_path)
    
    # Get player colors and cluster them
    player_indices = [i for i, p in enumerate(predictions) if p.get('predicted_role') == 'player']
    player_colors = [jersey_colors[i] for i in player_indices]
    
    if len(set([str(c) for c in jersey_colors])) == 1:
        return None  # Skip if all colors are the same
    
    clusters = cluster_player_colors(player_colors, k=2)
    
    # Analyze each unclassified prediction
    unclassified_data = []
    for i, pred in enumerate(predictions):
        if pred.get('predicted_role') == 'unclassified':
            unclassified_color = jersey_colors[i]
            is_reassigned = check_point_in_clusters(unclassified_color, clusters)
            
            unclassified_data.append({
                'color': unclassified_color,
                'bbox': pred['bbox'],
                'reassigned': is_reassigned,
                'prediction': pred
            })
    
    return {
        'player_colors': player_colors,
        'clusters': clusters,
        'unclassified_data': unclassified_data,
        'image_path': image_path,
        'jersey_colors': jersey_colors
    }

def visualize_clustering_3d(analysis_results, show_non_reassigned_only=True):
    """Create 3D visualization of color clustering"""
    fig = plt.figure(figsize=(15, 10))
    
    # Create subplots for different views
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224)
    
    all_player_colors = []
    all_non_reassigned = []
    all_reassigned = []
    
    frame_count = 0
    for frame_data in analysis_results:
        if frame_data is None:
            continue
            
        frame_count += 1
        player_colors = np.array(frame_data['player_colors'])
        clusters = frame_data['clusters']
        unclassified_data = frame_data['unclassified_data']
        
        all_player_colors.extend(player_colors)
        
        # Separate reassigned and non-reassigned
        non_reassigned = [data['color'] for data in unclassified_data if not data['reassigned']]
        reassigned = [data['color'] for data in unclassified_data if data['reassigned']]
        
        all_non_reassigned.extend(non_reassigned)
        all_reassigned.extend(reassigned)
    
    if not all_player_colors:
        print("No data to visualize")
        return
    
    # Convert to numpy arrays
    all_player_colors = np.array(all_player_colors)
    all_non_reassigned = np.array(all_non_reassigned) if all_non_reassigned else np.empty((0, 3))
    all_reassigned = np.array(all_reassigned) if all_reassigned else np.empty((0, 3))
    
    # Plot 1: Full 3D view
    ax1.scatter(all_player_colors[:, 0], all_player_colors[:, 1], all_player_colors[:, 2], 
               c='blue', alpha=0.6, s=30, label='Player Colors')
    
    if len(all_reassigned) > 0:
        ax1.scatter(all_reassigned[:, 0], all_reassigned[:, 1], all_reassigned[:, 2], 
                   c='green', alpha=0.8, s=50, label='Reassigned', marker='^')
    
    if len(all_non_reassigned) > 0:
        ax1.scatter(all_non_reassigned[:, 0], all_non_reassigned[:, 1], all_non_reassigned[:, 2], 
                   c='red', alpha=0.8, s=50, label='Non-Reassigned', marker='s')
    
    ax1.set_xlabel('Red')
    ax1.set_ylabel('Green')
    ax1.set_zlabel('Blue')
    ax1.set_title('3D Color Space - All Data')
    ax1.legend()
    
    # Plot 2: Focus on non-reassigned (if any)
    if len(all_non_reassigned) > 0:
        ax2.scatter(all_player_colors[:, 0], all_player_colors[:, 1], all_player_colors[:, 2], 
                   c='lightblue', alpha=0.3, s=20, label='Player Colors')
        ax2.scatter(all_non_reassigned[:, 0], all_non_reassigned[:, 1], all_non_reassigned[:, 2], 
                   c='red', alpha=1.0, s=80, label='Non-Reassigned', marker='s')
        ax2.set_title('Focus: Non-Reassigned Samples')
    else:
        ax2.text(0.5, 0.5, 0.5, 'No Non-Reassigned\nSamples Found', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('No Non-Reassigned Samples')
    
    ax2.set_xlabel('Red')
    ax2.set_ylabel('Green') 
    ax2.set_zlabel('Blue')
    ax2.legend()
    
    # Plot 3: Different angle view
    ax3.scatter(all_player_colors[:, 0], all_player_colors[:, 1], all_player_colors[:, 2], 
               c='blue', alpha=0.6, s=30, label='Player Colors')
    
    if len(all_non_reassigned) > 0:
        ax3.scatter(all_non_reassigned[:, 0], all_non_reassigned[:, 1], all_non_reassigned[:, 2], 
                   c='red', alpha=0.8, s=50, label='Non-Reassigned', marker='s')
    
    ax3.view_init(elev=20, azim=45)  # Different viewing angle
    ax3.set_xlabel('Red')
    ax3.set_ylabel('Green')
    ax3.set_zlabel('Blue')
    ax3.set_title('Alternative View')
    ax3.legend()
    
    # Plot 4: Statistics and color distances
    if len(all_non_reassigned) > 0:
        # Calculate distances from non-reassigned to player clusters
        distances_to_players = []
        for non_reassigned_color in all_non_reassigned:
            min_dist = float('inf')
            for player_color in all_player_colors:
                dist = np.linalg.norm(np.array(non_reassigned_color) - np.array(player_color))
                min_dist = min(min_dist, dist)
            distances_to_players.append(min_dist)
        
        ax4.hist(distances_to_players, bins=20, alpha=0.7, color='red', edgecolor='black')
        ax4.set_xlabel('Distance to Nearest Player Color')
        ax4.set_ylabel('Count')
        ax4.set_title('Distance Distribution: Non-Reassigned to Players')
        ax4.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_dist = np.mean(distances_to_players)
        median_dist = np.median(distances_to_players)
        ax4.axvline(mean_dist, color='blue', linestyle='--', label=f'Mean: {mean_dist:.1f}')
        ax4.axvline(median_dist, color='green', linestyle='--', label=f'Median: {median_dist:.1f}')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No Non-Reassigned\nSamples to Analyze', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('No Distance Analysis Available')
    
    plt.tight_layout()
    
    # Print summary statistics
    print(f"\n=== CLUSTERING ANALYSIS SUMMARY ===")
    print(f"Frames analyzed: {frame_count}")
    print(f"Total player colors: {len(all_player_colors)}")
    print(f"Total reassigned unclassified: {len(all_reassigned)}")
    print(f"Total non-reassigned unclassified: {len(all_non_reassigned)}")
    
    if len(all_non_reassigned) > 0:
        print(f"\n=== NON-REASSIGNED ANALYSIS ===")
        non_reassigned_colors = all_non_reassigned
        print(f"RGB ranges for non-reassigned:")
        print(f"  Red: {np.min(non_reassigned_colors[:, 0]):.1f} - {np.max(non_reassigned_colors[:, 0]):.1f}")
        print(f"  Green: {np.min(non_reassigned_colors[:, 1]):.1f} - {np.max(non_reassigned_colors[:, 1]):.1f}")
        print(f"  Blue: {np.min(non_reassigned_colors[:, 2]):.1f} - {np.max(non_reassigned_colors[:, 2]):.1f}")
        
        # Print some individual non-reassigned colors
        print(f"\nSample non-reassigned colors (RGB):")
        for i, color in enumerate(non_reassigned_colors[:5]):  # Show first 5
            print(f"  {i+1}: ({color[0]:.0f}, {color[1]:.0f}, {color[2]:.0f})")
    
    plt.show()

def collect_analysis_data(base_dir='prtreid_output', image_base_dir='test', 
                         confidence_threshold=3.564, max_frames=20):
    """Collect clustering analysis data from multiple frames"""
    
    analysis_results = []
    
    # Process each SNGS directory
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
            
        # Get all frame directories
        frame_dirs = [d for d in os.listdir(sngs_path) 
                     if d.startswith('000') and os.path.isdir(os.path.join(sngs_path, d))]
        
        for frame_dir in sorted(frame_dirs):
            if frames_processed >= max_frames:
                break
                
            reid_results_path = os.path.join(sngs_path, frame_dir, 'reid_results.json')
            
            if not os.path.exists(reid_results_path):
                continue
                
            # Load predictions and apply confidence threshold
            predictions, has_unclassified = load_frame_data(reid_results_path, confidence_threshold)
            
            if has_unclassified:
                # Get image path
                frame_num = frame_dir
                image_path = os.path.join(image_base_dir, sngs_dir, 'img1', f'{frame_num}.jpg')
                
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found at {image_path}")
                    continue
                
                # Analyze clustering for this frame
                try:
                    frame_analysis = analyze_frame_clustering(predictions, image_path)
                    if frame_analysis is not None:
                        analysis_results.append(frame_analysis)
                        frames_processed += 1
                        print(f"  Processed frame {frame_dir} ({frames_processed}/{max_frames})")
                except Exception as e:
                    print(f"  Error processing frame {frame_dir}: {e}")
                    continue
    
    return analysis_results

if __name__ == "__main__":
    print("Collecting clustering analysis data...")
    
    # Configuration
    config = {
        'base_dir': 'prtreid_output',
        'image_base_dir': 'test',
        'confidence_threshold': 3.564,
        'max_frames': 30  # Limit for faster analysis
    }
    
    # Collect data
    analysis_results = collect_analysis_data(**config)
    
    if not analysis_results:
        print("No frames with unclassified detections found!")
    else:
        print(f"Found {len(analysis_results)} frames with unclassified detections")
        
        # Create visualization
        print("Creating 3D visualization...")
        visualize_clustering_3d(analysis_results) 