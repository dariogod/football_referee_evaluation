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

def create_analysis_report(analysis_results, output_dir='clustering_analysis'):
    """Create comprehensive analysis report with saved figures"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all data
    all_player_colors = []
    all_non_reassigned = []
    all_reassigned = []
    all_cluster_centers = []
    
    # Analysis per frame
    frame_summaries = []
    
    for i, frame_data in enumerate(analysis_results):
        if frame_data is None:
            continue
            
        player_colors = np.array(frame_data['player_colors'])
        clusters = frame_data['clusters']
        unclassified_data = frame_data['unclassified_data']
        
        all_player_colors.extend(player_colors)
        
        # Collect cluster centers
        for cluster in clusters:
            all_cluster_centers.append(cluster['center'])
        
        # Separate reassigned and non-reassigned
        non_reassigned = [data['color'] for data in unclassified_data if not data['reassigned']]
        reassigned = [data['color'] for data in unclassified_data if data['reassigned']]
        
        all_non_reassigned.extend(non_reassigned)
        all_reassigned.extend(reassigned)
        
        # Frame summary
        frame_summaries.append({
            'frame_idx': i,
            'image_path': frame_data['image_path'],
            'num_players': len(player_colors),
            'num_clusters': len(clusters),
            'num_reassigned': len(reassigned),
            'num_non_reassigned': len(non_reassigned),
            'non_reassigned_colors': non_reassigned
        })
    
    # Convert to numpy arrays
    all_player_colors = np.array(all_player_colors)
    all_non_reassigned = np.array(all_non_reassigned) if all_non_reassigned else np.empty((0, 3))
    all_reassigned = np.array(all_reassigned) if all_reassigned else np.empty((0, 3))
    all_cluster_centers = np.array(all_cluster_centers) if all_cluster_centers else np.empty((0, 3))
    
    # 1. Overall 3D visualization
    print("Creating overall 3D visualization...")
    fig = plt.figure(figsize=(20, 15))
    
    # Main 3D plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(all_player_colors[:, 0], all_player_colors[:, 1], all_player_colors[:, 2], 
               c='blue', alpha=0.6, s=30, label='Player Colors')
    
    if len(all_cluster_centers) > 0:
        ax1.scatter(all_cluster_centers[:, 0], all_cluster_centers[:, 1], all_cluster_centers[:, 2], 
                   c='black', s=100, marker='X', label='Cluster Centers')
    
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
    
    # Focus on non-reassigned
    ax2 = fig.add_subplot(222, projection='3d')
    if len(all_non_reassigned) > 0:
        ax2.scatter(all_player_colors[:, 0], all_player_colors[:, 1], all_player_colors[:, 2], 
                   c='lightblue', alpha=0.3, s=20, label='Player Colors')
        ax2.scatter(all_non_reassigned[:, 0], all_non_reassigned[:, 1], all_non_reassigned[:, 2], 
                   c='red', alpha=1.0, s=80, label='Non-Reassigned', marker='s')
        
        if len(all_cluster_centers) > 0:
            ax2.scatter(all_cluster_centers[:, 0], all_cluster_centers[:, 1], all_cluster_centers[:, 2], 
                       c='black', s=150, marker='X', label='Cluster Centers')
        
        ax2.set_title('Focus: Non-Reassigned Samples')
    else:
        ax2.text(0.5, 0.5, 0.5, 'No Non-Reassigned\nSamples Found', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('No Non-Reassigned Samples')
    
    ax2.set_xlabel('Red')
    ax2.set_ylabel('Green')
    ax2.set_zlabel('Blue')
    ax2.legend()
    
    # 2D projections
    ax3 = fig.add_subplot(223)
    ax3.scatter(all_player_colors[:, 0], all_player_colors[:, 1], c='blue', alpha=0.6, s=20, label='Players')
    
    if len(all_cluster_centers) > 0:
        ax3.scatter(all_cluster_centers[:, 0], all_cluster_centers[:, 1], 
                   c='black', s=100, marker='X', label='Cluster Centers')
    
    if len(all_non_reassigned) > 0:
        ax3.scatter(all_non_reassigned[:, 0], all_non_reassigned[:, 1], 
                   c='red', alpha=0.8, s=40, label='Non-Reassigned', marker='s')
    
    ax3.set_xlabel('Red')
    ax3.set_ylabel('Green')
    ax3.set_title('2D Projection: Red vs Green')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Distance analysis
    ax4 = fig.add_subplot(224)
    if len(all_non_reassigned) > 0 and len(all_cluster_centers) > 0:
        distances_to_clusters = []
        for non_reassigned_color in all_non_reassigned:
            min_dist = float('inf')
            for cluster_center in all_cluster_centers:
                dist = np.linalg.norm(np.array(non_reassigned_color) - np.array(cluster_center))
                min_dist = min(min_dist, dist)
            distances_to_clusters.append(min_dist)
        
        ax4.hist(distances_to_clusters, bins=20, alpha=0.7, color='red', edgecolor='black')
        ax4.set_xlabel('Distance to Nearest Cluster Center')
        ax4.set_ylabel('Count')
        ax4.set_title('Distance Distribution: Non-Reassigned to Cluster Centers')
        ax4.grid(True, alpha=0.3)
        
        # Add statistics
        mean_dist = np.mean(distances_to_clusters)
        median_dist = np.median(distances_to_clusters)
        ax4.axvline(mean_dist, color='blue', linestyle='--', label=f'Mean: {mean_dist:.1f}')
        ax4.axvline(median_dist, color='green', linestyle='--', label=f'Median: {median_dist:.1f}')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No Distance Analysis\nAvailable', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('No Distance Analysis Available')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_clustering_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create detailed color analysis
    print("Creating detailed color analysis...")
    if len(all_non_reassigned) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Detailed Color Analysis of Non-Reassigned Samples', fontsize=16)
        
        # RGB channel histograms
        channels = ['Red', 'Green', 'Blue']
        colors = ['red', 'green', 'blue']
        
        for i, (channel, color) in enumerate(zip(channels, colors)):
            ax = axes[0, i]
            
            # Plot player colors
            ax.hist(all_player_colors[:, i], bins=30, alpha=0.5, label='Players', color='lightblue', density=True)
            
            # Plot non-reassigned
            ax.hist(all_non_reassigned[:, i], bins=20, alpha=0.8, label='Non-Reassigned', color=color, density=True)
            
            ax.set_xlabel(f'{channel} Value')
            ax.set_ylabel('Density')
            ax.set_title(f'{channel} Channel Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2D scatter plots
        projections = [('Red', 'Green', 0, 1), ('Red', 'Blue', 0, 2), ('Green', 'Blue', 1, 2)]
        
        for i, (x_label, y_label, x_idx, y_idx) in enumerate(projections):
            ax = axes[1, i]
            
            # Plot player colors
            ax.scatter(all_player_colors[:, x_idx], all_player_colors[:, y_idx], 
                      c='lightblue', alpha=0.5, s=20, label='Players')
            
            # Plot cluster centers
            if len(all_cluster_centers) > 0:
                ax.scatter(all_cluster_centers[:, x_idx], all_cluster_centers[:, y_idx], 
                          c='black', s=100, marker='X', label='Cluster Centers')
            
            # Plot non-reassigned
            ax.scatter(all_non_reassigned[:, x_idx], all_non_reassigned[:, y_idx], 
                      c='red', alpha=0.8, s=40, label='Non-Reassigned', marker='s')
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f'{x_label} vs {y_label}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detailed_color_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Generate text report
    print("Generating text report...")
    report_path = os.path.join(output_dir, 'clustering_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("CLUSTERING ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"SUMMARY STATISTICS:\n")
        f.write(f"Frames analyzed: {len(frame_summaries)}\n")
        f.write(f"Total player colors: {len(all_player_colors)}\n")
        f.write(f"Total cluster centers: {len(all_cluster_centers)}\n")
        f.write(f"Total reassigned unclassified: {len(all_reassigned)}\n")
        f.write(f"Total non-reassigned unclassified: {len(all_non_reassigned)}\n")
        f.write(f"Reassignment success rate: {len(all_reassigned)/(len(all_reassigned)+len(all_non_reassigned))*100:.1f}%\n\n")
        
        if len(all_non_reassigned) > 0:
            f.write("NON-REASSIGNED COLOR ANALYSIS:\n")
            f.write(f"RGB ranges for non-reassigned:\n")
            f.write(f"  Red: {np.min(all_non_reassigned[:, 0]):.1f} - {np.max(all_non_reassigned[:, 0]):.1f}\n")
            f.write(f"  Green: {np.min(all_non_reassigned[:, 1]):.1f} - {np.max(all_non_reassigned[:, 1]):.1f}\n")
            f.write(f"  Blue: {np.min(all_non_reassigned[:, 2]):.1f} - {np.max(all_non_reassigned[:, 2]):.1f}\n\n")
            
            f.write(f"Average non-reassigned color (RGB):\n")
            avg_color = np.mean(all_non_reassigned, axis=0)
            f.write(f"  ({avg_color[0]:.1f}, {avg_color[1]:.1f}, {avg_color[2]:.1f})\n\n")
            
            f.write(f"Standard deviation of non-reassigned colors:\n")
            std_color = np.std(all_non_reassigned, axis=0)
            f.write(f"  ({std_color[0]:.1f}, {std_color[1]:.1f}, {std_color[2]:.1f})\n\n")
        
        if len(all_cluster_centers) > 0:
            f.write("CLUSTER CENTER ANALYSIS:\n")
            f.write(f"Average cluster center (RGB):\n")
            avg_center = np.mean(all_cluster_centers, axis=0)
            f.write(f"  ({avg_center[0]:.1f}, {avg_center[1]:.1f}, {avg_center[2]:.1f})\n\n")
        
        f.write("FRAME-BY-FRAME BREAKDOWN:\n")
        for summary in frame_summaries:
            f.write(f"Frame {summary['frame_idx']+1}: {os.path.basename(summary['image_path'])}\n")
            f.write(f"  Players: {summary['num_players']}, Clusters: {summary['num_clusters']}\n")
            f.write(f"  Reassigned: {summary['num_reassigned']}, Non-reassigned: {summary['num_non_reassigned']}\n")
            if summary['non_reassigned_colors']:
                f.write(f"  Non-reassigned colors: {summary['non_reassigned_colors']}\n")
            f.write("\n")
    
    print(f"Analysis complete! Results saved to: {output_dir}")
    print(f"- overall_clustering_analysis.png: Main 3D visualization")
    print(f"- detailed_color_analysis.png: Detailed color channel analysis")
    print(f"- clustering_analysis_report.txt: Text summary report")

def collect_and_analyze(base_dir='prtreid_output', image_base_dir='test', 
                       confidence_threshold=3.564, max_frames=50, output_dir='clustering_analysis'):
    """Collect data and create comprehensive analysis"""
    
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
                    frame_analysis = analyze_frame_clustering(predictions, image_path)
                    if frame_analysis is not None:
                        analysis_results.append(frame_analysis)
                        frames_processed += 1
                        print(f"  Processed frame {frame_dir} ({frames_processed}/{max_frames})")
                except Exception as e:
                    print(f"  Error processing frame {frame_dir}: {e}")
                    continue
    
    if not analysis_results:
        print("No frames with unclassified detections found!")
        return
    
    print(f"\nCollected {len(analysis_results)} frames with unclassified detections")
    create_analysis_report(analysis_results, output_dir)

if __name__ == "__main__":
    print("Starting comprehensive clustering analysis...")
    
    config = {
        'base_dir': 'prtreid_output',
        'image_base_dir': 'test',
        'confidence_threshold': 3.564,
        'max_frames': 50,
        'output_dir': 'clustering_analysis'
    }
    
    collect_and_analyze(**config) 