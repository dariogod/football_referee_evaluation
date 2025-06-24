#!/usr/bin/env python3
"""
Superimpose heatmaps from three different visualizers by multiplying their normalized scores
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy.stats import gamma
from sklearn.neighbors import KernelDensity
from scipy.interpolate import griddata
import seaborn as sns
import time

# Import the visualizer classes
from decision_critical_zone_visualizer import DecisionCriticalZoneVisualizer
from angle_duel_visualizer import AngleDuelVisualizer
from general_position_visualizer import GeneralPositionVisualizer


class HeatmapSuperimposer:
    def __init__(self):
        print("🔧 Initializing HeatmapSuperimposer...")
        self.pitch_length = 105  # meters
        self.pitch_width = 68    # meters
        
        # High resolution grid for calculations
        self.resolution = 2
        print(f"📐 Creating coordinate grid with resolution: {self.resolution} points per meter...")
        self.x_coords = np.linspace(0, self.pitch_length, self.pitch_length * self.resolution)
        self.y_coords = np.linspace(0, self.pitch_width, self.pitch_width * self.resolution)
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords)
        print(f"📐 Grid shape: {self.X.shape} ({self.X.size:,} total points)")
        
        # Initialize visualizers
        print("🔧 Initializing visualizer components...")
        self.critical_zone_viz = DecisionCriticalZoneVisualizer()
        self.duel_viz = AngleDuelVisualizer() 
        self.general_viz = GeneralPositionVisualizer()
        print("✅ HeatmapSuperimposer initialization complete!")
    
    def S_distance(self, x, y, cluster_centers):
        """
        Distance-based score function using gamma distribution
        Returns normalized score (0-1) based on proximity to decision critical zones
        """
        print(f"🎯 Computing S_distance with {len(cluster_centers)} cluster centers...")
        start_time = time.time()
        
        if len(cluster_centers) == 0:
            print("⚠️  No cluster centers found, returning neutral score")
            return np.ones_like(x) * 0.5  # neutral score if no clusters
        
        # Convert to numpy arrays for broadcasting
        centers = np.array(cluster_centers)
        
        # Calculate distances from each point to all cluster centers
        scores = np.ones_like(x)
        for i, center in enumerate(centers):
            print(f"  📍 Processing cluster {i+1}/{len(centers)} at position {center}")
            distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            # Apply gamma scoring
            gamma_scores = self.critical_zone_viz.gamma_score(distances)
            scores *= gamma_scores
        
        # Normalize to 0-1 range
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        elapsed = time.time() - start_time
        print(f"✅ S_distance computed in {elapsed:.2f}s (range: {scores.min():.3f} - {scores.max():.3f})")
        return scores
    
    def S_duel_angle(self, x, y, duels):
        """
        Angle-based score function using cosine scoring
        Returns normalized score (0-1) based on angular positioning relative to duels
        """
        print(f"⚔️  Computing S_duel_angle with {len(duels)} duels...")
        start_time = time.time()
        
        if len(duels) == 0:
            print("⚠️  No duels found, returning neutral score")
            return np.ones_like(x) * 0.5  # neutral score if no duels
        
        scores = np.ones_like(x)
        
        for i, duel in enumerate(duels):
            print(f"  ⚔️  Processing duel {i+1}/{len(duels)} at center {duel['center']}")
            duel_center = duel['center']
            duel_line_angle = duel['angle']
            
            # Calculate angles from duel center to each grid point
            dx = x - duel_center[0]
            dy = y - duel_center[1]
            point_angles = np.arctan2(dy, dx)
            
            # Calculate relative angles to the duel line
            relative_angles = np.abs(point_angles - duel_line_angle)
            
            # Normalize angles to [0, π/2] (acute angles)
            relative_angles = np.mod(relative_angles, 2 * np.pi)
            relative_angles = np.minimum(relative_angles, 2 * np.pi - relative_angles)
            relative_angles = np.minimum(relative_angles, np.pi - relative_angles)
            relative_angles = np.degrees(relative_angles)
            
            # Apply cosine scoring function
            angle_scores = self.duel_viz.cosine_score_function(relative_angles)
            scores *= angle_scores
        
        # Normalize to 0-1 range
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        elapsed = time.time() - start_time
        print(f"✅ S_duel_angle computed in {elapsed:.2f}s (range: {scores.min():.3f} - {scores.max():.3f})")
        return scores
    
    def S_general_position(self, x, y, use_symmetrical=True):
        """
        General position score function using KDE of historical referee positions
        Returns normalized score (0-1) based on typical referee positioning
        """
        print(f"📊 Computing S_general_position (symmetrical: {use_symmetrical})...")
        start_time = time.time()
        
        try:
            # Load referee positions
            print("📁 Loading referee position data...")
            referee_x, referee_y = self.general_viz.load_referee_positions()
            print(f"📁 Loaded {len(referee_x)} referee positions")
            
            if use_symmetrical:
                print("🔄 Creating symmetrical data...")
                # Add symmetrical data
                symmetrical_x = list(referee_x)
                symmetrical_y = list(referee_y)
                
                # Add 180-degree rotated positions
                for rx, ry in zip(referee_x, referee_y):
                    rotated_x = self.pitch_length - rx
                    rotated_y = self.pitch_width - ry
                    symmetrical_x.append(rotated_x)
                    symmetrical_y.append(rotated_y)
                
                data_x, data_y = symmetrical_x, symmetrical_y
                print(f"🔄 Symmetrical data created: {len(data_x)} total points")
            else:
                data_x, data_y = referee_x, referee_y
            
            # Create KDE model
            print("🧮 Training KDE model...")
            kde_data = np.column_stack([data_x, data_y])
            kde = KernelDensity(bandwidth=1.5, kernel='gaussian')
            kde.fit(kde_data)
            
            # Evaluate KDE at each grid point
            print(f"🧮 Evaluating KDE on {x.size:,} grid points...")
            grid_points = np.column_stack([x.ravel(), y.ravel()])
            log_density = kde.score_samples(grid_points)
            density = np.exp(log_density).reshape(x.shape)
            
            # Normalize to 0-1 range
            if density.max() > density.min():
                density = (density - density.min()) / (density.max() - density.min())
            
            elapsed = time.time() - start_time
            print(f"✅ S_general_position computed in {elapsed:.2f}s (range: {density.min():.3f} - {density.max():.3f})")
            return density
            
        except FileNotFoundError:
            print("⚠️  Warning: Referee positions file not found, using uniform distribution")
            return np.ones_like(x) * 0.5
    
    def S_overall(self, x, y, cluster_centers, duels, use_symmetrical=True):
        """
        Overall score function combining all three heatmaps
        Returns normalized score (0-1) from the product of all individual scores
        """
        print("🔗 Computing S_overall by combining all score functions...")
        start_time = time.time()
        
        # Get individual scores
        distance_scores = self.S_distance(x, y, cluster_centers)
        angle_scores = self.S_duel_angle(x, y, duels)
        position_scores = self.S_general_position(x, y, use_symmetrical)
        
        print("🔗 Multiplying score functions...")
        # Multiply all scores together
        combined_scores = distance_scores * angle_scores * position_scores
        
        # Normalize to 0-1 range
        if combined_scores.max() > combined_scores.min():
            combined_scores = (combined_scores - combined_scores.min()) / (combined_scores.max() - combined_scores.min())
        
        elapsed = time.time() - start_time
        print(f"✅ S_overall computed in {elapsed:.2f}s (range: {combined_scores.min():.3f} - {combined_scores.max():.3f})")
        return combined_scores, distance_scores, angle_scores, position_scores
    
    def visualize_superimposed_heatmap(self, detections, frame_id, output_dir):
        """
        Create and visualize the superimposed heatmap for a single frame
        """
        print(f"\n🎨 Creating superimposed heatmap for frame {frame_id}...")
        overall_start_time = time.time()
        
        # Find detections for the specific frame
        print(f"🔍 Looking for detections in frame {frame_id}...")
        frame_detections = next((item for item in detections if item["frame_id"] == frame_id), None)
        
        if frame_detections is None:
            print(f"❌ No detections found for frame {frame_id}")
            return
        
        print(f"✅ Found detections for frame {frame_id}")
        
        # Extract data needed for each score function
        print("📊 Extracting game state data...")
        
        # 1. Get cluster centers for distance scoring
        print("  🎯 Extracting player positions and clusters...")
        player_positions, player_teams = self.critical_zone_viz.extract_player_positions(frame_detections)
        cluster_centers = self.critical_zone_viz.find_clusters(player_positions, player_teams) if len(player_positions) > 0 else []
        print(f"  🎯 Found {len(player_positions)} players, {len(cluster_centers)} clusters")
        
        # 2. Get duels for angle scoring
        print("  ⚔️  Extracting duels...")
        if len(player_positions) > 0:
            player_pos_duel, player_teams_duel, referee_position = self.duel_viz.extract_player_positions(frame_detections)
            duels = self.duel_viz.find_duels(player_pos_duel, player_teams_duel)
            print(f"  ⚔️  Found {len(duels)} duels, referee at {referee_position}")
        else:
            duels = []
            referee_position = None
            print("  ⚔️  No duels found")
        
        # 3. Calculate overall score
        print("\n🧮 Computing combined score functions...")
        overall_scores, distance_scores, angle_scores, position_scores = self.S_overall(
            self.X, self.Y, cluster_centers, duels, use_symmetrical=True
        )
        
        # Create single plot visualization
        print("\n🎨 Creating visualization...")
        viz_start_time = time.time()
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(f'Superimposed Heatmap - Frame {frame_id}', fontsize=16)
        
        # Load pitch image if available
        pitch_img = None
        if os.path.exists("src/utils/pitch_2.png"):
            try:
                pitch_img = plt.imread("src/utils/pitch_2.png")
                print("✅ Loaded pitch background image")
            except:
                print("⚠️  Could not load pitch background image")
        
        extent = [0, self.pitch_length, 0, self.pitch_width]
        
        # Show pitch background if available
        if pitch_img is not None:
            ax.imshow(pitch_img, extent=extent, alpha=1)
        
        # Overlay superimposed heatmap
        print("  🎨 Plotting superimposed heatmap...")
        im = ax.imshow(overall_scores, cmap='YlOrRd', alpha=0.8, extent=extent)
        ax.set_title('Superimposed Referee Position Heatmap', fontsize=14)
        ax.set_xlabel('Pitch Length (m)', fontsize=12)
        ax.set_ylabel('Pitch Width (m)', fontsize=12)
        ax.set_xlim(0, self.pitch_length)
        ax.set_ylim(0, self.pitch_width)
        ax.set_aspect('equal')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Optimal Position Score', fontsize=12)
        
        # Draw players and referee if available
        if len(player_positions) > 0:
            for j, (pos, team) in enumerate(zip(player_positions, player_teams)):
                color = 'white' if team == 1 else 'red' if team == 2 else 'black'
                ax.plot(pos[0], 68-pos[1], 'o', color=color, markersize=8, 
                       markeredgecolor='black', markeredgewidth=0)
        
        if referee_position is not None:
            ax.plot(referee_position[0], 68-referee_position[1], 'o', color='yellow', 
                   markersize=10, markeredgecolor='black', markeredgewidth=0)
            
            # Calculate and print detailed scores for referee position
            print(f"\n📊 REFEREE POSITION ANALYSIS - Frame {frame_id}")
            print(f"{'='*50}")
            print(f"🏃 Referee Position: ({referee_position[0]:.2f}, {referee_position[1]:.2f})")
            
            # Get referee position indices in the grid
            ref_x_idx = int(referee_position[0] / self.pitch_length * (len(self.x_coords) - 1))
            ref_y_idx = int(referee_position[1] / self.pitch_width * (len(self.y_coords) - 1))
            
            if 0 <= ref_x_idx < len(self.x_coords) and 0 <= ref_y_idx < len(self.y_coords):
                # Extract individual scores at referee position
                ref_distance_score = distance_scores[ref_y_idx, ref_x_idx]
                ref_angle_score = angle_scores[ref_y_idx, ref_x_idx]
                ref_position_score = position_scores[ref_y_idx, ref_x_idx]
                ref_final_score = overall_scores[ref_y_idx, ref_x_idx]
                
                print(f"🎯 Distance Score (S_distance):     {ref_distance_score:.4f}")
                print(f"⚔️  Angle Score (S_duel_angle):     {ref_angle_score:.4f}")
                print(f"📍 Position Score (S_general):      {ref_position_score:.4f}")
                print(f"{'─'*50}")
                print(f"🏆 FINAL COMBINED SCORE:           {ref_final_score:.4f}")
                print(f"{'='*50}")
                
                # Add performance evaluation
                if ref_final_score >= 0.8:
                    performance = "🟢 EXCELLENT"
                elif ref_final_score >= 0.6:
                    performance = "🟡 GOOD"
                elif ref_final_score >= 0.4:
                    performance = "🟠 AVERAGE"
                else:
                    performance = "🔴 NEEDS IMPROVEMENT"
                
                print(f"📈 Performance Rating: {performance}")
                print(f"{'='*50}\n")
            else:
                print("⚠️  Referee position is outside the grid bounds")
                        
        # Add text summary
        summary_text = f"""Frame {frame_id} | Clusters: {len(cluster_centers)} | Duels: {len(duels)} | Score Range: {overall_scores.min():.3f}-{overall_scores.max():.3f}"""
        
        if referee_position is not None:
            # Evaluate referee position
            ref_x_idx = int(referee_position[0] / self.pitch_length * (len(self.x_coords) - 1))
            ref_y_idx = int(referee_position[1] / self.pitch_width * (len(self.y_coords) - 1))
            if 0 <= ref_x_idx < len(self.x_coords) and 0 <= ref_y_idx < len(self.y_coords):
                ref_score = overall_scores[ref_y_idx, ref_x_idx]
                summary_text += f" | Referee Score: {ref_score:.3f}"
        
        fig.text(0.5, 0.02, summary_text, fontsize=10, ha='center', va='bottom')
        
        viz_elapsed = time.time() - viz_start_time
        print(f"✅ Visualization created in {viz_elapsed:.2f}s")
        
        # Save the figure
        print("💾 Saving figure...")
        output_path = os.path.join(output_dir, f"superimposed_heatmap_{frame_id:06d}.png")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        overall_elapsed = time.time() - overall_start_time
        print(f"✅ Saved superimposed heatmap for frame {frame_id} to {output_path}")
        print(f"⏱️  Total time for frame {frame_id}: {overall_elapsed:.2f}s")
        return output_path
    
    def visualize_multiple_frames(self, detections, frame_ids, output_dir):
        """Visualize superimposed heatmaps for multiple frames"""
        print(f"\n🚀 Starting batch processing of {len(frame_ids)} frames...")
        total_start_time = time.time()
        
        for i, frame_id in enumerate(frame_ids):
            print(f"\n{'='*60}")
            print(f"🔄 Processing frame {i+1}/{len(frame_ids)}: {frame_id}")
            print(f"{'='*60}")
            self.visualize_superimposed_heatmap(detections, frame_id, output_dir)
        
        total_elapsed = time.time() - total_start_time
        print(f"\n🎉 Batch processing complete!")
        print(f"⏱️  Total processing time: {total_elapsed:.2f}s")
        print(f"⏱️  Average time per frame: {total_elapsed/len(frame_ids):.2f}s")


def main():
    """Example usage"""
    print("🏈 Football Referee Evaluation - Heatmap Superimposition")
    print("="*60)
    
    # Load detection data
    detections_path = "data/example/predictions/role_assignment/detections.json"
    
    if not os.path.exists(detections_path):
        print(f"❌ Detections file not found: {detections_path}")
        return
        
    print("📁 Loading detection data...")
    with open(detections_path, "r") as f:
        detections = json.load(f)
    print(f"✅ Loaded {len(detections)} frames of detection data")
    
    # Initialize superimposer
    superimposer = HeatmapSuperimposer()
    
    # Example frames to visualize
    test_frames = [212, 400, 460]
    
    # Output directory
    output_dir = "data/example/images_for_paper/superimposed"
    
    # Create superimposed heatmaps
    superimposer.visualize_multiple_frames(detections, test_frames, output_dir)


if __name__ == "__main__":
    main()
