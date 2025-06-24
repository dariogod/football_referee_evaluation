import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.stats import gamma
import seaborn as sns


class DecisionCriticalZoneVisualizer:
    def __init__(self, pitch_image_path="src/utils/pitch_2.png"):
        """
        Initialize the decision critical zone visualizer
        
        Args:
            pitch_image_path: Path to the pitch background image
        """
        self.pitch_image_path = pitch_image_path
        
        # Define colors for different roles (in BGR format for OpenCV)
        self.role_colors = {
            "TEAM A": (255, 255, 255),  # white
            "TEAM B": (0, 0, 255),      # red
            "REF": (0, 255, 255),       # yellow
            "GK": (0, 0, 0)             # black
        }
        
        # Scale factor for pitch image (4x as in original code)
        self.scale_factor = 4
        
        # Gamma distribution parameters
        self.gamma_alpha = 5.05
        self.gamma_theta = 2.83
        
    def load_pitch_image(self):
        """Load and scale the pitch background image"""
        pitch_img = cv2.imread(self.pitch_image_path)
        if pitch_img is None:
            raise ValueError(f"Could not load pitch image from {self.pitch_image_path}")
        
        # Get pitch dimensions and scale up
        height, width, _ = pitch_img.shape
        scaled_img = cv2.resize(pitch_img, (width * self.scale_factor, height * self.scale_factor))
        
        return scaled_img
    
    def extract_player_positions(self, frame_detections):
        """Extract player positions and team information from frame detections"""
        player_positions = []
        player_teams = []
        
        for detection in frame_detections["detections"]:
            coords = detection["minimap_coordinates"]
            role = detection.get("role", "UNKNOWN")
            
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
        
        return np.array(player_positions), np.array(player_teams)
    
    def find_clusters(self, player_positions, player_teams, eps=4, min_samples=2):
        """Find clusters of players from opposing teams"""
        if len(player_positions) == 0:
            return []
        
        # Run DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(player_positions)
        labels = clustering.labels_
        
        # Number of clusters (excluding noise points with label -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        cluster_centers = []
        
        # For each cluster, check if it contains players from both teams
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_points = player_positions[cluster_mask]
            cluster_teams = player_teams[cluster_mask]
            
            # Check if cluster contains players from both teams
            unique_teams = np.unique(cluster_teams)
            if len(unique_teams) >= 2:  # Both teams present
                # Calculate cluster center
                center_x, center_y = np.mean(cluster_points, axis=0)
                cluster_centers.append([center_x, center_y])
        
        return cluster_centers
    
    def gamma_score(self, distance, alpha=None, theta=None):
        """Calculate the normalized gamma score for a given distance"""
        if alpha is None:
            alpha = self.gamma_alpha
        if theta is None:
            theta = self.gamma_theta
            
        mode = (alpha - 1) * theta
        g_max = gamma.pdf(mode, alpha, scale=theta)
        return gamma.pdf(distance, alpha, scale=theta) / g_max
    
    def create_heatmap(self, cluster_centers, referee_position=None, pitch_width=105, pitch_height=68):
        """Create a heatmap overlay based on gamma distribution scores"""
        # Create coordinate grids with high resolution
        x_coords = np.linspace(0, pitch_width, 1050)  # 10 points per meter
        y_coords = np.linspace(0, pitch_height, 680)   # 10 points per meter
        X, Y = np.meshgrid(x_coords, y_coords)
        
        if len(cluster_centers) == 0:
            return np.zeros_like(X), x_coords, y_coords
        
        # Reshape cluster centers for broadcasting
        centers = np.array(cluster_centers)
        centers_x = centers[:, 0].reshape(-1, 1, 1)
        centers_y = centers[:, 1].reshape(-1, 1, 1)
        
        # Calculate distances to all centers at once
        distances = np.sqrt((X - centers_x)**2 + (Y - centers_y)**2)
        
        # Calculate heatmap using gamma scoring
        heatmap = np.prod(self.gamma_score(distances), axis=0)
        
        # If referee position is provided, calculate and print their score
        if referee_position is not None:
            ref_distances = np.sqrt(np.sum((centers - referee_position)**2, axis=1))
            ref_score = np.prod(self.gamma_score(ref_distances))
            ref_score_normalized = (ref_score - heatmap.min()) / (heatmap.max() - heatmap.min())
            print(f"Referee score for frame: {ref_score_normalized:.4f}")
        
        # Normalize heatmap values to 0-1 range
        if heatmap.max() > heatmap.min():
            heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            heatmap_normalized = heatmap
        
        return heatmap_normalized, x_coords, y_coords
    
    def get_referee_position(self, frame_detections):
        """Extract referee position from frame detections"""
        for detection in frame_detections["detections"]:
            if detection.get("role") == "REF" and detection.get("minimap_coordinates"):
                coords = detection["minimap_coordinates"]
                x = coords["x"]
                y = coords["y"]
                x_max = coords["x_max"]
                y_max = coords["y_max"]
                norm_x = x / x_max * 105
                norm_y = y / y_max * 68
                return np.array([norm_x, norm_y])
        return None
    
    def visualize_clusters(self, detections, frame_id, output_path):
        """Visualize decision critical zones (clusters) for a single frame"""
        # Load pitch image
        pitch_img = self.load_pitch_image()
        pitch_height, pitch_width, _ = pitch_img.shape
        
        # Find detections for the specific frame
        frame_detections = next((item for item in detections if item["frame_id"] == frame_id), None)
        
        if frame_detections is None:
            print(f"No detections found for frame {frame_id}")
            return
        
        # Extract player positions
        player_positions, player_teams = self.extract_player_positions(frame_detections)
        
        if len(player_positions) == 0:
            print(f"No player positions found for frame {frame_id}")
            return
        
        # Find clusters
        cluster_centers = self.find_clusters(player_positions, player_teams)
        
        # Create a copy of the pitch image to draw on
        result_img = pitch_img.copy()
        
        # Draw all players first
        for i, (pos, team) in enumerate(zip(player_positions, player_teams)):
            norm_x = int(pos[0] / 105 * pitch_width)
            norm_y = int(pos[1] / 68 * pitch_height)
            
            role = "TEAM A" if team == 1 else "TEAM B" if team == 2 else "UNKNOWN"
            color = self.role_colors.get(role, (0, 0, 0))
            
            # Draw player as a circle
            cv2.circle(result_img, (norm_x, norm_y), 60, color, -1)
        
        # Draw referee if present
        referee_position = self.get_referee_position(frame_detections)
        if referee_position is not None:
            ref_x = int(referee_position[0] / 105 * pitch_width)
            ref_y = int(referee_position[1] / 68 * pitch_height)
            cv2.circle(result_img, (ref_x, ref_y), 60, self.role_colors["REF"], -1)
        
        # Draw cluster centers and circles
        for center in cluster_centers:
            center_x_px = int(center[0] / 105 * pitch_width)
            center_y_px = int(center[1] / 68 * pitch_height)
            
            # Draw circle around the cluster (estimated radius)
            radius = 200  # Adjust this value as needed
            cv2.circle(result_img, (center_x_px, center_y_px), radius, (0, 255, 0), 8)
            
            # Draw cluster center with green dot
            cv2.circle(result_img, (center_x_px, center_y_px), 20, (0, 255, 0), -1)
        
        # Save the image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result_img)
        print(f"Saved decision critical zones for frame {frame_id} with {len(cluster_centers)} clusters to {output_path}")
    
    def visualize_heatmap(self, detections, frame_id, output_path, output_dir=None):
        """Visualize gamma distribution heatmap for a single frame"""
        # Find detections for the specific frame
        frame_detections = next((item for item in detections if item["frame_id"] == frame_id), None)
        
        if frame_detections is None:
            print(f"No detections found for frame {frame_id}")
            return
        
        # Extract player positions and find clusters
        player_positions, player_teams = self.extract_player_positions(frame_detections)
        
        if len(player_positions) == 0:
            print(f"No player positions found for frame {frame_id}")
            return
        
        cluster_centers = self.find_clusters(player_positions, player_teams)
        
        if len(cluster_centers) == 0:
            print(f"No valid cluster centers found for frame {frame_id}")
            return
        
        # Get referee position
        referee_position = self.get_referee_position(frame_detections)
        
        # Create heatmap
        heatmap, x_coords, y_coords = self.create_heatmap(cluster_centers, referee_position)
        
        # Load the cluster visualization image as background instead of raw pitch
        if output_dir:
            cluster_image_path = os.path.join(output_dir, f"minimap_decision_critical_zones_{frame_id:06d}.png")
        else:
            # Try to construct path from output_path
            cluster_image_path = output_path.replace("minimap_heatmap_", "minimap_decision_critical_zones_")
        
        if os.path.exists(cluster_image_path):
            cluster_img = cv2.imread(cluster_image_path)
            if cluster_img is not None:
                cluster_rgb = cv2.cvtColor(cluster_img, cv2.COLOR_BGR2RGB)
            else:
                print(f"Could not load cluster image from {cluster_image_path}, using pitch image instead")
                pitch_img = self.load_pitch_image()
                cluster_rgb = cv2.cvtColor(pitch_img, cv2.COLOR_BGR2RGB)
        else:
            print(f"Cluster image not found at {cluster_image_path}, using pitch image instead")
            pitch_img = self.load_pitch_image()
            cluster_rgb = cv2.cvtColor(pitch_img, cv2.COLOR_BGR2RGB)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Display cluster image as background
        ax.imshow(cluster_rgb, extent=[0, 105, 0, 68], alpha=0.8)
        
        # Overlay heatmap
        extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]
        im = ax.imshow(heatmap, cmap='YlOrRd', alpha=0.7, extent=extent)
        
        # Draw players on top of heatmap for better visibility
        for i, (pos, team) in enumerate(zip(player_positions, player_teams)):
            role = "TEAM A" if team == 1 else "TEAM B" if team == 2 else "UNKNOWN"
            color = 'white' if role == "TEAM A" else 'red' if role == "TEAM B" else 'black'
            ax.plot(pos[0], 68-pos[1], 'o', color=color, markersize=10, markeredgecolor='black', markeredgewidth=0)
        
        # Draw referee on top of heatmap
        if referee_position is not None:
            ax.plot(referee_position[0], 68-referee_position[1], 'o', color='yellow', markersize=10, 
                   markeredgecolor='black', markeredgewidth=0)
        
        # # Draw cluster centers on top
        # for center in cluster_centers:
        #     ax.plot(center[0], 68-center[1], 'o', color='green', markersize=8, 
        #            markeredgecolor='black', markeredgewidth=0)
        
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 68)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Save the figure
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved heatmap for frame {frame_id} to {output_path}")
    
    def visualize_multiple_frames(self, detections, frame_ids, output_dir, include_heatmap=True):
        """Visualize multiple frames with both cluster and heatmap visualizations"""
        for frame_id in frame_ids:
            # Create cluster visualization first
            cluster_output_path = os.path.join(output_dir, f"minimap_decision_critical_zones_{frame_id:06d}.png")
            self.visualize_clusters(detections, frame_id, cluster_output_path)
            
            # Create heatmap visualization if requested (using cluster image as background)
            if include_heatmap:
                heatmap_output_path = os.path.join(output_dir, f"minimap_heatmap_{frame_id:06d}.png")
                self.visualize_heatmap(detections, frame_id, heatmap_output_path, output_dir)


def main():
    """Example usage"""
    # Load detection data
    detections_path = "data/example/predictions/role_assignment/detections.json"
    
    if not os.path.exists(detections_path):
        print(f"Detections file not found: {detections_path}")
        return
        
    with open(detections_path, "r") as f:
        detections = json.load(f)
    
    # Initialize visualizer
    visualizer = DecisionCriticalZoneVisualizer()
    
    # Example frames to visualize
    interesting_frames = [212, 400, 460]
    
    # Output directory
    output_dir = "data/example/images_for_paper"
    
    # Visualize frames
    visualizer.visualize_multiple_frames(detections, interesting_frames, output_dir)


if __name__ == "__main__":
    main() 