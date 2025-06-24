import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns


class GeneralPositionVisualizer:
    def __init__(self, pitch_image_path="src/utils/pitch_2.png", referee_positions_path="plots/ground_truth/referee_positions.json"):
        """
        Initialize the general position visualizer
        
        Args:
            pitch_image_path: Path to the pitch background image
            referee_positions_path: Path to the referee positions JSON file
        """
        self.pitch_image_path = pitch_image_path
        self.referee_positions_path = referee_positions_path
        
        # Cache directory for background heatmaps
        self.cache_dir = "cache/heatmap_backgrounds"
        self.heatmap_cache_path = os.path.join(self.cache_dir, "referee_heatmap.png")
        self.heatmap_symmetrical_cache_path = os.path.join(self.cache_dir, "referee_heatmap_symmetrical.png")
        
        # Define colors for different roles (consistent with other visualizers)
        self.role_colors = {
            "TEAM A": 'white',
            "TEAM B": 'red', 
            "REF": 'yellow',
            "GK": 'black'
        }
        
        # KDE parameters (matching show_ground_truth_distributions_from_labels.py)
        self.kde_threshold = 0.00
        self.kde_bandwidth_adjustment = 1.5
        self.kde_levels = 100
        
        # Pitch dimensions
        self.pitch_length = 105  # meters
        self.pitch_width = 68    # meters
        
    def _is_cache_valid(self, cache_path):
        """Check if cache file exists and is newer than the referee positions file"""
        if not os.path.exists(cache_path):
            return False
        
        if not os.path.exists(self.referee_positions_path):
            return False
            
        cache_mtime = os.path.getmtime(cache_path)
        referee_mtime = os.path.getmtime(self.referee_positions_path)
        
        return cache_mtime > referee_mtime
    
    def _create_and_cache_heatmap_background(self, use_symmetrical=False):
        """Create heatmap background and cache it"""
        print(f"Creating and caching {'symmetrical ' if use_symmetrical else ''}heatmap background...")
        
        # Load referee positions
        referee_x, referee_y = self.load_referee_positions()
        
        if use_symmetrical:
            # Create symmetrical data by adding 180-degree rotated positions
            symmetrical_x = []
            symmetrical_y = []
            
            # Add original positions
            symmetrical_x.extend(referee_x)
            symmetrical_y.extend(referee_y)
            
            # Add 180-degree rotated positions around the center of the pitch
            for x, y in zip(referee_x, referee_y):
                rotated_x = self.pitch_length - x
                rotated_y = self.pitch_width - y
                symmetrical_x.append(rotated_x)
                symmetrical_y.append(rotated_y)
            
            data_x, data_y = symmetrical_x, symmetrical_y
            cache_path = self.heatmap_symmetrical_cache_path
        else:
            data_x, data_y = referee_x, referee_y
            cache_path = self.heatmap_cache_path
        
        # Create matplotlib figure for the heatmap background
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Load and display pitch image as background if available
        if os.path.exists(self.pitch_image_path):
            try:
                pitch_img = plt.imread(self.pitch_image_path)
                ax.imshow(pitch_img, extent=[0, self.pitch_length, self.pitch_width, 0])
            except Exception as e:
                print(f"Warning: Could not load pitch image: {e}")
        
        # Create KDE plot
        sns.kdeplot(
            x=data_x,
            y=data_y,
            cmap="YlOrRd",
            fill=True,
            alpha=0.7,
            levels=self.kde_levels,
            thresh=self.kde_threshold,
            bw_adjust=self.kde_bandwidth_adjustment,
            ax=ax
        )
        
        # Set plot properties
        ax.set_xlim(0, self.pitch_length)
        ax.set_ylim(0, self.pitch_width)
        ax.invert_yaxis()  # Invert y-axis to match image coordinates
        ax.axis('off')  # Remove axes for clean background
        
        # Save the heatmap background
        os.makedirs(self.cache_dir, exist_ok=True)
        plt.savefig(cache_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0)
        plt.close()
        
        print(f"Heatmap background cached to {cache_path}")
    
    def _get_heatmap_background(self, use_symmetrical=False):
        """Get cached heatmap background or create it if needed"""
        cache_path = self.heatmap_symmetrical_cache_path if use_symmetrical else self.heatmap_cache_path
        
        # Check if cache is valid
        if not self._is_cache_valid(cache_path):
            self._create_and_cache_heatmap_background(use_symmetrical)
        else:
            print(f"Using cached {'symmetrical ' if use_symmetrical else ''}heatmap background...")
        
        return cache_path

    def load_referee_positions(self):
        """Load referee positions from JSON file"""
        if not os.path.exists(self.referee_positions_path):
            raise FileNotFoundError(f"Referee positions file not found: {self.referee_positions_path}")
            
        with open(self.referee_positions_path, 'r') as f:
            data = json.load(f)
        
        positions = data['positions']
        referee_x = [pos['x'] for pos in positions]
        referee_y = [pos['y'] for pos in positions]
        
        return referee_x, referee_y
    
    def extract_player_positions(self, frame_detections):
        """Extract player positions and roles from frame detections"""
        players = []
        
        for detection in frame_detections["detections"]:
            coords = detection["minimap_coordinates"]
            role = detection.get("role", "UNKNOWN")
            
            # Skip if coordinates are None
            if coords is None:
                continue
            
            x = coords["x"]
            y = coords["y"]
            x_max = coords["x_max"]
            y_max = coords["y_max"]
            
            # Normalize to pitch dimensions (in meters)
            norm_x = x / x_max * self.pitch_length
            norm_y = y / y_max * self.pitch_width
            
            players.append({
                'x': norm_x,
                'y': norm_y,
                'role': role
            })
        
        return players
    
    def visualize_frame(self, detections, frame_id, output_path):
        """
        Visualize referee KDE with player positions for a single frame
        
        Args:
            detections: List of detection data
            frame_id: Frame number to visualize
            output_path: Path to save the output image
        """
        # Find detections for the specific frame
        frame_detections = next((item for item in detections if item["frame_id"] == frame_id), None)
        
        if frame_detections is None:
            print(f"No detections found for frame {frame_id}")
            return
        
        # Extract player positions
        players = self.extract_player_positions(frame_detections)
        
        if len(players) == 0:
            print(f"No player positions found for frame {frame_id}")
            return
        
        # Get cached heatmap background
        try:
            heatmap_bg_path = self._get_heatmap_background(use_symmetrical=False)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Load and display cached heatmap background
        heatmap_img = plt.imread(heatmap_bg_path)
        ax.imshow(heatmap_img, extent=[0, self.pitch_length, self.pitch_width, 0])
        
        # Draw players on top of heatmap
        for player in players:
            color = self.role_colors.get(player['role'], 'gray')
            marker_size = 120
            
            ax.scatter(player['x'], player['y'], 
                      c=color, s=marker_size, 
                      edgecolors='black', linewidths=0,
                      alpha=0.9, zorder=10)
        
        # Set plot properties
        ax.set_xlim(0, self.pitch_length)
        ax.set_ylim(0, self.pitch_width)
        ax.invert_yaxis()  # Invert y-axis to match image coordinates
        ax.set_xlabel('X Position (meters)', fontsize=12)
        ax.set_ylabel('Y Position (meters)', fontsize=12)
        ax.set_title(f'Referee Position Distribution (KDE) with Players - Frame {frame_id}', fontsize=14)
        
        # Save the figure
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved general position visualization for frame {frame_id} to {output_path}")
    
    def visualize_frame_symmetrical(self, detections, frame_id, output_path):
        """
        Visualize referee KDE with symmetrical data and player positions for a single frame
        
        Args:
            detections: List of detection data
            frame_id: Frame number to visualize
            output_path: Path to save the output image
        """
        # Find detections for the specific frame
        frame_detections = next((item for item in detections if item["frame_id"] == frame_id), None)
        
        if frame_detections is None:
            print(f"No detections found for frame {frame_id}")
            return
        
        # Extract player positions
        players = self.extract_player_positions(frame_detections)
        
        if len(players) == 0:
            print(f"No player positions found for frame {frame_id}")
            return
        
        # Get cached symmetrical heatmap background
        try:
            heatmap_bg_path = self._get_heatmap_background(use_symmetrical=True)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Load and display cached heatmap background
        heatmap_img = plt.imread(heatmap_bg_path)
        ax.imshow(heatmap_img, extent=[0, self.pitch_length, self.pitch_width, 0])
        
        # Draw players on top of heatmap
        for player in players:
            color = self.role_colors.get(player['role'], 'gray')
            marker_size = 120
            
            ax.scatter(player['x'], player['y'], 
                      c=color, s=marker_size, 
                      edgecolors='black', linewidths=0,
                      alpha=0.9, zorder=10)
        
        # Set plot properties
        ax.set_xlim(0, self.pitch_length)
        ax.set_ylim(0, self.pitch_width)
        ax.invert_yaxis()  # Invert y-axis to match image coordinates
        ax.set_xlabel('X Position (meters)', fontsize=12)
        ax.set_ylabel('Y Position (meters)', fontsize=12)
        ax.set_title(f'Referee Position Distribution (Symmetrical KDE) with Players - Frame {frame_id}', fontsize=14)
        
        # Save the figure
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved symmetrical general position visualization for frame {frame_id} to {output_path}")
    
    def visualize_multiple_frames(self, detections, frame_ids, output_dir, use_symmetrical=False):
        """
        Visualize multiple frames
        
        Args:
            detections: List of detection data
            frame_ids: List of frame numbers to visualize
            output_dir: Directory to save output images
            use_symmetrical: Whether to use symmetrical referee data
        """
        for frame_id in frame_ids:
            if use_symmetrical:
                output_path = os.path.join(output_dir, f"general_position_symmetrical_{frame_id:06d}.png")
                self.visualize_frame_symmetrical(detections, frame_id, output_path)
            else:
                output_path = os.path.join(output_dir, f"general_position_{frame_id:06d}.png")
                self.visualize_frame(detections, frame_id, output_path)


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
    visualizer = GeneralPositionVisualizer()
    
    # Example frames to visualize
    interesting_frames = [212, 400, 460]
    
    # Output directory
    output_dir = "data/example/images_for_paper"
    
    # Visualize frames (both original and symmetrical)
    visualizer.visualize_multiple_frames(detections, interesting_frames, output_dir, use_symmetrical=False)
    visualizer.visualize_multiple_frames(detections, interesting_frames, output_dir, use_symmetrical=True)


if __name__ == "__main__":
    main() 