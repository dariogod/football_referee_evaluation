import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt


class AngleDuelVisualizer:
    def __init__(self, pitch_image_path="src/utils/pitch_2.png"):
        """
        Initialize the angle duel visualizer
        
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
        
    def load_pitch_image(self):
        """Load and scale the pitch background image"""
        pitch_img = cv2.imread(self.pitch_image_path)
        if pitch_img is None:
            raise ValueError(f"Could not load pitch image from {self.pitch_image_path}")
        
        # Get pitch dimensions and scale up
        height, width, _ = pitch_img.shape
        scaled_img = cv2.resize(pitch_img, (width * self.scale_factor, height * self.scale_factor))
        
        return scaled_img
    
    def cosine_score_function(self, angle):
        """
        Piecewise function using cosine transition from 0 to 1 between 0-45 degrees,
        then constant at 1 from 45-90 degrees.
        """
        score = np.zeros_like(angle)
        
        # For angles 0 to 45: cosine transition
        mask1 = angle <= 45
        t = angle[mask1] / 45  # Normalize to [0,1]
        score[mask1] = (1 - np.cos(np.pi * t)) / 2
        
        # For angles > 45: constant at 1
        mask2 = angle > 45
        score[mask2] = 1
        
        return score
    
    def extract_player_positions(self, frame_detections):
        """Extract player positions and team information from frame detections"""
        player_positions = []
        player_teams = []
        referee_position = None
        
        for detection in frame_detections["detections"]:
            coords = detection["minimap_coordinates"]
            role = detection.get("role", "UNKNOWN")
            
            # Store referee position separately
            if role == "REF" and coords is not None:
                x = coords["x"]
                y = coords["y"]
                x_max = coords["x_max"]
                y_max = coords["y_max"]
                norm_x = x / x_max * 105  # 105m is pitch width
                norm_y = y / y_max * 68   # 68m is pitch height
                referee_position = np.array([norm_x, norm_y])
                continue
            
            # Skip if coordinates are None
            if coords is None:
                continue
            
            # Determine team based on role
            team = 1 if role == "TEAM A" else 2 if role == "TEAM B" else 0
            
            x = coords["x"]
            y = coords["y"]
            x_max = coords["x_max"]
            y_max = coords["y_max"]
            
            # Normalize to pitch dimensions (in meters)
            norm_x = x / x_max * 105  # 105m is pitch width
            norm_y = y / y_max * 68   # 68m is pitch height
            
            player_positions.append([norm_x, norm_y])
            player_teams.append(team)
        
        return np.array(player_positions), np.array(player_teams), referee_position
    
    def find_duels(self, player_positions, player_teams, distance_threshold=3):
        """Find duels (players from opposing teams within threshold distance)"""
        duels = []
        
        if len(player_positions) < 2:
            return duels
        
        for i in range(len(player_positions)):
            for j in range(i+1, len(player_positions)):
                # Check if players are from different teams
                if player_teams[i] != player_teams[j]:
                    # Calculate distance between players
                    dist = np.sqrt(np.sum((player_positions[i] - player_positions[j])**2))
                    
                    # If distance is less than threshold, it's a duel
                    if dist < distance_threshold:
                        # Get midpoint and orientation of the duel
                        duel_center = (player_positions[i] + player_positions[j]) / 2
                        duel_vector = player_positions[j] - player_positions[i]
                        duel_angle = np.arctan2(duel_vector[1], duel_vector[0])
                        
                        duels.append({
                            'center': duel_center,
                            'angle': duel_angle,
                            'players': [player_positions[i], player_positions[j]],
                            'player_indices': [i, j]
                        })
        
        return duels
    
    def create_angular_heatmap(self, duels, image_size=(105, 68)):
        """Create a heatmap showing angular scoring around each duel"""
        
        # Create coordinate grid with high resolution
        x = np.linspace(0, image_size[0], 2100)  # 20 points per meter
        y = np.linspace(0, image_size[1], 1360)  # 20 points per meter
        X, Y = np.meshgrid(x, y)
        
        # Initialize heatmap
        heatmap = np.ones_like(X)
        
        # For each duel position
        for duel in duels:
            duel_center = duel['center']
            duel_line_angle = duel['angle']
            
            # Calculate distances from duel center
            dx = X - duel_center[0]
            dy = Y - duel_center[1]
            
            # Calculate angles from duel center to each grid point
            point_angles = np.arctan2(dy, dx)
            
            # Calculate relative angles to the duel line
            relative_angles = np.abs(point_angles - duel_line_angle)
            
            # Ensure angles are between 0 and 360 degrees
            relative_angles = np.mod(relative_angles, 2 * np.pi)
            
            # Ensure between 0 and 180 degrees
            relative_angles = np.minimum(relative_angles, 2 * np.pi - relative_angles)
            
            # Take the acute angle
            relative_angles = np.minimum(relative_angles, np.pi - relative_angles)
            
            relative_angles = np.degrees(relative_angles)
            
            # Apply cosine scoring function
            scores = self.cosine_score_function(relative_angles)
            
            # Multiply to heatmap
            heatmap *= scores
        
        # Normalize heatmap to 0-1 range
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        return heatmap, x, y
    
    def evaluate_referee_position(self, heatmap, x_coords, y_coords, referee_position):
        """Evaluate referee position based on the heatmap"""
        if referee_position is None:
            return None
        
        # Convert referee position to heatmap coordinates
        ref_x = referee_position[0] / 105 * (len(x_coords) - 1)
        ref_y = referee_position[1] / 68 * (len(y_coords) - 1)
        
        # Get heatmap value at referee position
        ref_x_idx = int(ref_x)
        ref_y_idx = int(ref_y)
        
        if 0 <= ref_x_idx < len(x_coords) and 0 <= ref_y_idx < len(y_coords):
            score = heatmap[ref_y_idx, ref_x_idx]
            print(f"Referee position {referee_position} has score {score}")
            return {
                'position': referee_position,
                'score': score,
                'evaluation': 'Good' if score > 0.7 else 'Fair' if score > 0.4 else 'Poor'
            }
        return None
    
    def calculate_duel_angles(self, duels, referee_position):
        """Calculate angles between duel lines and referee sight lines"""
        angles = []
        
        if referee_position is None:
            return angles
        
        for i, duel in enumerate(duels):
            duel_center = duel['center']
            duel_vector = duel['players'][1] - duel['players'][0]
            ref_to_duel_vector = duel_center - referee_position
            
            # Calculate unit vectors
            duel_unit = duel_vector / np.linalg.norm(duel_vector)
            ref_unit = ref_to_duel_vector / np.linalg.norm(ref_to_duel_vector)
            
            # Calculate dot product and angle
            dot_product = np.dot(duel_unit, ref_unit)
            # Clip to handle floating point errors
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle_rad = np.arccos(dot_product)
            angle_deg = np.degrees(angle_rad)
            
            # Always get the acute angle (less than 90 degrees)
            if angle_deg > 90:
                angle_deg = 180 - angle_deg
            
            angles.append(angle_deg)
            print(f"Angle between duel {i+1} and referee line: {angle_deg:.2f} degrees")
        
        return angles
    
    def visualize_duels(self, detections, frame_id, output_path):
        """Visualize duels for a single frame"""
        # Load pitch image
        pitch_img = self.load_pitch_image()
        pitch_height, pitch_width, _ = pitch_img.shape
        
        # Find detections for the specific frame
        frame_detections = next((item for item in detections if item["frame_id"] == frame_id), None)
        
        if frame_detections is None:
            print(f"No detections found for frame {frame_id}")
            return
        
        # Extract player positions
        player_positions, player_teams, referee_position = self.extract_player_positions(frame_detections)
        
        if len(player_positions) == 0:
            print(f"No player positions found for frame {frame_id}")
            return
        
        # Find duels
        duels = self.find_duels(player_positions, player_teams)
        
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
        if referee_position is not None:
            ref_x = int(referee_position[0] / 105 * pitch_width)
            ref_y = int(referee_position[1] / 68 * pitch_height)
            cv2.circle(result_img, (ref_x, ref_y), 60, self.role_colors["REF"], -1)
        
        # Draw duels
        for duel in duels:
            players = duel['players']
            
            # Convert coordinates to image pixels
            x1_px = int(players[0][0] / 105 * pitch_width)
            y1_px = int(players[0][1] / 68 * pitch_height)
            x2_px = int(players[1][0] / 105 * pitch_width)
            y2_px = int(players[1][1] / 68 * pitch_height)
            
            # Draw a line connecting the two players
            cv2.line(result_img, (x1_px, y1_px), (x2_px, y2_px), (0, 255, 0), 8)
            
            # Draw dots for the players in the duel
            cv2.circle(result_img, (x1_px, y1_px), 20, (0, 255, 0), -1)
            cv2.circle(result_img, (x2_px, y2_px), 20, (0, 255, 0), -1)
        
        # Draw lines from referee to the two closest duels if referee is present
        if referee_position is not None and duels:
            ref_x_px = int(referee_position[0] / 105 * pitch_width)
            ref_y_px = int(referee_position[1] / 68 * pitch_height)
            
            # Calculate distances from referee to all duels
            duel_distances = []
            for idx, duel in enumerate(duels):
                duel_midpoint = duel['center']
                dist = np.sqrt(np.sum((referee_position - duel_midpoint)**2))
                duel_distances.append((idx, dist))
            
            # Sort duels by distance to referee
            duel_distances.sort(key=lambda x: x[1])
            
            # Draw lines to the two closest duels if available
            for k in range(min(2, len(duel_distances))):
                closest_duel_idx = duel_distances[k][0]
                duel = duels[closest_duel_idx]
                duel_midpoint = duel['center']
                
                # Convert midpoint to image pixels
                mid_x_px = int(duel_midpoint[0] / 105 * pitch_width)
                mid_y_px = int(duel_midpoint[1] / 68 * pitch_height)
                
                # Draw a line from referee to duel midpoint
                cv2.line(result_img, (ref_x_px, ref_y_px), (mid_x_px, mid_y_px), 
                        (0, 255, 0), 8, cv2.LINE_AA)
            
            # Calculate and print angles
            self.calculate_duel_angles(duels, referee_position)
        
        # Save the image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result_img)
        print(f"Saved duel visualization for frame {frame_id} with {len(duels)} duels to {output_path}")
    
    def visualize_angular_heatmap(self, detections, frame_id, output_path):
        """Visualize angular heatmap for a single frame"""
        # Find detections for the specific frame
        frame_detections = next((item for item in detections if item["frame_id"] == frame_id), None)
        
        if frame_detections is None:
            print(f"No detections found for frame {frame_id}")
            return
        
        # Extract player positions
        player_positions, player_teams, referee_position = self.extract_player_positions(frame_detections)
        
        if len(player_positions) == 0:
            print(f"No player positions found for frame {frame_id}")
            return
        
        # Find duels
        duels = self.find_duels(player_positions, player_teams)
        
        if len(duels) == 0:
            print(f"No duels found for frame {frame_id}")
            return
        
        # Create angular heatmap
        heatmap, x_coords, y_coords = self.create_angular_heatmap(duels)
        
        # Evaluate referee position
        ref_evaluation = self.evaluate_referee_position(heatmap, x_coords, y_coords, referee_position)
        
        # Load the existing duels visualization as background
        output_dir = os.path.dirname(output_path)
        duels_image_path = os.path.join(output_dir, f"minimap_duels_{frame_id:06d}.png")
        
        if not os.path.exists(duels_image_path):
            print(f"Duels image not found: {duels_image_path}")
            print("Make sure to generate duels visualization first")
            return
        
        # Load the duels image
        background_img = cv2.imread(duels_image_path)
        if background_img is None:
            print(f"Could not load duels image from {duels_image_path}")
            return
        
        # Create matplotlib figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Convert background image from BGR to RGB
        background_rgb = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
        
        # Display background image with duels visualization
        ax.imshow(background_rgb, extent=[0, 105, 0, 68], alpha=0.7)
        
        # Overlay heatmap
        extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]
        im = ax.imshow(heatmap, cmap='YlOrRd', alpha=0.7, extent=extent)
        
        
        # Draw referee on top of heatmap
        if referee_position is not None:
            ax.plot(referee_position[0], 68-referee_position[1], 'o', color='yellow', markersize=10, 
                   markeredgecolor='black', markeredgewidth=0)
            
        # Draw players on top of heatmap
        for i, (pos, team) in enumerate(zip(player_positions, player_teams)):
            role = "TEAM A" if team == 1 else "TEAM B" if team == 2 else "UNKNOWN"
            color = 'white' if role == "TEAM A" else 'red' if role == "TEAM B" else 'black'
            ax.plot(pos[0], 68-pos[1], 'o', color=color, markersize=10, 
                   markeredgecolor='black', markeredgewidth=0)
        
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 68)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Save the figure
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved angular heatmap for frame {frame_id} with {len(duels)} duels to {output_path}")
    
    def visualize_multiple_frames(self, detections, frame_ids, output_dir, include_heatmap=True):
        """Visualize multiple frames with both duel and heatmap visualizations"""
        for frame_id in frame_ids:
            # Create duel visualization
            duel_output_path = os.path.join(output_dir, f"minimap_duels_{frame_id:06d}.png")
            self.visualize_duels(detections, frame_id, duel_output_path)
            
            # Create heatmap visualization if requested
            if include_heatmap:
                heatmap_output_path = os.path.join(output_dir, f"minimap_duel_heatmap_{frame_id:06d}.png")
                self.visualize_angular_heatmap(detections, frame_id, heatmap_output_path)


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
    visualizer = AngleDuelVisualizer()
    
    # Example frames to visualize
    interesting_frames = [212, 400, 460]
    
    # Output directory
    output_dir = "data/example/images_for_paper"
    
    # Visualize frames
    visualizer.visualize_multiple_frames(detections, interesting_frames, output_dir)


if __name__ == "__main__":
    main() 