import cv2
import numpy as np
import json
import os


class MinimapVisualizer:
    def __init__(self, pitch_image_path="src/utils/pitch_2.png"):
        """
        Initialize the minimap visualizer
        
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
    
    def visualize_frame(self, detections, frame_id, output_path):
        """
        Visualize players for a single frame
        
        Args:
            detections: List of detection data
            frame_id: Frame number to visualize
            output_path: Path to save the output image
        """
        # Load pitch image
        pitch_img = self.load_pitch_image()
        pitch_height, pitch_width, _ = pitch_img.shape
        
        # Find detections for the specific frame
        frame_detections = next((item for item in detections if item["frame_id"] == frame_id), None)
        
        if frame_detections is None:
            print(f"No detections found for frame {frame_id}")
            return
        
        # Create a copy of the pitch image to draw on
        result_img = pitch_img.copy()
        
        # Draw players
        for detection in frame_detections["detections"]:
            coords = detection["minimap_coordinates"]
            if coords is None:
                continue
                
            x = coords["x"]
            y = coords["y"]
            x_max = coords["x_max"]
            y_max = coords["y_max"]
            
            # Normalize to pitch dimensions (with scaling factor applied)
            norm_x = int(x / x_max * pitch_width)
            norm_y = int(y / y_max * pitch_height)
            
            # Get player role and determine color
            role = detection.get("role", "UNKNOWN")
            color = self.role_colors.get(role, (0, 0, 0))  # default to black
            
            # Draw player as a circle (radius scaled with image)
            radius = 60  # Adjusted for 4x scale
            cv2.circle(result_img, (norm_x, norm_y), radius, color, -1)
        
        # Save the image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result_img)
        print(f"Saved minimap visualization for frame {frame_id} to {output_path}")
    
    def visualize_multiple_frames(self, detections, frame_ids, output_dir):
        """
        Visualize multiple frames
        
        Args:
            detections: List of detection data
            frame_ids: List of frame numbers to visualize
            output_dir: Directory to save output images
        """
        for frame_id in frame_ids:
            output_path = os.path.join(output_dir, f"minimap_{frame_id:06d}.png")
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
    visualizer = MinimapVisualizer()
    
    # Example frames to visualize
    interesting_frames = [212, 400, 460]
    
    # Output directory
    output_dir = "data/example/images_for_paper"
    
    # Visualize frames
    visualizer.visualize_multiple_frames(detections, interesting_frames, output_dir)


if __name__ == "__main__":
    main() 