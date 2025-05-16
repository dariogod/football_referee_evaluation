import os
import cv2
import numpy as np
import json
import sys
from src.utils.perspective_transform.perspective_transformer import PerspectiveTransformer
from src.utils.custom_types import FrameDetections, MinimapCoordinates
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoordinateTransformer:
    def __init__(self):
        self.perspective_transform = PerspectiveTransformer()
    
    def transform_matrix(self, M, point, src_size, dst_size):
        """
        Transform a point using homography matrix and scale to target size
        """
        h, w = src_size
        dst_h, dst_w = dst_size
        
        # Apply homography to the point
        point_array = np.array([point[0] * 1280 / w, point[1] * 720 / h, 1])
        warped_point = np.dot(M, point_array)
        warped_point = warped_point[:2] / warped_point[2]
        
        # Scale to target size (115x74 is the standard pitch dimensions)
        x_scaled = int(warped_point[0] * dst_w / 115)
        y_scaled = int(warped_point[1] * dst_h / 74)
        
        return (x_scaled, y_scaled)
    
    def transform_image(self, M, image, dst_size):
        """
        Transform an image using homography matrix and scale to target size
        """
        dst_h, dst_w = dst_size

        # Resize input image to match the dimensions used in homography calculation
        resized_image = cv2.resize(image, (1280, 720))
        
        # Create output image of desired size
        warped = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
        
        # For each pixel in the output image
        for y_out in range(dst_h):
            for x_out in range(dst_w):
                # Convert output coordinates to pitch coordinates (115x74)
                x_pitch = x_out * 115 / dst_w
                y_pitch = y_out * 74 / dst_h
                
                # Apply inverse homography to get input image coordinates
                point = np.array([x_pitch, y_pitch, 1])
                inv_warped = np.dot(np.linalg.inv(M), point)
                inv_warped = inv_warped[:2] / inv_warped[2]
                
                # Scale back to input image coordinates
                x_in = int(inv_warped[0])
                y_in = int(inv_warped[1])
                
                # Copy pixel if within bounds
                if 0 <= x_in < 1280 and 0 <= y_in < 720:
                    warped[y_out, x_out] = resized_image[y_in, x_in]
                    
        return warped
        
    def transform(
            self, 
            input_path: str, 
            detections: list[FrameDetections], 
            intermediate_results_folder: str | None = None 
        ) -> list[FrameDetections]:
        """
        Calculate minimap coordinates for all detections and update the detections object
        
        Args:
            input_path: Path to the input video file
            detections: List of detection objects for each frame
            store_results: Whether to store results (images, JSONs) to disk (default: True)
            
        Returns:
            Updated detections with minimap coordinates
        """
        # Setup output directories if storing results
        if intermediate_results_folder is not None:
            os.makedirs(intermediate_results_folder, exist_ok=True)
            warped_images_dir = os.path.join(intermediate_results_folder, 'warped_images')
            os.makedirs(warped_images_dir, exist_ok=True)
        
        # Load pitch.jpg as background
        pitch_img = cv2.imread('src/utils/pitch.png')
        gt_h, gt_w, _ = pitch_img.shape
        circle_radius = max(2, int(gt_w / 115))
        
        # Video capture
        cap = cv2.VideoCapture(input_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        homography_data = {}
        last_M = None
        processed_frames = 0
        
        # Create VideoWriter for minimap if storing results
        minimap_video_writer = None
        if intermediate_results_folder is not None:
            fps = cap.get(cv2.CAP_PROP_FPS)
            minimap_video_path = os.path.join(intermediate_results_folder, 'minimap.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            minimap_video_writer = cv2.VideoWriter(minimap_video_path, fourcc, fps, (gt_w, gt_h))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Calculate homography matrix every 2 frames to save processing time
            if processed_frames % 2 == 0:
                M, warped_image = self.perspective_transform.homography_matrix(frame)
                last_M = M
                homography_data[str(processed_frames)] = M.tolist()
                
                # Save warped image if storing results
                if intermediate_results_folder is not None:
                    image_filename = f"frame_{processed_frames:06d}.jpg"
                    warped_image_path = os.path.join(warped_images_dir, image_filename)
                    cv2.imwrite(warped_image_path, warped_image)
                    
                    # Save high-res warped image
                    warped_image_high_res = self.transform_image(M, frame, (540, 960))
                    warped_image_high_res_path = os.path.join(warped_images_dir, f"frame_{processed_frames:06d}_high_res.jpg")
                    cv2.imwrite(warped_image_high_res_path, warped_image_high_res)
            else:
                # Use the last calculated homography matrix
                M = last_M if last_M is not None else self.perspective_transform.homography_matrix(frame)[0]
            
            # Find detection for current frame
            for frame_detections in detections:
                if frame_detections.frame_id == processed_frames:
                    # Create minimap visualization if storing results
                    if intermediate_results_folder is not None:
                        bg_img = pitch_img.copy()

                    for detection in frame_detections.detections:
                        x1, y1, x2, y2 = detection.bbox.as_list()
                        # Use bottom center of bounding box for better positioning
                        center_x = x1 + (x2 - x1)/2
                        center_y = y1 + (y2 - y1)
                        
                        # Transform coordinates to minimap
                        minimap_coords = self.transform_matrix(M, (center_x, center_y), (h, w), (gt_h, gt_w))

                        # Check if coordinates are within bounds
                        if 0 <= minimap_coords[0] < gt_w and 0 <= minimap_coords[1] < gt_h:
                            # Add minimap coordinates to detection
                            detection.minimap_coordinates = MinimapCoordinates(x=minimap_coords[0], y=minimap_coords[1], x_max=gt_w, y_max=gt_h)
                            
                        # Draw on visualization if storing results
                        if intermediate_results_folder is not None and 0 <= minimap_coords[0] < gt_w and 0 <= minimap_coords[1] < gt_h:
                            # Draw player as circle, color based on team
                            color = (0, 0, 0)
                            cv2.circle(bg_img, minimap_coords, circle_radius, color, -1)
                    
                    # Write frame to minimap video if storing results
                    if intermediate_results_folder is not None:
                        minimap_video_writer.write(bg_img)
                    break
            
            processed_frames += 1
        
        cap.release()
        if minimap_video_writer is not None:
            minimap_video_writer.release()
        
        # Save files if storing results
        if intermediate_results_folder is not None:
            # Save homography matrices
            output_file = os.path.join(intermediate_results_folder, "homography.json")
            with open(output_file, 'w') as f:
                json.dump(homography_data, f)
            
            # Save updated detections
            detections_file = os.path.join(intermediate_results_folder, "detections.json")
            with open(detections_file, 'w') as f:
                raw_detections = [frame_detections.model_dump() for frame_detections in detections]
                json.dump(raw_detections, f, indent=4)
        
        return detections