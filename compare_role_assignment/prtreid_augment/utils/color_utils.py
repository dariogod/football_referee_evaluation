import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import sys
import os

# Add parent directories to path to import from role_assignment_V2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from role_assingment_V2.color_conversions import RGBColor255, rgb_to_lab, lab_to_rgb_255, LABColor

def extract_jersey_colors(predictions, image_path):
    """Extract jersey colors for all predictions in a frame
    
    Args:
        predictions: List of predictions with bbox information
        image_path: Path to the frame image
        
    Returns:
        jersey_colors: List of RGB tuples for each prediction
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
        
    jersey_colors = []
    
    for pred in predictions:
        # Get bbox
        bbox = pred['bbox']
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Crop person (top 50% for torso)
        person_crop = image[y1:y1+y2, x1:x1+x2]
        
        if person_crop.size == 0:
            raise ValueError(f"Person crop is empty for prediction: {pred}")
            
        # Get upper 50% for torso
        height = person_crop.shape[0]
        torso_crop = person_crop[:height//2, :]
        
        # Extract jersey color
        jersey_rgb = get_jersey_color_simple(torso_crop)
        jersey_colors.append((jersey_rgb.r, jersey_rgb.g, jersey_rgb.b))
    
    return jersey_colors


def get_jersey_color_simple(roi):
    """Extract jersey color using K-means clustering with anchor point approach"""
    if roi.size == 0 or roi.shape[0] < 2 or roi.shape[1] < 2:
        return RGBColor255(r=128, g=128, b=128)  # Default gray
    
    # Convert BGR to RGB
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # Get the distance of each pixel from the bottom center (anchor_point) of the ROI
    # Pixels near this point are more likely to be the jersey
    height, width = roi.shape[:2]
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    anchor_y, anchor_x = height - 1, width / 2
    
    squared_distances = (y_coords - anchor_y)**2 + (x_coords - anchor_x)**2
    
    # Reshape to pixels
    pixels = roi_rgb.reshape(-1, 3)
    
    # Convert to LAB color space for better clustering
    lab_pixels = rgb_to_lab(pixels)
    
    # K-means clustering with 2 clusters
    kmeans = KMeans(n_clusters=2, n_init=3, random_state=42)
    labels = kmeans.fit_predict(lab_pixels)
    
    # Convert centers back to RGB
    centers_lab = [LABColor.from_array(center) for center in kmeans.cluster_centers_]
    centers_rgb = []
    for center in centers_lab:
        rgb_color = lab_to_rgb_255(center)
        centers_rgb.append([rgb_color.r, rgb_color.g, rgb_color.b])
    centers_rgb = np.array(centers_rgb)
    
    # Calculate average squared distance to anchor for each cluster
    avg_distances = []
    for cluster_idx in range(2):
        cluster_mask = labels == cluster_idx
        if np.sum(cluster_mask) > 0:  # Avoid division by zero
            avg_dist = np.mean(squared_distances.reshape(-1)[cluster_mask])
            avg_distances.append((cluster_idx, avg_dist))
    
    # Sort clusters by average distance to anchor (ascending)
    avg_distances.sort(key=lambda x: x[1])
    
    # Closest cluster to anchor is jersey
    jersey_idx = avg_distances[0][0]
    jersey_color = centers_rgb[jersey_idx]
    
    return RGBColor255(r=int(jersey_color[0]), g=int(jersey_color[1]), b=int(jersey_color[2]))


def cluster_player_colors(player_colors, k=2):
    """Cluster player colors using K-means and create convex hulls
    
    Args:
        player_colors: List of RGB tuples for players
        k: Number of clusters (default 2)
        
    Returns:
        clusters: List of cluster dictionaries with centers and convex hulls
    """
    if len(player_colors) < k:
        return []
    
    # Convert to numpy array
    colors_array = np.array(player_colors)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(colors_array)
    
    clusters = []
    
    for cluster_idx in range(k):
        # Get points in this cluster
        cluster_points = colors_array[labels == cluster_idx]
        
        if len(cluster_points) < 3:
            # Not enough points for convex hull, use bounding box
            min_vals = np.min(cluster_points, axis=0)
            max_vals = np.max(cluster_points, axis=0)
            
            clusters.append({
                'center': kmeans.cluster_centers_[cluster_idx],
                'points': cluster_points,
                'hull': None,
                'bbox': {'min': min_vals, 'max': max_vals}
            })
        else:
            # Create convex hull
            try:
                hull = ConvexHull(cluster_points)
                clusters.append({
                    'center': kmeans.cluster_centers_[cluster_idx],
                    'points': cluster_points,
                    'hull': hull,
                    'bbox': None
                })
            except:
                # Fallback to bounding box if convex hull fails
                min_vals = np.min(cluster_points, axis=0)
                max_vals = np.max(cluster_points, axis=0)
                
                clusters.append({
                    'center': kmeans.cluster_centers_[cluster_idx],
                    'points': cluster_points,
                    'hull': None,
                    'bbox': {'min': min_vals, 'max': max_vals}
                })
    
    return clusters 