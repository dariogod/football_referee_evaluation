from color_conversions import RGBColor255, LABColor, rgb_to_lab, lab_to_rgb_255
from custom_types import BBox, Person, PersonWithJerseyColor
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def assign_jersey_colors(full_image: np.ndarray, persons: list[Person], visualize: bool = False) -> list[PersonWithJerseyColor]:
    """Assign jersey colors to persons using K-means clustering on LAB color space
    
    Args:
        full_image: The full image containing all persons
        persons: List of Person objects with bounding boxes
        visualize: If True, shows a plot with ROIs and detected colors
    
    Returns:
        List of PersonWithJerseyColor objects
    """
    result = []
    rois = []  # Store ROIs for visualization
    
    for person in persons:
        # Crop ROI (upper half of person for jersey detection)
        roi = crop_person_roi(full_image, person.bbox)
        jersey_color = get_jersey_color(roi)
        
        result.append(PersonWithJerseyColor(
            id=person.id,
            bbox=person.bbox,
            pitch_coord=person.pitch_coord,
            gt_role=person.gt_role,
            jersey_color=jersey_color
        ))
        
        if visualize:
            rois.append(roi)
    
    if visualize:
        visualize_jersey_detection(persons, rois, result)
    
    return result

def crop_person_roi(full_image: np.ndarray, bbox: BBox, y_range: tuple[float, float] = (0.0, 0.5)) -> np.ndarray:
    """Crop the upper portion of person bounding box for jersey detection"""
    x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
    
    # Get person crop
    person_crop = full_image[y1:y2, x1:x2]
    
    if person_crop.size == 0:
        return np.array([])
    
    # Focus on upper portion (jersey area)
    height = person_crop.shape[0]
    y_start = int(height * y_range[0])
    y_end = int(height * y_range[1])
    
    if y_end <= y_start:
        return np.array([])
    
    roi = person_crop[y_start:y_end, :]
    
    # Downsample if too large for faster processing
    if roi.size > 0:
        h, w = roi.shape[:2]
        max_size = 64
        if h > max_size or w > max_size:
            ratio = min(max_size / h, max_size / w)
            new_size = (int(w * ratio), int(h * ratio))
            roi = cv2.resize(roi, new_size, interpolation=cv2.INTER_AREA)
    
    return roi

def get_jersey_color(roi: np.ndarray) -> RGBColor255:
    """Extract jersey color using K-means clustering in LAB color space"""
    if roi.size == 0 or roi.shape[0] < 2 or roi.shape[1] < 2:
        return RGBColor255(r=128, g=128, b=128)  # Default gray
    
    # Convert BGR to RGB
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # Calculate distances from bottom center (jersey anchor point)
    height, width = roi.shape[:2]
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    anchor_y, anchor_x = height - 1, width / 2
    squared_distances = (y_coords - anchor_y)**2 + (x_coords - anchor_x)**2
    
    # Convert to LAB color space for better clustering
    pixels = roi_rgb.reshape(-1, 3)
    lab_pixels = rgb_to_lab(pixels)
    
    # K-means clustering with 2 clusters (jersey vs background)
    kmeans = KMeans(n_clusters=2, n_init=3, random_state=42)
    labels = kmeans.fit_predict(lab_pixels)
    centers_lab = [LABColor.from_array(center) for center in kmeans.cluster_centers_]
    
    # Calculate average distance to anchor point for each cluster
    avg_distances = []
    for cluster_idx in range(2):
        cluster_mask = labels == cluster_idx
        if np.sum(cluster_mask) > 0:
            avg_dist = np.mean(squared_distances.reshape(-1)[cluster_mask])
            avg_distances.append((cluster_idx, avg_dist))
    
    # Jersey is the cluster closest to the anchor point (bottom center)
    avg_distances.sort(key=lambda x: x[1])
    jersey_cluster_idx = avg_distances[0][0]
    
    # Convert LAB back to RGB
    jersey_lab = centers_lab[jersey_cluster_idx]
    jersey_rgb = lab_to_rgb_255(jersey_lab)
    
    return jersey_rgb

def visualize_jersey_detection(persons: list[Person], rois: list[np.ndarray], results: list[PersonWithJerseyColor]) -> None:
    """Visualize jersey detection results showing ROIs and detected colors
    
    Args:
        persons: List of Person objects
        rois: List of ROI images for each person
        results: List of PersonWithJerseyColor objects with detected colors
    """
    fig, axes = plt.subplots(2, len(persons), figsize=(4*len(persons), 8))
    if len(persons) == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle('Jersey Color Detection Results', fontsize=16)
    
    for i, (person, roi, result) in enumerate(zip(persons, rois, results)):
        if roi.size > 0:
            # Show ROI in top row
            axes[0, i].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            axes[0, i].set_title(f'Person {person.id} ROI')
            axes[0, i].axis('off')
            
            # Show detected color in bottom row
            jersey_color = result.jersey_color
            color_patch = np.full((100, 100, 3), [jersey_color.r, jersey_color.g, jersey_color.b], dtype=np.uint8)
            axes[1, i].imshow(color_patch)
            axes[1, i].set_title(f'Detected Color\nRGB({jersey_color.r}, {jersey_color.g}, {jersey_color.b})')
            axes[1, i].axis('off')
        else:
            # Handle empty ROI case
            axes[0, i].text(0.5, 0.5, 'No ROI', ha='center', va='center', transform=axes[0, i].transAxes)
            axes[0, i].set_title(f'Person {person.id} ROI')
            axes[0, i].axis('off')
            
            axes[1, i].text(0.5, 0.5, 'No Color', ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].set_title('No Detection')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show() 