import numpy as np
from scipy.spatial import Delaunay

def check_point_in_clusters(point, clusters):
    """Check if a point (RGB color) falls within any of the clusters
    
    Args:
        point: RGB tuple to check
        clusters: List of cluster dictionaries with hulls or bboxes
        
    Returns:
        bool: True if point is inside any cluster
    """
    point_array = np.array(point)
    
    for cluster in clusters:
        if cluster['hull'] is not None:
            # Check using convex hull
            if point_in_hull(point_array, cluster['hull'].points[cluster['hull'].vertices]):
                return True
        elif cluster['bbox'] is not None:
            # Check using bounding box
            if point_in_bbox(point_array, cluster['bbox']):
                return True
    
    return False


def point_in_hull(point, hull_points):
    """Check if a point is inside a convex hull using Delaunay triangulation
    
    Args:
        point: numpy array of coordinates
        hull_points: numpy array of hull vertices
        
    Returns:
        bool: True if point is inside hull
    """
    if len(hull_points) < 3:
        return False
    
    try:
        # Create Delaunay triangulation of the hull points
        tri = Delaunay(hull_points)
        # Check if point is inside
        return tri.find_simplex(point) >= 0
    except:
        # If triangulation fails, use simple distance check
        # Check if point is close to any hull point
        distances = np.linalg.norm(hull_points - point, axis=1)
        return np.min(distances) < 30  # Threshold in RGB space


def point_in_bbox(point, bbox):
    """Check if a point is inside a bounding box
    
    Args:
        point: numpy array of coordinates
        bbox: dict with 'min' and 'max' arrays
        
    Returns:
        bool: True if point is inside bbox
    """
    return np.all(point >= bbox['min']) and np.all(point <= bbox['max'])


def expand_clusters(clusters, expansion_factor=1.1):
    """Expand clusters slightly to be more inclusive
    
    Args:
        clusters: List of cluster dictionaries
        expansion_factor: Factor to expand by (1.1 = 10% expansion)
        
    Returns:
        Expanded clusters
    """
    expanded_clusters = []
    
    for cluster in clusters:
        new_cluster = cluster.copy()
        
        if cluster['bbox'] is not None:
            # Expand bounding box
            center = (cluster['bbox']['min'] + cluster['bbox']['max']) / 2
            size = cluster['bbox']['max'] - cluster['bbox']['min']
            new_size = size * expansion_factor
            
            new_cluster['bbox'] = {
                'min': center - new_size / 2,
                'max': center + new_size / 2
            }
        
        expanded_clusters.append(new_cluster)
    
    return expanded_clusters 