from pydantic import BaseModel
from enum import Enum
from color_conversions import RGBColor255, rgb_to_lab
from custom_types import BBox, Person, PersonWithJerseyColor, PersonWithRole
from color_assigner_v2 import assign_jersey_colors
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import hdbscan
from typing import List, Tuple
import matplotlib.pyplot as plt

class GoodnessMethod(Enum):
    SILHOUETTE = "silhouette"
    CALINSKI_HARABASZ = "calinski_harabasz"
    DAVIES_BOULDIN = "davies_bouldin"
    INERTIA = "inertia"
    HDBSCAN = "hdbscan"

def calculate_goodness(X: np.ndarray, labels: np.ndarray, kmeans_model: KMeans, method: GoodnessMethod) -> float:
    """Calculate clustering goodness using different metrics."""
    unique_labels = np.unique(labels)
    n_samples = len(X)
    
    # Check if we have enough samples and clusters for the metrics
    if len(unique_labels) < 2 or n_samples < 3:
        return float('-inf') if method in [GoodnessMethod.SILHOUETTE, GoodnessMethod.CALINSKI_HARABASZ, GoodnessMethod.HDBSCAN] else float('inf')
    
    try:
        if method == GoodnessMethod.SILHOUETTE:
            return silhouette_score(X, labels)
        elif method == GoodnessMethod.CALINSKI_HARABASZ:
            return calinski_harabasz_score(X, labels)
        elif method == GoodnessMethod.DAVIES_BOULDIN:
            return -davies_bouldin_score(X, labels)  # Negative because lower is better
        elif method == GoodnessMethod.INERTIA:
            return -kmeans_model.inertia_  # Negative because lower is better
        elif method == GoodnessMethod.HDBSCAN:
            # For HDBSCAN, we use silhouette score as the goodness metric
            return silhouette_score(X, labels)
        else:
            raise ValueError(f"Unknown goodness method: {method}")
    except Exception:
        return float('-inf') if method in [GoodnessMethod.SILHOUETTE, GoodnessMethod.CALINSKI_HARABASZ, GoodnessMethod.HDBSCAN] else float('inf')

def find_outliers(X: np.ndarray, labels: np.ndarray, kmeans_model: KMeans, n_outliers: int) -> List[int]:
    """Find the most outlying points based on distance to cluster centers."""
    distances = []
    for i, point in enumerate(X):
        cluster_center = kmeans_model.cluster_centers_[labels[i]]
        distance = np.linalg.norm(point - cluster_center)
        distances.append((distance, i))
    
    # Sort by distance (descending) and return indices of top n_outliers
    distances.sort(reverse=True)
    return [idx for _, idx in distances[:n_outliers]]

def perform_hdbscan_clustering(persons_with_color: List[PersonWithJerseyColor]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Perform HDBSCAN clustering on scaled jersey colors in LAB space.
    Returns: (scaled_X, labels, outlier_indices)
    """
    # Extract jersey colors and convert to LAB
    X = []
    valid_persons = []
    for person in persons_with_color:
        if person.jersey_color is not None:
            # Convert RGB jersey color to LAB
            lab_color = rgb_to_lab(person.jersey_color)
            X.append([lab_color.l, lab_color.a, lab_color.b])
            valid_persons.append(person)
    
    if len(X) < 2:
        return np.array([]), np.array([]), []
    
    X = np.array(X)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
    labels = clusterer.fit_predict(X_scaled)
    
    # Find outliers (points labeled as -1 by HDBSCAN)
    outlier_indices = [i for i, label in enumerate(labels) if label == -1]
    
    return X_scaled, labels, outlier_indices

def perform_clustering_with_outlier_removal(persons_with_color: List[PersonWithJerseyColor], 
                                          goodness_method: GoodnessMethod) -> Tuple[int, List[float], List[List[int]]]:
    """
    Perform K-means clustering with iterative outlier removal on jersey colors in LAB space.
    Returns: (best_n_outliers, goodness_scores, outlier_indices_per_step)
    """
    # Extract jersey colors and convert to LAB
    X = []
    valid_persons = []
    for person in persons_with_color:
        if person.jersey_color is not None:
            # Convert RGB jersey color to LAB
            lab_color = rgb_to_lab(person.jersey_color)
            X.append([lab_color.l, lab_color.a, lab_color.b])
            valid_persons.append(person)
    
    if len(X) < 2:
        return 0, [0.0], [[]]
    
    X = np.array(X)
    goodness_scores = []
    outlier_indices_per_step = []
    
    current_indices = list(range(len(X)))
    
    # Perform clustering for 0 to 6 outliers removed
    for n_outliers in range(7):  # 0, 1, 2, 3, 4, 5, 6
        if len(current_indices) < 2:
            goodness_scores.append(float('-inf'))
            outlier_indices_per_step.append([])
            continue
            
        current_X = X[current_indices]
        
        # Perform K-means with k=2
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(current_X)
        
        # Calculate goodness
        goodness = calculate_goodness(current_X, labels, kmeans, goodness_method)
        goodness_scores.append(goodness)
        
        if n_outliers < 6:  # Don't find outliers on the last iteration
            # Find outliers for next iteration
            outlier_local_indices = find_outliers(current_X, labels, kmeans, 1)
            if outlier_local_indices:
                # Convert local index to global index
                global_outlier_idx = current_indices[outlier_local_indices[0]]
                outlier_indices_per_step.append([global_outlier_idx])
                # Remove outlier from current_indices
                current_indices.remove(global_outlier_idx)
            else:
                outlier_indices_per_step.append([])
        else:
            outlier_indices_per_step.append([])
    
    # Apply elbow method to find best number of outliers
    best_n_outliers = find_elbow_point(goodness_scores)
    
    return best_n_outliers, goodness_scores, outlier_indices_per_step

def find_elbow_point(scores: List[float]) -> int:
    """Find the elbow point in the goodness scores using the elbow method."""
    if len(scores) < 3:
        return 0
    
    # Convert to numpy array for easier computation
    scores = np.array(scores)
    n_points = len(scores)
    
    # Handle case where all scores are the same, invalid, or contain infinite values
    if np.all(np.isnan(scores)) or np.all(np.isinf(scores)):
        return 0
    
    # Filter out infinite and NaN values for elbow calculation
    finite_mask = np.isfinite(scores)
    if np.sum(finite_mask) < 3:
        return 0
    
    # Get indices and values of finite scores
    finite_indices = np.where(finite_mask)[0]
    finite_scores = scores[finite_mask]
    
    # Check if all finite scores are the same
    if np.allclose(finite_scores, finite_scores[0]):
        return 0
    
    # Use first and last finite points for the line
    first_finite_idx = finite_indices[0]
    last_finite_idx = finite_indices[-1]
    first_point = np.array([first_finite_idx, finite_scores[0]])
    last_point = np.array([last_finite_idx, finite_scores[-1]])
    
    # Check if the line has zero length (first and last points are the same)
    line_vector = last_point - first_point
    line_length = np.linalg.norm(line_vector)
    
    if line_length == 0:
        return 0
    
    # Calculate distances from each finite point to the line
    distances = np.zeros(n_points)
    for i, finite_idx in enumerate(finite_indices):
        point = np.array([finite_idx, finite_scores[i]])
        # Distance from point to line
        cross_product = np.cross(line_vector, first_point - point)
        distance = np.abs(cross_product) / line_length
        distances[finite_idx] = distance
    
    # Return the index with maximum distance (elbow point) among finite values
    return finite_indices[np.argmax(distances[finite_indices])]

def assign_roles(full_image: np.ndarray, persons_with_color: List[PersonWithJerseyColor], 
                 goodness_method: GoodnessMethod = GoodnessMethod.SILHOUETTE) -> List[PersonWithRole]:
    """
    Assign roles to persons using K-means clustering on jersey colors with outlier removal.
    Position is only used to distinguish between left and right teams.
    
    Args:
        full_image: The full image array
        persons_with_color: List of persons with jersey colors
        goodness_method: The clustering goodness method to use
    """
    if not persons_with_color:
        return []
    
    # Extract valid persons with jersey colors and convert to LAB
    valid_persons = []
    X_colors = []
    positions = []  # Store positions for left/right distinction
    for person in persons_with_color:
        if person.jersey_color is not None:
            # Only include persons with pitch coordinates for clustering
            if person.pitch_coord is not None:
                valid_persons.append(person)
                # Convert RGB jersey color to LAB
                lab_color = rgb_to_lab(person.jersey_color)
                X_colors.append([lab_color.l, lab_color.a, lab_color.b])
                # Store position for left/right distinction
                positions.append([person.pitch_coord.x_bottom_middle, person.pitch_coord.y_bottom_middle])
    
    if len(X_colors) < 2:
        # Handle case where we don't have enough valid persons for clustering
        result = []
        for person in persons_with_color:
            if person.pitch_coord is None:
                result.append(PersonWithRole(**person.dict(), pred_role="unknown"))
            else:
                result.append(PersonWithRole(**person.dict(), pred_role="referee"))
        return result
    
    X_colors = np.array(X_colors)
    positions = np.array(positions)
    
    # Use the specified goodness method
    hdbscan_labels = None  # Initialize for scope
    try:
        if goodness_method == GoodnessMethod.HDBSCAN:
            # Use HDBSCAN clustering on valid_persons (not all persons_with_color)
            X_scaled, hdbscan_labels, hdbscan_outliers = perform_hdbscan_clustering(valid_persons)
            best_n_outliers = len(hdbscan_outliers)
            outlier_indices_per_step = [hdbscan_outliers] if hdbscan_outliers else [[]]
        else:
            # Use K-means with outlier removal on valid_persons
            best_n_outliers, _, outlier_indices_per_step = perform_clustering_with_outlier_removal(valid_persons, goodness_method)
    except Exception as e:
        print(f"Error with method {goodness_method}: {e}")
        raise Exception(f"Clustering failed with method {goodness_method}: {e}")
    
    # Collect all outliers up to best_n_outliers
    all_outlier_indices = set()
    for i in range(best_n_outliers):
        if i < len(outlier_indices_per_step):
            all_outlier_indices.update(outlier_indices_per_step[i])
    
    # Get non-outlier indices
    non_outlier_indices = [i for i in range(len(valid_persons)) if i not in all_outlier_indices]
    
    # Perform final clustering on non-outliers using jersey colors
    if len(non_outlier_indices) >= 2:
        final_X_colors = X_colors[non_outlier_indices]
        
        if goodness_method == GoodnessMethod.HDBSCAN:
            # For HDBSCAN, we already have the clustering results
            # Filter the HDBSCAN labels to only include non-outliers
            labels = []
            for i in non_outlier_indices:
                # Find the corresponding label from HDBSCAN results
                if i < len(hdbscan_labels):
                    labels.append(hdbscan_labels[i])
                else:
                    labels.append(-1)  # Fallback for any missing labels
            labels = np.array(labels)
            
            # If HDBSCAN found more than 2 clusters, merge smaller clusters
            unique_labels = np.unique(labels[labels != -1])
            if len(unique_labels) > 2:
                # Keep only the two largest clusters
                cluster_sizes = [(label, np.sum(labels == label)) for label in unique_labels]
                cluster_sizes.sort(key=lambda x: x[1], reverse=True)
                keep_labels = [cluster_sizes[0][0], cluster_sizes[1][0]]
                
                # Create a new list for final non-outlier indices and labels
                final_non_outlier_indices = []
                final_labels = []
                
                # Reassign other clusters as outliers
                for i, label in enumerate(labels):
                    if label in keep_labels:
                        final_non_outlier_indices.append(non_outlier_indices[i])
                        # Relabel the kept clusters as 0 and 1
                        if label == keep_labels[0]:
                            final_labels.append(0)
                        else:
                            final_labels.append(1)
                    else:
                        # Add to outliers
                        original_idx = non_outlier_indices[i]
                        all_outlier_indices.add(original_idx)
                
                # Update non_outlier_indices and labels
                non_outlier_indices = final_non_outlier_indices
                labels = np.array(final_labels)
                
            elif len(unique_labels) == 1:
                # Only one cluster found, treat all as same team
                labels[labels != -1] = 0
                # Create a dummy second cluster if we have enough points
                if len(non_outlier_indices) > 1:
                    labels = np.append(labels, [1])
                    non_outlier_indices.append(non_outlier_indices[0])  # Duplicate first person for second cluster
        else:
            # Use K-means clustering
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(final_X_colors)
        
        # Use average pitch coordinates to distinguish between left and right teams
        # Calculate average x position for each cluster using pitch coordinates
        final_positions = positions[non_outlier_indices]
        
        # Get all persons in each cluster and calculate average pitch coordinates
        cluster_0_persons = [valid_persons[non_outlier_indices[i]] for i in range(len(final_positions)) if i < len(labels) and labels[i] == 0]
        cluster_1_persons = [valid_persons[non_outlier_indices[i]] for i in range(len(final_positions)) if i < len(labels) and labels[i] == 1]
        
        if cluster_0_persons and cluster_1_persons:
            # Calculate average x coordinate from pitch_coord for each cluster
            cluster_0_avg_x = np.mean([person.pitch_coord.x_bottom_middle for person in cluster_0_persons])
            cluster_1_avg_x = np.mean([person.pitch_coord.x_bottom_middle for person in cluster_1_persons])
            
            # Assign roles based on average x position (leftmost cluster gets player_left)
            if cluster_0_avg_x < cluster_1_avg_x:
                left_cluster, right_cluster = 0, 1
            else:
                left_cluster, right_cluster = 1, 0
        else:
            left_cluster, right_cluster = 0, 1
    else:
        labels = []
        left_cluster = right_cluster = 0
    
    # Create result list
    result = []
    
    for person in persons_with_color:
        if person.pitch_coord is None:
            # No pitch coordinates, assign as unknown
            result.append(PersonWithRole(**person.dict(), pred_role="unknown"))
        elif person.jersey_color is None:
            # No jersey color, assign as referee
            result.append(PersonWithRole(**person.dict(), pred_role="unknown"))
        else:
            # Find this person in valid_persons
            person_idx = None
            for i, valid_person in enumerate(valid_persons):
                if valid_person.id == person.id:
                    person_idx = i
                    break

            if person_idx is None:
                result.append(PersonWithRole(**person.dict(), pred_role="unknown"))
            elif person_idx in all_outlier_indices:
                result.append(PersonWithRole(**person.dict(), pred_role="referee"))
            else:
                # Non-outlier, assign based on cluster
                try:
                    non_outlier_pos = non_outlier_indices.index(person_idx)
                    if non_outlier_pos < len(labels):
                        if labels[non_outlier_pos] == left_cluster:
                            result.append(PersonWithRole(**person.dict(), pred_role="player_left"))
                        else:
                            result.append(PersonWithRole(**person.dict(), pred_role="player_right"))
                    else:
                        result.append(PersonWithRole(**person.dict(), pred_role="referee"))
                except ValueError:
                    # person_idx not in non_outlier_indices, treat as referee
                    result.append(PersonWithRole(**person.dict(), pred_role="referee"))
    
    return result