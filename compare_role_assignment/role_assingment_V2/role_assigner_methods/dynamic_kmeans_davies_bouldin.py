import numpy as np
from typing import List, Tuple, Set
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_role_assigner import BaseRoleAssigner
from custom_types import PersonWithJerseyColor


class DynamicKMeansDaviesBouldin(BaseRoleAssigner):
    """Dynamic K-means role assigner using Davies-Bouldin score for goodness metric."""
    
    def __init__(self):
        super().__init__()
    
    def perform_clustering(self, valid_persons: List[PersonWithJerseyColor], 
                         X_colors: np.ndarray) -> Tuple[int, Set[int], np.ndarray]:
        """
        Perform K-means clustering with iterative outlier removal using Davies-Bouldin score.
        Note: X_colors is already in the appropriate color space (RGB, LAB, or HSV).
        """
        # Use X_colors directly - it's already in the right color space
        X = X_colors.copy()
        
        goodness_scores = []
        outlier_indices_per_step = []
        current_indices = list(range(len(X)))
        
        # Perform clustering for 0 to 6 outliers removed
        for n_outliers in range(7):  # 0, 1, 2, 3, 4, 5, 6
            if len(current_indices) < 2:
                goodness_scores.append(float('inf'))  # For Davies-Bouldin, lower is better
                outlier_indices_per_step.append([])
                continue
                
            current_X = X[current_indices]
            
            # Perform K-means with k=2
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(current_X)
            
            # Calculate goodness using Davies-Bouldin score
            goodness = self._calculate_davies_bouldin_score(current_X, labels)
            goodness_scores.append(goodness)
            
            if n_outliers < 6:  # Don't find outliers on the last iteration
                # Find outliers for next iteration
                outlier_local_indices = self._find_outliers(current_X, labels, kmeans, 1)
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
        # For Davies-Bouldin, we need to negate scores since lower is better
        negated_scores = [-score if np.isfinite(score) else float('-inf') for score in goodness_scores]
        best_n_outliers = self._find_elbow_point(negated_scores)
        
        # Collect all outliers up to best_n_outliers
        all_outlier_indices = set()
        for i in range(best_n_outliers):
            if i < len(outlier_indices_per_step):
                all_outlier_indices.update(outlier_indices_per_step[i])
        
        # Get final non-outlier indices
        non_outlier_indices = [i for i in range(len(X)) if i not in all_outlier_indices]
        
        # Perform final clustering on non-outliers
        labels = np.array([])
        if len(non_outlier_indices) >= 2:
            final_X = X[non_outlier_indices]
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(final_X)
        
        return best_n_outliers, all_outlier_indices, labels
    
    def _calculate_davies_bouldin_score(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Davies-Bouldin score for clustering quality (lower is better)."""
        unique_labels = np.unique(labels)
        n_samples = len(X)
        
        # Check if we have enough samples and clusters for the metric
        if len(unique_labels) < 2 or n_samples < 3:
            return float('inf')
        
        try:
            return davies_bouldin_score(X, labels)
        except Exception:
            return float('inf')
    
    def _find_outliers(self, X: np.ndarray, labels: np.ndarray, 
                      kmeans: KMeans, n_outliers: int) -> List[int]:
        """Find the most outlying points based on distance to cluster centers."""
        distances = []
        for i, point in enumerate(X):
            cluster_center = kmeans.cluster_centers_[labels[i]]
            distance = np.linalg.norm(point - cluster_center)
            distances.append((distance, i))
        
        # Sort by distance (descending) and return indices of top n_outliers
        distances.sort(reverse=True)
        return [idx for _, idx in distances[:n_outliers]]
    
    def _find_elbow_point(self, scores: List[float]) -> int:
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
        
        # Check if the line has zero length
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