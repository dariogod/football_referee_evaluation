import numpy as np
from typing import List, Tuple, Set
import hdbscan
from sklearn.cluster import KMeans
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_role_assigner import BaseRoleAssigner
from custom_types import PersonWithJerseyColor


class HDBSCANRoleAssigner(BaseRoleAssigner):
    """HDBSCAN-based role assigner for team/referee classification."""
    
    def __init__(self):
        super().__init__()
    
    def perform_clustering(self, valid_persons: List[PersonWithJerseyColor], 
                         X_colors: np.ndarray) -> Tuple[int, Set[int], np.ndarray]:
        """
        Use HDBSCAN to find outliers (referees), then K-means on non-outliers for teams.
        Note: X_colors is already in the appropriate color space (RGB, LAB, or HSV).
        """
        # Use X_colors directly - it's already in the right color space
        X = X_colors.copy()
        
        if len(X) < 2:
            return 0, set(), np.array([])
        
        # Perform HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
        labels = clusterer.fit_predict(X)
        
        # Find outliers (points labeled as -1 by HDBSCAN)
        outlier_indices = set(i for i, label in enumerate(labels) if label == -1)
        best_n_outliers = len(outlier_indices)
        
        # Get unique cluster labels (excluding outliers)
        unique_labels = np.unique(labels[labels != -1])
        
        # If HDBSCAN found more than 2 clusters, keep only the two largest
        if len(unique_labels) > 2:
            # Count cluster sizes
            cluster_sizes = [(label, np.sum(labels == label)) for label in unique_labels]
            cluster_sizes.sort(key=lambda x: x[1], reverse=True)
            keep_labels = [cluster_sizes[0][0], cluster_sizes[1][0]]
            
            # Create new labels array
            new_labels = []
            new_outlier_indices = set()
            
            for i, label in enumerate(labels):
                if label == -1:
                    # Already an outlier
                    new_outlier_indices.add(i)
                elif label in keep_labels:
                    # Keep this cluster, relabel to 0 or 1
                    if label == keep_labels[0]:
                        new_labels.append(0)
                    else:
                        new_labels.append(1)
                else:
                    # This cluster becomes an outlier
                    new_outlier_indices.add(i)
            
            # Update outlier indices
            outlier_indices = new_outlier_indices
            best_n_outliers = len(outlier_indices)
            
            # Create final labels for non-outliers only
            non_outlier_indices = [i for i in range(len(labels)) if i not in outlier_indices]
            labels = np.array(new_labels)
            
        elif len(unique_labels) == 2:
            # Perfect case: exactly 2 clusters
            # Relabel clusters as 0 and 1
            non_outlier_indices = [i for i in range(len(labels)) if labels[i] != -1]
            label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
            labels = np.array([label_map[labels[i]] for i in non_outlier_indices])
            
        elif len(unique_labels) == 1:
            # Only one cluster found
            non_outlier_indices = [i for i in range(len(labels)) if labels[i] != -1]
            # All non-outliers get label 0
            labels = np.zeros(len(non_outlier_indices), dtype=int)
            
        else:
            # No clusters found (all outliers)
            labels = np.array([])
        
        return best_n_outliers, outlier_indices, labels 