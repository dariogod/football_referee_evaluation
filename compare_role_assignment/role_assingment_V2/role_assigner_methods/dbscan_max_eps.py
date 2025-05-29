import numpy as np
from typing import List, Tuple, Set
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_role_assigner import BaseRoleAssigner
from custom_types import PersonWithJerseyColor


class DBScanMaxEps(BaseRoleAssigner):
    """Role assigner using DBSCAN with dynamic eps adjustment and elbow method."""
    
    def __init__(self):
        super().__init__()
    
    def perform_clustering(self, valid_persons: List[PersonWithJerseyColor], 
                         X_colors: np.ndarray) -> Tuple[int, Set[int], np.ndarray]:
        """
        Perform DBSCAN clustering with dynamic eps adjustment.
        
        Args:
            valid_persons: List of persons with valid jersey colors and pitch coordinates
            X_colors: Array of colors for valid persons (can be RGB, LAB, or HSV)
            
        Returns:
            Tuple of (best_n_outliers, all_outlier_indices, labels)
            where labels are the cluster assignments for non-outliers
        """
        if len(X_colors) < 2:
            # Not enough data for clustering
            return 0, set(), np.array([0] * len(X_colors))
        
        # Standardize the feature space
        scaler = StandardScaler()
        X_colors_scaled = scaler.fit_transform(X_colors)
                
        # DBSCAN parameters
        min_samples = max([3, math.floor(0.25 * len(X_colors))])
        min_eps = 0.01
        max_eps = 1.0
        
        # Binary search to find the smallest eps that yields optimal_clusters
        best_labels = None
        best_eps = None
        
        while max_eps - min_eps > 0.01:  # Convergence threshold
            current_eps = (min_eps + max_eps) / 2
            dbscan = DBSCAN(eps=current_eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_colors_scaled)
            
            unique_clusters = np.unique(labels)
            non_outlier_clusters = [c for c in unique_clusters if c != -1]
            num_clusters = len(non_outlier_clusters)
            
            if num_clusters == 2:
                # Found target clusters, try larger eps to find maximum (save current result)
                best_labels = labels
                best_eps = current_eps
                min_eps = current_eps
            elif num_clusters < 2:
                # Too few clusters, need smaller eps
                max_eps = current_eps
            else:
                # Too many clusters, need larger eps
                min_eps = current_eps
        
        # If binary search didn't find a solution, use k-means
        if best_labels is None:
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
            best_labels = kmeans.fit_predict(X_colors_scaled)
        
        # Process clustering results - separate into teams and outliers
        # First, count members in each cluster
        cluster_sizes = Counter(best_labels)
        
        # Remove -1 (DBSCAN outliers) from consideration for largest clusters
        if -1 in cluster_sizes:
            del cluster_sizes[-1]
        
        # Find the two largest clusters
        largest_clusters = [cluster for cluster, _ in cluster_sizes.most_common(2)]
        
        # If fewer than 2 clusters found, handle edge case
        if len(largest_clusters) < 2:
            if len(largest_clusters) == 1:
                # Create a second cluster label
                largest_clusters.append(max(largest_clusters) + 1)
            else:
                # No clusters found, create two default clusters
                largest_clusters = [0, 1]
        
        # Determine outliers
        all_outlier_indices = set()
        team_labels = []
        
        for i, label in enumerate(best_labels):
            if label == -1 or label not in largest_clusters:
                all_outlier_indices.add(i)
            else:
                # Map to 0 or 1 for team assignment
                if label == largest_clusters[0]:
                    team_labels.append(0)
                else:
                    team_labels.append(1)
        
        # Convert team_labels to numpy array
        labels = np.array(team_labels)
        
        # Count outliers
        best_n_outliers = len(all_outlier_indices)
        
        return best_n_outliers, all_outlier_indices, labels 