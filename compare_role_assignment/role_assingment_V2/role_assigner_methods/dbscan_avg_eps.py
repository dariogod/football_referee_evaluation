import numpy as np
from typing import List, Tuple, Set
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score
from collections import Counter
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_role_assigner import BaseRoleAssigner
from custom_types import PersonWithJerseyColor


class DBScanAvgEps(BaseRoleAssigner):
    """Role assigner using DBSCAN with dynamic eps adjustment and iterative optimization."""
    
    def __init__(self):
        super().__init__()
    
    def perform_clustering(self, valid_persons: List[PersonWithJerseyColor], 
                         X_colors: np.ndarray) -> Tuple[int, Set[int], np.ndarray]:
        """
        Perform DBSCAN clustering with iterative optimization using Calinski-Harabasz index.
        
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
        search_min_eps = 0.01
        search_max_eps = 1.0
        
        # Find the minimum eps that yields 2 clusters
        min_eps_for_2_clusters = self._find_eps_for_n_clusters(X_colors_scaled, min_samples, 2, search_min_eps, search_max_eps, find_min=True)
        
        # Find the maximum eps that yields 2 clusters  
        max_eps_for_2_clusters = self._find_eps_for_n_clusters(X_colors_scaled, min_samples, 2, search_min_eps, search_max_eps, find_min=False)
        
        # If we found both min and max eps, use iterative optimization
        if min_eps_for_2_clusters is not None and max_eps_for_2_clusters is not None:
            best_labels, best_outlier_indices = self._iterative_optimization(
                X_colors_scaled, min_samples, min_eps_for_2_clusters, max_eps_for_2_clusters
            )
        elif min_eps_for_2_clusters is not None:
            # Use minimum eps if only that was found
            dbscan = DBSCAN(eps=min_eps_for_2_clusters, min_samples=min_samples)
            best_labels = dbscan.fit_predict(X_colors_scaled)
            best_outlier_indices = set(np.where(best_labels == -1)[0])
        elif max_eps_for_2_clusters is not None:
            # Use maximum eps if only that was found
            dbscan = DBSCAN(eps=max_eps_for_2_clusters, min_samples=min_samples)
            best_labels = dbscan.fit_predict(X_colors_scaled)
            best_outlier_indices = set(np.where(best_labels == -1)[0])
        else:
            # If no eps found for 2 clusters, use k-means
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
            best_labels = kmeans.fit_predict(X_colors_scaled)
            best_outlier_indices = set()
        
        # Process clustering results to get final team assignments
        team_labels, all_outlier_indices = self._process_clustering_results(best_labels, best_outlier_indices)
        
        return len(all_outlier_indices), all_outlier_indices, team_labels
    
    def _find_eps_for_n_clusters(self, X_colors_scaled, min_samples, n_clusters, search_min_eps, search_max_eps, find_min=True):
        """Find eps value that yields exactly n_clusters using binary search."""
        eps_for_n_clusters = None
        temp_min = search_min_eps
        temp_max = search_max_eps
        
        while temp_max - temp_min > 0.01:  # Convergence threshold
            current_eps = (temp_min + temp_max) / 2
            dbscan = DBSCAN(eps=current_eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_colors_scaled)
            
            unique_clusters = np.unique(labels)
            non_outlier_clusters = [c for c in unique_clusters if c != -1]
            num_clusters = len(non_outlier_clusters)
            
            if num_clusters == n_clusters:
                eps_for_n_clusters = current_eps
                if find_min:
                    temp_max = current_eps  # Try smaller eps to find minimum
                else:
                    temp_min = current_eps  # Try larger eps to find maximum
            elif num_clusters < n_clusters:
                # Too few clusters, need smaller eps
                temp_max = current_eps
            else:
                # Too many clusters, need larger eps
                temp_min = current_eps
        
        return eps_for_n_clusters
    
    def _iterative_optimization(self, X_colors_scaled, min_samples, min_eps, max_eps):
        """Iteratively optimize clustering using Calinski-Harabasz index."""
        
        # Get clustering results for min and max eps
        dbscan_min = DBSCAN(eps=min_eps, min_samples=min_samples)
        labels_min = dbscan_min.fit_predict(X_colors_scaled)
        
        dbscan_max = DBSCAN(eps=max_eps, min_samples=min_samples)
        labels_max = dbscan_max.fit_predict(X_colors_scaled)
        
        # Find samples that are outliers in min_eps but in clusters in max_eps
        min_outliers = set(np.where(labels_min == -1)[0])
        max_outliers = set(np.where(labels_max == -1)[0])
        border_samples = min_outliers - max_outliers  # Samples that are outliers in min but not in max
        
        if len(border_samples) == 0:
            # No border samples to add, return min_eps clustering
            return labels_min, min_outliers
        
        # Calculate cluster centers from max_eps clustering
        cluster_centers = self._calculate_cluster_centers(X_colors_scaled, labels_max)
        
        # Sort border samples by distance to nearest cluster center
        border_samples_with_distance = []
        for sample_idx in border_samples:
            sample_point = X_colors_scaled[sample_idx]
            min_distance = float('inf')
            
            for center in cluster_centers.values():
                distance = np.linalg.norm(sample_point - center)
                min_distance = min(min_distance, distance)
            
            border_samples_with_distance.append((sample_idx, min_distance))
        
        # Sort by distance (closest first)
        border_samples_with_distance.sort(key=lambda x: x[1])
        border_samples_sorted = [sample_idx for sample_idx, _ in border_samples_with_distance]
        
        # Start with min_eps clustering and iteratively add border samples
        best_labels = labels_min.copy()
        best_score = -1
        best_outliers = min_outliers.copy()
        
        current_labels = labels_min.copy()
        current_outliers = min_outliers.copy()
        
        # Calculate initial score if we have at least 2 non-outlier clusters
        non_outlier_mask = current_labels != -1
        if np.sum(non_outlier_mask) > 0 and len(np.unique(current_labels[non_outlier_mask])) >= 2:
            try:
                current_score = calinski_harabasz_score(X_colors_scaled[non_outlier_mask], current_labels[non_outlier_mask])
                if current_score > best_score:
                    best_score = current_score
                    best_labels = current_labels.copy()
                    best_outliers = current_outliers.copy()
            except:
                pass
        
        # Iteratively add border samples
        for sample_idx in border_samples_sorted:
            # Assign sample to the cluster it belongs to in max_eps clustering
            current_labels[sample_idx] = labels_max[sample_idx]
            current_outliers.discard(sample_idx)
            
            # Calculate Calinski-Harabasz score
            non_outlier_mask = current_labels != -1
            if np.sum(non_outlier_mask) > 0:
                unique_clusters = np.unique(current_labels[non_outlier_mask])
                if len(unique_clusters) >= 2:
                    try:
                        score = calinski_harabasz_score(X_colors_scaled[non_outlier_mask], current_labels[non_outlier_mask])
                        if score > best_score:
                            best_score = score
                            best_labels = current_labels.copy()
                            best_outliers = current_outliers.copy()
                    except:
                        pass
        
        return best_labels, best_outliers
    
    def _calculate_cluster_centers(self, X_colors_scaled, labels):
        """Calculate cluster centers for non-outlier clusters."""
        cluster_centers = {}
        unique_clusters = np.unique(labels)
        
        for cluster_id in unique_clusters:
            if cluster_id != -1:  # Skip outliers
                cluster_mask = labels == cluster_id
                cluster_points = X_colors_scaled[cluster_mask]
                center = np.mean(cluster_points, axis=0)
                cluster_centers[cluster_id] = center
        
        return cluster_centers
    
    def _process_clustering_results(self, best_labels, best_outlier_indices):
        """Process clustering results to get final team assignments."""
        # Count members in each cluster
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
        
        # Determine final outliers and team assignments
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
        
        return labels, all_outlier_indices 