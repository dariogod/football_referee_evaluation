from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Set
from custom_types import PersonWithJerseyColor, PersonWithRole
from color_conversions import rgb_to_lab, RGBColor255
from skimage import color
import matplotlib.pyplot as plt
import os


class BaseRoleAssigner(ABC):
    """Base class for role assignment algorithms."""
    
    def __init__(self):
        self.method_name = self.__class__.__name__
        self.current_frame_number = None  # Track current frame for plotting
    
    @abstractmethod
    def perform_clustering(self, valid_persons: List[PersonWithJerseyColor], 
                         X_colors: np.ndarray) -> Tuple[int, Set[int], np.ndarray]:
        """
        Perform clustering and outlier detection.
        
        Args:
            valid_persons: List of persons with valid jersey colors and pitch coordinates
            X_colors: Array of RGB colors for valid persons (shape: n_persons x 3)
            
        Returns:
            Tuple of (best_n_outliers, all_outlier_indices, labels)
            where labels are the cluster assignments for non-outliers
        """
        pass
    
    def plot_clustering_results(self, X_colors_rgb: np.ndarray, X_colors_lab: np.ndarray, 
                               all_outlier_indices: Set[int], non_outlier_indices: List[int], 
                               labels: np.ndarray, left_cluster: int, right_cluster: int):
        """
        Plot clustering results for LAB colorspace using cluster center colors.
        
        Args:
            X_colors_rgb: RGB color array (for calculating cluster centers)
            X_colors_lab: LAB color array (for plotting coordinates)
            all_outlier_indices: Set of outlier indices
            non_outlier_indices: List of non-outlier indices
            labels: Cluster labels for non-outliers
            left_cluster: Cluster label for left team
            right_cluster: Cluster label for right team
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Normalize RGB colors for matplotlib (0-1 range)
        rgb_normalized = X_colors_rgb / 255.0
        
        # Plot outliers with crosses
        if all_outlier_indices:
            outlier_indices_list = list(all_outlier_indices)
            outlier_colors = rgb_normalized[outlier_indices_list]
            outlier_lab = X_colors_lab[outlier_indices_list]
            
            ax.scatter(outlier_lab[:, 2], outlier_lab[:, 1], outlier_lab[:, 0],  # b, a, L
                      c=outlier_colors, marker='x', s=100, label='Outliers', 
                      linewidths=2)
        
        # Calculate cluster centers and plot teams with cluster center colors
        if len(non_outlier_indices) > 0 and len(labels) > 0:
            # Get indices for first cluster (left team)
            first_cluster_mask = labels == left_cluster
            if np.any(first_cluster_mask):
                first_cluster_indices = np.array(non_outlier_indices)[first_cluster_mask]
                first_cluster_lab = X_colors_lab[first_cluster_indices]
                
                # Calculate cluster center in RGB space and normalize
                first_cluster_rgb = X_colors_rgb[first_cluster_indices]
                first_cluster_center_rgb = np.mean(first_cluster_rgb, axis=0) / 255.0
                
                ax.scatter(first_cluster_lab[:, 2], first_cluster_lab[:, 1], first_cluster_lab[:, 0],  # b, a, L
                          c=[first_cluster_center_rgb], marker='o', s=80, label='Team A', 
                          edgecolors='black', linewidths=0.5)
            
            # Get indices for second cluster (right team)
            second_cluster_mask = labels == right_cluster  
            if np.any(second_cluster_mask):
                second_cluster_indices = np.array(non_outlier_indices)[second_cluster_mask]
                second_cluster_lab = X_colors_lab[second_cluster_indices]
                
                # Calculate cluster center in RGB space and normalize
                second_cluster_rgb = X_colors_rgb[second_cluster_indices]
                second_cluster_center_rgb = np.mean(second_cluster_rgb, axis=0) / 255.0
                
                ax.scatter(second_cluster_lab[:, 2], second_cluster_lab[:, 1], second_cluster_lab[:, 0],  # b, a, L
                          c=[second_cluster_center_rgb], marker='o', s=80, label='Team B', 
                          edgecolors='black', linewidths=0.5)
        
        # Set labels and title
        ax.set_xlabel('b* (Blue-Yellow)')
        ax.set_ylabel('a* (Green-Red)')
        ax.set_zlabel('L* (Lightness)')
        
        # Add legend
        ax.legend()
        
        # Set viewing angle for best visualization
        ax.view_init(elev=20, azim=45)
        
        # Create output directory and save plot
        output_dir = f"clustering_plots_{self.method_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"frame_{self.current_frame_number:03d}_lab_clustering.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Saved clustering plot to: {output_path}")
    
    def assign_roles(self, full_image: np.ndarray, 
                    persons_with_color: List[PersonWithJerseyColor],
                    frame_number: int = None) -> List[PersonWithRole]:
        """
        Assign roles to persons using clustering on jersey colors.
        Position is only used to distinguish between left and right teams.
        Clustering is performed in RGB, LAB, and HSV color spaces.
        
        Args:
            full_image: The full image array (not used but kept for compatibility)
            persons_with_color: List of persons with jersey colors
            frame_number: Optional frame number for plotting purposes
        """
        # Set current frame number for plotting
        self.current_frame_number = frame_number
        
        if not persons_with_color:
            return []
        
        # Extract valid persons with jersey colors (RGB format)
        valid_persons = []
        X_colors_rgb = []
        positions = []  # Store positions for left/right distinction
        
        for person in persons_with_color:
            if person.jersey_color is not None and person.pitch_coord is not None:
                valid_persons.append(person)
                X_colors_rgb.append([person.jersey_color.r, person.jersey_color.g, person.jersey_color.b])
                # Store position for left/right distinction
                positions.append([person.pitch_coord.x_bottom_middle, 
                                person.pitch_coord.y_bottom_middle])
        
        if len(X_colors_rgb) < 2:
            # Handle case where we don't have enough valid persons for clustering
            return self._assign_default_roles(persons_with_color)
        
        X_colors_rgb = np.array(X_colors_rgb)
        positions = np.array(positions)
        
        # Prepare colors in different color spaces
        # LAB colors
        X_colors_lab = []
        for rgb_color in X_colors_rgb:
            lab_color = rgb_to_lab(RGBColor255(r=int(rgb_color[0]), g=int(rgb_color[1]), b=int(rgb_color[2])))
            X_colors_lab.append([lab_color.l, lab_color.a, lab_color.b])
        X_colors_lab = np.array(X_colors_lab)
        
        # HSV colors
        # Convert RGB (0-255) to normalized RGB (0-1) for skimage
        X_colors_rgb_normalized = X_colors_rgb / 255.0
        X_colors_hsv = color.rgb2hsv(X_colors_rgb_normalized.reshape(1, -1, 3)).reshape(-1, 3)
        # Scale HSV to more meaningful ranges: H (0-360), S (0-100), V (0-100)
        X_colors_hsv[:, 0] *= 360  # Hue
        X_colors_hsv[:, 1] *= 100  # Saturation
        X_colors_hsv[:, 2] *= 100  # Value
        
        # Perform clustering for each color space
        clustering_results = {}
        
        for color_space, X_colors in [("rgb", X_colors_rgb), ("lab", X_colors_lab), ("hsv", X_colors_hsv)]:
            try:
                print(f"Performing clustering for {color_space} space")
                best_n_outliers, all_outlier_indices, labels = self.perform_clustering(
                    valid_persons, X_colors)
                
                # Get non-outlier indices
                non_outlier_indices = [i for i in range(len(valid_persons)) 
                                     if i not in all_outlier_indices]
                
                # Determine left and right clusters
                if len(non_outlier_indices) >= 2 and len(labels) > 0:
                    left_cluster, right_cluster = self._determine_left_right_clusters(
                        valid_persons, non_outlier_indices, labels, positions)
                else:
                    left_cluster = right_cluster = 0
                
                clustering_results[color_space] = {
                    "all_outlier_indices": all_outlier_indices,
                    "non_outlier_indices": non_outlier_indices,
                    "labels": labels,
                    "left_cluster": left_cluster,
                    "right_cluster": right_cluster
                }
                
                # Plot clustering results for LAB colorspace on frame 10
                if color_space == "lab" and frame_number == 10:
                    self.plot_clustering_results(
                        X_colors_rgb, X_colors_lab, all_outlier_indices, 
                        non_outlier_indices, labels, left_cluster, right_cluster
                    )
                
            except Exception as e:
                print(f"Error with method {self.method_name} in {color_space} space: {e}")
                # Store default results for this color space
                clustering_results[color_space] = {
                    "all_outlier_indices": set(),
                    "non_outlier_indices": list(range(len(valid_persons))),
                    "labels": np.zeros(len(valid_persons)),
                    "left_cluster": 0,
                    "right_cluster": 0
                }
        
        # Create result list with roles from all color spaces
        return self._create_result_list_multi_color(persons_with_color, valid_persons, 
                                                   clustering_results)
    
    def _assign_default_roles(self, persons_with_color: List[PersonWithJerseyColor]) -> List[PersonWithRole]:
        """Assign default roles when clustering is not possible."""
        result = []
        for person in persons_with_color:
            result.append(PersonWithRole(**person.dict(), pred_role={
                "rgb": "player",
                "lab": "player", 
                "hsv": "player"
            }))
        return result
    
    def _determine_left_right_clusters(self, valid_persons: List[PersonWithJerseyColor],
                                     non_outlier_indices: List[int], 
                                     labels: np.ndarray,
                                     positions: np.ndarray) -> Tuple[int, int]:
        """Determine which cluster is left team and which is right team."""
        # Get all persons in each cluster and calculate average pitch coordinates
        cluster_0_persons = []
        cluster_1_persons = []
        
        for i, idx in enumerate(non_outlier_indices):
            if i < len(labels):
                if labels[i] == 0:
                    cluster_0_persons.append(valid_persons[idx])
                elif labels[i] == 1:
                    cluster_1_persons.append(valid_persons[idx])
        
        if cluster_0_persons and cluster_1_persons:
            # Calculate average x coordinate from pitch_coord for each cluster
            cluster_0_avg_x = np.mean([person.pitch_coord.x_bottom_middle 
                                     for person in cluster_0_persons])
            cluster_1_avg_x = np.mean([person.pitch_coord.x_bottom_middle 
                                     for person in cluster_1_persons])
            
            # Assign roles based on average x position (leftmost cluster gets player_left)
            if cluster_0_avg_x < cluster_1_avg_x:
                return 0, 1
            else:
                return 1, 0
        else:
            return 0, 1
    
    def _create_result_list_multi_color(self, persons_with_color: List[PersonWithJerseyColor],
                                      valid_persons: List[PersonWithJerseyColor],
                                      clustering_results: dict) -> List[PersonWithRole]:
        """Create the final result list with assigned roles for all color spaces."""
        result = []
        
        for person in persons_with_color:
            # Initialize pred_role dict
            pred_role = {"rgb": "player", "lab": "player", "hsv": "player"}
            
            if person.pitch_coord is None or person.jersey_color is None:
                # No pitch coordinates or jersey color, assign as unknown for all color spaces
                result.append(PersonWithRole(**person.dict(), pred_role=pred_role))
            else:
                # Find this person in valid_persons
                person_idx = None
                for i, valid_person in enumerate(valid_persons):
                    if valid_person.id == person.id:
                        person_idx = i
                        break
                
                if person_idx is None:
                    result.append(PersonWithRole(**person.dict(), pred_role=pred_role))
                else:
                    # Assign role for each color space
                    for color_space in ["rgb", "lab", "hsv"]:
                        cluster_data = clustering_results[color_space]
                        
                        if person_idx in cluster_data["all_outlier_indices"]:
                            pred_role[color_space] = "referee"
                        else:
                            # Non-outlier, assign based on cluster
                            try:
                                non_outlier_pos = cluster_data["non_outlier_indices"].index(person_idx)
                                if non_outlier_pos < len(cluster_data["labels"]):
                                    if cluster_data["labels"][non_outlier_pos] == cluster_data["left_cluster"]:
                                        pred_role[color_space] = "player_left"
                                    else:
                                        pred_role[color_space] = "player_right"
                                else:
                                    pred_role[color_space] = "player"
                            except ValueError:
                                # person_idx not in non_outlier_indices, treat as referee
                                pred_role[color_space] = "player"
                    
                    result.append(PersonWithRole(**person.dict(), pred_role=pred_role))
        
        return result 