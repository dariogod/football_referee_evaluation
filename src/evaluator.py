from src.utils.custom_types import FrameDetections, Detection, MinimapCoordinates
import logging
from typing import Any
from pydantic import BaseModel
import numpy as np
from sklearn.cluster import DBSCAN
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PotentialActionPoint(BaseModel):
    player_coordinates: list[MinimapCoordinates]
    cluster_center: MinimapCoordinates
    referee_distance: float

class Duel(BaseModel):
    player_1: MinimapCoordinates
    player_2: MinimapCoordinates
    referee_angle: float

class EvaluationResult(BaseModel):
    referee_position: MinimapCoordinates
    inside_rectangle: bool
    potential_action_points: list[PotentialActionPoint]
    duels: list[Duel]

class Evaluator:
    def __init__(self):
        pass

    def _is_referee_in_rectangle(self, detection: Detection) -> bool:
        x_max = detection.minimap_coordinates.x_max
        y_max = detection.minimap_coordinates.y_max
        referee_x = detection.minimap_coordinates.x
        referee_y = detection.minimap_coordinates.y

        scale_x = x_max / 105
        scale_y = y_max / 68
        scale = (scale_x + scale_y) / 2

        goal_width = 7.32
        middle_y = y_max / 2

        lower_box_boundary_y = middle_y - (goal_width / 2) * scale - 16.5 * scale
        upper_box_boundary_y = middle_y + (goal_width / 2) * scale + 16.5 * scale

        left_box_boundary_x = 0 + 16.5 * scale
        right_box_boundary_x = x_max - 16.5 * scale

        if (
            referee_x > left_box_boundary_x and referee_x < right_box_boundary_x and
            referee_y > lower_box_boundary_y and referee_y < upper_box_boundary_y
        ):
            return True
        
        return False
    
    def _get_distance(self, coordinates1: MinimapCoordinates, coordinates2: MinimapCoordinates) -> float:
        point1 = np.array([coordinates1.x, coordinates1.y])
        point2 = np.array([coordinates2.x, coordinates2.y])
        return np.linalg.norm(point1 - point2)
    
    def _find_potential_action_points(self, detections: list[Detection], referee_position: MinimapCoordinates) -> list[PotentialActionPoint]:
        player_detections = [
            d for d in detections 
                if d.role != "REF" 
                and d.role != "OOB" 
                and d.class_name == "person"
                and d.minimap_coordinates is not None
        ]
        
        if len(player_detections) < 2:
            return np.array([])
        
        try:
            x_max = player_detections[0].minimap_coordinates.x_max
            y_max = player_detections[0].minimap_coordinates.y_max
        except Exception as e:
            logger.error(f"Error getting x_max and y_max for player detections: {player_detections[0]}")
            raise e
            
        # Extract coordinates for clustering
        coordinates = np.array(
            [
                [d.minimap_coordinates.x, d.minimap_coordinates.y] for d in player_detections
            ]
        )
        
        scale_x = x_max / 105
        scale_y = y_max / 68
        scale = (scale_x + scale_y) / 2
        
        # Apply DBSCAN with eps=2 meters in pixel units
        eps_pixels = 2 * scale  # Convert 2 meters to pixels
        db = DBSCAN(eps=eps_pixels, min_samples=2).fit(coordinates)
        
        # Calculate cluster centers (centroid of points in each cluster)
        potential_action_points: list[PotentialActionPoint] = []
        labels = db.labels_
        
        # -1 label means noise (not part of any cluster)
        unique_labels = set(labels) - {-1}
        
        for label in unique_labels:
            cluster_points = coordinates[labels == label]
            player_coordinates = [
                MinimapCoordinates(x=point[0], y=point[1], x_max=x_max, y_max=y_max)
                for point in cluster_points
            ]
            center = np.mean(cluster_points, axis=0)
            cluster_center = MinimapCoordinates(
                x=int(center[0]),
                y=int(center[1]),
                x_max=x_max,
                y_max=y_max
            )

            potential_action_points.append(
                PotentialActionPoint(
                    player_coordinates=player_coordinates,
                    cluster_center=cluster_center,
                    referee_distance=self._get_distance(referee_position, cluster_center)
                )
            )
        return potential_action_points
    
    def _get_angle(self, player1_position: MinimapCoordinates, player2_position: MinimapCoordinates, referee_position: MinimapCoordinates) -> float:
        point1 = np.array([player1_position.x, player1_position.y])
        point2 = np.array([player2_position.x, player2_position.y])

        middle_point = np.array([(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2])
        referee_point = np.array([referee_position.x, referee_position.y])

        # Calculate vectors
        vector1 = middle_point - point1
        vector2 = referee_point - middle_point
        
        # Calculate the angle between vectors
        angle = np.degrees(np.arctan2(np.linalg.det([vector1, vector2]), np.dot(vector1, vector2)))

        while angle < 0:
            angle += 360
            
        # First ensure angle is between 0 and 180
        if angle > 180:
            angle = angle % 180
            
        # Then if it's larger than 90, take the complementary angle
        if angle > 90:
            angle = 180 - angle
            
        return angle

    
    def _find_duels(self, detections: list[Detection], referee_position: MinimapCoordinates) -> list[Duel]:
        player_detections = [
            d for d in detections 
                if d.role != "REF" 
                and d.role != "OOB" 
                and d.class_name == "person"
                and d.minimap_coordinates is not None
        ]

        if len(player_detections) < 2:
            return []

        x_max = player_detections[0].minimap_coordinates.x_max
        y_max = player_detections[0].minimap_coordinates.y_max

        scale_x = x_max / 105
        scale_y = y_max / 68
        scale = (scale_x + scale_y) / 2

        duels: list[Duel] = []
        already_included: list[tuple[int, int]] = []
        for detection in player_detections:
            for other_detection in player_detections:
                if detection == other_detection:
                    continue

                if tuple(sorted((detection.track_id, other_detection.track_id))) in already_included:
                    continue

                distance = self._get_distance(detection.minimap_coordinates, other_detection.minimap_coordinates)
                if distance < 2 * scale: # 2 meters
                    angle = self._get_angle(detection.minimap_coordinates, other_detection.minimap_coordinates, referee_position)
                    already_included.append(tuple(sorted((detection.track_id, other_detection.track_id))))
                    duels.append(
                        Duel(
                            player_1=detection.minimap_coordinates,
                            player_2=other_detection.minimap_coordinates,
                            referee_angle=angle
                        )
                    )
        return duels
                
    def evaluate_frame(self, frame_detections: FrameDetections) -> EvaluationResult | None:
        referee_detections: list[Detection] = []
        for detection in frame_detections.detections:
            if detection.role == "REF":
                referee_detections.append(detection)

        if len(referee_detections) == 0:
            logger.warning(f"No referee detections found for frame {frame_detections.frame_id}")
            return None
        
        if len(referee_detections) > 1:
            logger.warning(f"Multiple referee detections found for frame {frame_detections.frame_id}")
            return None
        
        referee_detection = referee_detections[0]

        if referee_detection.minimap_coordinates is None:
            logger.warning(f"No minimap coordinates found for referee detection for frame {frame_detections.frame_id}")
            return None

        # Check if referee is in the rectangle
        inside_rectangle = self._is_referee_in_rectangle(referee_detection)

        # Get distance to potential action points
        potential_action_points = self._find_potential_action_points(frame_detections.detections, referee_detection.minimap_coordinates)

        # Get duels
        duels = self._find_duels(frame_detections.detections, referee_detection.minimap_coordinates)

        return EvaluationResult(
            referee_position=referee_detection.minimap_coordinates,
            inside_rectangle=inside_rectangle,
            potential_action_points=potential_action_points,
            duels=duels
        )
    
    def evaluate_video(self, detections: list[FrameDetections], intermediate_results_folder: str | None = None) -> dict[int, EvaluationResult | None]:
        evaluation_results: dict[int, EvaluationResult | None] = {}
        for frame_detections in detections:
            frame_id = frame_detections.frame_id
            evaluation_result = self.evaluate_frame(frame_detections)
            evaluation_results[frame_id] = evaluation_result

        if intermediate_results_folder is not None:
            if not os.path.exists(intermediate_results_folder):
                os.makedirs(intermediate_results_folder)

            with open(os.path.join(intermediate_results_folder, "evaluation_results.json"), "w") as f:
                json.dump(
                    {
                        frame_id: evaluation_result.model_dump() if evaluation_result is not None else None
                        for frame_id, evaluation_result in evaluation_results.items()
                    },
                    f
                )

        return evaluation_results