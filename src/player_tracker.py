import json
import numpy as np
import cv2
import os
import torch
from typing import Dict, List, Optional, Literal, Any
from ultralytics import YOLO
from transformers import DFineForObjectDetection, AutoImageProcessor
import math
from tqdm import tqdm
from src.utils.custom_types import BBox, Detection, TrackInfo, FrameDetections
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasePlayerTracker(ABC):
    """Abstract base class for player trackers."""
    
    # Class mapping for COCO dataset
    # 'person' is class 0 and 'sports ball' is class 32
    class_mapping: Dict[int, str] = {0: 'person', 32: 'sports ball'}
    
    @abstractmethod
    def __init__(self):
        """Initialize the tracker."""
        pass
    
    @abstractmethod
    def track_players(self, input_path: str, intermediate_results_folder: str | None = None) -> List[FrameDetections]:
        """
        Track players in a video.
        
        Args:
            input_path: Path to the input video file
            intermediate_results_folder: Optional folder to save intermediate results
            
        Returns:
            List of FrameDetections objects containing tracking results
        """
        pass
    
    def calculate_iou(self, box1: BBox, box2: BBox) -> float:
        """Calculate IoU between two bounding boxes"""
        # Convert to lists if needed
        box1_list = box1.as_list()
        box2_list = box2.as_list()
        
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1_list
        x1_2, y1_2, x2_2, y2_2 = box2_list
        
        # Calculate area of each box
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate intersection coordinates
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        # Calculate intersection area
        w = max(0, xi2 - xi1)
        h = max(0, yi2 - yi1)
        intersection = w * h
        
        # Calculate IoU
        union = area1 + area2 - intersection
        iou = intersection / union if union > 0 else 0
        
        return iou
    
    def setup_video_processing(self, input_path: str, intermediate_results_folder: str | None = None) -> tuple[cv2.VideoCapture, Any, Any, int, int, float, int, int, float, int, Any, List[FrameDetections]]:
        """Setup common video processing elements."""
        # Check if file exists
        if not os.path.isfile(input_path):
            logger.error(f"Video file not found: {input_path}")
            raise FileNotFoundError(f"Video file not found: {input_path}")
            
        # Open the input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Set up font
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = height / 1000
        font_thickness = math.floor(font_face)
        text_height = cv2.getTextSize("1234567890", font_face, font_scale, font_thickness)[0][1]

        # Set up output elements
        out = None
        output_video_path = None
        output_json_path = None
        
        # Set up output directory only if we're storing results
        if intermediate_results_folder is not None:
            os.makedirs(intermediate_results_folder, exist_ok=True)
            
            # Define output paths
            output_video_path = os.path.join(intermediate_results_folder, 'tracking_results.mp4')
            output_json_path = os.path.join(intermediate_results_folder, 'detections.json')
            
            # Set up video writer
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        # List to store all detections
        all_detections: List[FrameDetections] = []
        
        return (cap, out, output_json_path, width, height, fps, total_frames, 
                font_face, font_scale, font_thickness, text_height, all_detections)
    
    def process_frame_output(self, frame: np.ndarray, frame_detections: List[Detection], 
                            out: Any, font_face: int, font_scale: float, 
                            font_thickness: int, text_height: int) -> None:
        """Process and write output frame with detections."""
        if out is None:
            return
            
        # Draw detections on frame
        for detection in frame_detections:
            x1, y1, x2, y2 = detection.bbox.x1, detection.bbox.y1, detection.bbox.x2, detection.bbox.y2
            track_id = detection.track_id
            class_name = detection.class_name
            
            # Different colors for different classes
            color = (0, 0, 0) if class_name == 'person' else (255, 255, 255)
            
            # Draw bounding box
            cv2.rectangle(img=frame, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=font_thickness)
            
            # Draw track ID
            if track_id is not None:
                cv2.putText(
                    img=frame, 
                    text=f"ID: {track_id}", 
                    org=(x1, y1 - int(text_height * 0.5)),  # Adjust offset based on text height
                    fontFace=font_face, 
                    fontScale=font_scale, 
                    color=(0, 0, 0), 
                    thickness=font_thickness, 
                    lineType=cv2.LINE_AA)
        
        # Write frame to output video
        out.write(frame)


class YoloPlayerTracker(BasePlayerTracker):
    """Player tracker using YOLO model."""
    
    def __init__(self):
        """Initialize the YOLO tracker."""
        super().__init__()
        
        model_path = 'models/yolo11n.pt'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        if not os.path.exists(model_path):
            logger.info(f"Model file {model_path} not found. Downloading...")
            # YOLO automatically downloads and caches the model if it doesn't exist
            self.model = YOLO("yolov11n.pt")  # This will download from Ultralytics
            # Save the model to the specified path
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(self.model.model.state_dict(), model_path)
            logger.info(f"Model downloaded and saved to {model_path}")
        else:
            self.model = YOLO(model_path)
    
    def track_players(self, input_path: str, intermediate_results_folder: str | None = None) -> List[FrameDetections]:
        """
        Track players in a video using YOLO.
        
        Args:
            input_path: Path to the input video file
            intermediate_results_folder: Optional folder to save intermediate results
            
        Returns:
            List of FrameDetections objects containing tracking results
        """
        (cap, out, output_json_path, width, height, fps, total_frames, 
         font_face, font_scale, font_thickness, text_height, all_detections) = self.setup_video_processing(
             input_path, intermediate_results_folder)
        
        frame_count: int = 0
        
        # Create a progress bar
        progress_bar = tqdm(total=total_frames, desc="Tracking players with YOLO", unit="frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            
            frame_detections: List[Detection] = []
            
            # Use YOLO to detect and track persons and sports balls in the frame
            results = self.model.track(frame, persist=True, verbose=False, classes=[0, 32])
            
            # Process detection results
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.class_mapping.get(cls_id, 'unknown')

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else None
                    
                    # Store detection data
                    bbox = BBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2))
                    detection = Detection(
                        bbox=bbox,
                        confidence=float(conf),
                        track_id=int(track_id) if track_id is not None else None,
                        class_name=class_name,
                    )
                    frame_detections.append(detection)
            
            # Save detection data for this frame
            frame_data = FrameDetections(
                frame_id=int(frame_count),
                detections=frame_detections
            )
            all_detections.append(frame_data)
            
            if intermediate_results_folder is not None:
                self.process_frame_output(
                    frame, frame_detections, out, 
                    font_face, font_scale, font_thickness, text_height
                )
            
            frame_count += 1
            progress_bar.update(1)
        
        # Close progress bar
        progress_bar.close()
        
        # Release resources
        cap.release()
        if intermediate_results_folder is not None:
            out.release()
        
            detections_json = [frame_data.model_dump() for frame_data in all_detections]
            with open(output_json_path, 'w') as f:
                json.dump(detections_json, f, indent=4)

        cv2.destroyAllWindows()
        return all_detections


class DFinePlayerTracker(BasePlayerTracker):
    """Player tracker using D-FINE model."""
    
    def __init__(self):
        """Initialize the D-FINE tracker."""
        super().__init__()
        
        # Load the D-FINE model and image processor from saved paths
        model_save_path = 'models/dfine_model'
        processor_save_path = 'models/dfine_processor'
        
        # Create directories if they don't exist
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(processor_save_path, exist_ok=True)
        
        model_exists = os.path.exists(os.path.join(model_save_path, 'config.json'))
        processor_exists = os.path.exists(os.path.join(processor_save_path, 'config.json'))
        
        if not model_exists or not processor_exists:
            logger.info("D-FINE model or processor not found locally. Downloading from Hugging Face...")
            
            # Download model and processor from Hugging Face
            remote_model_name = "IDEA-Research/D-FINE"
            
            # Download and save the model
            if not model_exists:
                logger.info(f"Downloading D-FINE model from {remote_model_name}...")
                model = DFineForObjectDetection.from_pretrained(remote_model_name)
                model.save_pretrained(model_save_path)
                logger.info(f"Model downloaded and saved to {model_save_path}")
            else:
                model = DFineForObjectDetection.from_pretrained(model_save_path)
            
            # Download and save the processor
            if not processor_exists:
                logger.info(f"Downloading D-FINE image processor from {remote_model_name}...")
                processor = AutoImageProcessor.from_pretrained(remote_model_name, use_fast=False)
                processor.save_pretrained(processor_save_path)
                logger.info(f"Processor downloaded and saved to {processor_save_path}")
            else:
                processor = AutoImageProcessor.from_pretrained(processor_save_path, use_fast=False)
            
            self.model = model
            self.image_processor = processor
        else:
            # Load existing models
            self.model = DFineForObjectDetection.from_pretrained(model_save_path)
            self.image_processor = AutoImageProcessor.from_pretrained(processor_save_path, use_fast=False)
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Tracking parameters
        self.next_id: int = 0
        self.tracks: Dict[str, Dict[int, TrackInfo]] = {}  # Dictionary to store active tracks
        self.max_age: int = 30  # Maximum number of frames a track can be inactive before being removed
        self.iou_threshold: float = 0.3  # Minimum IoU to consider as a match
    
    def update_tracks(self, detections: List[Detection], class_name: str) -> None:
        """Associate detections with existing tracks based on IoU"""
        # If no tracks exist yet, create new tracks for all detections
        if len(self.tracks.get(class_name, {})) == 0:
            new_tracks: Dict[int, TrackInfo] = {}
            for detection in detections:
                track_id = self.next_id
                self.next_id += 1
                new_tracks[track_id] = TrackInfo(
                    bbox=detection.bbox,
                    age=0,
                    confidence=detection.confidence,
                    last_seen=0  # Frame counter when last seen
                )
                detection.track_id = track_id
            self.tracks[class_name] = new_tracks
            return
        
        # Calculate IoU between each detection and each track
        matched_track_ids: List[int] = []
        matched_detection_indices: List[int] = []
        
        for i, detection in enumerate(detections):
            max_iou: float = -1
            best_track_id: int = -1
            
            for track_id, track in self.tracks[class_name].items():
                if track_id in matched_track_ids:
                    continue
                
                iou = self.calculate_iou(detection.bbox, track.bbox)
                if iou > max_iou and iou >= self.iou_threshold:
                    max_iou = iou
                    best_track_id = track_id
            
            if best_track_id != -1:
                matched_track_ids.append(best_track_id)
                matched_detection_indices.append(i)
                detections[i].track_id = best_track_id
                # Update track information
                self.tracks[class_name][best_track_id].bbox = detection.bbox
                self.tracks[class_name][best_track_id].age = 0
                self.tracks[class_name][best_track_id].confidence = detection.confidence
                self.tracks[class_name][best_track_id].last_seen = 0
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detection_indices:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[class_name][track_id] = TrackInfo(
                    bbox=detection.bbox,
                    age=0,
                    confidence=detection.confidence,
                    last_seen=0
                )
                detection.track_id = track_id
        
        # Update age of all tracks and remove old ones
        tracks_to_remove: List[int] = []
        for track_id in self.tracks[class_name]:
            if track_id not in matched_track_ids:
                self.tracks[class_name][track_id].age += 1
                self.tracks[class_name][track_id].last_seen += 1
                if self.tracks[class_name][track_id].age > self.max_age:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[class_name][track_id]
    
    def track_players(self, input_path: str, intermediate_results_folder: str | None = None) -> List[FrameDetections]:
        """
        Track players in a video using D-FINE.
        
        Args:
            input_path: Path to the input video file
            intermediate_results_folder: Optional folder to save intermediate results
            
        Returns:
            List of FrameDetections objects containing tracking results
        """
        (cap, out, output_json_path, width, height, fps, total_frames, 
         font_face, font_scale, font_thickness, text_height, all_detections) = self.setup_video_processing(
             input_path, intermediate_results_folder)
        
        frame_count: int = 0
        
        # Initialize tracking for each class
        for class_name in self.class_mapping.values():
            if class_name not in self.tracks:
                self.tracks[class_name] = {}
        
        # Create a progress bar
        progress_bar = tqdm(total=total_frames, desc="Tracking players with D-FINE", unit="frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            
            frame_detections: List[Detection] = []
            
            # Convert BGR to RGB (D-FINE expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_height, img_width = rgb_frame.shape[:2]
            
            # Prepare inputs for D-FINE
            inputs = self.image_processor(images=rgb_frame, return_tensors="pt").to(self.device)
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Process results
            results = self.image_processor.post_process_object_detection(
                outputs, 
                threshold=0.5,  # Confidence threshold
                target_sizes=[(img_height, img_width)]
            )[0]  # Get first image in batch
            
            # Group detections by class for tracking
            class_detections: Dict[str, List[Detection]] = {class_name: [] for class_name in self.class_mapping.values()}
            
            # Extract detection results
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                cls_id = label.item()
                
                # Only process persons and sports balls
                if cls_id in self.class_mapping:
                    class_name = self.class_mapping[cls_id]
                    
                    # Convert box coordinates to integers
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    conf = float(score.cpu().numpy())
                    
                    # Create detection object (track_id will be assigned by the tracker)
                    bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
                    detection = Detection(
                        bbox=bbox,
                        confidence=float(conf),
                        track_id=None,  # Will be assigned by tracker
                        class_name=class_name,
                    )
                    
                    # Add to class-specific detection list
                    class_detections[class_name].append(detection)
            
            # Update tracks for each class
            for class_name, detections in class_detections.items():
                self.update_tracks(detections, class_name)
                frame_detections.extend(detections)
            
            # Save detection data for this frame
            frame_data = FrameDetections(
                frame_id=int(frame_count),
                detections=frame_detections
            )
            all_detections.append(frame_data)
            
            if intermediate_results_folder is not None:
                self.process_frame_output(
                    frame, frame_detections, out, 
                    font_face, font_scale, font_thickness, text_height
                )
            
            frame_count += 1
            progress_bar.update(1)
        
        # Close progress bar
        progress_bar.close()
        
        # Release resources
        cap.release()
        if intermediate_results_folder is not None:
            out.release()
        
            detections_json = [frame_data.model_dump() for frame_data in all_detections]
            with open(output_json_path, 'w') as f:
                json.dump(detections_json, f, indent=4)

        cv2.destroyAllWindows()
        return all_detections