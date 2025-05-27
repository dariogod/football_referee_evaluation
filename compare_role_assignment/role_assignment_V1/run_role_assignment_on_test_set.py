import json
import os
import sys
from typing import List, Dict, Any

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.custom_types import Detection, FrameDetections, BBox, MinimapCoordinates
from src.role_assigner import RoleAssigner
from src.color_assigner import ColorAssigner
from tqdm import tqdm
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


def load_oracle_detections(labels_path: str) -> List[FrameDetections]:
    """
    Load oracle detections from Labels-GameState.json file and convert them to FrameDetections format.
    
    Args:
        labels_path: Path to the Labels-GameState.json file
        
    Returns:
        List of FrameDetections objects
    """
    # Load the pitch image to get actual minimap dimensions
    pitch_img = cv2.imread('src/utils/pitch.png')
    if pitch_img is None:
        # Use default dimensions if pitch.png not found
        minimap_height, minimap_width = 680, 1050
    else:
        minimap_height, minimap_width, _ = pitch_img.shape
    
    with open(labels_path, 'r') as f:
        data = json.load(f)
    
    # Group annotations by image_id (frame)
    frame_annotations: Dict[str, List[Dict[str, Any]]] = {}
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        if image_id not in frame_annotations:
            frame_annotations[image_id] = []
        frame_annotations[image_id].append(annotation)
    
    # Create FrameDetections for each frame
    all_detections: List[FrameDetections] = []
    
    for image in data['images']:
        if not image['is_labeled']:
            continue
            
        image_id = image['image_id']
        # Frame ID is typically the last 6 digits of image_id minus 1 (0-indexed)
        frame_id = int(image_id[-6:]) - 1
        
        frame_detections: List[Detection] = []
        
        if image_id in frame_annotations:
            for annotation in frame_annotations[image_id]:
                # Process player (1), goalkeeper (2), and referee (3) detections
                # Skip ball (4) and other categories
                if annotation['category_id'] not in [1, 2, 3]:
                    continue
                
                # Extract bounding box
                bbox_data = annotation['bbox_image']
                bbox = BBox(
                    x1=int(bbox_data['x']),
                    y1=int(bbox_data['y']),
                    x2=int(bbox_data['x'] + bbox_data['w']),
                    y2=int(bbox_data['y'] + bbox_data['h'])
                )
                
                # Extract minimap coordinates from bbox_pitch
                bbox_pitch = annotation.get('bbox_pitch')
                minimap_coords = None
                if bbox_pitch:
                    # Use bottom middle position from pitch coordinates
                    # Convert from meters to pixels
                    # Standard pitch is 105m x 68m
                    # We need to load the actual minimap dimensions from pitch.png
                    x_meters = bbox_pitch['x_bottom_middle']
                    y_meters = bbox_pitch['y_bottom_middle']
                    
                    # Convert pitch coordinates (in meters, centered at 0,0) to minimap pixels
                    # Pitch coordinates: x from -52.5 to 52.5, y from -34 to 34
                    # Need to map to minimap pixel coordinates
                    
                    # Convert from pitch meters (centered at 0,0) to pixels (top-left origin)
                    minimap_x = int((x_meters + 52.5) / 105 * minimap_width)
                    minimap_y = int((y_meters + 34) / 68 * minimap_height)
                    
                    minimap_coords = MinimapCoordinates(
                        x=minimap_x,
                        y=minimap_y,
                        x_max=minimap_width,
                        y_max=minimap_height
                    )
                
                # Get oracle role from attributes for comparison
                attributes = annotation.get('attributes', {})
                oracle_role = attributes.get('role', 'unknown')
                oracle_team = attributes.get('team', None)
                oracle_jersey = attributes.get('jersey', None)
                
                # Store oracle info in a comment for debugging
                # Note: We'll let RoleAssigner determine the actual role
                
                # Create Detection object
                detection = Detection(
                    bbox=bbox,
                    roi_bbox=None,
                    confidence=1.0,  # Oracle detection
                    track_id=annotation['track_id'],
                    class_name="person",
                    jersey_color=None,  # Will be assigned by ColorAssigner
                    role=None,  # Will be assigned by RoleAssigner
                    minimap_coordinates=minimap_coords
                )
                
                # Store oracle info for later comparison (not part of Detection model)
                detection._oracle_role = oracle_role
                detection._oracle_team = oracle_team
                detection._oracle_jersey = oracle_jersey
                
                frame_detections.append(detection)
        
        # Create FrameDetections object
        frame_data = FrameDetections(
            frame_id=frame_id,
            detections=frame_detections
        )
        all_detections.append(frame_data)
    
    return all_detections


def save_detections_with_oracle(detections: List[FrameDetections], output_dir: str):
    """Save detections with oracle information preserved."""
    detections_with_oracle = []
    for frame_detections in detections:
        frame_dict = frame_detections.model_dump()
        # Add oracle information to each detection
        for i, detection in enumerate(frame_detections.detections):
            if hasattr(detection, '_oracle_role'):
                frame_dict['detections'][i]['_oracle_role'] = detection._oracle_role
            if hasattr(detection, '_oracle_team'):
                frame_dict['detections'][i]['_oracle_team'] = detection._oracle_team
            if hasattr(detection, '_oracle_jersey'):
                frame_dict['detections'][i]['_oracle_jersey'] = detection._oracle_jersey
        detections_with_oracle.append(frame_dict)
    
    with open(os.path.join(output_dir, "detections_with_oracle.json"), "w") as f:
        json.dump(detections_with_oracle, f, indent=4)


def extract_player_images_first_frame(video_path: str, detections: List[FrameDetections], output_dir: str):
    """
    Extract and save images of all players from the first frame.
    
    Args:
        video_path: Path to the video file
        detections: List of FrameDetections with role assignments
        output_dir: Directory to save player images
    """
    # Create directory for player images
    player_images_dir = os.path.join(output_dir, 'player_images_first_frame')
    os.makedirs(player_images_dir, exist_ok=True)
    
    # Open video to get first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read first frame from video")
        return
    
    # Get detections from first frame
    if not detections or detections[0].frame_id != 0:
        print("Warning: No detections found for first frame")
        return
    
    first_frame_detections = detections[0]
    
    # Create a figure to display all players
    num_detections = len(first_frame_detections.detections)
    if num_detections == 0:
        print("No detections in first frame")
        return
    
    # Calculate grid dimensions for display
    cols = min(6, num_detections)
    rows = (num_detections + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    
    # Extract and display each player
    for idx, detection in enumerate(first_frame_detections.detections):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]
        
        # Extract player image
        x1, y1, x2, y2 = detection.bbox.x1, detection.bbox.y1, detection.bbox.x2, detection.bbox.y2
        player_img = frame[y1:y2, x1:x2]
        
        # Convert BGR to RGB for matplotlib
        player_img_rgb = cv2.cvtColor(player_img, cv2.COLOR_BGR2RGB)
        
        # Display player image
        ax.imshow(player_img_rgb)
        
        # Add title with track ID and role
        title = f"Track {detection.track_id}"
        if detection.role:
            title += f"\n{detection.role}"
        
        # Add oracle info if available
        if hasattr(detection, '_oracle_role'):
            oracle_info = f"\n(Oracle: {detection._oracle_role}"
            if hasattr(detection, '_oracle_team') and detection._oracle_team:
                oracle_info += f" - {detection._oracle_team}"
            oracle_info += ")"
            title += oracle_info
            
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        
        # Save individual player image
        player_filename = f"track_{detection.track_id:03d}_{detection.role or 'UNK'}.jpg"
        player_filepath = os.path.join(player_images_dir, player_filename)
        cv2.imwrite(player_filepath, player_img)
    
    # Remove empty subplots
    for idx in range(num_detections, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row][col].axis('off')
    
    # Save the complete figure
    plt.tight_layout()
    plt.savefig(os.path.join(player_images_dir, 'all_players_first_frame.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {num_detections} player images to {player_images_dir}")


def check_if_already_processed(video_dir: str, output_dir: str) -> bool:
    """Check if a video has already been processed."""
    video_name = os.path.basename(video_dir)
    video_output_dir = os.path.join(output_dir, video_name)
    
    # Check if the key output files exist
    required_files = [
        'summary.json',
        'detections_with_oracle.json',
        'role_comparison.json'
    ]
    
    return all(os.path.exists(os.path.join(video_output_dir, f)) for f in required_files)


def run_role_assignment_on_video(video_dir: str, output_dir: str, show_first_frame_players: bool = True, force_reprocess: bool = False) -> Dict[str, Any]:
    """
    Run role assignment on a single video using oracle detections.
    
    Args:
        video_dir: Path to the video directory containing Labels-GameState.json and video file
        output_dir: Directory to save results
        
    Returns:
        Dictionary with results
    """
    video_name = os.path.basename(video_dir)
    labels_path = os.path.join(video_dir, 'Labels-GameState.json')
    video_path = os.path.join(video_dir, f'{video_name}.mp4')
    
    # Check if already processed and skip if not forcing reprocess
    if not force_reprocess and check_if_already_processed(video_dir, output_dir):
        print(f"Video {video_name} already processed, loading existing results...")
        video_output_dir = os.path.join(output_dir, video_name)
        
        # Load existing summary
        summary_path = os.path.join(video_output_dir, 'summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                return json.load(f)
        else:
            return {"error": "Summary file not found in existing results"}
    
    # Check if files exist
    if not os.path.exists(labels_path):
        print(f"Warning: Labels file not found: {labels_path}")
        return {"error": "Labels file not found"}
    
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found: {video_path}")
        return {"error": "Video file not found"}
    
    # Load oracle detections
    print(f"Loading oracle detections from {labels_path}")
    detections = load_oracle_detections(labels_path)
    
    # Initialize ColorAssigner and RoleAssigner
    color_assigner = ColorAssigner()
    role_assigner = RoleAssigner()
    
    # Create output directory for this video
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # First assign colors to each detection
    print(f"Assigning colors to {len(detections)} frames")
    try:
        # Run color assignment
        detections_with_colors = color_assigner.process_video(
            input_path=video_path,
            detections=detections,
            intermediate_results_folder=None  # Don't save intermediate color results
        )
        
        # Then run role assignment
        print(f"Running role assignment on {len(detections_with_colors)} frames")
        updated_detections = role_assigner.assign_roles(
            input_path=video_path,
            detections=detections_with_colors,
            intermediate_results_folder=video_output_dir
        )
        
        # Save detections with oracle information preserved
        save_detections_with_oracle(updated_detections, video_output_dir)
        
        # Extract role assignments
        role_assignments = []
        unique_tracks = set()
        for frame_det in updated_detections:
            for det in frame_det.detections:
                if det.track_id is not None:
                    unique_tracks.add(det.track_id)
                    
        # Get final role for each track
        track_roles = {}
        for frame_det in updated_detections:
            for det in frame_det.detections:
                if det.track_id is not None and det.role is not None:
                    track_roles[det.track_id] = det.role
        
        # Save summary
        summary = {
            "video_name": video_name,
            "num_frames": len(detections),
            "num_tracks": len(unique_tracks),
            "track_roles": track_roles,
            "role_counts": {
                "TEAM A": sum(1 for role in track_roles.values() if role == "TEAM A"),
                "TEAM B": sum(1 for role in track_roles.values() if role == "TEAM B"),
                "REF": sum(1 for role in track_roles.values() if role == "REF"),
                "GK": sum(1 for role in track_roles.values() if role == "GK"),
                "OOB": sum(1 for role in track_roles.values() if role == "OOB"),
                "UNK": sum(1 for role in track_roles.values() if role == "UNK")
            }
        }
        
        # Save summary
        with open(os.path.join(video_output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Extract and save player images from first frame if requested
        if show_first_frame_players:
            extract_player_images_first_frame(video_path, updated_detections, video_output_dir)
            
        return summary
        
    except Exception as e:
        print(f"Error processing {video_name}: {str(e)}")
        return {"error": str(e)}


def calculate_metrics_for_video(video_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    Calculate metrics for a single video by comparing predicted vs ground truth roles.
    Only processes every 10th frame like the metrics_calculation.py script.
    
    Args:
        video_dir: Path to video directory with results
        output_dir: Output directory containing the results
        
    Returns:
        Dictionary with metrics for this video
    """
    video_name = os.path.basename(video_dir)
    video_output_dir = os.path.join(output_dir, video_name)
    
    # Load the detections with role assignments and oracle information
    detections_path = os.path.join(video_output_dir, 'detections_with_oracle.json')
    if not os.path.exists(detections_path):
        return {"error": "Detections file with oracle info not found"}
    
    with open(detections_path, 'r') as f:
        detections_data = json.load(f)
    
    # Calculate metrics for every 10th frame
    metrics = {
        'player': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
        'goalkeeper': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
        'referee': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    }
    
    frame_comparisons = []
    
    for frame_data in detections_data:
        frame_id = frame_data['frame_id']
        
        # Only process every 10th frame
        if frame_id % 10 != 0:
            continue
            
        for detection in frame_data['detections']:
            # Get oracle role from the detection dict
            oracle_role = detection.get('_oracle_role', 'unknown')
            predicted_role = detection.get('role', 'UNK')
            
            # Map our role names to the metrics categories
            oracle_mapped = map_role_to_category(oracle_role)
            predicted_mapped = map_role_to_category(predicted_role)
            
            # Store comparison
            frame_comparisons.append({
                'frame_id': frame_id,
                'track_id': detection.get('track_id'),
                'oracle_role': oracle_role,
                'predicted_role': predicted_role,
                'oracle_mapped': oracle_mapped,
                'predicted_mapped': predicted_mapped
            })
            
            # Calculate confusion matrix elements for each category
            for category in ['player', 'goalkeeper', 'referee']:
                tp, fp, fn, tn = calculate_confusion_matrix_elements(
                    oracle_mapped, predicted_mapped, category
                )
                metrics[category]['tp'] += tp
                metrics[category]['fp'] += fp
                metrics[category]['fn'] += fn
                metrics[category]['tn'] += tn
    
    # Save frame-by-frame comparison
    comparison_path = os.path.join(video_output_dir, 'role_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(frame_comparisons, f, indent=4)
    
    return {
        'video_name': video_name,
        'metrics': metrics,
        'num_comparisons': len(frame_comparisons)
    }


def map_role_to_category(role: str) -> str:
    """Map role names to standard categories for metrics calculation."""
    role_lower = role.lower()
    if role_lower in ['player', 'team a', 'team b']:
        return 'player'
    elif role_lower in ['goalkeeper', 'gk']:
        return 'goalkeeper'
    elif role_lower in ['referee', 'ref']:
        return 'referee'
    else:
        return 'unknown'


def calculate_confusion_matrix_elements(true_role: str, predicted_role: str, category: str) -> tuple:
    """Calculate TP, FP, FN, TN for a specific category."""
    tp = fp = fn = tn = 0
    
    if true_role == category and predicted_role == category:
        tp = 1  # True Positive
    elif true_role != category and predicted_role == category:
        fp = 1  # False Positive
    elif true_role == category and predicted_role != category:
        fn = 1  # False Negative
    else:
        tn = 1  # True Negative
        
    return tp, fp, fn, tn


def calculate_final_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """Calculate accuracy, precision, recall, and F1 score."""
    total = tp + fp + fn + tn
    
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def main():
    """Run role assignment on all videos in the test set."""
    test_dir = "data/SoccerNet/SN-GSR-2025/test"
    output_dir = "compare_role_assignment/results"
    
    # Don't extract player images for full test set (too slow)
    extract_first_frame_players = False
    
    # Set to True to force reprocessing of all videos, False to skip already processed ones
    force_reprocess = False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all video directories
    video_dirs = [d for d in glob.glob(os.path.join(test_dir, "SNGS-*")) if os.path.isdir(d)]
    video_dirs.sort()
    
    print(f"Found {len(video_dirs)} videos in test set")
    
    # Check how many are already processed
    already_processed = sum(1 for video_dir in video_dirs if check_if_already_processed(video_dir, output_dir))
    print(f"Already processed: {already_processed}/{len(video_dirs)} videos")
    
    if not force_reprocess and already_processed > 0:
        print("Skipping already processed videos. Set force_reprocess=True to reprocess all.")
    
    # Process each video
    all_results = []
    all_metrics = []
    
    for video_dir in tqdm(video_dirs, desc="Processing videos"):
        video_name = os.path.basename(video_dir)
        
        # Check if already processed
        if not force_reprocess and check_if_already_processed(video_dir, output_dir):
            print(f"\n{video_name} already processed, loading existing results...")
        else:
            print(f"\nProcessing {video_name}")
            
        result = run_role_assignment_on_video(video_dir, output_dir, 
                                            show_first_frame_players=extract_first_frame_players,
                                            force_reprocess=force_reprocess)
        result["video_path"] = video_dir
        all_results.append(result)
        
        # Calculate metrics for this video
        if "error" not in result:
            metrics_result = calculate_metrics_for_video(video_dir, output_dir)
            all_metrics.append(metrics_result)
    
    # Calculate overall metrics across all videos
    overall_metrics = {
        'player': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
        'goalkeeper': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
        'referee': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    }
    
    for metrics_result in all_metrics:
        if "error" not in metrics_result:
            for category in ['player', 'goalkeeper', 'referee']:
                for metric in ['tp', 'fp', 'fn', 'tn']:
                    overall_metrics[category][metric] += metrics_result['metrics'][category][metric]
    
    # Calculate final metrics for each category
    final_metrics = {}
    for category in ['player', 'goalkeeper', 'referee']:
        tp = overall_metrics[category]['tp']
        fp = overall_metrics[category]['fp']
        fn = overall_metrics[category]['fn']
        tn = overall_metrics[category]['tn']
        
        calculated = calculate_final_metrics(tp, fp, fn, tn)
        
        final_metrics[category] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            **calculated
        }
    
    # Save overall summary
    summary = {
        "total_videos": len(video_dirs),
        "successful": sum(1 for r in all_results if "error" not in r),
        "failed": sum(1 for r in all_results if "error" in r),
        "results": all_results,
        "overall_metrics": final_metrics,
        "per_video_metrics": all_metrics
    }
    
    with open(os.path.join(output_dir, 'overall_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Save metrics in CSV format like metrics_calculation.py
    
    # Overall metrics CSV
    overall_data = []
    for category in ['player', 'goalkeeper', 'referee']:
        metrics = final_metrics[category]
        overall_data.append({
            'role': category,
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'fn': metrics['fn'],
            'tn': metrics['tn'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        })
    
    overall_df = pd.DataFrame(overall_data)
    overall_csv_path = os.path.join(output_dir, 'role_assignment_metrics.csv')
    overall_df.to_csv(overall_csv_path, index=False)
    
    # Per-video metrics CSV
    video_data = []
    for metrics_result in all_metrics:
        if "error" not in metrics_result:
            video_name = metrics_result['video_name']
            for category in ['player', 'goalkeeper', 'referee']:
                metrics = metrics_result['metrics'][category]
                calculated = calculate_final_metrics(
                    metrics['tp'], metrics['fp'], metrics['fn'], metrics['tn']
                )
                
                video_data.append({
                    'video': video_name,
                    'role': category,
                    'tp': metrics['tp'],
                    'fp': metrics['fp'],
                    'fn': metrics['fn'],
                    'tn': metrics['tn'],
                    'accuracy': calculated['accuracy'],
                    'precision': calculated['precision'],
                    'recall': calculated['recall'],
                    'f1_score': calculated['f1_score']
                })
    
    video_df = pd.DataFrame(video_data)
    video_csv_path = os.path.join(output_dir, 'role_assignment_metrics_per_video.csv')
    video_df.to_csv(video_csv_path, index=False)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"ROLE ASSIGNMENT PROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"Total videos: {summary['total_videos']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    
    # Print role distribution across all videos
    total_roles = {
        "TEAM A": 0,
        "TEAM B": 0,
        "REF": 0,
        "GK": 0,
        "OOB": 0,
        "UNK": 0
    }
    
    for result in all_results:
        if "role_counts" in result:
            for role, count in result["role_counts"].items():
                total_roles[role] += count
    
    print(f"\nTotal role assignments across all videos:")
    for role, count in total_roles.items():
        print(f"  {role}: {count}")
    
    # Print overall metrics
    print(f"\n{'='*80}")
    print(f"OVERALL ROLE CLASSIFICATION METRICS")
    print(f"{'='*80}")
    
    for category in ['player', 'goalkeeper', 'referee']:
        metrics = final_metrics[category]
        print(f"\n{category.upper()} METRICS:")
        print("-" * 40)
        print(f"True Positives (TP):  {metrics['tp']:,}")
        print(f"False Positives (FP): {metrics['fp']:,}")
        print(f"False Negatives (FN): {metrics['fn']:,}")
        print(f"True Negatives (TN):  {metrics['tn']:,}")
        print(f"Accuracy:             {metrics['accuracy']:.4f}")
        print(f"Precision:            {metrics['precision']:.4f}")
        print(f"Recall:               {metrics['recall']:.4f}")
        print(f"F1 Score:             {metrics['f1_score']:.4f}")
    
    print(f"\nResults saved to:")
    print(f"  - Overall summary: {os.path.join(output_dir, 'overall_summary.json')}")
    print(f"  - Overall metrics: {overall_csv_path}")
    print(f"  - Per-video metrics: {video_csv_path}")


if __name__ == "__main__":
    main()
