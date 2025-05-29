import os
import json
from utils.data_loader import load_frame_data, save_augmented_results
from utils.color_utils import extract_jersey_colors, cluster_player_colors
from utils.geometry_utils import check_point_in_clusters

def process_prtreid_predictions(base_dir='prtreid_output', image_base_dir='test', 
                               confidence_threshold=3.564, output_dir='augmented_output'):
    """Process prtreid predictions and reassign unclassified based on jersey color clustering
    
    Args:
        base_dir: Directory containing prtreid_output with reid_results.json files
        image_base_dir: Base directory for image files
        confidence_threshold: Confidence threshold for player class
        output_dir: Directory to save augmented results
    """
    # Statistics
    total_frames = 0
    frames_with_unclassified = 0
    total_reassigned = 0
    
    # Process each SNGS directory
    sngs_dirs = [d for d in os.listdir(base_dir) if d.startswith('SNGS-')]
    print(f"Found {len(sngs_dirs)} SNGS directories")
    
    for sngs_dir in sorted(sngs_dirs):
        print(f"Processing SNGS directory: {sngs_dir}")
        sngs_path = os.path.join(base_dir, sngs_dir)
        if not os.path.isdir(sngs_path):
            continue
            
        # Get all frame directories
        frame_dirs = [d for d in os.listdir(sngs_path) 
                     if d.startswith('000') and os.path.isdir(os.path.join(sngs_path, d))]
        
        for frame_dir in sorted(frame_dirs):
            total_frames += 1
            reid_results_path = os.path.join(sngs_path, frame_dir, 'reid_results.json')
            
            if not os.path.exists(reid_results_path):
                continue
                
            # Load predictions and apply confidence threshold
            predictions, has_unclassified = load_frame_data(reid_results_path, confidence_threshold)
            
            if has_unclassified:
                frames_with_unclassified += 1
                
                # Get image path
                frame_num = frame_dir
                image_path = os.path.join(image_base_dir, sngs_dir, 'img1', f'{frame_num}.jpg')
                
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found at {image_path}")
                    continue
                
                # Process frame with jersey color clustering
                augmented_predictions, num_reassigned = process_frame_with_clustering(
                    predictions, image_path
                )
                
                total_reassigned += num_reassigned
                
                # Save augmented results
                output_path = os.path.join(output_dir, sngs_dir, frame_dir)
                save_augmented_results(augmented_predictions, output_path)
            else:
                # No unclassified, just copy original results
                output_path = os.path.join(output_dir, sngs_dir, frame_dir)
                save_augmented_results(predictions, output_path)
    
    # Print statistics
    print(f"\nProcessing complete!")
    print(f"Total frames: {total_frames}")
    print(f"Frames with unclassified: {frames_with_unclassified}")
    print(f"Total predictions reassigned: {total_reassigned}")
    

def process_frame_with_clustering(predictions, image_path):
    """Process a single frame with jersey color clustering
    
    Args:
        predictions: List of predictions for the frame
        image_path: Path to the frame image
        
    Returns:
        augmented_predictions: Updated predictions
        num_reassigned: Number of predictions reassigned to player
    """
    # Separate players and unclassified
    players = [p for p in predictions if p.get('predicted_role') == 'player']
    unclassified = [p for p in predictions if p.get('predicted_role') == 'unclassified']
    
    if len(players) < 2 or len(unclassified) == 0:
        # Not enough players to cluster or no unclassified to reassign
        return predictions, 0
    
    # Extract jersey colors for all relevant predictions
    jersey_colors = extract_jersey_colors(predictions, image_path)

    assert len(jersey_colors) == len(predictions)
    
    # Cluster player colors (k=2)
    player_indices = [i for i, p in enumerate(predictions) if p.get('predicted_role') == 'player']
    player_colors = [jersey_colors[i] for i in player_indices]

    assert len(player_colors) == len(players)

    if len(set(jersey_colors)) == 1:
        raise ValueError(f"All jersey colors are the same: {jersey_colors}")
    
    clusters = cluster_player_colors(player_colors, k=2)
    
    # Check each unclassified prediction
    num_reassigned = 0
    augmented_predictions = predictions.copy()
    
    for i, pred in enumerate(predictions):
        if pred.get('predicted_role') == 'unclassified':
            unclassified_color = jersey_colors[i]
            
            # Check if color falls within any cluster
            if check_point_in_clusters(unclassified_color, clusters):
                # Reassign to player
                augmented_predictions[i] = pred.copy()
                augmented_predictions[i]['predicted_role'] = 'player'
                augmented_predictions[i]['reassigned'] = True
                num_reassigned += 1
    
    return augmented_predictions, num_reassigned


if __name__ == "__main__":
    # Configuration
    config = {
        'base_dir': 'prtreid_output',
        'image_base_dir': 'test',
        'confidence_threshold': 3.564,
        'output_dir': 'augmented_output'
    }
    
    print("Starting augmented prtreid processing...")
    print(f"Configuration: {config}")
    
    process_prtreid_predictions(**config)
