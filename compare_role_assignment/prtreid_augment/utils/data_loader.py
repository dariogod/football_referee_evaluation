import json
import os

def load_frame_data(reid_results_path, confidence_threshold):
    """Load predictions from a reid_results.json file and apply confidence threshold
    
    Args:
        reid_results_path: Path to reid_results.json file
        confidence_threshold: Threshold for player confidence
        
    Returns:
        predictions: List of predictions with low-confidence players marked as unclassified
        has_unclassified: Boolean indicating if there are any unclassified predictions
    """
    with open(reid_results_path, 'r') as f:
        data = json.load(f)
    
    predictions = []
    has_unclassified = False
    
    for detection in data:
        pred = detection.copy()
        
        # Apply confidence threshold
        if 'predicted_role' in pred and 'role_confidence' in pred:
            if pred['predicted_role'] == 'player' and pred['role_confidence'] < confidence_threshold:
                pred['predicted_role'] = 'unclassified'
                pred['original_role'] = 'player'
                has_unclassified = True
        
        if pred.get('predicted_role') == 'unclassified':
            has_unclassified = True
            
        predictions.append(pred)
    
    return predictions, has_unclassified


def save_augmented_results(predictions, output_path):
    """Save augmented predictions to output directory
    
    Args:
        predictions: List of augmented predictions
        output_path: Output directory path
    """
    os.makedirs(output_path, exist_ok=True)
    
    output_file = os.path.join(output_path, 'augmented_reid_results.json')
    
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2) 