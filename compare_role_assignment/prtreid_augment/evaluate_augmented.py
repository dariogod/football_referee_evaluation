import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

def load_augmented_predictions(base_dir):
    """Load all augmented predictions from augmented_reid_results.json files"""
    all_predictions = []
    all_ground_truth = []
    reassigned_count = 0
    total_detections = 0
    
    # Get all SNGS directories
    sngs_dirs = [d for d in os.listdir(base_dir) if d.startswith('SNGS-')]
    print(f"Found {len(sngs_dirs)} SNGS directories")
    
    for sngs_dir in sorted(sngs_dirs):
        sngs_path = os.path.join(base_dir, sngs_dir)
        if not os.path.isdir(sngs_path):
            continue
            
        # Get all frame directories
        frame_dirs = [d for d in os.listdir(sngs_path) 
                     if d.startswith('000') and os.path.isdir(os.path.join(sngs_path, d))]
        
        for frame_dir in sorted(frame_dirs):
            results_path = os.path.join(sngs_path, frame_dir, 'augmented_reid_results.json')
            
            if os.path.exists(results_path):
                try:
                    with open(results_path, 'r') as f:
                        data = json.load(f)
                    
                    for detection in data:
                        if 'true_role' in detection and 'predicted_role' in detection:
                            total_detections += 1
                            all_ground_truth.append(detection['true_role'])
                            all_predictions.append(detection['predicted_role'])
                            
                            if detection.get('reassigned', False):
                                reassigned_count += 1
                            
                except Exception as e:
                    print(f"Error processing {results_path}: {e}")
    
    print(f"Loaded {len(all_predictions)} predictions total")
    print(f"Reassigned {reassigned_count} predictions back to 'player'")
    print(f"Reassignment rate: {reassigned_count/total_detections*100:.2f}%" if total_detections > 0 else "No detections")
    
    return all_predictions, all_ground_truth

def compare_with_original(original_dir='prtreid_output', augmented_dir='augmented_output', confidence_threshold=3.564):
    """Compare augmented results with original results"""
    print("="*80)
    print("COMPARISON: Original vs Augmented Predictions")
    print("="*80)
    
    # Load original predictions
    print("\nOriginal predictions:")
    from confusion_matrix import load_all_predictions
    orig_pred, orig_true = load_all_predictions(original_dir, confidence_threshold)
    
    # Load augmented predictions
    print("\nAugmented predictions:")
    aug_pred, aug_true = load_augmented_predictions(augmented_dir)
    
    # Calculate improvements
    if len(orig_pred) == len(aug_pred):
        orig_correct = sum(1 for p, t in zip(orig_pred, orig_true) if p == t)
        aug_correct = sum(1 for p, t in zip(aug_pred, aug_true) if p == t)
        
        orig_acc = orig_correct / len(orig_pred) * 100
        aug_acc = aug_correct / len(aug_pred) * 100
        
        print(f"\nOriginal accuracy: {orig_acc:.2f}%")
        print(f"Augmented accuracy: {aug_acc:.2f}%")
        print(f"Improvement: {aug_acc - orig_acc:.2f}%")
        
        # Count class changes
        changes = sum(1 for o, a in zip(orig_pred, aug_pred) if o != a)
        print(f"\nTotal predictions changed: {changes}")
        
        # Analyze changes
        unclassified_to_player = sum(1 for o, a in zip(orig_pred, aug_pred) 
                                    if o == 'unclassified' and a == 'player')
        print(f"Unclassified → Player: {unclassified_to_player}")
    
    return aug_pred, aug_true

if __name__ == "__main__":
    import sys
    
    # Check if we should compare or just evaluate
    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        # Compare mode
        y_pred, y_true = compare_with_original()
    else:
        # Direct evaluation mode
        base_dir = 'augmented_output'
        print(f"Evaluating augmented predictions from {base_dir}")
        y_pred, y_true = load_augmented_predictions(base_dir)
    
    if len(y_pred) == 0:
        print("No predictions found!")
        exit(1)
    
    # Get classes
    classes = sorted(list(set(y_true + y_pred)))
    print(f"\nClasses found: {classes}")
    
    # Print class distribution
    print("\nCLASS DISTRIBUTION:")
    print("-" * 40)
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    
    print("True labels:")
    for cls, count in sorted(true_counts.items()):
        print(f"  {cls:<12}: {count:>6} ({count/len(y_true)*100:>5.1f}%)")
    
    print("\nPredicted labels:")
    for cls, count in sorted(pred_counts.items()):
        print(f"  {cls:<12}: {count:>6} ({count/len(y_pred)*100:>5.1f}%)")
    
    # Print classification report
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, labels=classes, digits=4))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Augmented Role Classification Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Role', fontsize=12)
    plt.ylabel('True Role', fontsize=12)
    plt.tight_layout()
    plt.savefig('augmented_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved to augmented_confusion_matrix.png") 