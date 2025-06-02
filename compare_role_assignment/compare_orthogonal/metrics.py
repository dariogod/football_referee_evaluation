import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import sys
from contextlib import redirect_stdout

def load_data(file_path):
    """Load the combined role predictions data."""
    with open(file_path, 'r') as f:
        return json.load(f)

def normalize_role(role):
    """Normalize role labels to group player types."""
    if role in ['player_left', 'player_right']:
        return 'player'
    return role

def extract_predictions_and_ground_truth(data, exclude_datasets=None):
    """Extract all predictions and ground truth labels from the data."""
    if exclude_datasets is None:
        exclude_datasets = []
    
    results = {
        'dbscan_rgb': {'gt': [], 'pred': []},
        'dbscan_lab': {'gt': [], 'pred': []},
        'dbscan_hsv': {'gt': [], 'pred': []},
        'prtreid': {'gt': [], 'pred': []}
    }
    
    for dataset_id, dataset_data in data.items():
        if dataset_id in exclude_datasets:
            print(f"Excluding dataset: {dataset_id}")
            continue
            
        for frame_id, detections in dataset_data.items():
            for detection in detections:
                gt_role = normalize_role(detection['gt_role'])
                
                # DBSCAN predictions
                dbscan_preds = detection.get('dbscan_pred_role', {})
                for color_space in ['rgb', 'lab', 'hsv']:
                    if color_space in dbscan_preds:
                        pred_role = normalize_role(dbscan_preds[color_space])
                        results[f'dbscan_{color_space}']['gt'].append(gt_role)
                        results[f'dbscan_{color_space}']['pred'].append(pred_role)
                
                # PRTREID predictions
                if 'prtreid_data' in detection:
                    prtreid_pred = detection['prtreid_data'].get('predicted_role')
                    if prtreid_pred in ["other", "ball"]:
                        continue
                    if prtreid_pred:
                        pred_role = normalize_role(prtreid_pred)
                        results['prtreid']['gt'].append(gt_role)
                        results['prtreid']['pred'].append(pred_role)
    
    return results

def calculate_metrics_for_class(y_true, y_pred, target_class, all_classes):
    """Calculate TP, FP, FN, TN, accuracy, precision, recall, F1 for a specific class."""
    # Convert to binary classification for the target class
    y_true_binary = [1 if label == target_class else 0 for label in y_true]
    y_pred_binary = [1 if label == target_class else 0 for label in y_pred]
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def print_metrics_table(method_name, metrics_by_class):
    """Print metrics in a formatted table."""
    print(f"\n{method_name.upper()} METRICS:")
    print("="*80)
    print(f"{'Class':<12} {'TP':<6} {'FP':<6} {'FN':<6} {'TN':<6} {'Accuracy':<10} {'Precision':<10} {'Recall':<8} {'F1':<8}")
    print("-"*80)
    
    for class_name, metrics in metrics_by_class.items():
        print(f"{class_name:<12} {metrics['tp']:<6} {metrics['fp']:<6} {metrics['fn']:<6} {metrics['tn']:<6} "
              f"{metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<8.4f} {metrics['f1']:<8.4f}")

def print_confusion_matrix(method_name, y_true, y_pred, classes):
    """Print confusion matrix in a formatted table."""
    # Get all unique labels that appear in the data
    unique_labels = sorted(list(set(y_true + y_pred)))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    print(f"\n{method_name.upper()} CONFUSION MATRIX:")
    print("="*60)
    
    # Print header
    header = "Actual\\Predicted" + "".join(f"{label:>12}" for label in unique_labels)
    print(header)
    print("-" * len(header))
    
    # Print matrix rows
    for i, true_label in enumerate(unique_labels):
        row = f"{true_label:<15}"
        for j, pred_label in enumerate(unique_labels):
            row += f"{cm[i][j]:>12}"
        print(row)
    
    # Print totals
    print("-" * len(header))
    total_row = "Total" + "".join(f"{cm.sum(axis=0)[j]:>12}" for j in range(len(unique_labels)))
    print(total_row)
    
    # Print overall accuracy
    overall_accuracy = np.trace(cm) / np.sum(cm)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")

def main():
    # Load data
    data_file = 'combined_role_predictions_updated.json'
    data = load_data(data_file)
    
    # Exclude specified datasets
    exclude_datasets = ['SNGS-125', 'SNGS-190']
    
    # Extract predictions and ground truth
    results = extract_predictions_and_ground_truth(data, exclude_datasets)
    
    # Define classes
    classes = ['player', 'goalkeeper', 'referee']
    
    # Redirect output to metrics.txt
    with open('metrics.txt', 'w') as f:
        with redirect_stdout(f):
            print("FOOTBALL REFEREE EVALUATION - ROLE CLASSIFICATION METRICS")
            print("="*60)
            print(f"Excluded datasets: {', '.join(exclude_datasets)}")
            print(f"Classes evaluated: {', '.join(classes)}")
            
            # Calculate and print metrics for each method
            for method_name, method_data in results.items():
                if not method_data['gt']:  # Skip if no data
                    print(f"\nNo data available for {method_name}")
                    continue
                    
                print(f"\nTotal predictions for {method_name}: {len(method_data['gt'])}")
                
                # Print confusion matrix
                print_confusion_matrix(method_name, method_data['gt'], method_data['pred'], classes)
                
                # Get all unique classes in the data
                all_classes_in_data = set(method_data['gt'] + method_data['pred'])
                
                # Calculate metrics for each class
                metrics_by_class = {}
                for class_name in classes:
                    if class_name in all_classes_in_data:
                        metrics = calculate_metrics_for_class(
                            method_data['gt'], 
                            method_data['pred'], 
                            class_name, 
                            all_classes_in_data
                        )
                        metrics_by_class[class_name] = metrics
                    else:
                        print(f"Warning: No instances of class '{class_name}' found in {method_name} data")
                
                # Print metrics table
                if metrics_by_class:
                    print_metrics_table(method_name, metrics_by_class)
            
            # Print summary statistics
            print(f"\n\nSUMMARY STATISTICS:")
            print("="*40)
            for method_name, method_data in results.items():
                if method_data['gt']:
                    print(f"{method_name}: {len(method_data['gt'])} total predictions")
            
            print(f"\nDatasets included: {len([k for k in data.keys() if k not in exclude_datasets])}")
            print(f"Datasets excluded: {len(exclude_datasets)}")

if __name__ == "__main__":
    main()
