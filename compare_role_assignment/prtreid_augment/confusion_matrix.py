import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict, Counter

def load_all_predictions(base_dir, confidence_threshold):
    """Load all predictions from reid_results.json files across all SNGS-XXX/000YYY directories"""
    all_predictions = []
    all_ground_truth = []
    total_detections = 0
    reassigned_count = 0
    
    # Get all SNGS directories
    sngs_dirs = [d for d in os.listdir(base_dir) if d.startswith('SNGS-')]
    print(f"Found {len(sngs_dirs)} SNGS directories")
    
    for sngs_dir in sorted(sngs_dirs):
        sngs_path = os.path.join(base_dir, sngs_dir)
        if not os.path.isdir(sngs_path):
            continue
            
        # Get all frame directories (000XXX)
        frame_dirs = [d for d in os.listdir(sngs_path) if d.startswith('000') and os.path.isdir(os.path.join(sngs_path, d))]
        
        for frame_dir in sorted(frame_dirs):
            reid_results_path = os.path.join(sngs_path, frame_dir, 'reid_results.json')
            
            if os.path.exists(reid_results_path):
                try:
                    with open(reid_results_path, 'r') as f:
                        data = json.load(f)
                    
                    for detection in data:
                        if 'true_role' in detection and 'predicted_role' in detection:
                            total_detections += 1
                            all_ground_truth.append(detection['true_role'])
                            
                            # Check role_confidence and reassign if below threshold
                            predicted_role = detection['predicted_role']
                            if 'role_confidence' in detection:
                                role_confidence = detection['role_confidence']
                                if predicted_role == 'player' and role_confidence < confidence_threshold:
                                    predicted_role = 'unclassified'
                                    reassigned_count += 1
                            
                            all_predictions.append(predicted_role)
                            
                except json.JSONDecodeError as e:
                    print(f"Error reading {reid_results_path}: {e}")
                except Exception as e:
                    print(f"Error processing {reid_results_path}: {e}")
    
    print(f"Loaded {len(all_predictions)} predictions total")
    print(f"Reassigned {reassigned_count} predictions to 'unclassified' due to low confidence (< {confidence_threshold})")
    print(f"Reassignment rate: {reassigned_count/total_detections*100:.2f}%" if total_detections > 0 else "No detections found")
    
    return all_predictions, all_ground_truth

def compute_metrics_per_class(y_true, y_pred, classes):
    """Compute TP, FP, FN, TN, Accuracy, Precision, Recall, F1 for each class"""
    metrics = {}
    
    for cls in classes:
        # Convert to binary classification for this class
        y_true_binary = [1 if label == cls else 0 for label in y_true]
        y_pred_binary = [1 if pred == cls else 0 for pred in y_pred]
        
        # Calculate confusion matrix elements
        tp = sum(1 for true, pred in zip(y_true_binary, y_pred_binary) if true == 1 and pred == 1)
        fp = sum(1 for true, pred in zip(y_true_binary, y_pred_binary) if true == 0 and pred == 1)
        fn = sum(1 for true, pred in zip(y_true_binary, y_pred_binary) if true == 1 and pred == 0)
        tn = sum(1 for true, pred in zip(y_true_binary, y_pred_binary) if true == 0 and pred == 0)
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[cls] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Role Classification Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Role', fontsize=12)
    plt.ylabel('True Role', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return cm

def print_detailed_metrics(metrics, classes, y_true, y_pred):
    """Print detailed metrics in a formatted table"""
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION METRICS")
    print("="*80)
    
    # Print header
    print(f"{'Class':<12} {'TP':<6} {'FP':<6} {'FN':<6} {'TN':<6} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8}")
    print("-" * 80)
    
    for cls in classes:
        m = metrics[cls]
        print(f"{cls:<12} {m['TP']:<6} {m['FP']:<6} {m['FN']:<6} {m['TN']:<6} "
              f"{m['Accuracy']:<8.4f} {m['Precision']:<8.4f} {m['Recall']:<8.4f} {m['F1']:<8.4f}")
    
    # Calculate overall metrics
    total_correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    overall_accuracy = total_correct / len(y_true)
    
    # Macro averages
    macro_precision = np.mean([metrics[cls]['Precision'] for cls in classes])
    macro_recall = np.mean([metrics[cls]['Recall'] for cls in classes])
    macro_f1 = np.mean([metrics[cls]['F1'] for cls in classes])
    
    print("-" * 80)
    print(f"{'Overall':<12} {'':<24} {overall_accuracy:<8.4f} {macro_precision:<8.4f} {macro_recall:<8.4f} {macro_f1:<8.4f}")
    print("="*80)

def print_class_distribution(y_true, y_pred):
    """Print class distribution for true labels and predictions"""
    print("\nCLASS DISTRIBUTION:")
    print("-" * 40)
    
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    
    print("True labels:")
    for cls, count in sorted(true_counts.items()):
        percentage = count / len(y_true) * 100
        print(f"  {cls:<12}: {count:>6} ({percentage:>5.1f}%)")
    
    print("\nPredicted labels:")
    for cls, count in sorted(pred_counts.items()):
        percentage = count / len(y_pred) * 100
        print(f"  {cls:<12}: {count:>6} ({percentage:>5.1f}%)")

if __name__ == "__main__":

    confidence_threshold = 3.564

    # Define base directory
    base_dir = "prtreid_output"
    
    print("Starting evaluation of prtreid_output...")
    print("Loading all predictions...")
    
    # Load all predictions
    y_pred, y_true = load_all_predictions(base_dir, confidence_threshold)
    
    if len(y_pred) == 0:
        print("No predictions found! Check the directory structure.")
        exit(1)
    
    # Define classes
    classes = sorted(list(set(y_true + y_pred)))
    print(f"Found classes: {classes}")
    
    # Print class distribution
    print_class_distribution(y_true, y_pred)
    
    # Compute metrics for each class
    metrics = compute_metrics_per_class(y_true, y_pred, classes)
    
    # Print detailed metrics
    print_detailed_metrics(metrics, classes, y_true, y_pred)
    
    # Create and save confusion matrix
    cm = plot_confusion_matrix(y_true, y_pred, classes, save_path=f"confusion_matrix_confidence_threshold_{confidence_threshold}.png")
    
    # Print confusion matrix as text
    print(f"\nCONFUSION MATRIX:")
    print(f"Classes: {classes}")
    print("Rows = True labels, Columns = Predicted labels")
    print(cm)
    
    # Generate sklearn classification report for comparison
    print(f"\nSKLEARN CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, labels=classes, digits=4))
