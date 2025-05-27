#!/usr/bin/env python3
"""Recalculate metrics from existing role assignment results."""

import json
import os
import sys
from typing import Dict, Any, List
import pandas as pd
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def calculate_metrics_for_video(video_name: str, results_dir: str) -> Dict[str, Any]:
    """
    Calculate metrics for a single video by comparing predicted vs ground truth roles.
    Only processes every 10th frame.
    """
    video_output_dir = os.path.join(results_dir, video_name)
    
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


def create_confusion_matrices(results_dir: str, all_metrics: List[Dict[str, Any]]):
    """Create and save confusion matrices for role classification."""
    
    # Collect all predictions and ground truth labels
    all_true_labels = []
    all_pred_labels = []
    
    for video_dir in glob.glob(os.path.join(results_dir, "SNGS-*")):
        video_name = os.path.basename(video_dir)
        comparison_path = os.path.join(video_dir, 'role_comparison.json')
        
        if os.path.exists(comparison_path):
            with open(comparison_path, 'r') as f:
                comparisons = json.load(f)
            
            for comp in comparisons:
                oracle_mapped = comp['oracle_mapped']
                predicted_mapped = comp['predicted_mapped']
                
                # Only include known categories
                if oracle_mapped in ['player', 'goalkeeper', 'referee'] and predicted_mapped in ['player', 'goalkeeper', 'referee']:
                    all_true_labels.append(oracle_mapped)
                    all_pred_labels.append(predicted_mapped)
    
    if not all_true_labels:
        print("No valid labels found for confusion matrix")
        return
    
    # Define the labels in order
    labels = ['player', 'goalkeeper', 'referee']
    
    # Create confusion matrix
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=labels)
    
    # Create a more detailed confusion matrix plot
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Number of Predictions'})
    
    plt.title('Confusion Matrix - Role Classification\n(Ground Truth vs Predicted)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Role', fontsize=12)
    plt.ylabel('True Role', fontsize=12)
    
    # Add percentage annotations
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Add percentage text to each cell
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='red')
    
    plt.tight_layout()
    
    # Save the confusion matrix
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create normalized confusion matrix
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Proportion'})
    
    plt.title('Normalized Confusion Matrix - Role Classification\n(Row-wise Normalization)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Role', fontsize=12)
    plt.ylabel('True Role', fontsize=12)
    plt.tight_layout()
    
    # Save normalized confusion matrix
    cm_norm_path = os.path.join(results_dir, 'confusion_matrix_normalized.png')
    plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_csv_path = os.path.join(results_dir, 'confusion_matrix.csv')
    cm_df.to_csv(cm_csv_path)
    
    # Generate and save classification report
    report = classification_report(all_true_labels, all_pred_labels, 
                                 target_names=labels, output_dict=True)
    
    report_df = pd.DataFrame(report).transpose()
    report_csv_path = os.path.join(results_dir, 'classification_report.csv')
    report_df.to_csv(report_csv_path)
    
    # Print confusion matrix info
    print(f"\n{'='*60}")
    print("CONFUSION MATRIX")
    print(f"{'='*60}")
    print(f"Total predictions analyzed: {len(all_true_labels):,}")
    print(f"\nConfusion Matrix:")
    print(f"{'':>12} {'Player':>8} {'GK':>8} {'Referee':>8}")
    for i, true_label in enumerate(labels):
        print(f"{true_label:>12} {cm[i][0]:>8} {cm[i][1]:>8} {cm[i][2]:>8}")
    
    print(f"\nFiles saved:")
    print(f"  - Confusion matrix plot: {cm_path}")
    print(f"  - Normalized confusion matrix: {cm_norm_path}")
    print(f"  - Confusion matrix CSV: {cm_csv_path}")
    print(f"  - Classification report: {report_csv_path}")


def main():
    """Recalculate metrics from existing role assignment results."""
    results_dir = "compare_role_assignment/results"
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found!")
        return
    
    # Get all video result directories
    video_dirs = [d for d in glob.glob(os.path.join(results_dir, "SNGS-*")) if os.path.isdir(d)]
    video_dirs.sort()
    
    print(f"Found {len(video_dirs)} video result directories")
    
    # Calculate metrics for each video
    all_metrics = []
    
    for video_dir in tqdm(video_dirs, desc="Recalculating metrics"):
        video_name = os.path.basename(video_dir)
        
        # Check if detections_with_oracle.json exists
        detections_path = os.path.join(video_dir, 'detections_with_oracle.json')
        if not os.path.exists(detections_path):
            print(f"Warning: {video_name} missing detections_with_oracle.json, skipping...")
            continue
            
        metrics_result = calculate_metrics_for_video(video_name, results_dir)
        if "error" not in metrics_result:
            all_metrics.append(metrics_result)
        else:
            print(f"Error processing {video_name}: {metrics_result['error']}")
    
    # Calculate overall metrics across all videos
    overall_metrics = {
        'player': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
        'goalkeeper': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
        'referee': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    }
    
    for metrics_result in all_metrics:
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
    
    # Save metrics in CSV format
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
    overall_csv_path = os.path.join(results_dir, 'role_assignment_metrics.csv')
    overall_df.to_csv(overall_csv_path, index=False)
    
    # Per-video metrics CSV
    video_data = []
    for metrics_result in all_metrics:
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
    video_csv_path = os.path.join(results_dir, 'role_assignment_metrics_per_video.csv')
    video_df.to_csv(video_csv_path, index=False)
    
    # Print overall metrics
    print(f"\n{'='*80}")
    print(f"CORRECTED ROLE CLASSIFICATION METRICS")
    print(f"{'='*80}")
    print(f"Processed {len(all_metrics)} videos successfully")
    
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
    
    print(f"\nCorrected results saved to:")
    print(f"  - Overall metrics: {overall_csv_path}")
    print(f"  - Per-video metrics: {video_csv_path}")
    
    # Create confusion matrices
    create_confusion_matrices(results_dir, all_metrics)


if __name__ == "__main__":
    main() 