import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

def load_all_predictions_with_confidence(base_dir):
    """Load all predictions with confidence scores from reid_results.json files"""
    all_data = []
    
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
                        if all(key in detection for key in ['true_role', 'predicted_role', 'role_confidence']):
                            all_data.append({
                                'true_role': detection['true_role'],
                                'predicted_role': detection['predicted_role'],
                                'role_confidence': detection['role_confidence'],
                                'sngs_dir': sngs_dir,
                                'frame_dir': frame_dir
                            })
                            
                except json.JSONDecodeError as e:
                    print(f"Error reading {reid_results_path}: {e}")
                except Exception as e:
                    print(f"Error processing {reid_results_path}: {e}")
    
    print(f"Loaded {len(all_data)} predictions with confidence scores")
    return all_data

def plot_confidence_distribution_by_predicted_role(data, save_path=None):
    """Plot confidence distribution grouped by predicted role"""
    # Group data by predicted role
    role_confidences = defaultdict(list)
    
    for item in data:
        role_confidences[item['predicted_role']].append(item['role_confidence'])
    
    # Get all unique roles
    roles = sorted(role_confidences.keys())
    roles.remove("other")
    n_roles = len(roles)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Plot histogram for each role
    for i, role in enumerate(roles):
        if i < len(axes):
            confidences = role_confidences[role]
            
            axes[i].hist(confidences, bins=30, alpha=0.7, color=plt.cm.Set3(i), edgecolor='black')
            axes[i].set_title(f'Confidence Distribution - {role}\n(n={len(confidences)})', fontsize=12)
            axes[i].set_xlabel('Confidence Score')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
            # Add statistics
            mean_conf = np.mean(confidences)
            median_conf = np.median(confidences)
            std_conf = np.std(confidences)
            
            axes[i].axvline(mean_conf, color='red', linestyle='--', label=f'Mean: {mean_conf:.3f}')
            axes[i].axvline(median_conf, color='orange', linestyle='--', label=f'Median: {median_conf:.3f}')
            axes[i].legend()
    
    # Hide unused subplots
    for i in range(n_roles, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Role Confidence Distribution by Predicted Role', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confidence distribution plot saved to {save_path}")
    
def plot_confidence_boxplot(data, save_path=None):
    """Create box plot comparing confidence distributions across roles"""
    # Prepare data for box plot
    plot_data = []
    for item in data:
        plot_data.append({
            'Predicted Role': item['predicted_role'],
            'True Role': item['true_role'],
            'Confidence': item['role_confidence'],
            'Correct': item['predicted_role'] == item['true_role']
        })
    
    df = pd.DataFrame(plot_data)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Box plot by predicted role
    sns.boxplot(data=df, x='Predicted Role', y='Confidence', ax=ax1)
    ax1.set_title('Confidence Distribution by Predicted Role')
    ax1.tick_params(axis='x', rotation=45)
    
    # Box plot by correctness
    sns.boxplot(data=df, x='Correct', y='Confidence', ax=ax2)
    ax2.set_title('Confidence Distribution: Correct vs Incorrect Predictions')
    ax2.set_xlabel('Prediction Correctness')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Box plot saved to {save_path}")
        
    return df

def plot_confidence_vs_accuracy(data, save_path=None):
    """Plot confidence threshold vs FP, FN, TP, TN counts for the player class"""
    # Prepare data for player class analysis
    plot_data = []
    for item in data:
        # Convert to binary classification for player class
        true_player = item['true_role'] == 'player'
        pred_player = item['predicted_role'] == 'player'
        
        plot_data.append({
            'confidence': item['role_confidence'],
            'true_player': true_player,
            'pred_player': pred_player,
            'predicted_role': item['predicted_role']
        })
    
    df = pd.DataFrame(plot_data)
    
    # Define confidence thresholds to test
    # Use the meaningful range for player predictions with finer granularity in critical region
    fine_thresholds = np.arange(3.4, 3.95, 0.02)  # Fine steps in main distribution
    coarse_thresholds = np.concatenate([
        np.arange(0.0, 3.4, 0.1),     # Coarse steps for lower confidence
        fine_thresholds,               # Fine steps for main distribution  
        np.arange(3.95, 5.0, 0.1)     # Coarse steps for higher confidence
    ])
    
    confidence_thresholds = sorted(coarse_thresholds)
    
    # Calculate TP, FP, FN, TN counts for each threshold
    fps = []  # False Positive counts
    fns = []  # False Negative counts
    tps = []  # True Positive counts  
    tns = []  # True Negative counts
    total_predictions = []  # Total predictions above threshold
    
    for threshold in confidence_thresholds:
        # Consider only predictions with confidence >= threshold
        above_threshold = df[df['confidence'] >= threshold]
        
        if len(above_threshold) > 0:
            # Calculate confusion matrix elements for player class
            tp = len(above_threshold[(above_threshold['true_player'] == True) & (above_threshold['pred_player'] == True)])
            fp = len(above_threshold[(above_threshold['true_player'] == False) & (above_threshold['pred_player'] == True)])
            fn = len(above_threshold[(above_threshold['true_player'] == True) & (above_threshold['pred_player'] == False)])
            tn = len(above_threshold[(above_threshold['true_player'] == False) & (above_threshold['pred_player'] == False)])
            
            total_pred = len(above_threshold)
        else:
            tp = fp = fn = tn = total_pred = 0
        
        fps.append(fp)
        fns.append(fn)
        tps.append(tp)
        tns.append(tn)
        total_predictions.append(total_pred)
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Get percentile info for reference lines
    player_data = [item for item in data if item['predicted_role'] == 'player']
    player_confidences = [item['role_confidence'] for item in player_data]
    
    p10 = np.percentile(player_confidences, 10)
    p50 = np.percentile(player_confidences, 50)
    p90 = np.percentile(player_confidences, 90)
    
    # Plot 1: FP and FN counts vs confidence threshold
    ax1.plot(confidence_thresholds, fps, 'o-', linewidth=2, markersize=3, color='red', label='FP (False Positives)')
    ax1.plot(confidence_thresholds, fns, 's-', linewidth=2, markersize=3, color='blue', label='FN (False Negatives)')
    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Count')
    ax1.set_title('False Positives and False Negatives vs Confidence Threshold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add percentile lines
    ax1.axvline(p10, color='gray', linestyle='--', alpha=0.7, label=f'P10 ({p10:.3f})')
    ax1.axvline(p50, color='gray', linestyle='-', alpha=0.7, label=f'P50 ({p50:.3f})')
    ax1.axvline(p90, color='gray', linestyle='--', alpha=0.7, label=f'P90 ({p90:.3f})')
    ax1.legend()
    
    # Plot 2: TP and TN counts vs confidence threshold
    ax2.plot(confidence_thresholds, tps, 'o-', linewidth=2, markersize=3, color='green', label='TP (True Positives)')
    ax2.plot(confidence_thresholds, tns, 's-', linewidth=2, markersize=3, color='orange', label='TN (True Negatives)')
    ax2.set_xlabel('Confidence Threshold')
    ax2.set_ylabel('Count')
    ax2.set_title('True Positives and True Negatives vs Confidence Threshold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add percentile lines
    ax2.axvline(p10, color='gray', linestyle='--', alpha=0.7)
    ax2.axvline(p50, color='gray', linestyle='-', alpha=0.7)
    ax2.axvline(p90, color='gray', linestyle='--', alpha=0.7)
    
    # Plot 3: All confusion matrix elements together
    ax3.plot(confidence_thresholds, tps, 'o-', linewidth=2, markersize=3, color='green', label='TP (True Positives)')
    ax3.plot(confidence_thresholds, fps, 's-', linewidth=2, markersize=3, color='red', label='FP (False Positives)')
    ax3.plot(confidence_thresholds, fns, '^-', linewidth=2, markersize=3, color='blue', label='FN (False Negatives)')
    ax3.plot(confidence_thresholds, tns, 'v-', linewidth=2, markersize=3, color='orange', label='TN (True Negatives)')
    ax3.set_xlabel('Confidence Threshold')
    ax3.set_ylabel('Count')
    ax3.set_title('All Confusion Matrix Elements vs Confidence Threshold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Add percentile lines
    ax3.axvline(p10, color='gray', linestyle='--', alpha=0.7)
    ax3.axvline(p50, color='gray', linestyle='-', alpha=0.7)
    ax3.axvline(p90, color='gray', linestyle='--', alpha=0.7)
    
    # Plot 4: Total predictions above threshold
    ax4.plot(confidence_thresholds, total_predictions, 'o-', linewidth=2, markersize=3, color='purple')
    ax4.set_xlabel('Confidence Threshold')
    ax4.set_ylabel('Number of Predictions')
    ax4.set_title('Total Predictions with Confidence ≥ Threshold')
    ax4.grid(True, alpha=0.3)
    
    # Add percentile lines
    ax4.axvline(p10, color='gray', linestyle='--', alpha=0.7)
    ax4.axvline(p50, color='gray', linestyle='-', alpha=0.7)
    ax4.axvline(p90, color='gray', linestyle='--', alpha=0.7)
    
    plt.suptitle('Player Class: Confusion Matrix Elements vs Confidence Threshold\n(Predictions with confidence ≥ threshold)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confidence threshold vs TP/FP/FN/TN counts plot saved to {save_path}")

def print_confidence_statistics(data):
    """Print detailed confidence statistics"""
    print("\n" + "="*80)
    print("CONFIDENCE STATISTICS")
    print("="*80)
    
    # Overall statistics
    all_confidences = [item['role_confidence'] for item in data]
    print(f"Total predictions: {len(all_confidences)}")
    print(f"Overall confidence - Mean: {np.mean(all_confidences):.4f}, "
          f"Median: {np.median(all_confidences):.4f}, "
          f"Std: {np.std(all_confidences):.4f}")
    print(f"Confidence range: {np.min(all_confidences):.4f} - {np.max(all_confidences):.4f}")
    
    # Statistics by predicted role
    print("\nSTATISTICS BY PREDICTED ROLE:")
    print("-" * 80)
    print(f"{'Role':<12} {'Count':<8} {'Mean':<8} {'Median':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-" * 80)
    
    role_stats = defaultdict(list)
    for item in data:
        role_stats[item['predicted_role']].append(item['role_confidence'])
    
    for role in sorted(role_stats.keys()):
        confidences = role_stats[role]
        print(f"{role:<12} {len(confidences):<8} {np.mean(confidences):<8.4f} "
              f"{np.median(confidences):<8.4f} {np.std(confidences):<8.4f} "
              f"{np.min(confidences):<8.4f} {np.max(confidences):<8.4f}")
    
    # Statistics by correctness
    print("\nSTATISTICS BY PREDICTION CORRECTNESS:")
    print("-" * 80)
    
    correct_confidences = [item['role_confidence'] for item in data if item['predicted_role'] == item['true_role']]
    incorrect_confidences = [item['role_confidence'] for item in data if item['predicted_role'] != item['true_role']]
    
    print(f"Correct predictions ({len(correct_confidences)}):")
    if correct_confidences:
        print(f"  Mean: {np.mean(correct_confidences):.4f}, "
              f"Median: {np.median(correct_confidences):.4f}, "
              f"Std: {np.std(correct_confidences):.4f}")
    
    print(f"Incorrect predictions ({len(incorrect_confidences)}):")
    if incorrect_confidences:
        print(f"  Mean: {np.mean(incorrect_confidences):.4f}, "
              f"Median: {np.median(incorrect_confidences):.4f}, "
              f"Std: {np.std(incorrect_confidences):.4f}")

if __name__ == "__main__":
    # Define base directory
    base_dir = "prtreid_output"
    
    print("Starting confidence distribution analysis...")
    print("Loading all predictions with confidence scores...")
    
    # Load all predictions with confidence
    data = load_all_predictions_with_confidence(base_dir)
    
    if len(data) == 0:
        print("No predictions with confidence scores found! Check the directory structure and data format.")
        exit(1)
    
    # Print confidence statistics
    print_confidence_statistics(data)
    
    # Create confidence distribution plots
    print("\nGenerating confidence distribution plots...")
    
    # Plot confidence distributions by predicted role
    plot_confidence_distribution_by_predicted_role(data, save_path="confidence_distribution_by_role.png")
    
    # Create box plots
    df = plot_confidence_boxplot(data, save_path="confidence_boxplot.png")
    
    # Plot confidence threshold vs all confusion matrix elements for player class
    plot_confidence_vs_accuracy(data, save_path="confidence_threshold_vs_confusion_matrix.png")
    
    print("\nConfidence distribution analysis complete!")
    print("Generated plots:")
    print("- confidence_distribution_by_role.png")
    print("- confidence_boxplot.png") 
    print("- confidence_threshold_vs_confusion_matrix.png")
