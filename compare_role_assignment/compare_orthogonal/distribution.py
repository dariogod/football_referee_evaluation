import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import pandas as pd

def simplify_role(role: str) -> str:
    """Simplify role names - convert player_left/player_right to just 'player'."""
    if role in ['player_left', 'player_right']:
        return 'player'
    return role

def load_combined_dataset(file_path: str = "compare_role_assignment/compare_orthogonal/combined_role_predictions_updated.json") -> Dict:
    """Load the combined dataset from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_referee_accuracy_per_clip(dataset: Dict, exclude_clips: List[str] = None) -> Dict:
    """
    Calculate referee accuracy per clip for both models.
    
    Args:
        dataset: The combined dataset
        exclude_clips: List of clip names to exclude from analysis
    
    Returns:
        Dictionary with clip names as keys and accuracy stats as values
    """
    if exclude_clips is None:
        exclude_clips = []
    
    clip_stats = {}
    excluded_count = 0
    
    for clip_name, clip_data in dataset.items():
        # Skip excluded clips
        if clip_name in exclude_clips:
            excluded_count += 1
            continue
            
        referee_detections = []
        
        for frame_str, frame_data in clip_data.items():
            for detection in frame_data:
                # Only process referee detections
                gt_role_simplified = simplify_role(detection['gt_role'])
                if gt_role_simplified != 'referee':
                    continue
                    
                # DBSCAN
                dbscan_pred_dict = detection.get('dbscan_pred_role', {})
                dbscan_lab_pred_val = dbscan_pred_dict.get('lab')
                dbscan_lab_simplified = simplify_role(dbscan_lab_pred_val) # simplify_role handles None input gracefully
                
                # For PRTReid, handle the mapped prediction
                prtreid_mapped_simplified = None # Default to None
                prtreid_data = detection.get('prtreid_data')
                if prtreid_data is not None:
                    prtreid_mapped_pred_val = prtreid_data.get('mapped_predicted_role')
                    if prtreid_mapped_pred_val is not None:
                        prtreid_mapped_simplified = simplify_role(prtreid_mapped_pred_val)
                
                referee_detections.append({
                    'dbscan_correct': gt_role_simplified == dbscan_lab_simplified and dbscan_lab_simplified is not None,
                    'prtreid_correct': prtreid_mapped_simplified == gt_role_simplified if prtreid_mapped_simplified is not None else False
                })
        
        # Calculate accuracies for this clip
        if referee_detections:
            total_referee = len(referee_detections)
            prtreid_correct = sum(1 for det in referee_detections if det['prtreid_correct'])
            dbscan_correct = sum(1 for det in referee_detections if det['dbscan_correct'])
            
            clip_stats[clip_name] = {
                'total_referee': total_referee,
                'prtreid_correct': prtreid_correct,
                'dbscan_correct': dbscan_correct,
                'prtreid_accuracy': (prtreid_correct / total_referee) * 100,
                'dbscan_accuracy': (dbscan_correct / total_referee) * 100
            }
    
    if exclude_clips:
        print(f"Excluded {excluded_count} clips ({', '.join(exclude_clips)}) from referee accuracy analysis")
    
    return clip_stats

def create_referee_accuracy_distribution(clip_stats: Dict, output_path: str = "compare_role_assignment/compare_orthogonal/referee_accuracy_distribution.png"):
    """
    Create a bar chart showing referee accuracy per clip for both models.
    
    Args:
        clip_stats: Dictionary with clip statistics
        output_path: Path to save the plot
    """
    # Sort clips by name for consistent ordering
    sorted_clips = sorted(clip_stats.keys())
    
    # Extract data for plotting
    clip_names = []
    prtreid_accuracies = []
    dbscan_accuracies = []
    total_referees = []
    
    for clip in sorted_clips:
        stats = clip_stats[clip]
        clip_names.append(clip)
        prtreid_accuracies.append(stats['prtreid_accuracy'])
        dbscan_accuracies.append(stats['dbscan_accuracy'])
        total_referees.append(stats['total_referee'])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # Set the width of bars and positions
    bar_width = 0.35
    x_positions = np.arange(len(clip_names))
    
    # Create bars
    bars1 = ax.bar(x_positions - bar_width/2, prtreid_accuracies, bar_width, 
                   label='PRTReid', color='red', alpha=0.7)
    bars2 = ax.bar(x_positions + bar_width/2, dbscan_accuracies, bar_width,
                   label='DBSCAN (LAB)', color='blue', alpha=0.7)
    
    # Customize the plot
    ax.set_xlabel('Clips', fontweight='bold', fontsize=12)
    ax.set_ylabel('Referee Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('Referee Accuracy Distribution by Clip', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(clip_names, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)  # Set y-axis limit to 0-105%
    
    # Add value labels on bars
    def add_value_labels(bars, values, totals):
        for bar, value, total in zip(bars, values, totals):
            height = bar.get_height()
            ax.annotate(f'{value:.0f}%\n({total})',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', 
                       fontsize=8, fontweight='bold')
    
    add_value_labels(bars1, prtreid_accuracies, total_referees)
    add_value_labels(bars2, dbscan_accuracies, total_referees)
    
    # Add statistics text box
    total_clips = len(clip_names)
    avg_prtreid = np.mean(prtreid_accuracies)
    avg_dbscan = np.mean(dbscan_accuracies)
    total_referee_detections = sum(total_referees)
    
    stats_text = f"""STATISTICS:
Total Clips: {total_clips}
Total Referee Detections: {total_referee_detections:,}

Average Accuracies:
PRTReid: {avg_prtreid:.1f}%
DBSCAN: {avg_dbscan:.1f}%

Difference: {avg_prtreid - avg_dbscan:+.1f}%"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Referee accuracy distribution saved to: {output_path}")
    
    # Also display the plot
    plt.show()
    
    return {
        'total_clips': total_clips,
        'total_referee_detections': total_referee_detections,
        'avg_prtreid_accuracy': avg_prtreid,
        'avg_dbscan_accuracy': avg_dbscan,
        'accuracy_difference': avg_prtreid - avg_dbscan
    }

def print_detailed_clip_stats(clip_stats: Dict):
    """Print detailed statistics for each clip."""
    print("\n" + "="*80)
    print("DETAILED REFEREE ACCURACY BY CLIP")
    print("="*80)
    
    # Sort clips by the difference in accuracy (PRTReid - DBSCAN)
    sorted_clips = sorted(clip_stats.items(), 
                         key=lambda x: x[1]['prtreid_accuracy'] - x[1]['dbscan_accuracy'], 
                         reverse=True)
    
    print(f"{'Clip':<15} {'Referees':<8} {'PRTReid':<10} {'DBSCAN':<10} {'Difference':<12} {'Winner'}")
    print("-" * 80)
    
    for clip, stats in sorted_clips:
        diff = stats['prtreid_accuracy'] - stats['dbscan_accuracy']
        winner = "PRTReid" if diff > 0 else "DBSCAN" if diff < 0 else "Tie"
        
        print(f"{clip:<15} {stats['total_referee']:<8} {stats['prtreid_accuracy']:<10.1f} "
              f"{stats['dbscan_accuracy']:<10.1f} {diff:<12.1f} {winner}")

def calculate_statistical_measures(clip_stats: Dict) -> Dict:
    """
    Calculate mean, median, and variance for both methods.
    
    Args:
        clip_stats: Dictionary with clip statistics
    
    Returns:
        Dictionary with statistical measures for both methods
    """
    # Extract accuracy arrays
    prtreid_accuracies = [stats['prtreid_accuracy'] for stats in clip_stats.values()]
    dbscan_accuracies = [stats['dbscan_accuracy'] for stats in clip_stats.values()]
    
    # Calculate statistics for PRTReid
    prtreid_mean = np.mean(prtreid_accuracies)
    prtreid_median = np.median(prtreid_accuracies)
    prtreid_variance = np.var(prtreid_accuracies, ddof=1)  # Sample variance
    prtreid_std = np.std(prtreid_accuracies, ddof=1)  # Sample standard deviation
    
    # Calculate statistics for DBSCAN
    dbscan_mean = np.mean(dbscan_accuracies)
    dbscan_median = np.median(dbscan_accuracies)
    dbscan_variance = np.var(dbscan_accuracies, ddof=1)  # Sample variance
    dbscan_std = np.std(dbscan_accuracies, ddof=1)  # Sample standard deviation
    
    return {
        'prtreid': {
            'mean': prtreid_mean,
            'median': prtreid_median,
            'variance': prtreid_variance,
            'std_dev': prtreid_std,
            'min': np.min(prtreid_accuracies),
            'max': np.max(prtreid_accuracies)
        },
        'dbscan': {
            'mean': dbscan_mean,
            'median': dbscan_median,
            'variance': dbscan_variance,
            'std_dev': dbscan_std,
            'min': np.min(dbscan_accuracies),
            'max': np.max(dbscan_accuracies)
        },
        'differences': {
            'mean_diff': prtreid_mean - dbscan_mean,
            'median_diff': prtreid_median - dbscan_median,
            'variance_diff': prtreid_variance - dbscan_variance
        }
    }

def print_statistical_measures(stats: Dict):
    """Print detailed statistical measures in a formatted table."""
    print("\n" + "="*80)
    print("STATISTICAL MEASURES FOR REFEREE ACCURACY")
    print("="*80)
    
    print(f"{'Measure':<15} {'PRTReid':<12} {'DBSCAN':<12} {'Difference':<12}")
    print("-" * 80)
    
    prtreid = stats['prtreid']
    dbscan = stats['dbscan']
    diff = stats['differences']
    
    print(f"{'Mean':<15} {prtreid['mean']:<12.2f} {dbscan['mean']:<12.2f} {diff['mean_diff']:<12.2f}")
    print(f"{'Median':<15} {prtreid['median']:<12.2f} {dbscan['median']:<12.2f} {diff['median_diff']:<12.2f}")
    print(f"{'Variance':<15} {prtreid['variance']:<12.2f} {dbscan['variance']:<12.2f} {diff['variance_diff']:<12.2f}")
    print(f"{'Std Dev':<15} {prtreid['std_dev']:<12.2f} {dbscan['std_dev']:<12.2f} {prtreid['std_dev'] - dbscan['std_dev']:<12.2f}")
    print(f"{'Min':<15} {prtreid['min']:<12.2f} {dbscan['min']:<12.2f} {prtreid['min'] - dbscan['min']:<12.2f}")
    print(f"{'Max':<15} {prtreid['max']:<12.2f} {dbscan['max']:<12.2f} {prtreid['max'] - dbscan['max']:<12.2f}")

def main():
    """Main function to create referee accuracy distribution."""
    # Define clips to exclude
    exclude_clips = ['SNGS-125', 'SNGS-190', 'SNGS-146']
    
    print("Loading combined dataset...")
    dataset = load_combined_dataset()
    
    print("Calculating referee accuracy per clip...")
    clip_stats = calculate_referee_accuracy_per_clip(dataset, exclude_clips=exclude_clips)
    
    print("Creating referee accuracy distribution plot...")
    summary_stats = create_referee_accuracy_distribution(clip_stats)
    
    print("Calculating statistical measures...")
    statistical_measures = calculate_statistical_measures(clip_stats)
    
    print("Printing detailed statistics...")
    print_detailed_clip_stats(clip_stats)
    
    # Print statistical measures
    print_statistical_measures(statistical_measures)
    
    print(f"\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total clips analyzed: {summary_stats['total_clips']}")
    print(f"Total referee detections: {summary_stats['total_referee_detections']:,}")
    print(f"Average PRTReid accuracy: {summary_stats['avg_prtreid_accuracy']:.1f}%")
    print(f"Average DBSCAN accuracy: {summary_stats['avg_dbscan_accuracy']:.1f}%")
    print(f"Average difference (PRTReid - DBSCAN): {summary_stats['accuracy_difference']:+.1f}%")
    
    # Save detailed stats to JSON
    output_file = "compare_role_assignment/compare_orthogonal/referee_accuracy_by_clip.json"
    with open(output_file, 'w') as f:
        json.dump({
            'clip_statistics': clip_stats,
            'summary': summary_stats,
            'statistical_measures': statistical_measures,
            'excluded_clips': exclude_clips
        }, f, indent=2)
    print(f"\nDetailed statistics saved to: {output_file}")

if __name__ == "__main__":
    main()
