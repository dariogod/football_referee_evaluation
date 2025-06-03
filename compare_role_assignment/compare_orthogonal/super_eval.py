import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib_venn import venn2, venn2_circles
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import sys
from contextlib import redirect_stdout

OUTPUT_BASE_DIR = Path("compare_role_assignment/compare_orthogonal/")
OUTPUT_SUBFOLDER = "super_eval"
SUPER_EVAL_DIR = OUTPUT_BASE_DIR / OUTPUT_SUBFOLDER

# Placeholder for missing predictions in sklearn metrics
NO_PREDICTION_PLACEHOLDER = '_NO_PRED_'

# --- Helper Functions (from existing scripts) ---

def simplify_role(role: str) -> str:
    """Simplify role names - convert player_left/player_right to just 'player'. Handles None input."""
    if role is None:
        return None
    if role in ['player_left', 'player_right']:
        return 'player'
    return role

def load_combined_dataset(file_path: str = "compare_role_assignment/compare_orthogonal/combined_role_predictions_updated.json") -> Dict:
    """Load the combined dataset from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

# --- Unified Data Extraction ---

def extract_all_detection_data(dataset: Dict, exclude_clips: List[str] = None) -> List[Dict]:
    """
    Extracts and processes all detection data from the raw dataset.
    This is the single source of truth for all subsequent analyses.
    """
    if exclude_clips is None:
        exclude_clips = []
    
    all_processed_detections = []
    detection_id_counter = 0
    excluded_detection_count = 0
    
    for clip_name, clip_data in dataset.items():
        if clip_name in exclude_clips:
            excluded_detection_count += sum(len(frame_data) for frame_data in clip_data.values())
            continue
            
        for frame_str, frame_detections in clip_data.items():
            for raw_detection in frame_detections:
                gt_role_simplified = simplify_role(raw_detection['gt_role'])
                
                # DBSCAN predictions
                dbscan_preds = raw_detection.get('dbscan_pred_role', {})
                
                dbscan_lab_raw = dbscan_preds.get('lab')
                dbscan_lab_simplified = simplify_role(dbscan_lab_raw)
                
                dbscan_rgb_raw = dbscan_preds.get('rgb')
                dbscan_rgb_simplified = simplify_role(dbscan_rgb_raw)
                
                dbscan_hsv_raw = dbscan_preds.get('hsv')
                dbscan_hsv_simplified = simplify_role(dbscan_hsv_raw)

                # PRTReid mapped prediction
                prtreid_mapped_simplified = None
                prtreid_data = raw_detection.get('prtreid_data')
                if prtreid_data is not None:
                    prtreid_mapped_raw = prtreid_data.get('mapped_predicted_role')
                    prtreid_mapped_simplified = simplify_role(prtreid_mapped_raw)

                processed_detection = {
                    'id': detection_id_counter,
                    'clip': clip_name,
                    'frame': raw_detection['frame'],
                    'track_id': raw_detection['track_id'],
                    'gt_role': gt_role_simplified,
                    
                    'dbscan_lab_pred': dbscan_lab_simplified,
                    'dbscan_lab_correct': gt_role_simplified == dbscan_lab_simplified and dbscan_lab_simplified is not None,
                    
                    'dbscan_rgb_pred': dbscan_rgb_simplified,
                    'dbscan_rgb_correct': gt_role_simplified == dbscan_rgb_simplified and dbscan_rgb_simplified is not None,

                    'dbscan_hsv_pred': dbscan_hsv_simplified,
                    'dbscan_hsv_correct': gt_role_simplified == dbscan_hsv_simplified and dbscan_hsv_simplified is not None,
                    
                    'prtreid_pred': prtreid_mapped_simplified,
                    'prtreid_correct': gt_role_simplified == prtreid_mapped_simplified and prtreid_mapped_simplified is not None,
                }
                all_processed_detections.append(processed_detection)
                detection_id_counter += 1
                
    if exclude_clips:
        print(f"Unified data extraction: Excluded {len(exclude_clips)} clips ({', '.join(exclude_clips)}) with {excluded_detection_count:,} detections.")
    
    print(f"Unified data extraction: Processed {len(all_processed_detections):,} detections from {len(dataset) - len(exclude_clips)} clips.")
    return all_processed_detections

# --- Functions from distribution.py (Referee Accuracy Distribution) ---

def calculate_referee_accuracy_per_clip_unified(all_detections_data: List[Dict]) -> Dict:
    """Calculate referee accuracy per clip for PRTReid and DBSCAN LAB from unified data."""
    clip_stats = defaultdict(lambda: {'total_referee': 0, 'prtreid_correct_count': 0, 'dbscan_lab_correct_count': 0})
    
    for det in all_detections_data:
        if det['gt_role'] == 'referee':
            clip_name = det['clip']
            clip_stats[clip_name]['total_referee'] += 1
            if det['prtreid_correct']:
                clip_stats[clip_name]['prtreid_correct_count'] += 1
            if det['dbscan_lab_correct']: # Specifically DBSCAN LAB for this comparison
                clip_stats[clip_name]['dbscan_lab_correct_count'] += 1
                
    # Calculate accuracies
    final_clip_stats = {}
    for clip_name, counts in clip_stats.items():
        total_referee = counts['total_referee']
        if total_referee > 0:
            final_clip_stats[clip_name] = {
                'total_referee': total_referee,
                'prtreid_correct': counts['prtreid_correct_count'],
                'dbscan_correct': counts['dbscan_lab_correct_count'], # Renaming for consistency with original function
                'prtreid_accuracy': (counts['prtreid_correct_count'] / total_referee) * 100,
                'dbscan_accuracy': (counts['dbscan_lab_correct_count'] / total_referee) * 100
            }
    return final_clip_stats

def create_referee_accuracy_distribution_unified(clip_stats: Dict, output_dir: Path):
    output_path = output_dir / "referee_accuracy_distribution.png"
    sorted_clips = sorted(clip_stats.keys())
    
    clip_names = [clip for clip in sorted_clips if clip_stats[clip]['total_referee'] > 0] # Ensure non-empty
    if not clip_names:
        print("No clips with referee detections found for distribution plot.")
        return {}

    prtreid_accuracies = [clip_stats[clip]['prtreid_accuracy'] for clip in clip_names]
    dbscan_accuracies = [clip_stats[clip]['dbscan_accuracy'] for clip in clip_names]
    total_referees = [clip_stats[clip]['total_referee'] for clip in clip_names]
        
    fig, ax = plt.subplots(figsize=(max(20, len(clip_names)*0.5), 8)) # Dynamic width
    bar_width = 0.35
    x_positions = np.arange(len(clip_names))
    
    bars1 = ax.bar(x_positions - bar_width/2, prtreid_accuracies, bar_width, label='PRTReid', color='red', alpha=0.7)
    bars2 = ax.bar(x_positions + bar_width/2, dbscan_accuracies, bar_width, label='DBSCAN (LAB)', color='blue', alpha=0.7)
    
    ax.set_xlabel('Clips', fontweight='bold', fontsize=12)
    ax.set_ylabel('Referee Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('Referee Accuracy Distribution by Clip (PRTReid vs DBSCAN LAB)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(clip_names, rotation=45, ha='right', fontsize=max(6, 10 - len(clip_names)//10)) # Dynamic fontsize
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)
    
    def add_value_labels(bars, values, totals):
        for bar, value, total in zip(bars, values, totals):
            height = bar.get_height()
            ax.annotate(f'{value:.0f}%\n({total})', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    add_value_labels(bars1, prtreid_accuracies, total_referees)
    add_value_labels(bars2, dbscan_accuracies, total_referees)
    
    total_clips_plot = len(clip_names)
    avg_prtreid = np.mean(prtreid_accuracies) if prtreid_accuracies else 0
    avg_dbscan = np.mean(dbscan_accuracies) if dbscan_accuracies else 0
    total_referee_detections_plot = sum(total_referees)
    
    stats_text = f"""STATISTICS:
Total Clips in Plot: {total_clips_plot}
Total Referee Detections in Plot: {total_referee_detections_plot:,}

Average Accuracies (Plot):
PRTReid: {avg_prtreid:.1f}%
DBSCAN (LAB): {avg_dbscan:.1f}%

Difference: {avg_prtreid - avg_dbscan:+.1f}%"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Referee accuracy distribution saved to: {output_path}")
    plt.close(fig)
    
    return {
        'total_clips_in_plot': total_clips_plot,
        'total_referee_detections_in_plot': total_referee_detections_plot,
        'avg_prtreid_accuracy_plot': avg_prtreid,
        'avg_dbscan_accuracy_plot': avg_dbscan,
        'accuracy_difference_plot': avg_prtreid - avg_dbscan
    }

def print_detailed_referee_clip_stats_unified(clip_stats: Dict):
    print("\n" + "="*80)
    print("DETAILED REFEREE ACCURACY BY CLIP (PRTReid vs DBSCAN LAB)")
    print("="*80)
    
    if not clip_stats:
        print("No clip statistics available for detailed referee stats.")
        return

    sorted_clips = sorted(clip_stats.items(), 
                         key=lambda x: (x[1]['prtreid_accuracy'] - x[1]['dbscan_accuracy']) if x[1]['total_referee'] > 0 else -float('inf'), 
                         reverse=True)
    
    print(f"{'Clip':<15} {'Referees':<8} {'PRTReid':<10} {'DBSCAN':<10} {'Difference':<12} {'Winner'}")
    print("-" * 80)
    
    for clip, stats in sorted_clips:
        if stats['total_referee'] == 0: continue
        diff = stats['prtreid_accuracy'] - stats['dbscan_accuracy']
        winner = "PRTReid" if diff > 0 else "DBSCAN" if diff < 0 else "Tie"
        print(f"{clip:<15} {stats['total_referee']:<8} {stats['prtreid_accuracy']:<10.1f} "
              f"{stats['dbscan_accuracy']:<10.1f} {diff:<12.1f} {winner}")

def calculate_referee_statistical_measures_unified(clip_stats: Dict) -> Dict:
    if not clip_stats: return {}
    prtreid_accuracies = [stats['prtreid_accuracy'] for stats in clip_stats.values() if stats['total_referee'] > 0]
    dbscan_accuracies = [stats['dbscan_accuracy'] for stats in clip_stats.values() if stats['total_referee'] > 0]
    
    if not prtreid_accuracies or not dbscan_accuracies: # Ensure there's data to process
        return {
            'prtreid': {'mean': 0, 'median': 0, 'variance': 0, 'std_dev': 0, 'min': 0, 'max': 0},
            'dbscan': {'mean': 0, 'median': 0, 'variance': 0, 'std_dev': 0, 'min': 0, 'max': 0},
            'differences': {'mean_diff': 0, 'median_diff': 0, 'variance_diff': 0}
        }

    return {
        'prtreid': {
            'mean': np.mean(prtreid_accuracies), 'median': np.median(prtreid_accuracies),
            'variance': np.var(prtreid_accuracies, ddof=1 if len(prtreid_accuracies)>1 else 0),
            'std_dev': np.std(prtreid_accuracies, ddof=1 if len(prtreid_accuracies)>1 else 0),
            'min': np.min(prtreid_accuracies), 'max': np.max(prtreid_accuracies)
        },
        'dbscan': {
            'mean': np.mean(dbscan_accuracies), 'median': np.median(dbscan_accuracies),
            'variance': np.var(dbscan_accuracies, ddof=1 if len(dbscan_accuracies)>1 else 0),
            'std_dev': np.std(dbscan_accuracies, ddof=1 if len(dbscan_accuracies)>1 else 0),
            'min': np.min(dbscan_accuracies), 'max': np.max(dbscan_accuracies)
        },
        'differences': {
            'mean_diff': np.mean(prtreid_accuracies) - np.mean(dbscan_accuracies),
            'median_diff': np.median(prtreid_accuracies) - np.median(dbscan_accuracies),
            'variance_diff': np.var(prtreid_accuracies, ddof=1 if len(prtreid_accuracies)>1 else 0) - np.var(dbscan_accuracies, ddof=1 if len(dbscan_accuracies)>1 else 0)
        }
    }

def print_referee_statistical_measures_unified(stats: Dict):
    print("\n" + "="*80)
    print("STATISTICAL MEASURES FOR REFEREE ACCURACY (PRTReid vs DBSCAN LAB)")
    print("="*80)
    if not stats or 'prtreid' not in stats:
        print("No statistical measures available for referees.")
        return

    print(f"{'Measure':<15} {'PRTReid':<12} {'DBSCAN':<12} {'Difference':<12}")
    print("-" * 80)
    prtreid = stats['prtreid']; dbscan = stats['dbscan']; diff = stats['differences']
    print(f"{'Mean':<15} {prtreid['mean']:<12.2f} {dbscan['mean']:<12.2f} {diff['mean_diff']:<12.2f}")
    print(f"{'Median':<15} {prtreid['median']:<12.2f} {dbscan['median']:<12.2f} {diff['median_diff']:<12.2f}")
    print(f"{'Variance':<15} {prtreid['variance']:<12.2f} {dbscan['variance']:<12.2f} {diff['variance_diff']:<12.2f}")
    print(f"{'Std Dev':<15} {prtreid['std_dev']:<12.2f} {dbscan['std_dev']:<12.2f} {prtreid['std_dev'] - dbscan['std_dev']:<12.2f}")
    print(f"{'Min':<15} {prtreid['min']:<12.2f} {dbscan['min']:<12.2f} {prtreid['min'] - dbscan['min']:<12.2f}")
    print(f"{'Max':<15} {prtreid['max']:<12.2f} {dbscan['max']:<12.2f} {prtreid['max'] - dbscan['max']:<12.2f}")

# --- Functions from metrics.py (Sklearn-based Metrics) ---

def prepare_sklearn_data(all_detections_data: List[Dict]) -> Dict[str, Dict[str, List[str]]]:
    """Prepare data for sklearn: y_true and y_pred for each method."""
    sklearn_data = {
        'prtreid': {'gt': [], 'pred': []},
        'dbscan_lab': {'gt': [], 'pred': []},
        'dbscan_rgb': {'gt': [], 'pred': []},
        'dbscan_hsv': {'gt': [], 'pred': []}
    }
    for det in all_detections_data:
        gt = det['gt_role'] if det['gt_role'] is not None else NO_PREDICTION_PLACEHOLDER
        
        sklearn_data['prtreid']['gt'].append(gt)
        sklearn_data['prtreid']['pred'].append(det['prtreid_pred'] if det['prtreid_pred'] is not None else NO_PREDICTION_PLACEHOLDER)
        
        sklearn_data['dbscan_lab']['gt'].append(gt)
        sklearn_data['dbscan_lab']['pred'].append(det['dbscan_lab_pred'] if det['dbscan_lab_pred'] is not None else NO_PREDICTION_PLACEHOLDER)

        sklearn_data['dbscan_rgb']['gt'].append(gt)
        sklearn_data['dbscan_rgb']['pred'].append(det['dbscan_rgb_pred'] if det['dbscan_rgb_pred'] is not None else NO_PREDICTION_PLACEHOLDER)

        sklearn_data['dbscan_hsv']['gt'].append(gt)
        sklearn_data['dbscan_hsv']['pred'].append(det['dbscan_hsv_pred'] if det['dbscan_hsv_pred'] is not None else NO_PREDICTION_PLACEHOLDER)
    return sklearn_data

def calculate_metrics_for_class_sklearn(y_true, y_pred, target_class):
    y_true_binary = [1 if label == target_class else 0 for label in y_true]
    y_pred_binary = [1 if label == target_class else 0 for label in y_pred]
    
    # Handle cases with no positive instances in true or pred for the target class
    if sum(y_true_binary) == 0 and sum(y_pred_binary) == 0: # No true positives, no predicted positives
        tn = len(y_true_binary)
        tp, fp, fn = 0,0,0
    elif sum(y_true_binary) > 0 or sum(y_pred_binary) > 0 : # Ensure there's something to compute on
        cm_result = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
        if len(cm_result) == 4: # Standard case
            tn, fp, fn, tp = cm_result
        elif len(cm_result) == 1: # All samples fall into one category (e.g. all TN or all TP)
            if y_true_binary[0] == 0 and y_pred_binary[0] == 0 : # All TN
                tn, fp, fn, tp = cm_result[0],0,0,0
            elif y_true_binary[0] == 1 and y_pred_binary[0] == 1: # All TP
                tn, fp, fn, tp = 0,0,0,cm_result[0]
            elif y_true_binary[0] == 1 and y_pred_binary[0] == 0: # All FN
                 tn, fp, fn, tp = 0,0,cm_result[0],0
            elif y_true_binary[0] == 0 and y_pred_binary[0] == 1: # ALL FP
                 tn, fp, fn, tp = 0,cm_result[0],0,0
            else: # Should not happen
                 tn, fp, fn, tp = 0,0,0,0 
        else: # Should not happen with labels=[0,1]
             tn, fp, fn, tp = 0,0,0,0 
    else: # Should not happen
        tn, fp, fn, tp = 0,0,0,0


    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn), 
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1
    }

def print_metrics_table_sklearn(method_name, metrics_by_class):
    print(f"\n{method_name.upper()} SKLEARN METRICS:")
    print("="*90)
    print(f"{'Class':<15} {'TP':<6} {'FP':<6} {'FN':<6} {'TN':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<8} {'F1':<8}")
    print("-"*90)
    for class_name, metrics in metrics_by_class.items():
        print(f"{class_name:<15} {metrics['tp']:<6} {metrics['fp']:<6} {metrics['fn']:<6} {metrics['tn']:<10} "
              f"{metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<8.4f} {metrics['f1']:<8.4f}")

def print_confusion_matrix_sklearn(method_name, y_true, y_pred, classes_to_show):
    unique_labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=classes_to_show) # Use defined classes to ensure consistent matrix size
    
    print(f"\n{method_name.upper()} CONFUSION MATRIX (Overall):")
    print("="* (20 + len(classes_to_show) * 12))
    header = "Actual\Predicted" + "".join(f"{label:>12}" for label in classes_to_show)
    print(header)
    print("-" * len(header))
    for i, true_label in enumerate(classes_to_show):
        row_str = f"{true_label:<15}"
        for j, _ in enumerate(classes_to_show):
            row_str += f"{cm[i][j]:>12}"
        print(row_str)
    print("-" * len(header))
    
    overall_accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
    print(f"Overall Accuracy (for displayed classes): {overall_accuracy:.4f}")

# --- Functions from orthogonal_comparison.py (Venn Diagrams, Role/Clip Analysis for PRTReid vs DBSCAN LAB) ---

def get_correct_sets_for_venn(all_detections_data: List[Dict]) -> Tuple[Set[int], Set[int]]:
    prtreid_correct_ids = set()
    dbscan_lab_correct_ids = set()
    for det in all_detections_data:
        if det['prtreid_correct']:
            prtreid_correct_ids.add(det['id'])
        if det['dbscan_lab_correct']:
            dbscan_lab_correct_ids.add(det['id'])
    return prtreid_correct_ids, dbscan_lab_correct_ids

def create_venn_diagram_unified(prtreid_correct_ids: Set[int], dbscan_lab_correct_ids: Set[int], 
                                total_detections: int, output_dir: Path, title_suffix: str = "", filename_suffix: str = ""):
    output_path = output_dir / f"venn_diagram_prtvdbscanlab{filename_suffix}.png"
    
    both_correct = prtreid_correct_ids.intersection(dbscan_lab_correct_ids)
    prtreid_only = prtreid_correct_ids - dbscan_lab_correct_ids
    dbscan_only = dbscan_lab_correct_ids - prtreid_correct_ids
    neither_correct = total_detections - len(prtreid_correct_ids.union(dbscan_lab_correct_ids)) # Detections where neither of these two were correct
    
    plt.figure(figsize=(12, 7))
    plt.subplot(1, 2, 1)
    venn = venn2(subsets=(len(prtreid_only), len(dbscan_only), len(both_correct)), 
                 set_labels=('PRTReid Correct', 'DBSCAN (LAB) Correct'))
    if venn.get_patch_by_id('10'): venn.get_patch_by_id('10').set_facecolor('#ff9999'); venn.get_patch_by_id('10').set_alpha(0.7)
    if venn.get_patch_by_id('01'): venn.get_patch_by_id('01').set_facecolor('#9999ff'); venn.get_patch_by_id('01').set_alpha(0.7)
    if venn.get_patch_by_id('11'): venn.get_patch_by_id('11').set_facecolor('#99ff99'); venn.get_patch_by_id('11').set_alpha(0.7)
    venn2_circles(subsets=(len(prtreid_only), len(dbscan_only), len(both_correct)))
    plt.title(f'Model Correctness: PRTReid vs DBSCAN (LAB){title_suffix}\n(Total Detections Considered: {total_detections:,})', fontsize=12, fontweight='bold')
    
    plt.subplot(1, 2, 2)
    plt.axis('off')
    pr_acc = (len(prtreid_correct_ids) / total_detections * 100) if total_detections > 0 else 0
    db_acc = (len(dbscan_lab_correct_ids) / total_detections * 100) if total_detections > 0 else 0
    stats_text = f"""COMPARISON STATS ({title_suffix.strip()}):
Total Detections: {total_detections:,}
PRTReid Correct: {len(prtreid_correct_ids):,} ({pr_acc:.1f}%)
DBSCAN (LAB) Correct: {len(dbscan_lab_correct_ids):,} ({db_acc:.1f}%)
Both Correct: {len(both_correct):,} ({(len(both_correct)/total_detections*100) if total_detections > 0 else 0:.1f}%)
PRTReid Only: {len(prtreid_only):,} ({(len(prtreid_only)/total_detections*100) if total_detections > 0 else 0:.1f}%)
DBSCAN Only: {len(dbscan_only):,} ({(len(dbscan_only)/total_detections*100) if total_detections > 0 else 0:.1f}%)
Neither Correct (of these two): {neither_correct:,} ({(neither_correct/total_detections*100) if total_detections > 0 else 0:.1f}%)
Agreement (Both Correct or Both Incorrect): {(len(both_correct) + neither_correct) / total_detections * 100 if total_detections > 0 else 0:.1f}%
"""
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', 
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Venn diagram saved to: {output_path}")
    plt.close()
    return {
        'prtreid_only': len(prtreid_only), 'dbscan_only': len(dbscan_only), 
        'both_correct': len(both_correct), 'neither_correct_of_two': neither_correct,
        'prtreid_accuracy': pr_acc, 'dbscan_accuracy': db_acc, 'total_detections_in_venn': total_detections
    }

def analyze_by_role_unified(all_detections_data: List[Dict]) -> Dict:
    role_analysis = defaultdict(lambda: {'total': 0, 'prtreid_correct': 0, 'dbscan_lab_correct': 0, 'both_correct': 0})
    roles = sorted(list(set(det['gt_role'] for det in all_detections_data if det['gt_role'] is not None)))

    for det in all_detections_data:
        role = det['gt_role']
        if role is None: continue
        role_analysis[role]['total'] += 1
        if det['prtreid_correct']: role_analysis[role]['prtreid_correct'] += 1
        if det['dbscan_lab_correct']: role_analysis[role]['dbscan_lab_correct'] += 1
        if det['prtreid_correct'] and det['dbscan_lab_correct']: role_analysis[role]['both_correct'] += 1
            
    final_role_analysis = {}
    for role, counts in role_analysis.items():
        total = counts['total']
        final_role_analysis[role] = {
            'total': total,
            'prtreid_correct': counts['prtreid_correct'],
            'dbscan_lab_correct': counts['dbscan_lab_correct'],
            'both_correct': counts['both_correct'],
            'prtreid_accuracy': (counts['prtreid_correct'] / total * 100) if total > 0 else 0,
            'dbscan_lab_accuracy': (counts['dbscan_lab_correct'] / total * 100) if total > 0 else 0
        }
    return final_role_analysis

def create_role_comparison_plot_unified(role_analysis: Dict, output_dir: Path):
    output_path = output_dir / "role_comparison_prtvdbscanlab.png"
    roles = [r for r in role_analysis if role_analysis[r]['total'] > 0]
    if not roles: print("No roles with data for comparison plot."); return

    prtreid_accuracies = [role_analysis[role]['prtreid_accuracy'] for role in roles]
    dbscan_accuracies = [role_analysis[role]['dbscan_lab_accuracy'] for role in roles]
    
    x = np.arange(len(roles)); width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(roles)*1.5), 6))
    bars1 = ax.bar(x - width/2, prtreid_accuracies, width, label='PRTReid', color='#ff9999', alpha=0.8)
    bars2 = ax.bar(x + width/2, dbscan_accuracies, width, label='DBSCAN (LAB)', color='#9999ff', alpha=0.8)
    
    ax.set_xlabel('Roles', fontweight='bold'); ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Model Accuracy by Role (PRTReid vs DBSCAN LAB)', fontsize=14, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(roles, rotation=45, ha="right")
    ax.legend(); ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim(0, 105)

    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    autolabel(bars1); autolabel(bars2)
    plt.tight_layout(); plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Role comparison plot saved to: {output_path}"); plt.close(fig)

def create_role_specific_venn_diagrams_unified(all_detections_data: List[Dict], output_dir: Path) -> Dict:
    role_venn_stats = {}
    roles = sorted(list(set(det['gt_role'] for det in all_detections_data if det['gt_role'] is not None)))

    for role in roles:
        role_specific_detections = [det for det in all_detections_data if det['gt_role'] == role]
        if not role_specific_detections: continue

        prt_correct_ids, db_correct_ids = get_correct_sets_for_venn(role_specific_detections)
        stats = create_venn_diagram_unified(prt_correct_ids, db_correct_ids, len(role_specific_detections), 
                                            output_dir, title_suffix=f" - {role.upper()}", 
                                            filename_suffix=f"_{role.lower().replace(' ', '_')}")
        role_venn_stats[role] = stats
    return role_venn_stats

def analyze_by_clip_and_role_unified(all_detections_data: List[Dict], target_role: str = None) -> Dict:
    """Analyzes PRTReid vs DBSCAN LAB performance by clip, optionally for a specific role."""
    clip_analysis = defaultdict(lambda: {
        'total': 0, 'prtreid_correct': 0, 'dbscan_lab_correct': 0, 'both_correct': 0, 'both_wrong':0,
        'prtreid_only_correct':0, 'dbscan_lab_only_correct':0, 'role': target_role if target_role else 'all'
    })
    
    for det in all_detections_data:
        if target_role and det['gt_role'] != target_role:
            continue
        
        clip = det['clip']
        clip_analysis[clip]['total'] += 1
        if det['prtreid_correct']: clip_analysis[clip]['prtreid_correct'] += 1
        if det['dbscan_lab_correct']: clip_analysis[clip]['dbscan_lab_correct'] += 1
        if det['prtreid_correct'] and det['dbscan_lab_correct']: clip_analysis[clip]['both_correct'] += 1
        if not det['prtreid_correct'] and not det['dbscan_lab_correct']: clip_analysis[clip]['both_wrong'] +=1

    final_clip_analysis = {}
    for clip, counts in clip_analysis.items():
        total = counts['total']
        if total == 0: continue
        prt_acc = (counts['prtreid_correct'] / total * 100)
        db_acc = (counts['dbscan_lab_correct'] / total * 100)
        final_clip_analysis[clip] = {
            **counts,
            'prtreid_accuracy': prt_acc,
            'dbscan_lab_accuracy': db_acc,
            'accuracy_difference': prt_acc - db_acc, # PRT - DBSCAN
            'prtreid_only_correct': counts['prtreid_correct'] - counts['both_correct'],
            'dbscan_lab_only_correct': counts['dbscan_lab_correct'] - counts['both_correct']
        }
    return final_clip_analysis

def find_clips_with_significant_differences_unified(clip_analysis: Dict, threshold: float = 10.0) -> Dict:
    significant_clips = {'prtreid_better': [], 'dbscan_better': [], 'similar_performance': []}
    for clip, stats in clip_analysis.items():
        item = {'clip': clip, **stats}
        if abs(stats['accuracy_difference']) >= threshold:
            if stats['accuracy_difference'] > 0: significant_clips['prtreid_better'].append(item)
            else: significant_clips['dbscan_better'].append(item)
        else:
            significant_clips['similar_performance'].append(item)
    significant_clips['prtreid_better'].sort(key=lambda x: x['accuracy_difference'], reverse=True)
    significant_clips['dbscan_better'].sort(key=lambda x: abs(x['accuracy_difference']), reverse=True)
    return significant_clips

def save_analysis_json(data: Any, filename: str, output_dir: Path):
    output_path = output_dir / filename
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Analysis saved to: {output_path}")

# --- Main Execution Logic ---
def main_super_eval():
    SUPER_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define clips to exclude (consistent for all analyses in this script)
    # Using the most restrictive set from original scripts for this example.
    # This can be made a parameter or configured as needed.
    exclude_clips_list = [] 
    print(f"Starting Super Evaluation. Excluded clips: {exclude_clips_list}")
    print(f"All outputs will be saved to: {SUPER_EVAL_DIR.resolve()}")

    print("\nLoading combined dataset...")
    raw_dataset = load_combined_dataset()
    
    print("\nExtracting and processing all detection data...")
    all_detections = extract_all_detection_data(raw_dataset, exclude_clips=exclude_clips_list)
    save_analysis_json(all_detections, "all_processed_detections.json", SUPER_EVAL_DIR)

    # --- 1. Referee Accuracy Distribution Analysis (from distribution.py) ---
    print("\n" + "="*50); print("PART 1: REFEREE ACCURACY DISTRIBUTION ANALYSIS"); print("="*50)
    referee_clip_stats = calculate_referee_accuracy_per_clip_unified(all_detections)
    if referee_clip_stats:
        dist_summary = create_referee_accuracy_distribution_unified(referee_clip_stats, SUPER_EVAL_DIR)
        print_detailed_referee_clip_stats_unified(referee_clip_stats)
        referee_stat_measures = calculate_referee_statistical_measures_unified(referee_clip_stats)
        print_referee_statistical_measures_unified(referee_stat_measures)
        
        referee_analysis_summary = {
            'clip_statistics': referee_clip_stats,
            'plot_summary': dist_summary,
            'overall_statistical_measures': referee_stat_measures,
            'excluded_clips': exclude_clips_list
        }
        save_analysis_json(referee_analysis_summary, "referee_focused_distribution_analysis.json", SUPER_EVAL_DIR)
    else:
        print("No referee detections found after filtering; skipping referee distribution analysis.")

    # --- 2. Sklearn Metrics Analysis (from metrics.py) ---
    print("\n" + "="*50); print("PART 2: SKLEARN-BASED METRICS ANALYSIS"); print("="*50)
    sklearn_input_data = prepare_sklearn_data(all_detections)
    # Define classes for sklearn metrics (consistent order)
    # Include NO_PREDICTION_PLACEHOLDER if it might appear and you want to see it in matrix
    gt_labels_for_classes = set(det['gt_role'] for det in all_detections if det['gt_role'] is not None)
    defined_classes = sorted(list(gt_labels_for_classes.union({'player', 'goalkeeper', 'referee'}))) # Ensure common roles are present
    
    metrics_output_file = SUPER_EVAL_DIR / "sklearn_metrics_report.txt"
    with open(metrics_output_file, 'w') as f_metrics:
        with redirect_stdout(f_metrics): # Redirect print statements for this section
            print("FOOTBALL REFEREE EVALUATION - SUPER EVALUATION SCRIPT")
            print(f"SKLEARN-BASED ROLE CLASSIFICATION METRICS (Overall for {len(all_detections)} detections)")
            print(f"Excluded clips: {', '.join(exclude_clips_list) if exclude_clips_list else 'None'}")
            print(f"Classes considered for matrix: {', '.join(defined_classes)}")
            print(f"'{NO_PREDICTION_PLACEHOLDER}' used for missing/None predictions.")

            all_sklearn_metrics = {}
            for method_name, method_data in sklearn_input_data.items():
                if not method_data['gt']:
                    print(f"\nNo data available for {method_name}")
                    continue
                
                print(f"\nTotal predictions for {method_name}: {len(method_data['gt'])}")
                print_confusion_matrix_sklearn(method_name, method_data['gt'], method_data['pred'], defined_classes)
                
                metrics_by_class = {}
                for class_name in defined_classes: # Iterate over defined classes
                    # Filter y_true and y_pred to only include instances where true label OR predicted label is class_name or NO_PREDICTION_PLACEHOLDER
                    # This makes TP/FP/FN/TN specific to "class_name vs rest"
                    class_metrics = calculate_metrics_for_class_sklearn(method_data['gt'], method_data['pred'], class_name)
                    metrics_by_class[class_name] = class_metrics
                
                if metrics_by_class:
                    print_metrics_table_sklearn(method_name, metrics_by_class)
                all_sklearn_metrics[method_name] = metrics_by_class
            save_analysis_json(all_sklearn_metrics, "sklearn_detailed_metrics_by_class.json", SUPER_EVAL_DIR) # Save this outside redirect
    print(f"Sklearn metrics report saved to: {metrics_output_file}")


    # --- 3. Orthogonal Comparison (PRTReid vs DBSCAN LAB - from orthogonal_comparison.py) ---
    print("\n" + "="*50); print("PART 3: ORTHOGONAL COMPARISON (PRTReid vs DBSCAN LAB)"); print("="*50)
    prt_correct_ids, dbscan_lab_correct_ids = get_correct_sets_for_venn(all_detections)
    total_detections_for_venn = len(all_detections)
    
    overall_venn_stats = create_venn_diagram_unified(prt_correct_ids, dbscan_lab_correct_ids, total_detections_for_venn, SUPER_EVAL_DIR, title_suffix=" - Overall")
    
    role_specific_venn_stats = create_role_specific_venn_diagrams_unified(all_detections, SUPER_EVAL_DIR)
    
    role_analysis_prtvdb = analyze_by_role_unified(all_detections) # PRT vs DBSCAN LAB
    create_role_comparison_plot_unified(role_analysis_prtvdb, SUPER_EVAL_DIR)
    
    # Clip analysis (overall PRTReid vs DBSCAN LAB)
    clip_analysis_overall_prtvdb = analyze_by_clip_and_role_unified(all_detections, target_role=None) # None means all roles
    significant_clips_overall = find_clips_with_significant_differences_unified(clip_analysis_overall_prtvdb, threshold=10.0)
    
    # Referee-specific clip analysis (PRTReid vs DBSCAN LAB)
    clip_analysis_referee_prtvdb = analyze_by_clip_and_role_unified(all_detections, target_role='referee')
    significant_clips_referee = find_clips_with_significant_differences_unified(clip_analysis_referee_prtvdb, threshold=5.0)

    orthogonal_summary = {
        'overall_venn_stats': overall_venn_stats,
        'role_specific_venn_stats': role_specific_venn_stats,
        'accuracy_by_role_prtvdbscanlab': role_analysis_prtvdb,
        'clip_analysis_overall_prtvdbscanlab': {
            'analysis': clip_analysis_overall_prtvdb,
            'significant_clips_gt10pct_diff': significant_clips_overall
        },
        'clip_analysis_referee_prtvdbscanlab': {
            'analysis': clip_analysis_referee_prtvdb,
            'significant_clips_gt5pct_diff': significant_clips_referee
        },
        'methodology': {
            'compared': ['PRTReid (mapped_predicted_role)', 'DBSCAN (LAB color space)'],
            'role_simplification': 'player_left/right to player, None for missing',
            'correctness_criteria': 'Exact match with simplified ground truth role (None pred is incorrect)'
        },
        'excluded_clips': exclude_clips_list,
        'total_detections_analyzed': total_detections_for_venn
    }
    save_analysis_json(orthogonal_summary, "orthogonal_comparison_prtvdbscanlab.json", SUPER_EVAL_DIR)

    print("\n--- Orthogonal Comparison Summary (PRTReid vs DBSCAN LAB) ---")
    print(f"Total Detections Analyzed: {total_detections_for_venn}")
    print(f"Overall PRTReid Accuracy: {overall_venn_stats['prtreid_accuracy']:.1f}%")
    print(f"Overall DBSCAN (LAB) Accuracy: {overall_venn_stats['dbscan_accuracy']:.1f}%")
    print("See JSON and plots for detailed role, clip, and Venn breakdowns.")

    print(f"\n\nSuper Evaluation Complete. All results are in {SUPER_EVAL_DIR.resolve()}")

if __name__ == "__main__":
    main_super_eval() 