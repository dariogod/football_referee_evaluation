#!/usr/bin/env python3
"""
Script to calculate metrics for all 4 goodness methods from role assignment outputs.
Calculates TP, FP, FN, TN, accuracy, precision, recall, and F1 scores for each role.
Now supports multiple color spaces (rgb, lab, hsv) for each method.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict
import pandas as pd


def get_sngs_folders(test_dir: str = "test") -> List[Path]:
    """Get all SNGS folders that have role assignment JSON files."""
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"Test directory {test_dir} does not exist")
        return []
    
    sngs_folders = []
    for folder in test_path.iterdir():
        if folder.is_dir() and folder.name.startswith("SNGS-"):
            # Check if at least one role assignment file exists
            role_files = list(folder.glob("role_assignments_*.json"))
            if role_files:
                sngs_folders.append(folder)
    
    return sorted(sngs_folders)


def discover_methods(folders: List[Path]) -> List[str]:
    """Discover all available methods by scanning role assignment files."""
    methods = set()
    
    for folder in folders:
        role_files = list(folder.glob("role_assignments_*.json"))
        for file in role_files:
            # Extract method name from filename
            # Format is: role_assignments_<method>.json
            filename = file.stem  # Remove .json extension
            if filename.startswith("role_assignments_"):
                method = filename.replace("role_assignments_", "")
                if method:  # Ensure method name is not empty
                    methods.add(method)
    
    return sorted(list(methods))


def load_role_assignments(json_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load role assignments from JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return {}


def normalize_role(role: str) -> str:
    """Normalize role names to standard categories."""
    if role in ["player_left", "player_right"]:
        return "player"
    elif role == "goalkeeper":
        return "goalkeeper"
    elif role == "referee":
        return "referee"
    else:
        return "unknown"


def calculate_confusion_matrix(gt_roles: List[str], pred_roles: List[str], target_role: str) -> Tuple[int, int, int, int]:
    """
    Calculate TP, FP, FN, TN for a specific role.
    
    Args:
        gt_roles: List of ground truth roles
        pred_roles: List of predicted roles
        target_role: The role we're calculating metrics for
    
    Returns:
        Tuple of (TP, FP, FN, TN)
    """
    tp = fp = fn = tn = 0
    
    for gt, pred in zip(gt_roles, pred_roles):
        if gt == target_role and pred == target_role:
            tp += 1
        elif gt != target_role and pred == target_role:
            fp += 1
        elif gt == target_role and pred != target_role:
            fn += 1
        else:  # gt != target_role and pred != target_role
            tn += 1
    
    return tp, fp, fn, tn


def calculate_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """Calculate accuracy, precision, recall, and F1 score from confusion matrix values."""
    total = tp + fp + fn + tn
    
    # Accuracy
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }


def process_method_data(folders: List[Path], method: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Process all folders for a specific method and calculate aggregated metrics for each color space."""
    
    # Define color spaces
    color_spaces = ["rgb", "lab", "hsv"]
    
    # Aggregate data across all folders for each color space
    all_gt_roles = {cs: [] for cs in color_spaces}
    all_pred_roles = {cs: [] for cs in color_spaces}
    
    folders_processed = 0
    
    for folder in folders:
        role_file = folder / f"role_assignments_{method}.json"
        if not role_file.exists():
            continue
            
        data = load_role_assignments(role_file)
        if not data:
            continue
            
        # Extract all predictions and ground truths from this folder
        for frame_id, frame_data in data.items():
            for person in frame_data:
                gt_role = normalize_role(person['gt_role'])
                
                # Handle both old format (string) and new format (dict)
                if isinstance(person['pred_role'], str):
                    # Old format - use the same prediction for all color spaces
                    pred_role = normalize_role(person['pred_role'])
                    for cs in color_spaces:
                        if gt_role != "unknown" and pred_role != "unknown":
                            all_gt_roles[cs].append(gt_role)
                            all_pred_roles[cs].append(pred_role)
                else:
                    # New format - extract prediction for each color space
                    for cs in color_spaces:
                        if cs in person['pred_role']:
                            pred_role = normalize_role(person['pred_role'][cs])
                            # Only include known roles
                            if gt_role != "unknown" and pred_role != "unknown":
                                all_gt_roles[cs].append(gt_role)
                                all_pred_roles[cs].append(pred_role)
        
        folders_processed += 1
    
    print(f"  Processed {folders_processed} folders")
    
    # Calculate metrics for each color space
    color_space_metrics = {}
    
    for cs in color_spaces:
        if len(all_gt_roles[cs]) == 0:
            continue
            
        print(f"    Color space {cs}: {len(all_gt_roles[cs])} total predictions")
        
        # Calculate metrics for each role
        roles = ["player", "goalkeeper", "referee"]
        role_metrics = {}
        
        for role in roles:
            tp, fp, fn, tn = calculate_confusion_matrix(all_gt_roles[cs], all_pred_roles[cs], role)
            role_metrics[role] = calculate_metrics(tp, fp, fn, tn)
        
        # Calculate metrics for "outlier" class (goalkeeper + referee vs player)
        outlier_gt_roles = ["outlier" if role in ["goalkeeper", "referee"] else role for role in all_gt_roles[cs]]
        outlier_pred_roles = ["outlier" if role in ["goalkeeper", "referee"] else role for role in all_pred_roles[cs]]
        tp, fp, fn, tn = calculate_confusion_matrix(outlier_gt_roles, outlier_pred_roles, "outlier")
        role_metrics["outlier"] = calculate_metrics(tp, fp, fn, tn)
        
        # Calculate overall metrics (micro-average)
        total_tp = sum(role_metrics[role]['TP'] for role in roles)
        total_fp = sum(role_metrics[role]['FP'] for role in roles)
        total_fn = sum(role_metrics[role]['FN'] for role in roles)
        total_tn = sum(role_metrics[role]['TN'] for role in roles)
        
        role_metrics['All'] = calculate_metrics(total_tp, total_fp, total_fn, total_tn)
        
        color_space_metrics[cs] = role_metrics
    
    return color_space_metrics


def write_results_table(results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]], output_file: str = "results.txt"):
    """Write results in a clean table format to a file."""
    
    methods = list(results.keys())
    roles = ["player", "outlier", "goalkeeper", "referee", "All"]
    metrics = ['TP', 'FP', 'FN', 'TN', 'Accuracy', 'Precision', 'Recall', 'F1']
    color_spaces = ["rgb", "lab", "hsv"]
    
    with open(output_file, 'w') as f:
        for method in methods:
            f.write(f"\n{'='*80}\n")
            f.write(f"METHOD: {method.upper()}\n")
            f.write(f"{'='*80}\n")
            
            # Process each color space for this method
            for cs in color_spaces:
                if cs not in results[method]:
                    continue
                    
                f.write(f"\nColor Space: {cs.upper()}\n")
                f.write(f"{'-'*80}\n")
                
                # Create DataFrame for this method and color space
                data = []
                for role in roles:
                    row = [role]
                    for metric in metrics:
                        value = results[method][cs][role][metric]
                        if metric in ['TP', 'FP', 'FN', 'TN']:
                            row.append(f"{int(value)}")
                        else:
                            row.append(f"{value:.4f}")
                    data.append(row)
                
                df = pd.DataFrame(data, columns=['Role'] + metrics)
                f.write(df.to_string(index=False) + "\n")
        
        # Write summary comparison table for each color space
        for cs in color_spaces:
            f.write(f"\n{'='*80}\n")
            f.write(f"SUMMARY COMPARISON - {cs.upper()} (F1 Scores)\n")
            f.write(f"{'='*80}\n")
            
            summary_data = []
            for method in methods:
                if cs in results[method]:
                    row = [method]
                    for role in ["player", "outlier", "goalkeeper", "referee", "All"]:
                        f1_score = results[method][cs][role]['F1']
                        row.append(f"{f1_score:.4f}")
                    summary_data.append(row)
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data, columns=['Method', 'Player', 'Outlier', 'Goalkeeper', 'Referee', 'All'])
                f.write(summary_df.to_string(index=False) + "\n")
        
        # Write overall summary comparison table (best color space for each method)
        f.write(f"\n{'='*80}\n")
        f.write("BEST COLOR SPACE BY METHOD (Overall F1 Score)\n")
        f.write(f"{'='*80}\n")
        
        best_cs_data = []
        for method in methods:
            best_f1 = 0
            best_cs = ""
            for cs in color_spaces:
                if cs in results[method]:
                    f1 = results[method][cs]['All']['F1']
                    if f1 > best_f1:
                        best_f1 = f1
                        best_cs = cs
            
            if best_cs:
                row = [method, best_cs, f"{best_f1:.4f}"]
                # Add F1 scores for each role with the best color space
                for role in ["player", "outlier", "goalkeeper", "referee"]:
                    f1_score = results[method][best_cs][role]['F1']
                    row.append(f"{f1_score:.4f}")
                best_cs_data.append(row)
        
        if best_cs_data:
            best_cs_df = pd.DataFrame(best_cs_data, columns=['Method', 'Best CS', 'Overall F1', 'Player F1', 'Outlier F1', 'Goalkeeper F1', 'Referee F1'])
            f.write(best_cs_df.to_string(index=False) + "\n")


def main():
    """Main function to calculate metrics for all methods."""
    parser = argparse.ArgumentParser(description='Calculate metrics for all goodness methods with multiple color spaces')
    parser.add_argument('--test_dir', default='test',
                       help='Path to the test directory (default: test)')
    
    args = parser.parse_args()
    
    # Get all SNGS folders
    sngs_folders = get_sngs_folders(args.test_dir)
    
    if not sngs_folders:
        print(f"No SNGS folders with role assignment files found in {args.test_dir}")
        return
    
    print(f"Found {len(sngs_folders)} SNGS folders to analyze")
    print(f"Folders: {[folder.name for folder in sngs_folders]}")
    
    # Discover available methods
    methods = discover_methods(sngs_folders)
    
    if not methods:
        print("No role assignment files found in the SNGS folders")
        return
    
    print(f"\nDiscovered {len(methods)} method(s): {', '.join(methods)}")
    
    # Process each method
    results = {}
    
    for method in methods:
        print(f"\nProcessing method: {method}")
        results[method] = process_method_data(sngs_folders, method)
    
    # Print results
    write_results_table(results)
    print(f"\nResults written to results.txt")


if __name__ == "__main__":
    main()
