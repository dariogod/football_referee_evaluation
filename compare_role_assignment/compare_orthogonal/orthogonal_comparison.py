import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib_venn import venn2, venn2_circles
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set
import numpy as np

def simplify_role(role: str) -> str:
    """Simplify role names - convert player_left/player_right to just 'player'."""
    if role in ['player_left', 'player_right']:
        return 'player'
    return role

def load_combined_dataset(file_path: str = "compare_role_assignment/compare_orthogonal/combined_role_predictions_updated.json") -> Dict:
    """Load the combined dataset from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_predictions(dataset: Dict, exclude_clips: List[str] = None) -> Tuple[Set[int], Set[int], List[Dict]]:
    """
    Analyze predictions and return sets of correct predictions for each model.
    
    Args:
        dataset: The combined dataset
        exclude_clips: List of clip names to exclude from analysis
    
    Returns:
        - prtreid_correct: Set of detection IDs where PRTReid is correct
        - dbscan_correct: Set of detection IDs where DBSCAN (lab) is correct
        - all_detections: List of all detection data for detailed analysis
    """
    if exclude_clips is None:
        exclude_clips = []
    
    prtreid_correct = set()
    dbscan_correct = set()
    all_detections = []
    
    detection_id = 0
    excluded_count = 0
    
    for clip_name, clip_data in dataset.items():
        # Skip excluded clips
        if clip_name in exclude_clips:
            excluded_count += sum(len(frame_data) for frame_data in clip_data.values())
            continue
            
        for frame_str, frame_data in clip_data.items():
            for detection in frame_data:
                # Simplify roles
                gt_role_simplified = simplify_role(detection['gt_role'])
                
                # DBSCAN (LAB)
                dbscan_pred_dict = detection.get('dbscan_pred_role', {})
                dbscan_lab_pred_val = dbscan_pred_dict.get('lab')
                dbscan_lab_simplified = simplify_role(dbscan_lab_pred_val) # simplify_role handles None

                # For PRTReid, we need to handle the mapped prediction
                prtreid_mapped_simplified = None # Default to None
                prtreid_data = detection.get('prtreid_data')
                if prtreid_data is not None:
                    prtreid_mapped_pred_val = prtreid_data.get('mapped_predicted_role')
                    if prtreid_mapped_pred_val is not None:
                        prtreid_mapped_simplified = simplify_role(prtreid_mapped_pred_val)
                
                # Store detection info
                detection_info = {
                    'id': detection_id,
                    'clip': clip_name,
                    'frame': detection['frame'],
                    'track_id': detection['track_id'],
                    'gt_role': gt_role_simplified,
                    'dbscan_lab_pred': dbscan_lab_simplified,
                    'prtreid_pred': prtreid_mapped_simplified,
                    'dbscan_correct': gt_role_simplified == dbscan_lab_simplified and dbscan_lab_simplified is not None,
                    'prtreid_correct': prtreid_mapped_simplified == gt_role_simplified if prtreid_mapped_simplified is not None else False
                }
                
                all_detections.append(detection_info)
                
                # Check correctness
                if detection_info['dbscan_correct']:
                    dbscan_correct.add(detection_id)
                
                if detection_info['prtreid_correct']:
                    prtreid_correct.add(detection_id)
                
                detection_id += 1
    
    if exclude_clips:
        print(f"Excluded {len(exclude_clips)} clips ({', '.join(exclude_clips)}) with {excluded_count:,} detections from analysis")
    
    return prtreid_correct, dbscan_correct, all_detections

def create_venn_diagram(prtreid_correct: Set[int], dbscan_correct: Set[int], 
                       total_detections: int, output_path: str = "compare_role_assignment/compare_orthogonal/venn_diagram.png"):
    """Create and save a Venn diagram comparing the two models."""
    
    # Calculate intersections
    both_correct = prtreid_correct.intersection(dbscan_correct)
    prtreid_only = prtreid_correct - dbscan_correct
    dbscan_only = dbscan_correct - prtreid_correct
    both_wrong = total_detections - len(prtreid_correct.union(dbscan_correct))
    
    # Create the Venn diagram
    plt.figure(figsize=(12, 8))
    
    # Create subplot for Venn diagram
    plt.subplot(1, 2, 1)
    
    # Create Venn diagram
    venn = venn2(subsets=(len(prtreid_only), len(dbscan_only), len(both_correct)), 
                 set_labels=('PRTReid\nCorrect', 'DBSCAN (LAB)\nCorrect'))
    
    # Customize colors
    if venn.get_patch_by_id('10'):
        venn.get_patch_by_id('10').set_facecolor('#ff9999')  # PRTReid only
        venn.get_patch_by_id('10').set_alpha(0.7)
    if venn.get_patch_by_id('01'):
        venn.get_patch_by_id('01').set_facecolor('#9999ff')  # DBSCAN only
        venn.get_patch_by_id('01').set_alpha(0.7)
    if venn.get_patch_by_id('11'):
        venn.get_patch_by_id('11').set_facecolor('#99ff99')  # Both correct
        venn.get_patch_by_id('11').set_alpha(0.7)
    
    # Add circles
    venn2_circles(subsets=(len(prtreid_only), len(dbscan_only), len(both_correct)))
    
    plt.title(f'Model Correctness Comparison\n(Total Detections: {total_detections:,})', fontsize=14, fontweight='bold')
    
    # Create second subplot for statistics
    plt.subplot(1, 2, 2)
    plt.axis('off')
    
    # Calculate percentages
    prtreid_total_correct = len(prtreid_correct)
    dbscan_total_correct = len(dbscan_correct)
    
    prtreid_accuracy = (prtreid_total_correct / total_detections) * 100
    dbscan_accuracy = (dbscan_total_correct / total_detections) * 100
    both_accuracy = (len(both_correct) / total_detections) * 100
    
    # Create statistics text
    stats_text = f"""STATISTICS:

Total Detections: {total_detections:,}

PRTReid Correct: {prtreid_total_correct:,} ({prtreid_accuracy:.1f}%)
DBSCAN (LAB) Correct: {dbscan_total_correct:,} ({dbscan_accuracy:.1f}%)

Both Correct: {len(both_correct):,} ({both_accuracy:.1f}%)
PRTReid Only: {len(prtreid_only):,} ({len(prtreid_only)/total_detections*100:.1f}%)
DBSCAN Only: {len(dbscan_only):,} ({len(dbscan_only)/total_detections*100:.1f}%)
Both Wrong: {both_wrong:,} ({both_wrong/total_detections*100:.1f}%)

Agreement Rate: {(len(both_correct) + both_wrong) / total_detections * 100:.1f}%
"""
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Venn diagram saved to: {output_path}")
    
    return {
        'prtreid_only': len(prtreid_only),
        'dbscan_only': len(dbscan_only),
        'both_correct': len(both_correct),
        'both_wrong': both_wrong,
        'prtreid_accuracy': prtreid_accuracy,
        'dbscan_accuracy': dbscan_accuracy
    }

def analyze_by_role(all_detections: List[Dict]) -> Dict:
    """Analyze model performance by role."""
    role_analysis = {}
    
    # Group by role
    roles = set(det['gt_role'] for det in all_detections)
    
    for role in roles:
        role_detections = [det for det in all_detections if det['gt_role'] == role]
        total = len(role_detections)
        
        prtreid_correct = sum(1 for det in role_detections if det['prtreid_correct'])
        dbscan_correct = sum(1 for det in role_detections if det['dbscan_correct'])
        both_correct = sum(1 for det in role_detections if det['prtreid_correct'] and det['dbscan_correct'])
        
        role_analysis[role] = {
            'total': total,
            'prtreid_correct': prtreid_correct,
            'dbscan_correct': dbscan_correct,
            'both_correct': both_correct,
            'prtreid_accuracy': (prtreid_correct / total) * 100 if total > 0 else 0,
            'dbscan_accuracy': (dbscan_correct / total) * 100 if total > 0 else 0
        }
    
    return role_analysis

def create_role_comparison_plot(role_analysis: Dict, output_path: str = "compare_role_assignment/compare_orthogonal/role_comparison.png"):
    """Create a bar plot comparing model performance by role."""
    roles = list(role_analysis.keys())
    prtreid_accuracies = [role_analysis[role]['prtreid_accuracy'] for role in roles]
    dbscan_accuracies = [role_analysis[role]['dbscan_accuracy'] for role in roles]
    
    x = np.arange(len(roles))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, prtreid_accuracies, width, label='PRTReid', color='#ff9999', alpha=0.8)
    bars2 = ax.bar(x + width/2, dbscan_accuracies, width, label='DBSCAN (LAB)', color='#9999ff', alpha=0.8)
    
    ax.set_xlabel('Roles', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Model Accuracy by Role', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(roles)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Role comparison plot saved to: {output_path}")

def save_detailed_analysis(all_detections: List[Dict], venn_stats: Dict, role_analysis: Dict, 
                          output_path: str = "compare_role_assignment/compare_orthogonal/detailed_analysis.json"):
    """Save detailed analysis results to JSON."""
    analysis_results = {
        'summary': venn_stats,
        'role_analysis': role_analysis,
        'total_detections': len(all_detections),
        'methodology': {
            'role_simplification': 'player_left and player_right simplified to player',
            'models_compared': ['PRTReid (mapped predictions)', 'DBSCAN (LAB color space)'],
            'correctness_criteria': 'Exact match with ground truth role'
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"Detailed analysis saved to: {output_path}")

def analyze_by_clip(all_detections: List[Dict]) -> Dict:
    """Analyze model performance by clip and identify significant differences."""
    clip_analysis = {}
    
    # Group by clip
    clips = set(det['clip'] for det in all_detections)
    
    for clip in clips:
        clip_detections = [det for det in all_detections if det['clip'] == clip]
        total = len(clip_detections)
        
        if total == 0:
            continue
            
        prtreid_correct = sum(1 for det in clip_detections if det['prtreid_correct'])
        dbscan_correct = sum(1 for det in clip_detections if det['dbscan_correct'])
        both_correct = sum(1 for det in clip_detections if det['prtreid_correct'] and det['dbscan_correct'])
        both_wrong = sum(1 for det in clip_detections if not det['prtreid_correct'] and not det['dbscan_correct'])
        
        prtreid_accuracy = (prtreid_correct / total) * 100
        dbscan_accuracy = (dbscan_correct / total) * 100
        accuracy_diff = prtreid_accuracy - dbscan_accuracy
        
        clip_analysis[clip] = {
            'total': total,
            'prtreid_correct': prtreid_correct,
            'dbscan_correct': dbscan_correct,
            'both_correct': both_correct,
            'both_wrong': both_wrong,
            'prtreid_accuracy': prtreid_accuracy,
            'dbscan_accuracy': dbscan_accuracy,
            'accuracy_difference': accuracy_diff,
            'prtreid_only_correct': prtreid_correct - both_correct,
            'dbscan_only_correct': dbscan_correct - both_correct
        }
    
    return clip_analysis

def find_clips_with_significant_differences(clip_analysis: Dict, threshold: float = 10.0) -> Dict:
    """Find clips where one model significantly outperforms the other."""
    significant_clips = {
        'prtreid_better': [],
        'dbscan_better': [],
        'similar_performance': []
    }
    
    for clip, stats in clip_analysis.items():
        accuracy_diff = stats['accuracy_difference']
        
        if abs(accuracy_diff) >= threshold:
            if accuracy_diff > 0:
                significant_clips['prtreid_better'].append({
                    'clip': clip,
                    'prtreid_accuracy': stats['prtreid_accuracy'],
                    'dbscan_accuracy': stats['dbscan_accuracy'],
                    'difference': accuracy_diff,
                    'total_detections': stats['total']
                })
            else:
                significant_clips['dbscan_better'].append({
                    'clip': clip,
                    'prtreid_accuracy': stats['prtreid_accuracy'],
                    'dbscan_accuracy': stats['dbscan_accuracy'],
                    'difference': accuracy_diff,
                    'total_detections': stats['total']
                })
        else:
            significant_clips['similar_performance'].append({
                'clip': clip,
                'prtreid_accuracy': stats['prtreid_accuracy'],
                'dbscan_accuracy': stats['dbscan_accuracy'],
                'difference': accuracy_diff,
                'total_detections': stats['total']
            })
    
    # Sort by absolute difference
    significant_clips['prtreid_better'].sort(key=lambda x: x['difference'], reverse=True)
    significant_clips['dbscan_better'].sort(key=lambda x: abs(x['difference']), reverse=True)
    
    return significant_clips

def create_role_specific_venn_diagrams(all_detections: List[Dict], output_dir: str = "compare_role_assignment/compare_orthogonal/") -> Dict:
    """Create separate Venn diagrams for each role."""
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Group detections by role
    roles = set(det['gt_role'] for det in all_detections)
    role_venn_stats = {}
    
    for role in roles:
        role_detections = [det for det in all_detections if det['gt_role'] == role]
        
        if len(role_detections) == 0:
            continue
            
        # Create sets of correct predictions for this role
        prtreid_correct_role = set()
        dbscan_correct_role = set()
        
        for i, det in enumerate(role_detections):
            if det['prtreid_correct']:
                prtreid_correct_role.add(i)
            if det['dbscan_correct']:
                dbscan_correct_role.add(i)
        
        total_role_detections = len(role_detections)
        
        # Calculate intersections
        both_correct = prtreid_correct_role.intersection(dbscan_correct_role)
        prtreid_only = prtreid_correct_role - dbscan_correct_role
        dbscan_only = dbscan_correct_role - prtreid_correct_role
        both_wrong = total_role_detections - len(prtreid_correct_role.union(dbscan_correct_role))
        
        # Create the Venn diagram for this role
        plt.figure(figsize=(12, 8))
        
        # Create subplot for Venn diagram
        plt.subplot(1, 2, 1)
        
        # Create Venn diagram
        venn = venn2(subsets=(len(prtreid_only), len(dbscan_only), len(both_correct)), 
                     set_labels=('PRTReid\nCorrect', 'DBSCAN (LAB)\nCorrect'))
        
        # Customize colors
        if venn.get_patch_by_id('10'):
            venn.get_patch_by_id('10').set_facecolor('#ff9999')  # PRTReid only
            venn.get_patch_by_id('10').set_alpha(0.7)
        if venn.get_patch_by_id('01'):
            venn.get_patch_by_id('01').set_facecolor('#9999ff')  # DBSCAN only
            venn.get_patch_by_id('01').set_alpha(0.7)
        if venn.get_patch_by_id('11'):
            venn.get_patch_by_id('11').set_facecolor('#99ff99')  # Both correct
            venn.get_patch_by_id('11').set_alpha(0.7)
        
        # Add circles
        venn2_circles(subsets=(len(prtreid_only), len(dbscan_only), len(both_correct)))
        
        plt.title(f'Model Correctness Comparison - {role.upper()}\n(Total Detections: {total_role_detections:,})', 
                 fontsize=14, fontweight='bold')
        
        # Create second subplot for statistics
        plt.subplot(1, 2, 2)
        plt.axis('off')
        
        # Calculate percentages
        prtreid_total_correct = len(prtreid_correct_role)
        dbscan_total_correct = len(dbscan_correct_role)
        
        prtreid_accuracy = (prtreid_total_correct / total_role_detections) * 100 if total_role_detections > 0 else 0
        dbscan_accuracy = (dbscan_total_correct / total_role_detections) * 100 if total_role_detections > 0 else 0
        both_accuracy = (len(both_correct) / total_role_detections) * 100 if total_role_detections > 0 else 0
        
        # Create statistics text
        stats_text = f"""STATISTICS - {role.upper()}:

Total Detections: {total_role_detections:,}

PRTReid Correct: {prtreid_total_correct:,} ({prtreid_accuracy:.1f}%)
DBSCAN (LAB) Correct: {dbscan_total_correct:,} ({dbscan_accuracy:.1f}%)

Both Correct: {len(both_correct):,} ({both_accuracy:.1f}%)
PRTReid Only: {len(prtreid_only):,} ({len(prtreid_only)/total_role_detections*100:.1f}%)
DBSCAN Only: {len(dbscan_only):,} ({len(dbscan_only)/total_role_detections*100:.1f}%)
Both Wrong: {both_wrong:,} ({both_wrong/total_role_detections*100:.1f}%)

Agreement Rate: {(len(both_correct) + both_wrong) / total_role_detections * 100:.1f}%
"""
        
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        output_path = f"{output_dir}venn_diagram_{role}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Venn diagram for {role} saved to: {output_path}")
        plt.close()  # Close the figure to free memory
        
        # Store stats for this role
        role_venn_stats[role] = {
            'prtreid_only': len(prtreid_only),
            'dbscan_only': len(dbscan_only),
            'both_correct': len(both_correct),
            'both_wrong': both_wrong,
            'prtreid_accuracy': prtreid_accuracy,
            'dbscan_accuracy': dbscan_accuracy,
            'total_detections': total_role_detections,
            'agreement_rate': (len(both_correct) + both_wrong) / total_role_detections * 100 if total_role_detections > 0 else 0
        }
    
    return role_venn_stats

def analyze_by_clip_and_role(all_detections: List[Dict], target_role: str, threshold: float = 5.0) -> Dict:
    """
    Analyze model performance by clip for a specific role and identify significant differences.
    
    Args:
        all_detections: List of all detection data
        target_role: The role to analyze (e.g., 'referee', 'player', 'goalkeeper')
        threshold: Minimum percentage difference to consider significant
    
    Returns:
        Dictionary containing clip analysis for the specified role
    """
    # Filter detections for the target role
    role_detections = [det for det in all_detections if det['gt_role'] == target_role]
    
    clip_analysis = {}
    
    # Group by clip
    clips = set(det['clip'] for det in role_detections)
    
    for clip in clips:
        clip_detections = [det for det in role_detections if det['clip'] == clip]
        total = len(clip_detections)
        
        if total == 0:
            continue
            
        prtreid_correct = sum(1 for det in clip_detections if det['prtreid_correct'])
        dbscan_correct = sum(1 for det in clip_detections if det['dbscan_correct'])
        both_correct = sum(1 for det in clip_detections if det['prtreid_correct'] and det['dbscan_correct'])
        both_wrong = sum(1 for det in clip_detections if not det['prtreid_correct'] and not det['dbscan_correct'])
        
        prtreid_accuracy = (prtreid_correct / total) * 100
        dbscan_accuracy = (dbscan_correct / total) * 100
        accuracy_diff = prtreid_accuracy - dbscan_accuracy
        
        clip_analysis[clip] = {
            'role': target_role,
            'total': total,
            'prtreid_correct': prtreid_correct,
            'dbscan_correct': dbscan_correct,
            'both_correct': both_correct,
            'both_wrong': both_wrong,
            'prtreid_accuracy': prtreid_accuracy,
            'dbscan_accuracy': dbscan_accuracy,
            'accuracy_difference': accuracy_diff,
            'prtreid_only_correct': prtreid_correct - both_correct,
            'dbscan_only_correct': dbscan_correct - both_correct
        }
    
    return clip_analysis

def find_clips_with_significant_differences_by_role(clip_analysis: Dict, threshold: float = 5.0) -> Dict:
    """Find clips where one model significantly outperforms the other for a specific role."""
    significant_clips = {
        'prtreid_better': [],
        'dbscan_better': [],
        'similar_performance': []
    }
    
    for clip, stats in clip_analysis.items():
        accuracy_diff = stats['accuracy_difference']
        
        if abs(accuracy_diff) >= threshold:
            if accuracy_diff > 0:
                significant_clips['prtreid_better'].append({
                    'clip': clip,
                    'role': stats['role'],
                    'prtreid_accuracy': stats['prtreid_accuracy'],
                    'dbscan_accuracy': stats['dbscan_accuracy'],
                    'difference': accuracy_diff,
                    'total_detections': stats['total']
                })
            else:
                significant_clips['dbscan_better'].append({
                    'clip': clip,
                    'role': stats['role'],
                    'prtreid_accuracy': stats['prtreid_accuracy'],
                    'dbscan_accuracy': stats['dbscan_accuracy'],
                    'difference': accuracy_diff,
                    'total_detections': stats['total']
                })
        else:
            significant_clips['similar_performance'].append({
                'clip': clip,
                'role': stats['role'],
                'prtreid_accuracy': stats['prtreid_accuracy'],
                'dbscan_accuracy': stats['dbscan_accuracy'],
                'difference': accuracy_diff,
                'total_detections': stats['total']
            })
    
    # Sort by absolute difference
    significant_clips['prtreid_better'].sort(key=lambda x: x['difference'], reverse=True)
    significant_clips['dbscan_better'].sort(key=lambda x: abs(x['difference']), reverse=True)
    
    return significant_clips

def analyze_referee_clips(dataset_path: str = "compare_role_assignment/compare_orthogonal/combined_role_predictions_updated.json", exclude_clips: List[str] = None):
    """Specific function to analyze referee clips and print SNGS-XXX clips with significant differences."""
    if exclude_clips is None:
        exclude_clips = []
        
    print("="*80)
    print("REFEREE-SPECIFIC CLIP ANALYSIS")
    if exclude_clips:
        print(f"EXCLUDING CLIPS: {', '.join(exclude_clips)}")
    print("="*80)
    
    # Load data and analyze
    dataset = load_combined_dataset(dataset_path)
    prtreid_correct, dbscan_correct, all_detections = analyze_predictions(dataset, exclude_clips=exclude_clips)
    
    # Analyze clips specifically for referee role
    referee_clip_analysis = analyze_by_clip_and_role(all_detections, 'referee', threshold=5.0)
    
    # Find significant differences with a lower threshold for referees
    significant_referee_clips = find_clips_with_significant_differences_by_role(referee_clip_analysis, threshold=5.0)
    
    print(f"\nREFEREE ROLE ANALYSIS:")
    print(f"Total clips with referee detections: {len(referee_clip_analysis)}")
    
    total_referee_detections = sum(stats['total'] for stats in referee_clip_analysis.values())
    print(f"Total referee detections: {total_referee_detections:,}")
    
    print(f"\n" + "="*80)
    print("CLIPS WHERE PRTREID SIGNIFICANTLY OUTPERFORMS DBSCAN FOR REFEREES (≥5% difference)")
    print("="*80)
    
    if significant_referee_clips['prtreid_better']:
        print(f"\nFound {len(significant_referee_clips['prtreid_better'])} clips where PRTReid performs better:")
        for i, clip_data in enumerate(significant_referee_clips['prtreid_better'], 1):
            print(f"  {i:2d}. {clip_data['clip']}")
            print(f"      PRTReid: {clip_data['prtreid_accuracy']:.1f}% | DBSCAN: {clip_data['dbscan_accuracy']:.1f}% | Diff: +{clip_data['difference']:.1f}% | Referee detections: {clip_data['total_detections']}")
        
        print(f"\nSNGS-XXX clips to investigate (PRTReid better): {', '.join([clip['clip'] for clip in significant_referee_clips['prtreid_better']])}")
    else:
        print("No clips found where PRTReid significantly outperforms DBSCAN for referees.")
    
    print(f"\n" + "="*80)
    print("CLIPS WHERE DBSCAN SIGNIFICANTLY OUTPERFORMS PRTREID FOR REFEREES (≥5% difference)")
    print("="*80)
    
    if significant_referee_clips['dbscan_better']:
        print(f"\nFound {len(significant_referee_clips['dbscan_better'])} clips where DBSCAN performs better:")
        for i, clip_data in enumerate(significant_referee_clips['dbscan_better'], 1):
            print(f"  {i:2d}. {clip_data['clip']}")
            print(f"      PRTReid: {clip_data['prtreid_accuracy']:.1f}% | DBSCAN: {clip_data['dbscan_accuracy']:.1f}% | Diff: {clip_data['difference']:.1f}% | Referee detections: {clip_data['total_detections']}")
        
        print(f"\nSNGS-XXX clips to investigate (DBSCAN better): {', '.join([clip['clip'] for clip in significant_referee_clips['dbscan_better']])}")
    else:
        print("No clips found where DBSCAN significantly outperforms PRTReid for referees.")
    
    print(f"\n" + "="*80)
    print("ALL CLIPS WITH DIFFERENCES ≥5% FOR REFEREES")
    print("="*80)
    
    all_significant = significant_referee_clips['prtreid_better'] + significant_referee_clips['dbscan_better']
    if all_significant:
        all_clips = sorted([clip['clip'] for clip in all_significant])
        print(f"\nAll SNGS-XXX clips with ≥5% difference: {', '.join(all_clips)}")
        print(f"Total clips to investigate: {len(all_clips)}")
    else:
        print("No clips found with significant differences ≥5% for referees.")
    
    print(f"\n" + "="*80)
    print("DETAILED REFEREE CLIP BREAKDOWN")
    print("="*80)
    
    # Sort all clips by accuracy difference (absolute value)
    all_clips_sorted = sorted(referee_clip_analysis.items(), 
                             key=lambda x: abs(x[1]['accuracy_difference']), reverse=True)
    
    print(f"\nAll clips sorted by accuracy difference (showing all {len(all_clips_sorted)} clips):")
    for i, (clip, stats) in enumerate(all_clips_sorted, 1):
        diff_indicator = "+" if stats['accuracy_difference'] >= 0 else ""
        print(f"  {i:2d}. {clip}: PRTReid {stats['prtreid_accuracy']:.1f}% | DBSCAN {stats['dbscan_accuracy']:.1f}% | Diff: {diff_indicator}{stats['accuracy_difference']:.1f}% | Referees: {stats['total']}")
    
    # Save referee-specific analysis
    referee_output_path = "compare_role_assignment/compare_orthogonal/referee_clip_analysis.json"
    with open(referee_output_path, 'w') as f:
        json.dump({
            'referee_clip_analysis': referee_clip_analysis,
            'significant_differences': significant_referee_clips,
            'threshold_used': 5.0,
            'excluded_clips': exclude_clips,
            'summary': {
                'total_clips': len(referee_clip_analysis),
                'total_referee_detections': total_referee_detections,
                'clips_prtreid_better': len(significant_referee_clips['prtreid_better']),
                'clips_dbscan_better': len(significant_referee_clips['dbscan_better']),
                'clips_similar': len(significant_referee_clips['similar_performance'])
            }
        }, f, indent=2)
    print(f"\nReferee-specific clip analysis saved to: {referee_output_path}")
    
    return significant_referee_clips, referee_clip_analysis

def main():
    """Main function to run the orthogonal comparison analysis."""
    # Define clips to exclude from analysis
    exclude_clips = ['SNGS-125', 'SNGS-190']
    
    print("Loading combined dataset...")
    dataset = load_combined_dataset()
    
    print("Analyzing predictions...")
    prtreid_correct, dbscan_correct, all_detections = analyze_predictions(dataset, exclude_clips=exclude_clips)
    
    print("Creating overall Venn diagram...")
    venn_stats = create_venn_diagram(prtreid_correct, dbscan_correct, len(all_detections))
    
    print("Creating role-specific Venn diagrams...")
    role_venn_stats = create_role_specific_venn_diagrams(all_detections)
    
    print("Analyzing performance by role...")
    role_analysis = analyze_by_role(all_detections)
    
    print("Analyzing performance by clip...")
    clip_analysis = analyze_by_clip(all_detections)
    
    print("Finding clips with significant differences...")
    significant_clips = find_clips_with_significant_differences(clip_analysis, threshold=10.0)
    
    print("Creating role comparison plot...")
    create_role_comparison_plot(role_analysis)
    
    print("Saving detailed analysis...")
    save_detailed_analysis(all_detections, venn_stats, role_analysis)
    
    print("\n" + "="*70)
    print("ORTHOGONAL COMPARISON RESULTS")
    if exclude_clips:
        print(f"(EXCLUDING: {', '.join(exclude_clips)})")
    print("="*70)
    print(f"Total detections analyzed: {len(all_detections):,}")
    print(f"PRTReid accuracy: {venn_stats['prtreid_accuracy']:.1f}%")
    print(f"DBSCAN (LAB) accuracy: {venn_stats['dbscan_accuracy']:.1f}%")
    print(f"\nModel agreement breakdown:")
    print(f"  Both correct: {venn_stats['both_correct']:,} detections")
    print(f"  PRTReid only correct: {venn_stats['prtreid_only']:,} detections")
    print(f"  DBSCAN only correct: {venn_stats['dbscan_only']:,} detections")
    print(f"  Both wrong: {venn_stats['both_wrong']:,} detections")
    
    print(f"\nPerformance by role:")
    for role, stats in role_analysis.items():
        print(f"  {role.upper()}:")
        print(f"    Total: {stats['total']:,} detections")
        print(f"    PRTReid: {stats['prtreid_accuracy']:.1f}%")
        print(f"    DBSCAN (LAB): {stats['dbscan_accuracy']:.1f}%")
    
    # Print role-specific Venn diagram statistics
    print(f"\n" + "="*70)
    print("ROLE-SPECIFIC VENN DIAGRAM STATISTICS")
    print("="*70)
    for role, stats in role_venn_stats.items():
        print(f"\n{role.upper()} ({stats['total_detections']:,} detections):")
        print(f"  PRTReid accuracy: {stats['prtreid_accuracy']:.1f}%")
        print(f"  DBSCAN accuracy: {stats['dbscan_accuracy']:.1f}%")
        print(f"  Agreement rate: {stats['agreement_rate']:.1f}%")
        print(f"  Both correct: {stats['both_correct']:,}")
        print(f"  PRTReid only: {stats['prtreid_only']:,}")
        print(f"  DBSCAN only: {stats['dbscan_only']:,}")
        print(f"  Both wrong: {stats['both_wrong']:,}")
    
    # Print clips with significant differences
    print(f"\n" + "="*70)
    print("CLIPS WITH SIGNIFICANT PERFORMANCE DIFFERENCES (≥10%)")
    print("="*70)
    
    print(f"\nClips where PRTReid significantly outperforms DBSCAN ({len(significant_clips['prtreid_better'])} clips):")
    for i, clip_data in enumerate(significant_clips['prtreid_better'][:20], 1):  # Show top 20
        print(f"  {i:2d}. {clip_data['clip']}")
        print(f"      PRTReid: {clip_data['prtreid_accuracy']:.1f}% | DBSCAN: {clip_data['dbscan_accuracy']:.1f}% | Diff: +{clip_data['difference']:.1f}% | Detections: {clip_data['total_detections']}")
    
    if len(significant_clips['prtreid_better']) > 20:
        print(f"      ... and {len(significant_clips['prtreid_better']) - 20} more clips")
    
    print(f"\nClips where DBSCAN significantly outperforms PRTReid ({len(significant_clips['dbscan_better'])} clips):")
    for i, clip_data in enumerate(significant_clips['dbscan_better'][:20], 1):  # Show top 20
        print(f"  {i:2d}. {clip_data['clip']}")
        print(f"      PRTReid: {clip_data['prtreid_accuracy']:.1f}% | DBSCAN: {clip_data['dbscan_accuracy']:.1f}% | Diff: {clip_data['difference']:.1f}% | Detections: {clip_data['total_detections']}")
    
    if len(significant_clips['dbscan_better']) > 20:
        print(f"      ... and {len(significant_clips['dbscan_better']) - 20} more clips")
    
    print(f"\nClips with similar performance ({len(significant_clips['similar_performance'])} clips with <10% difference)")
    
    # Save clip analysis
    clip_output_path = "compare_role_assignment/compare_orthogonal/clip_analysis.json"
    with open(clip_output_path, 'w') as f:
        json.dump({
            'clip_analysis': clip_analysis,
            'significant_differences': significant_clips,
            'threshold_used': 10.0,
            'excluded_clips': exclude_clips
        }, f, indent=2)
    print(f"\nClip analysis saved to: {clip_output_path}")
    
    # Save role-specific Venn stats
    role_venn_output_path = "compare_role_assignment/compare_orthogonal/role_venn_stats.json"
    with open(role_venn_output_path, 'w') as f:
        json.dump(role_venn_stats, f, indent=2)
    print(f"Role-specific Venn statistics saved to: {role_venn_output_path}")
    
    print("\nAnalysis complete! Check the generated plots and detailed analysis files.")

if __name__ == "__main__":
    main()
    
    # Also run referee-specific analysis with exclusions
    print("\n" + "="*80)
    print("RUNNING REFEREE-SPECIFIC ANALYSIS")
    print("="*80)
    exclude_clips = ['SNGS-125', 'SNGS-190', 'SNGS-146']
    analyze_referee_clips(exclude_clips=exclude_clips)
