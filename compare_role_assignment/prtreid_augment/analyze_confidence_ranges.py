import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

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
                                'role_confidence': detection['role_confidence']
                            })
                            
                except json.JSONDecodeError as e:
                    print(f"Error reading {reid_results_path}: {e}")
                except Exception as e:
                    print(f"Error processing {reid_results_path}: {e}")
    
    print(f"Loaded {len(all_data)} predictions with confidence scores")
    return all_data

def analyze_confidence_ranges(data):
    """Analyze confidence ranges with detailed percentiles for each role"""
    print("\n" + "="*100)
    print("DETAILED CONFIDENCE RANGE ANALYSIS")
    print("="*100)
    
    # Group by predicted role
    role_confidences = defaultdict(list)
    for item in data:
        role_confidences[item['predicted_role']].append(item['role_confidence'])
    
    # Define percentiles to analyze
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    
    print(f"{'Role':<12} {'Count':<8} {'Min':<8} ", end="")
    for p in percentiles:
        print(f"P{p:<2}"[:3] + f"{'%':<5}", end="")
    print("Max")
    print("-" * 100)
    
    for role in sorted(role_confidences.keys()):
        confidences = np.array(role_confidences[role])
        
        print(f"{role:<12} {len(confidences):<8} {np.min(confidences):<8.3f} ", end="")
        for p in percentiles:
            print(f"{np.percentile(confidences, p):<8.3f}", end="")
        print(f"{np.max(confidences):<8.3f}")
    
    return role_confidences

def analyze_player_confidence_distribution(data):
    """Detailed analysis of player class confidence distribution"""
    print("\n" + "="*80)
    print("PLAYER CLASS CONFIDENCE DISTRIBUTION ANALYSIS")
    print("="*80)
    
    player_data = [item for item in data if item['predicted_role'] == 'player']
    player_confidences = [item['role_confidence'] for item in player_data]
    
    print(f"Total player predictions: {len(player_confidences)}")
    
    # Detailed percentiles
    percentiles = np.arange(0, 101, 5)  # Every 5%
    
    print("\nPercentile analysis:")
    print("Percentile  |  Confidence")
    print("-" * 25)
    for p in percentiles:
        conf_val = np.percentile(player_confidences, p)
        print(f"{p:>8}%   |  {conf_val:>10.4f}")
    
    # Find meaningful confidence ranges
    print(f"\nMeaningful ranges for player class:")
    print(f"Bottom 10%: {np.percentile(player_confidences, 0):.4f} - {np.percentile(player_confidences, 10):.4f}")
    print(f"Middle 80%: {np.percentile(player_confidences, 10):.4f} - {np.percentile(player_confidences, 90):.4f}")
    print(f"Top 10%:    {np.percentile(player_confidences, 90):.4f} - {np.percentile(player_confidences, 100):.4f}")
    
    # Suggest confidence bins
    min_conf = np.percentile(player_confidences, 1)   # Skip extreme outliers
    max_conf = np.percentile(player_confidences, 99)  # Skip extreme outliers
    
    print(f"\nSuggested confidence range for analysis (1st-99th percentile):")
    print(f"Range: {min_conf:.4f} - {max_conf:.4f}")
    
    # Suggest bin size
    range_size = max_conf - min_conf
    suggested_bins = max(10, int(range_size / 0.1))  # At least 10 bins, or one per 0.1 units
    bin_size = range_size / suggested_bins
    
    print(f"Suggested number of bins: {suggested_bins}")
    print(f"Suggested bin size: {bin_size:.4f}")
    
    return min_conf, max_conf, suggested_bins

def compare_correct_vs_incorrect_player_confidence(data):
    """Compare confidence distributions for correct vs incorrect player predictions"""
    print("\n" + "="*80)
    print("PLAYER PREDICTIONS: CORRECT vs INCORRECT CONFIDENCE COMPARISON")
    print("="*80)
    
    player_data = [item for item in data if item['predicted_role'] == 'player']
    
    correct_player = [item['role_confidence'] for item in player_data if item['true_role'] == 'player']
    incorrect_player = [item['role_confidence'] for item in player_data if item['true_role'] != 'player']
    
    print(f"Correct player predictions: {len(correct_player)}")
    print(f"Incorrect player predictions: {len(incorrect_player)}")
    
    if len(incorrect_player) > 0:
        print(f"\nCorrect player predictions confidence:")
        print(f"  Mean: {np.mean(correct_player):.4f}, Median: {np.median(correct_player):.4f}")
        print(f"  Range: {np.min(correct_player):.4f} - {np.max(correct_player):.4f}")
        print(f"  10th-90th percentile: {np.percentile(correct_player, 10):.4f} - {np.percentile(correct_player, 90):.4f}")
        
        print(f"\nIncorrect player predictions confidence:")
        print(f"  Mean: {np.mean(incorrect_player):.4f}, Median: {np.median(incorrect_player):.4f}")
        print(f"  Range: {np.min(incorrect_player):.4f} - {np.max(incorrect_player):.4f}")
        print(f"  10th-90th percentile: {np.percentile(incorrect_player, 10):.4f} - {np.percentile(incorrect_player, 90):.4f}")
        
        # Find overlap region
        correct_min = np.percentile(correct_player, 10)
        correct_max = np.percentile(correct_player, 90)
        incorrect_min = np.percentile(incorrect_player, 10)
        incorrect_max = np.percentile(incorrect_player, 90)
        
        overlap_start = max(correct_min, incorrect_min)
        overlap_end = min(correct_max, incorrect_max)
        
        if overlap_start < overlap_end:
            print(f"\nOverlap region (10th-90th percentiles): {overlap_start:.4f} - {overlap_end:.4f}")
        else:
            print(f"\nNo significant overlap between correct and incorrect predictions")

if __name__ == "__main__":
    # Load data
    base_dir = "prtreid_output"
    print("Loading all predictions with confidence scores...")
    data = load_all_predictions_with_confidence(base_dir)
    
    if len(data) == 0:
        print("No predictions found!")
        exit(1)
    
    # Analyze confidence ranges for all roles
    role_confidences = analyze_confidence_ranges(data)
    
    # Detailed analysis for player class
    min_conf, max_conf, suggested_bins = analyze_player_confidence_distribution(data)
    
    # Compare correct vs incorrect for player class
    compare_correct_vs_incorrect_player_confidence(data) 