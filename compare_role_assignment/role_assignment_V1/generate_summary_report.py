import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

def generate_summary_report():
    """Generate a comprehensive summary report of the role classification metrics"""
    
    # Read the CSV files
    overall_df = pd.read_csv('role_classification_metrics.csv')
    per_video_df = pd.read_csv('role_classification_metrics_per_video.csv')
    
    print("="*80)
    print("COMPREHENSIVE ROLE CLASSIFICATION METRICS REPORT")
    print("="*80)
    
    # Overall Summary
    print("\n1. OVERALL PERFORMANCE SUMMARY")
    print("-" * 50)
    
    for _, row in overall_df.iterrows():
        role = row['role']
        print(f"\n{role.upper()}:")
        print(f"  • Total Detections: {row['tp'] + row['fn']:,}")
        print(f"  • Correctly Classified: {row['tp']:,}")
        print(f"  • Misclassified: {row['fn']:,}")
        print(f"  • False Alarms: {row['fp']:,}")
        print(f"  • Accuracy: {row['accuracy']:.1%}")
        print(f"  • Precision: {row['precision']:.1%}")
        print(f"  • Recall: {row['recall']:.1%}")
        print(f"  • F1 Score: {row['f1_score']:.1%}")
    
    # Performance comparison
    print("\n2. ROLE PERFORMANCE COMPARISON")
    print("-" * 50)
    
    metrics_comparison = overall_df[['role', 'accuracy', 'precision', 'recall', 'f1_score']].copy()
    metrics_comparison = metrics_comparison.round(4)
    
    print("\nMetrics Comparison Table:")
    print(metrics_comparison.to_string(index=False))
    
    # Best and worst performing roles
    best_f1 = overall_df.loc[overall_df['f1_score'].idxmax()]
    worst_f1 = overall_df.loc[overall_df['f1_score'].idxmin()]
    
    print(f"\nBest F1 Score: {best_f1['role']} ({best_f1['f1_score']:.1%})")
    print(f"Worst F1 Score: {worst_f1['role']} ({worst_f1['f1_score']:.1%})")
    
    # Per-video analysis
    print("\n3. PER-VIDEO ANALYSIS")
    print("-" * 50)
    
    for role in ['player', 'goalkeeper', 'referee']:
        role_data = per_video_df[per_video_df['role'] == role]
        
        print(f"\n{role.upper()} Performance Across Videos:")
        print(f"  • Mean F1 Score: {role_data['f1_score'].mean():.1%}")
        print(f"  • Std F1 Score: {role_data['f1_score'].std():.3f}")
        print(f"  • Best Video F1: {role_data['f1_score'].max():.1%}")
        print(f"  • Worst Video F1: {role_data['f1_score'].min():.1%}")
        
        # Find best and worst performing videos for this role
        best_video = role_data.loc[role_data['f1_score'].idxmax(), 'video']
        worst_video = role_data.loc[role_data['f1_score'].idxmin(), 'video']
        
        print(f"  • Best Video: {best_video}")
        print(f"  • Worst Video: {worst_video}")
    
    # Confusion matrix analysis
    print("\n4. CONFUSION MATRIX ANALYSIS")
    print("-" * 50)
    
    total_detections = overall_df['tp'].sum() + overall_df['fn'].sum()
    total_predictions = overall_df['tp'].sum() + overall_df['fp'].sum()
    
    print(f"\nTotal Ground Truth Detections: {total_detections:,}")
    print(f"Total Predictions Made: {total_predictions:,}")
    
    # Class distribution
    print("\nGround Truth Class Distribution:")
    for _, row in overall_df.iterrows():
        role = row['role']
        role_total = row['tp'] + row['fn']
        percentage = (role_total / total_detections) * 100
        print(f"  • {role}: {role_total:,} ({percentage:.1f}%)")
    
    # Error analysis
    print("\n5. ERROR ANALYSIS")
    print("-" * 50)
    
    total_errors = overall_df['fp'].sum() + overall_df['fn'].sum()
    print(f"Total Classification Errors: {total_errors:,}")
    
    for _, row in overall_df.iterrows():
        role = row['role']
        false_negatives = row['fn']
        false_positives = row['fp']
        role_errors = false_negatives + false_positives
        error_rate = (role_errors / total_errors) * 100
        
        print(f"\n{role.upper()} Errors:")
        print(f"  • False Negatives (missed): {false_negatives:,}")
        print(f"  • False Positives (false alarms): {false_positives:,}")
        print(f"  • Total errors: {role_errors:,} ({error_rate:.1f}% of all errors)")
    
    # Video-level statistics
    print("\n6. VIDEO-LEVEL STATISTICS")
    print("-" * 50)
    
    video_stats = per_video_df.groupby('video').agg({
        'f1_score': ['mean', 'std'],
        'accuracy': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    video_stats.columns = ['_'.join(col).strip() for col in video_stats.columns]
    
    # Find best and worst performing videos overall
    video_overall_f1 = per_video_df.groupby('video')['f1_score'].mean()
    best_overall_video = video_overall_f1.idxmax()
    worst_overall_video = video_overall_f1.idxmin()
    
    print(f"Best Overall Video: {best_overall_video} (Mean F1: {video_overall_f1[best_overall_video]:.1%})")
    print(f"Worst Overall Video: {worst_overall_video} (Mean F1: {video_overall_f1[worst_overall_video]:.1%})")
    
    print(f"\nMean F1 Score Across All Videos: {video_overall_f1.mean():.1%}")
    print(f"Standard Deviation: {video_overall_f1.std():.3f}")
    
    # Recommendations
    print("\n7. RECOMMENDATIONS")
    print("-" * 50)
    
    print("\nBased on the analysis:")
    
    # Player performance
    player_metrics = overall_df[overall_df['role'] == 'player'].iloc[0]
    if player_metrics['f1_score'] > 0.95:
        print("✓ Player classification is performing excellently")
    elif player_metrics['f1_score'] > 0.90:
        print("✓ Player classification is performing well")
    else:
        print("⚠ Player classification needs improvement")
    
    # Goalkeeper performance
    goalkeeper_metrics = overall_df[overall_df['role'] == 'goalkeeper'].iloc[0]
    if goalkeeper_metrics['recall'] < 0.7:
        print("⚠ Goalkeeper detection has low recall - many goalkeepers are being missed")
    if goalkeeper_metrics['precision'] < 0.9:
        print("⚠ Goalkeeper classification has precision issues - false positives present")
    
    # Referee performance
    referee_metrics = overall_df[overall_df['role'] == 'referee'].iloc[0]
    if referee_metrics['recall'] < 0.8:
        print("⚠ Referee detection has moderate recall - some referees are being missed")
    
    print("\nGeneral recommendations:")
    print("• Focus on improving goalkeeper detection (lowest F1 score)")
    print("• Investigate videos with poor performance for targeted improvements")
    print("• Consider class imbalance - players are much more frequent than goalkeepers/referees")
    
    print("\n" + "="*80)
    print("REPORT GENERATION COMPLETED")
    print("="*80)

if __name__ == "__main__":
    generate_summary_report() 