import json
import os
import glob
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd

class RoleClassificationMetrics:
    """Calculate classification metrics for role assignment (player, goalkeeper, referee)"""
    
    def __init__(self, prtreid_output_dir: str):
        self.prtreid_output_dir = prtreid_output_dir
        self.roles = ['player', 'goalkeeper', 'referee']
        self.metrics = defaultdict(lambda: defaultdict(int))
        
    def calculate_confusion_matrix_elements(self, true_role: str, predicted_role: str, role: str) -> Tuple[int, int, int, int]:
        """
        Calculate TP, FP, FN, TN for a specific role given true and predicted labels
        
        Args:
            true_role: Ground truth role
            predicted_role: Predicted role
            role: The role we're calculating metrics for
            
        Returns:
            Tuple of (TP, FP, FN, TN)
        """
        tp = fp = fn = tn = 0
        
        if true_role == role and predicted_role == role:
            tp = 1  # True Positive
        elif true_role != role and predicted_role == role:
            fp = 1  # False Positive
        elif true_role == role and predicted_role != role:
            fn = 1  # False Negative
        else:
            tn = 1  # True Negative
            
        return tp, fp, fn, tn
    
    def process_video_data(self, video_path: str) -> Dict[str, Dict[str, int]]:
        """
        Process all frames in a video and calculate metrics
        
        Args:
            video_path: Path to video directory
            
        Returns:
            Dictionary with metrics for each role
        """
        video_metrics = defaultdict(lambda: defaultdict(int))
        
        # Check if aggregated file exists
        all_results_path = os.path.join(video_path, 'all_reid_results.json')
        
        if os.path.exists(all_results_path):
            # Use aggregated file
            with open(all_results_path, 'r') as f:
                all_data = json.load(f)
                
            for frame_name, detections in all_data.items():
                for detection in detections:
                    true_role = detection.get('true_role')
                    predicted_role = detection.get('predicted_role')
                    
                    if true_role and predicted_role:
                        for role in self.roles:
                            tp, fp, fn, tn = self.calculate_confusion_matrix_elements(
                                true_role, predicted_role, role
                            )
                            video_metrics[role]['tp'] += tp
                            video_metrics[role]['fp'] += fp
                            video_metrics[role]['fn'] += fn
                            video_metrics[role]['tn'] += tn
        else:
            # Process individual frame files
            frame_dirs = glob.glob(os.path.join(video_path, '*/'))
            
            for frame_dir in frame_dirs:
                reid_results_path = os.path.join(frame_dir, 'reid_results.json')
                
                if os.path.exists(reid_results_path):
                    with open(reid_results_path, 'r') as f:
                        detections = json.load(f)
                        
                    for detection in detections:
                        true_role = detection.get('true_role')
                        predicted_role = detection.get('predicted_role')
                        
                        if true_role and predicted_role:
                            for role in self.roles:
                                tp, fp, fn, tn = self.calculate_confusion_matrix_elements(
                                    true_role, predicted_role, role
                                )
                                video_metrics[role]['tp'] += tp
                                video_metrics[role]['fp'] += fp
                                video_metrics[role]['fn'] += fn
                                video_metrics[role]['tn'] += tn
        
        return video_metrics
    
    def calculate_metrics_from_confusion_matrix(self, tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
        """
        Calculate accuracy, precision, recall, and F1 score from confusion matrix elements
        
        Args:
            tp: True Positives
            fp: False Positives
            fn: False Negatives
            tn: True Negatives
            
        Returns:
            Dictionary with calculated metrics
        """
        total = tp + fp + fn + tn
        
        # Accuracy
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    def process_all_videos(self) -> Dict[str, Dict[str, any]]:
        """
        Process all videos and calculate overall metrics
        
        Returns:
            Dictionary with overall metrics for each role
        """
        overall_metrics = defaultdict(lambda: defaultdict(int))
        video_results = {}
        
        # Get all video directories
        video_dirs = glob.glob(os.path.join(self.prtreid_output_dir, 'SNGS-*/'))
        
        print(f"Found {len(video_dirs)} videos to process...")
        
        for video_dir in video_dirs:
            video_name = os.path.basename(video_dir.rstrip('/'))
            print(f"Processing video: {video_name}")
            
            video_metrics = self.process_video_data(video_dir)
            video_results[video_name] = video_metrics
            
            # Accumulate overall metrics
            for role in self.roles:
                for metric in ['tp', 'fp', 'fn', 'tn']:
                    overall_metrics[role][metric] += video_metrics[role][metric]
        
        # Calculate final metrics for each role
        final_results = {}
        
        for role in self.roles:
            tp = overall_metrics[role]['tp']
            fp = overall_metrics[role]['fp']
            fn = overall_metrics[role]['fn']
            tn = overall_metrics[role]['tn']
            
            calculated_metrics = self.calculate_metrics_from_confusion_matrix(tp, fp, fn, tn)
            
            final_results[role] = {
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
                **calculated_metrics
            }
        
        return final_results, video_results
    
    def print_results(self, results: Dict[str, Dict[str, any]]):
        """Print formatted results"""
        print("\n" + "="*80)
        print("ROLE CLASSIFICATION METRICS RESULTS")
        print("="*80)
        
        for role in self.roles:
            metrics = results[role]
            print(f"\n{role.upper()} METRICS:")
            print("-" * 40)
            print(f"True Positives (TP):  {metrics['tp']:,}")
            print(f"False Positives (FP): {metrics['fp']:,}")
            print(f"False Negatives (FN): {metrics['fn']:,}")
            print(f"True Negatives (TN):  {metrics['tn']:,}")
            print(f"Accuracy:             {metrics['accuracy']:.4f}")
            print(f"Precision:            {metrics['precision']:.4f}")
            print(f"Recall:               {metrics['recall']:.4f}")
            print(f"F1 Score:             {metrics['f1_score']:.4f}")
    
    def save_results_to_csv(self, results: Dict[str, Dict[str, any]], video_results: Dict, output_path: str = "role_classification_metrics.csv"):
        """Save results to CSV file"""
        
        # Overall results
        overall_data = []
        for role in self.roles:
            metrics = results[role]
            overall_data.append({
                'role': role,
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
        overall_df.to_csv(output_path, index=False)
        print(f"\nOverall results saved to: {output_path}")
        
        # Per-video results
        video_data = []
        for video_name, video_metrics in video_results.items():
            for role in self.roles:
                metrics = video_metrics[role]
                calculated = self.calculate_metrics_from_confusion_matrix(
                    metrics['tp'], metrics['fp'], metrics['fn'], metrics['tn']
                )
                
                video_data.append({
                    'video': video_name,
                    'role': role,
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
        video_output_path = output_path.replace('.csv', '_per_video.csv')
        video_df.to_csv(video_output_path, index=False)
        print(f"Per-video results saved to: {video_output_path}")

def main():
    """Main function to run the metrics calculation"""
    prtreid_output_dir = "prtreid_output"
    
    if not os.path.exists(prtreid_output_dir):
        print(f"Error: Directory '{prtreid_output_dir}' not found!")
        return
    
    # Initialize metrics calculator
    calculator = RoleClassificationMetrics(prtreid_output_dir)
    
    # Process all videos and calculate metrics
    print("Starting role classification metrics calculation...")
    overall_results, video_results = calculator.process_all_videos()
    
    # Print results
    calculator.print_results(overall_results)
    
    # Save results to CSV
    calculator.save_results_to_csv(overall_results, video_results)
    
    print("\nMetrics calculation completed!")

if __name__ == "__main__":
    main()
