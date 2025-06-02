import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional

def get_all_sngs_clips() -> List[str]:
    """Get all SNGS-XXX clip names from the role_assignment_V2/test directory."""
    role_assignment_dir = Path("compare_role_assignment/role_assingment_V2/test")
    sngs_dirs = glob.glob(str(role_assignment_dir / "SNGS-*"))
    clip_names = [Path(dir_path).name for dir_path in sngs_dirs]
    return sorted(clip_names)

def load_dbscan_data(clip_name: str) -> Dict[str, List[Dict]]:
    """Load DBSCAN role assignment data for a specific clip."""
    dbscan_path = Path(f"compare_role_assignment/role_assingment_V2/test/{clip_name}/role_assignments_dbscan.json")
    
    if not dbscan_path.exists():
        print(f"Warning: DBSCAN file not found for {clip_name}")
        return {}
    
    with open(dbscan_path, 'r') as f:
        return json.load(f)

def load_prtreid_data(clip_name: str, frame_num: int) -> List[Dict]:
    """Load PRTReid data for a specific clip and frame."""
    frame_str = f"{frame_num:06d}"
    prtreid_path = Path(f"compare_role_assignment/prtreid_augment/prtreid_output/{clip_name}/{frame_str}/reid_results.json")
    
    if not prtreid_path.exists():
        print(f"Warning: PRTReid file not found for {clip_name}, frame {frame_num}")
        return []
    
    with open(prtreid_path, 'r') as f:
        return json.load(f)

def get_available_frames(clip_name: str) -> List[int]:
    """Get all available frames (every 10th) for a specific clip from PRTReid output."""
    prtreid_clip_dir = Path(f"compare_role_assignment/prtreid_augment/prtreid_output/{clip_name}")
    
    if not prtreid_clip_dir.exists():
        print(f"Warning: PRTReid directory not found for {clip_name}")
        return []
    
    frame_dirs = [d for d in prtreid_clip_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    frames = sorted([int(d.name) for d in frame_dirs])
    return frames

def map_prtreid_to_dbscan_role(true_role: str, true_team: str, predicted_role: str) -> str:
    """Map PRTReid predictions to DBSCAN-style role format."""
    if predicted_role == "referee":
        return "referee"
    elif predicted_role == "goalkeeper":
        return "goalkeeper"
    elif predicted_role == "player":
        if true_team == "left":
            return "player_left"
        elif true_team == "right":
            return "player_right"
        else:
            return "player_unknown"
    else:
        return "unknown"

def combine_frame_data(dbscan_frame_data: List[Dict], prtreid_frame_data: List[Dict], frame_num: int) -> List[Dict]:
    """Combine DBSCAN and PRTReid data for a single frame."""
    combined_data = []
    
    # Create a mapping from track_id to PRTReid data
    prtreid_by_track_id = {item['track_id']: item for item in prtreid_frame_data}
    
    for dbscan_item in dbscan_frame_data:
        track_id = dbscan_item['id']
        
        # Start with DBSCAN data
        combined_item = {
            'frame': frame_num,
            'track_id': track_id,
            'bbox': dbscan_item['bbox'],
            'pitch_coord': dbscan_item['pitch_coord'],
            'gt_role': dbscan_item['gt_role'],
            'dbscan_pred_role': {
                'rgb': dbscan_item['pred_role']['rgb'],
                'lab': dbscan_item['pred_role']['lab'],
                'hsv': dbscan_item['pred_role']['hsv']
            }
        }
        
        # Add PRTReid data if available
        if track_id in prtreid_by_track_id:
            prtreid_item = prtreid_by_track_id[track_id]
            combined_item['prtreid_data'] = {
                'predicted_role': prtreid_item['predicted_role'],
                'role_confidence': prtreid_item['role_confidence'],
                'true_role': prtreid_item['true_role'],
                'true_team': prtreid_item['true_team'],
                'true_jersey': prtreid_item['true_jersey'],
                'mapped_predicted_role': map_prtreid_to_dbscan_role(
                    prtreid_item['true_role'], 
                    prtreid_item['true_team'], 
                    prtreid_item['predicted_role']
                )
            }
        else:
            combined_item['prtreid_data'] = None
            print(f"Warning: No PRTReid data found for track_id {track_id} in frame {frame_num}")
        
        combined_data.append(combined_item)
    
    return combined_data

def create_combined_dataset() -> Dict[str, Dict]:
    """Create the complete combined dataset for all SNGS clips and every 10th frame."""
    all_clips = get_all_sngs_clips()
    combined_dataset = {}
    
    print(f"Processing {len(all_clips)} SNGS clips...")
    
    for clip_name in all_clips:
        print(f"Processing {clip_name}...")
        
        # Load DBSCAN data for the entire clip
        dbscan_data = load_dbscan_data(clip_name)
        
        # Get available frames for this clip
        available_frames = get_available_frames(clip_name)
        
        clip_data = {}
        
        for frame_num in available_frames:
            frame_str = str(frame_num)
            
            # Check if frame exists in DBSCAN data
            if frame_str not in dbscan_data:
                print(f"Warning: Frame {frame_num} not found in DBSCAN data for {clip_name}")
                continue
            
            # Load PRTReid data for this frame
            prtreid_data = load_prtreid_data(clip_name, frame_num)
            
            # Combine the data
            combined_frame_data = combine_frame_data(
                dbscan_data[frame_str], 
                prtreid_data, 
                frame_num
            )
            
            clip_data[frame_str] = combined_frame_data
        
        combined_dataset[clip_name] = clip_data
        print(f"Completed {clip_name}: {len(clip_data)} frames processed")
    
    return combined_dataset

def save_combined_dataset(dataset: Dict, output_file: str = "combined_role_predictions.json"):
    """Save the combined dataset to a JSON file."""
    output_path = Path("compare_role_assignment/compare_orthogonal") / output_file
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Combined dataset saved to {output_path}")

def get_dataset_statistics(dataset: Dict) -> Dict:
    """Generate statistics about the combined dataset."""
    stats = {
        'total_clips': len(dataset),
        'total_frames': 0,
        'total_detections': 0,
        'clips_with_missing_prtreid': 0,
        'detections_with_missing_prtreid': 0,
        'role_distribution': {},
        'clips_summary': {}
    }
    
    for clip_name, clip_data in dataset.items():
        clip_stats = {
            'frames': len(clip_data),
            'detections': 0,
            'missing_prtreid': 0
        }
        
        for frame_str, frame_data in clip_data.items():
            stats['total_frames'] += 1
            clip_stats['detections'] += len(frame_data)
            stats['total_detections'] += len(frame_data)
            
            for detection in frame_data:
                # Count role distribution
                gt_role = detection['gt_role']
                if gt_role not in stats['role_distribution']:
                    stats['role_distribution'][gt_role] = 0
                stats['role_distribution'][gt_role] += 1
                
                # Count missing PRTReid data
                if detection['prtreid_data'] is None:
                    clip_stats['missing_prtreid'] += 1
                    stats['detections_with_missing_prtreid'] += 1
        
        if clip_stats['missing_prtreid'] > 0:
            stats['clips_with_missing_prtreid'] += 1
        
        stats['clips_summary'][clip_name] = clip_stats
    
    return stats

def main():
    """Main function to create and save the combined dataset."""
    print("Creating combined dataset of ground truth, DBSCAN, and PRTReid predictions...")
    print("This combines data for all SNGS-XXX clips for every 10th frame.")
    print("-" * 70)
    
    # Create the combined dataset
    combined_dataset = create_combined_dataset()
    
    # Save the dataset
    save_combined_dataset(combined_dataset)
    
    # Generate and display statistics
    stats = get_dataset_statistics(combined_dataset)
    
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Total clips processed: {stats['total_clips']}")
    print(f"Total frames processed: {stats['total_frames']}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Clips with missing PRTReid data: {stats['clips_with_missing_prtreid']}")
    print(f"Detections with missing PRTReid data: {stats['detections_with_missing_prtreid']}")
    
    print(f"\nRole distribution:")
    for role, count in stats['role_distribution'].items():
        print(f"  {role}: {count}")
    
    print(f"\nFirst 5 clips summary:")
    for i, (clip_name, clip_stats) in enumerate(list(stats['clips_summary'].items())[:5]):
        print(f"  {clip_name}: {clip_stats['frames']} frames, {clip_stats['detections']} detections, {clip_stats['missing_prtreid']} missing PRTReid")
    
    # Save statistics
    stats_path = Path("compare_role_assignment/compare_orthogonal/dataset_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to {stats_path}")
    
    print("\nCombined dataset creation completed!")

if __name__ == "__main__":
    main()
