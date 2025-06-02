import json
import math
from pathlib import Path
from typing import Dict, List, Any

# Define pitch dimensions in meters
PITCH_LENGTH = 105  # meters
PITCH_WIDTH = 68    # meters

# Define penalty box dimensions
PENALTY_AREA_LENGTH = 10  # meters from goal line
PENALTY_AREA_WIDTH = 40.32  # meters wide
PENALTY_AREA_Y_MIN = (PITCH_WIDTH - PENALTY_AREA_WIDTH) / 2  # 13.84 meters
PENALTY_AREA_Y_MAX = (PITCH_WIDTH + PENALTY_AREA_WIDTH) / 2  # 54.16 meters

def convert_to_pitch_coordinates(x_center, y_center):
    """Convert from center-based coordinates to pitch coordinates with (0,0) at bottom-left"""
    # Original: x in [-52.5, 52.5], y in [-34, 34]
    # Convert to: x in [0, 105], y in [0, 68]
    x_pitch = x_center + PITCH_LENGTH / 2
    y_pitch = y_center + PITCH_WIDTH / 2
    return x_pitch, y_pitch

def is_within_sideline(x_pitch, y_pitch, sideline_distance=5.0):
    """Check if position is within specified distance of any sideline"""
    # Check if within sideline_distance meters of any edge
    return (y_pitch <= sideline_distance or 
            y_pitch >= (PITCH_WIDTH - sideline_distance))

def is_inside_penalty_box(x_pitch, y_pitch):
    """Check if position is inside either penalty box"""
    # Left penalty box
    if (0 <= x_pitch <= PENALTY_AREA_LENGTH and 
        PENALTY_AREA_Y_MIN <= y_pitch <= PENALTY_AREA_Y_MAX):
        print(f"Left penalty box: {x_pitch}, {y_pitch}")
        return True
    # Right penalty box
    if ((PITCH_LENGTH - PENALTY_AREA_LENGTH) <= x_pitch <= PITCH_LENGTH and 
        PENALTY_AREA_Y_MIN <= y_pitch <= PENALTY_AREA_Y_MAX):
        print(f"Right penalty box: {x_pitch}, {y_pitch}")
        return True
    return False

def distance_to_center_circle(x_pitch, y_pitch):
    """Calculate distance from position to center circle (middle of pitch)"""
    center_x = PITCH_LENGTH / 2  # 52.5
    center_y = PITCH_WIDTH / 2   # 34
    return math.sqrt((x_pitch - center_x)**2 + (y_pitch - center_y)**2)

def update_role_predictions_for_frame(frame_data: List[Dict]) -> List[Dict]:
    """Update role predictions for a single frame according to the specified rules"""
    updated_frame_data = []
    
    for detection in frame_data:
        # Create a copy of the detection
        updated_detection = detection.copy()
        
        # Get pitch coordinates
        pitch_coord = detection.get('pitch_coord')
        if not pitch_coord:
            updated_frame_data.append(updated_detection)
            continue
            
        x_center = pitch_coord['x_bottom_middle']
        y_center = pitch_coord['y_bottom_middle']
        x_pitch, y_pitch = convert_to_pitch_coordinates(x_center, y_center)
        
        # Update dbscan_pred_role for each color space
        dbscan_pred_role = updated_detection['dbscan_pred_role'].copy()
        
        # Step 1: Change all referee predictions to "candidate_referee"
        for color_space in ['rgb', 'lab', 'hsv']:
            if dbscan_pred_role[color_space] == 'referee':
                dbscan_pred_role[color_space] = 'candidate_referee'
        
        updated_detection['dbscan_pred_role'] = dbscan_pred_role
        updated_frame_data.append(updated_detection)
    
    # Step 2: Apply rules for each color space independently
    for color_space in ['rgb', 'lab', 'hsv']:
        # Get all candidate_referee detections for this color space
        candidate_referees = []
        for i, detection in enumerate(updated_frame_data):
            if detection['dbscan_pred_role'][color_space] == 'candidate_referee':
                pitch_coord = detection.get('pitch_coord')
                if pitch_coord:
                    x_center = pitch_coord['x_bottom_middle']
                    y_center = pitch_coord['y_bottom_middle']
                    x_pitch, y_pitch = convert_to_pitch_coordinates(x_center, y_center)
                    candidate_referees.append({
                        'index': i,
                        'detection': detection,
                        'x_pitch': x_pitch,
                        'y_pitch': y_pitch,
                        'distance_to_center': distance_to_center_circle(x_pitch, y_pitch)
                    })
        
        # Rule 1: If within 5 meters of sideline → referee
        remaining_candidates = []
        for candidate in candidate_referees:
            if is_within_sideline(candidate['x_pitch'], candidate['y_pitch']):
                print(f"within sideline: {candidate['x_pitch']}, {candidate['y_pitch']}")
                updated_frame_data[candidate['index']]['dbscan_pred_role'][color_space] = 'referee'
            else:
                remaining_candidates.append(candidate)
        
        # Rule 2: Handle remaining candidates
        if len(remaining_candidates) == 1:
            # Single remaining candidate
            candidate = remaining_candidates[0]
            if is_inside_penalty_box(candidate['x_pitch'], candidate['y_pitch']):
                updated_frame_data[candidate['index']]['dbscan_pred_role'][color_space] = 'goalkeeper'
            else:
                updated_frame_data[candidate['index']]['dbscan_pred_role'][color_space] = 'referee'
                
        elif len(remaining_candidates) > 1:
            # Multiple remaining candidates
            # Sort by distance to center circle (closest first)
            remaining_candidates.sort(key=lambda x: x['distance_to_center'])
            
            # Closest to center circle → referee
            closest_candidate = remaining_candidates[0]
            print(f"closest to center circle: {closest_candidate['x_pitch']}, {closest_candidate['y_pitch']}")
            updated_frame_data[closest_candidate['index']]['dbscan_pred_role'][color_space] = 'referee'
            
            # All others → goalkeeper
            for candidate in remaining_candidates[1:]:
                updated_frame_data[candidate['index']]['dbscan_pred_role'][color_space] = 'goalkeeper'
    
    return updated_frame_data

def load_combined_predictions(file_path: str) -> Dict[str, Any]:
    """Load the combined role predictions JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_updated_predictions(data: Dict[str, Any], output_path: str):
    """Save the updated predictions to a new JSON file"""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def update_all_predictions(input_file: str, output_file: str):
    """Main function to update all role predictions according to the specified rules"""
    print("Loading combined role predictions...")
    data = load_combined_predictions(input_file)
    
    print(f"Processing {len(data)} clips...")
    
    total_frames = 0
    total_detections = 0
    updated_clips = 0
    
    for clip_name, clip_data in data.items():
        print(f"Processing clip: {clip_name}")
        updated_clips += 1
        
        for frame_str, frame_data in clip_data.items():
            total_frames += 1
            total_detections += len(frame_data)
            
            # Update predictions for this frame
            updated_frame_data = update_role_predictions_for_frame(frame_data)
            
            # Replace the frame data with updated version
            data[clip_name][frame_str] = updated_frame_data
    
    print(f"Processed {updated_clips} clips, {total_frames} frames, {total_detections} detections")
    
    # Verify no candidate_referee predictions remain
    candidate_count = 0
    for clip_name, clip_data in data.items():
        for frame_str, frame_data in clip_data.items():
            for detection in frame_data:
                for color_space in ['rgb', 'lab', 'hsv']:
                    if detection['dbscan_pred_role'][color_space] == 'candidate_referee':
                        candidate_count += 1
    
    if candidate_count > 0:
        print(f"WARNING: {candidate_count} candidate_referee predictions still remain!")
    else:
        print("✓ Successfully eliminated all candidate_referee predictions")
    
    print(f"Saving updated predictions to {output_file}...")
    save_updated_predictions(data, output_file)
    print("Update complete!")

def main():
    """Main function"""
    input_file = "compare_role_assignment/compare_orthogonal/combined_role_predictions.json"
    output_file = "compare_role_assignment/compare_orthogonal/combined_role_predictions_updated.json"
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} not found!")
        return
    
    print("="*70)
    print("UPDATING ROLE PREDICTIONS WITH GOALKEEPER LOGIC")
    print("="*70)
    print("Rules applied:")
    print("1. All referee predictions → candidate_referee")
    print("2. If within 5m of sideline → referee")
    print("3. If 1 remaining candidate:")
    print("   - Inside penalty box → goalkeeper")
    print("   - Outside penalty box → referee")
    print("4. If multiple remaining candidates:")
    print("   - Closest to center circle → referee")
    print("   - All others → goalkeeper")
    print("-" * 70)
    
    update_all_predictions(input_file, output_file)

if __name__ == "__main__":
    main()
