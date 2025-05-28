from utils import get_test_folders, convert_to_persons
from color_assigner_v2 import assign_jersey_colors
import cv2
import json
from pathlib import Path

    
print("Processing SoccerNet test set...")

# Get all test folders
test_folders = get_test_folders()

for folder in test_folders:
    print(f"Processing folder: {folder.name}")
    
    # Convert annotations to Person objects
    persons_data = convert_to_persons(folder)
    
    if not persons_data:
        print(f"  No valid frames found")
        continue
        
    print(f"  Found {len(persons_data)} frames with persons")
    
    # Dictionary to store all color assignments for this folder
    folder_color_assignments = {}
    
    # Process each frame
    for frame_id, frame_data in persons_data.items():
        image_path = frame_data['image_path']
        persons = frame_data['persons']
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Could not load image: {image_path}")
            continue
        
        # Assign jersey colors
        try:
            persons_with_color = assign_jersey_colors(image, persons, visualize=False)
            print(f"  Frame {frame_id}: Assigned colors to {len(persons_with_color)} persons")
            
            # Store color assignments for this frame
            frame_assignments = []
            for person_with_color in persons_with_color:
                frame_assignments.append(person_with_color.model_dump())
            
            folder_color_assignments[str(frame_id)] = frame_assignments
                
        except Exception as e:
            print(f"  Error assigning colors for frame {frame_id}: {e}")
    
    # Save color assignments to JSON file in the test folder
    if folder_color_assignments:
        output_file = folder / "color_assignments.json"
        with open(output_file, 'w') as f:
            json.dump(folder_color_assignments, f, indent=2)
        print(f"  Saved color assignments to {output_file}")
    else:
        print(f"  No color assignments to save for {folder.name}")

