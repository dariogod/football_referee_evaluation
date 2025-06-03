#!/usr/bin/env python3
"""
Test script to run DBScan role assigner and plot frame 10 clustering results in LAB colorspace.
"""

import json
import numpy as np
from pathlib import Path
from typing import List

from custom_types import BBox, PitchCoord, PersonWithJerseyColor
from color_conversions import RGBColor255
from role_assigner_methods.dbscan_role_assigner import DBScanRoleAssigner


def load_json_data(file_path: Path) -> dict:
    """Load JSON data from file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def convert_json_to_persons(frame_data: List[dict]) -> List[PersonWithJerseyColor]:
    """Convert JSON frame data to PersonWithJerseyColor objects."""
    persons = []
    for person_data in frame_data:
        bbox = BBox(**person_data['bbox'])
        
        pitch_coord = None
        if person_data['pitch_coord'] is not None:
            pitch_coord = PitchCoord(**person_data['pitch_coord'])
        
        jersey_color = None
        if person_data['jersey_color'] is not None:
            jersey_color = RGBColor255(**person_data['jersey_color'])
        
        person = PersonWithJerseyColor(
            id=person_data['id'],
            bbox=bbox,
            pitch_coord=pitch_coord,
            gt_role=person_data['gt_role'],
            jersey_color=jersey_color
        )
        persons.append(person)
    
    return persons


def main():
    """Test the DBScan role assigner and plot frame 10."""
    # Look for test folder with color_assignments.json
    test_dir = Path("test")
    if not test_dir.exists():
        print(f"Test directory {test_dir} does not exist")
        return
    
    # Find first test folder with color_assignments.json
    test_folder = None
    for folder in test_dir.iterdir():
        if folder.is_dir():
            color_assignments_file = folder / "color_assignments.json"
            if color_assignments_file.exists():
                test_folder = folder
                if test_folder.name != "SNGS-195":
                    continue
                break
    
    if test_folder is None:
        print("No test folder with color_assignments.json found")
        return
    
    print(f"Using test folder: {test_folder.name}")
    
    # Load color assignments
    color_assignments_path = test_folder / "color_assignments.json"
    try:
        data = load_json_data(color_assignments_path)
        print(f"Loaded {len(data)} frames from color_assignments.json")
    except Exception as e:
        print(f"Error loading {color_assignments_path}: {e}")
        return
    
    # Check if frame 10 exists
    if "10" not in data:
        print("Frame 10 not found in data")
        return
    
    # Create DBScan role assigner
    role_assigner = DBScanRoleAssigner()
    
    # Create a dummy image (since the role assignment function requires it but doesn't use it)
    dummy_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Process frame 10
    print("Processing frame 10...")
    frame_data = data["10"]
    persons_with_color = convert_json_to_persons(frame_data)
    
    print(f"Frame 10 has {len(persons_with_color)} persons")
    
    # Assign roles - this will trigger plotting for frame 10
    persons_with_roles = role_assigner.assign_roles(dummy_image, persons_with_color, frame_number=10)
    
    print(f"Role assignment complete. Found {len(persons_with_roles)} persons with roles.")
    
    # Print role summary
    role_counts = {}
    for person in persons_with_roles:
        for color_space in ["rgb", "lab", "hsv"]:
            role = person.pred_role[color_space]
            if color_space not in role_counts:
                role_counts[color_space] = {}
            role_counts[color_space][role] = role_counts[color_space].get(role, 0) + 1
    
    print("\nRole assignments by color space:")
    for color_space, counts in role_counts.items():
        print(f"  {color_space.upper()}: {counts}")


if __name__ == "__main__":
    main() 