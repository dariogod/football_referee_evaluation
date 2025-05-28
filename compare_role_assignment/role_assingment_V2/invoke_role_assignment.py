#!/usr/bin/env python3
"""
Script to invoke role assignment on color_assignments.json files from all test folders.
This script is a continuation of the color assignment process from invoke_on_test_set.py.
It loads the color_assignments.json files, converts them to PersonWithJerseyColor objects,
runs the role assignment algorithm with all methods, and saves separate JSON files for each method.
"""

import json
import numpy as np
import argparse
from typing import List, Dict, Any
from pathlib import Path

from custom_types import BBox, PitchCoord, PersonWithJerseyColor, PersonWithRole
from color_conversions import RGBColor255
from role_assigner_v2 import assign_roles, GoodnessMethod


def get_test_folders(test_dir: str = "test") -> List[Path]:
    """Get all test folders containing color_assignments.json files."""
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"Test directory {test_dir} does not exist")
        return []
    
    test_folders = []
    for folder in test_path.iterdir():
        if folder.is_dir():
            color_assignments_file = folder / "color_assignments.json"
            if color_assignments_file.exists():
                test_folders.append(folder)
    
    return sorted(test_folders)


def load_json_data(json_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load the JSON data from file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def convert_json_to_persons(frame_data: List[Dict[str, Any]]) -> List[PersonWithJerseyColor]:
    """Convert JSON frame data to PersonWithJerseyColor objects."""
    persons = []
    
    for person_data in frame_data:
        # Create BBox
        bbox = BBox(
            x1=person_data['bbox']['x1'],
            y1=person_data['bbox']['y1'],
            x2=person_data['bbox']['x2'],
            y2=person_data['bbox']['y2']
        )
        
        # Create PitchCoord (if available)
        pitch_coord = None
        if person_data.get('pitch_coord'):
            pitch_coord = PitchCoord(
                x_bottom_middle=person_data['pitch_coord']['x_bottom_middle'],
                y_bottom_middle=person_data['pitch_coord']['y_bottom_middle']
            )
        
        # Create RGBColor255 (if available)
        jersey_color = None
        if person_data.get('jersey_color'):
            jersey_color = RGBColor255(
                r=person_data['jersey_color']['r'],
                g=person_data['jersey_color']['g'],
                b=person_data['jersey_color']['b']
            )
        
        # Create PersonWithJerseyColor
        person = PersonWithJerseyColor(
            id=person_data['id'],
            bbox=bbox,
            pitch_coord=pitch_coord,
            gt_role=person_data['gt_role'],
            jersey_color=jersey_color
        )
        
        persons.append(person)
    
    return persons





def save_role_assignments_to_json(role_assignments: Dict[str, List[Dict[str, Any]]], output_path: Path):
    """Save role assignments to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(role_assignments, f, indent=2)


def process_folder(folder: Path, methods_to_test: List[GoodnessMethod], verbose: bool = False):
    """Process a single test folder and generate role assignments for all methods."""
    print(f"\nProcessing folder: {folder.name}")
    
    # Load color assignments
    color_assignments_path = folder / "color_assignments.json"
    try:
        data = load_json_data(color_assignments_path)
        print(f"  Loaded {len(data)} frames from color_assignments.json")
    except Exception as e:
        print(f"  Error loading {color_assignments_path}: {e}")
        return
    
    # Create a dummy image (since the role assignment function requires it but doesn't use it)
    dummy_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Process each method
    for method in methods_to_test:
        print(f"  Processing with method: {method.value}")
        
        # Dictionary to store role assignments for this method
        role_assignments = {}
        frames_processed = 0
        
        # Process each frame
        for frame_id, frame_data in data.items():
            try:
                # Convert to PersonWithJerseyColor objects
                persons_with_color = convert_json_to_persons(frame_data)
                
                # Assign roles
                persons_with_roles = assign_roles(dummy_image, persons_with_color, method)
                
                # Store role assignments for this frame
                frame_assignments = []
                for person_with_role in persons_with_roles:
                    frame_assignments.append(person_with_role.model_dump())
                
                role_assignments[str(frame_id)] = frame_assignments
                frames_processed += 1
                
                if verbose:
                    print(f"    Frame {frame_id}: processed {len(persons_with_roles)} persons")
                
            except Exception as e:
                print(f"    Error processing frame {frame_id} with method {method.value}: {e}")
        
        # Save role assignments for this method
        output_file = folder / f"role_assignments_{method.value}.json"
        save_role_assignments_to_json(role_assignments, output_file)
        print(f"    Saved role assignments to {output_file} ({frames_processed} frames)")


def main():
    """Main function to run role assignment on all test folders."""
    parser = argparse.ArgumentParser(description='Run role assignment on all color_assignments.json files')
    parser.add_argument('--test_dir', default='test',
                       help='Path to the test directory (default: test)')
    parser.add_argument('--methods', nargs='+', 
                       choices=['silhouette', 'calinski_harabasz', 'davies_bouldin', 'inertia', 'hdbscan'],
                       default=['silhouette', 'calinski_harabasz', 'davies_bouldin', 'inertia', 'hdbscan'],
                       help='Goodness methods to test (default: all methods)')
    parser.add_argument('--folders', nargs='+', default=None,
                       help='Specific folder names to process (default: all folders)')

    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed results for each frame')
    
    args = parser.parse_args()
    
    # Get all test folders
    test_folders = get_test_folders(args.test_dir)
    
    if not test_folders:
        print(f"No test folders with color_assignments.json found in {args.test_dir}")
        return
    
    # Filter folders if specific ones are requested
    if args.folders:
        test_folders = [folder for folder in test_folders if folder.name in args.folders]
        if not test_folders:
            print(f"None of the specified folders found: {args.folders}")
            return
    
    print(f"Found {len(test_folders)} test folders to process")
    
    # Convert method strings to enum values
    method_map = {
        'silhouette': GoodnessMethod.SILHOUETTE,
        'calinski_harabasz': GoodnessMethod.CALINSKI_HARABASZ,
        'davies_bouldin': GoodnessMethod.DAVIES_BOULDIN,
        'inertia': GoodnessMethod.INERTIA,
        'hdbscan': GoodnessMethod.HDBSCAN
    }
    methods_to_test = [method_map[method] for method in args.methods]
    
    # Process all folders
    for folder in test_folders:
        process_folder(folder, methods_to_test, args.verbose)
    
    print(f"\nProcessing complete. Processed {len(test_folders)} folders.")


if __name__ == "__main__":
    main()
