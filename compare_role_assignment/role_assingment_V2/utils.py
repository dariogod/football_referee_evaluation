import json
from pathlib import Path
from custom_types import BBox, Person, PitchCoord
import cv2

def extract_frame_id_from_filename(filename: str) -> int:
    """Extract frame ID from filename like '000010.jpg' -> 10"""
    return int(Path(filename).stem)

def create_person_from_annotation(annotation: dict) -> Person:
    """Create a Person object from annotation data"""
    bbox_data = annotation.get('bbox_image')
    
    if bbox_data is None:
        raise ValueError("bbox_image is missing or None in annotation")
    
    # Convert from center-based to corner-based coordinates
    x_center = bbox_data['x_center']
    y_center = bbox_data['y_center']
    width = bbox_data['w']
    height = bbox_data['h']
    
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
    
    # Extract pitch coordinates if available
    pitch_coord = None
    if 'bbox_pitch' in annotation and annotation['bbox_pitch'] is not None:
        pitch_data = annotation['bbox_pitch']
        pitch_coord = PitchCoord(
            x_bottom_middle=pitch_data['x_bottom_middle'],
            y_bottom_middle=pitch_data['y_bottom_middle']
        )
        
    # Use track_id as person id
    person_id = annotation['track_id']

    role = "unknown"
    if annotation["attributes"]["role"] == "player":
        team = annotation["attributes"]["team"]
        if team == "left":
            role = "player_left"
        elif team == "right":
            role = "player_right"
    elif annotation["attributes"]["role"] == "referee":
        role = "referee"
    elif annotation["attributes"]["role"] == "goalkeeper":
        role = "goalkeeper"
    
    return Person(
        id=person_id, 
        bbox=bbox, 
        pitch_coord=pitch_coord,
        gt_role=role
    )

def get_test_folders(test_base_path: str = "test") -> list:
    """Get list of all SNGS-XXX test folders"""
    test_path = Path(test_base_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_path}")
    
    folders = []
    for folder in test_path.iterdir():
        if folder.is_dir() and folder.name.startswith("SNGS-"):
            folders.append(folder)
    
    return folders

def convert_to_persons(test_folder_path: Path) -> dict:
    """Convert annotations to Person objects for frames with frame_id % 10 == 0"""
    
    labels_file = test_folder_path / "Labels-GameState.json"
    img_folder = test_folder_path / "img1"
    
    if not labels_file.exists():
        print(f"Warning: Labels file not found in {test_folder_path}")
        return {}
    
    if not img_folder.exists():
        print(f"Warning: img1 folder not found in {test_folder_path}")
        return {}
    
    # Load ground truth labels
    with open(labels_file, 'r') as f:
        gt_data = json.load(f)
    
    # Create mapping from image_id to annotations
    annotations_by_image = {}
    for annotation in gt_data['annotations']:
        # Only process category_id 1 (player), 2 (goalkeeper), 3 (referee)
        if annotation['category_id'] in [1, 2, 3]:
            image_id = annotation['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(annotation)
    
    # Process frames with frame_id % 10 == 0
    results = {}
    
    for image_info in gt_data['images']:
        filename = image_info['file_name']
        frame_id = extract_frame_id_from_filename(filename)
        
        # Only process frames where frame_id % 10 == 0
        if frame_id % 10 == 0:
            image_id = image_info['image_id']
            image_path = img_folder / filename
            
            if not image_path.exists():
                print(f"Warning: Image file {image_path} not found")
                continue
            
            # Get annotations for this image
            if image_id in annotations_by_image:
                persons = []
                for annotation in annotations_by_image[image_id]:
                    try:
                        person = create_person_from_annotation(annotation)
                        persons.append(person)
                    except Exception as e:
                        print(f"Error creating person from annotation: {e}")
                        continue
                
                if persons:
                    results[frame_id] = {
                        'image_path': str(image_path),
                        'persons': persons
                    }
    
    return results 