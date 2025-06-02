import json
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
from pathlib import Path
import glob

def load_labels(json_path):
    """Load and parse the Labels-GameState.json file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_image_filename_from_id(image_id):
    """Convert image_id to filename format (e.g., '3116000010' -> '000010.jpg')"""
    # Extract the last 6 digits and format as filename
    frame_num = int(image_id[-6:])
    return f"{frame_num:06d}.jpg"

def extract_person_crops(image_path, annotations, image_id):
    """Extract cropped images of all persons from a single frame"""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        return []
    
    crops = []
    person_info = []
    
    # Filter annotations for this specific image
    image_annotations = [ann for ann in annotations if ann.get('image_id') == image_id]
    
    for annotation in image_annotations:
        # Check if this is a person annotation
        if annotation.get('supercategory') == 'object' and annotation.get('category_id') <= 3:
            bbox = annotation.get('bbox_image', {})
            if bbox:
                x = int(bbox['x'])
                y = int(bbox['y'])
                w = int(bbox['w'])
                h = int(bbox['h'])
                
                # Extract the crop with some padding
                padding = 5
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image.shape[1], x + w + padding)
                y2 = min(image.shape[0], y + h + padding)
                
                crop = image[y1:y2, x1:x2]
                
                if crop.size > 0:
                    crops.append(crop)
                    
                    # Extract additional info for labeling
                    attributes = annotation.get('attributes', {})
                    info = {
                        # 'role': attributes.get('role', 'unknown'),
                        # 'jersey': attributes.get('jersey', ''),
                        # 'team': attributes.get('team', ''),
                        'track_id': annotation.get('track_id', ''),
                    }
                    person_info.append(info)
    
    return crops, person_info

def create_grid_image(crops, person_info, frame_name, sequence_name, output_size=(1920, 1080)):
    """Create a grid layout of all person crops"""
    if not crops:
        return None
    
    # Calculate grid dimensions
    num_crops = len(crops)
    grid_cols = math.ceil(math.sqrt(num_crops))
    grid_rows = math.ceil(num_crops / grid_cols)
    
    # Calculate crop size to fit the output
    margin = 10
    crop_width = (output_size[0] - margin * (grid_cols + 1)) // grid_cols
    crop_height = (output_size[1] - 80 - margin * (grid_rows + 1)) // grid_rows  # Reserve 80px for header
    
    # Create output image
    output_image = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    output_image.fill(50)  # Dark gray background
    
    # Convert to PIL for text rendering
    pil_image = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
        small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Add header text
    header_text = f"{sequence_name} - Frame {frame_name}"
    draw.text((margin, margin), header_text, fill=(255, 255, 255), font=font)
    
    # Place crops in grid
    for i, (crop, info) in enumerate(zip(crops, person_info)):
        row = i // grid_cols
        col = i % grid_cols
        
        # Calculate position
        x_pos = margin + col * (crop_width + margin)
        y_pos = 60 + margin + row * (crop_height + margin)
        
        # Resize crop to fit grid cell
        crop_resized = cv2.resize(crop, (crop_width, crop_height))
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        
        # Paste the crop
        pil_image.paste(Image.fromarray(crop_rgb), (x_pos, y_pos))
        
        # Add label with person information
        label_text = f"ID:{info['track_id']}"
        if 'role' in info and info['role']:
            label_text += f" {info['role']}"
        if 'jersey' in info and info['jersey']:
            label_text += f" #{info['jersey']}"
        if 'team' in info and info['team']:
            label_text += f" ({info['team']})"
        
        # Create a larger font for the label
        try:
            label_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 32)  # Increased from 16 to 32
        except:
            label_font = ImageFont.load_default()
        
        # Draw label background
        text_bbox = draw.textbbox((0, 0), label_text, font=label_font)  # Using larger font
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        label_y = y_pos + crop_height - text_height - 8  # Increased padding for larger text
        draw.rectangle([x_pos, label_y, x_pos + text_width + 16, y_pos + crop_height], 
                      fill=(0, 0, 0, 128))
        draw.text((x_pos + 8, label_y), label_text, fill=(255, 255, 255), font=label_font)  # Using larger font
    
    # Convert back to OpenCV format
    final_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return final_image

def process_sequence(sequence_path, output_dir):
    """Process a single SNGS sequence"""
    sequence_name = os.path.basename(sequence_path)
    print(f"Processing sequence: {sequence_name}")
    
    # Paths
    labels_path = os.path.join(sequence_path, "Labels-GameState.json")
    images_dir = os.path.join(sequence_path, "img1")
    
    if not os.path.exists(labels_path) or not os.path.exists(images_dir):
        print(f"Warning: Missing labels or images directory for {sequence_name}")
        return
    
    # Load labels
    try:
        labels_data = load_labels(labels_path)
    except Exception as e:
        print(f"Error loading labels for {sequence_name}: {e}")
        return
    
    # Get sequence info
    info = labels_data.get('info', {})
    seq_length = info.get('seq_length', 750)
    
    # Extract annotations
    annotations = labels_data.get('annotations', [])
    
    # Process every 10th frame
    processed_frames = 0
    for frame_num in range(10, seq_length + 1, 10):  # Start from frame 10, step by 10

        if frame_num > 50:
            break

        # Convert frame number to image_id format
        sequence_id = sequence_name.split('-')[1]  # Extract number from SNGS-XXX
        image_id = f"3{sequence_id}{frame_num:06d}"
        
        # Get corresponding filename
        filename = f"{frame_num:06d}.jpg"
        image_path = os.path.join(images_dir, filename)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Extract person crops
        crops, person_info = extract_person_crops(image_path, annotations, image_id)
        
        if not crops:
            print(f"No person detections found for frame {frame_num}")
            continue
        
        # Create grid image
        grid_image = create_grid_image(crops, person_info, filename, sequence_name)
        
        if grid_image is not None:
            # Save the grid image
            output_filename = f"{sequence_name}_frame_{frame_num:06d}_persons.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, grid_image)
            print(f"Saved: {output_filename} ({len(crops)} persons)")
            processed_frames += 1
    
    print(f"Completed {sequence_name}: {processed_frames} frames processed")

def main():
    """Main function to process all sequences"""
    # Get the script directory and construct paths relative to the workspace root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels to workspace root
    
    # Paths
    test_dir = os.path.join(workspace_root, "data", "SoccerNet", "SN-GSR-2025", "test")
    output_dir = os.path.join(script_dir, "output", "person_grids")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Looking for sequences in: {test_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find all SNGS sequences
    sequence_dirs = glob.glob(os.path.join(test_dir, "SNGS-*"))
    sequence_dirs.sort()
    
    print(f"Found {len(sequence_dirs)} sequences to process")
    
    # Process each sequence
    for sequence_path in sequence_dirs:
        if os.path.isdir(sequence_path):
            try:
                process_sequence(sequence_path, output_dir)
            except Exception as e:
                print(f"Error processing {sequence_path}: {e}")
                continue
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
