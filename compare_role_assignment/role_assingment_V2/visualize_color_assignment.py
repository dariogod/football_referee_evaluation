import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import colorsys
from color_conversions import RGBColor255, rgb_to_lab


def rgb_to_hsv(r, g, b):
    """Convert RGB (0-255) to HSV (H: 0-360, S: 0-100, V: 0-100)"""
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    return h * 360, s * 100, v * 100


def create_color_space_plots(colors_rgb, frame_num, output_dir):
    """Create 3D scatter plots for RGB, LAB, and HSV color spaces"""
    
    # Prepare data for each color space
    rgb_data = np.array(colors_rgb)
    lab_data = []
    hsv_data = []
    
    # Convert RGB to LAB and HSV
    for r, g, b in colors_rgb:
        # Convert to LAB
        rgb_obj = RGBColor255(r=int(r), g=int(g), b=int(b))
        lab_color = rgb_to_lab(rgb_obj)
        lab_data.append([lab_color.l, lab_color.a, lab_color.b])
        
        # Convert to HSV
        h, s, v = rgb_to_hsv(r, g, b)
        hsv_data.append([h, s, v])
    
    lab_data = np.array(lab_data)
    hsv_data = np.array(hsv_data)
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(20, 6))
    
    # Color for points (use the actual RGB colors)
    colors_normalized = rgb_data / 255.0
    
    # RGB subplot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(rgb_data[:, 0], rgb_data[:, 1], rgb_data[:, 2], 
                c=colors_normalized, s=100, edgecolors='black', linewidth=1)
    ax1.set_xlabel('Red (0-255)')
    ax1.set_ylabel('Green (0-255)')
    ax1.set_zlabel('Blue (0-255)')
    ax1.set_title(f'RGB Color Space - Frame {frame_num}')
    ax1.set_xlim(0, 255)
    ax1.set_ylim(0, 255)
    ax1.set_zlim(0, 255)
    
    # LAB subplot
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(lab_data[:, 0], lab_data[:, 1], lab_data[:, 2], 
                c=colors_normalized, s=100, edgecolors='black', linewidth=1)
    ax2.set_xlabel('L* (0-100)')
    ax2.set_ylabel('a* (-128 to 127)')
    ax2.set_zlabel('b* (-128 to 127)')
    ax2.set_title(f'LAB Color Space - Frame {frame_num}')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-128, 127)
    ax2.set_zlim(-128, 127)
    
    # HSV subplot
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(hsv_data[:, 0], hsv_data[:, 1], hsv_data[:, 2], 
                c=colors_normalized, s=100, edgecolors='black', linewidth=1)
    ax3.set_xlabel('Hue (0-360°)')
    ax3.set_ylabel('Saturation (0-100%)')
    ax3.set_zlabel('Value (0-100%)')
    ax3.set_title(f'HSV Color Space - Frame {frame_num}')
    ax3.set_xlim(0, 360)
    ax3.set_ylim(0, 100)
    ax3.set_zlim(0, 100)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / f'color_space_frame_{frame_num}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization for frame {frame_num} to {output_path}")


def process_all_folders():
    """Process all SNGS-XXX folders in the test directory"""
    base_path = Path("test")
    frames_to_process = [10, 100, 200, 300, 400, 500, 600, 700]
    
    # Get all SNGS-XXX folders
    sngs_folders = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("SNGS-")])
    
    for folder in sngs_folders:
        print(f"\nProcessing {folder.name}...")
        
        # Check if color_assignments.json exists
        color_assignments_path = folder / "color_assignments.json"
        if not color_assignments_path.exists():
            print(f"  Skipping {folder.name}: color_assignments.json not found")
            continue
        
        # Create output directory
        output_dir = folder / "color_assignment_visualization"
        output_dir.mkdir(exist_ok=True)
        
        # Load color assignments
        try:
            with open(color_assignments_path, 'r') as f:
                color_data = json.load(f)
        except Exception as e:
            print(f"  Error loading color_assignments.json: {e}")
            continue
        
        # Process each frame
        for frame_num in frames_to_process:
            frame_key = str(frame_num)
            
            if frame_key not in color_data:
                print(f"  Frame {frame_num} not found in {folder.name}")
                continue
            
            # Extract jersey colors for this frame
            frame_data = color_data[frame_key]
            colors_rgb = []
            
            for player in frame_data:
                if 'jersey_color' in player:
                    color = player['jersey_color']
                    colors_rgb.append([color['r'], color['g'], color['b']])
            
            if colors_rgb:
                # Create visualization
                create_color_space_plots(colors_rgb, frame_num, output_dir)
            else:
                print(f"  No jersey colors found for frame {frame_num}")


if __name__ == "__main__":
    # Change to the directory containing the test folder
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("Starting color assignment visualization...")
    process_all_folders()
    print("\nVisualization complete!")
