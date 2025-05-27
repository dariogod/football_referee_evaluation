#!/usr/bin/env python3
"""Test script to run role assignment on a single video."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_role_assignment_on_test_set import run_role_assignment_on_video
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def main():
    # Test on a single video
    test_video_dir = "data/SoccerNet/SN-GSR-2025/test/SNGS-116"
    output_dir = "compare_role_assignment/test_results"
    
    print(f"Testing on video: {test_video_dir}")
    result = run_role_assignment_on_video(test_video_dir, output_dir, show_first_frame_players=True)
    
    print("\nResult:")
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Video: {result.get('video_name', 'Unknown')}")
        print(f"Frames: {result.get('num_frames', 0)}")
        print(f"Tracks: {result.get('num_tracks', 0)}")
        print(f"Role counts: {result.get('role_counts', {})}")
        
        # Display the combined player images
        video_name = result.get('video_name', 'Unknown')
        player_images_path = os.path.join(output_dir, video_name, 'player_images_first_frame', 'all_players_first_frame.png')
        
        if os.path.exists(player_images_path):
            print(f"\nDisplaying player images from first frame...")
            img = mpimg.imread(player_images_path)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'All Players from First Frame - {video_name}')
            plt.tight_layout()
            plt.show()
        else:
            print(f"\nPlayer images not found at: {player_images_path}")

if __name__ == "__main__":
    main() 