#!/usr/bin/env python3
"""
Test script for the visualization modules
"""

import json
import os
import sys

# Add the parent directory to the path so we can import the visualizers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from example_predictions.minimap_visualizer import MinimapVisualizer
from example_predictions.decision_critical_zone_visualizer import DecisionCriticalZoneVisualizer
from example_predictions.angle_duel_visualizer import AngleDuelVisualizer
from example_predictions.general_position_visualizer import GeneralPositionVisualizer
from example_predictions.superimpose_heatmaps import HeatmapSuperimposer


def test_visualizers():
    """Test all visualizers with sample data"""
    
    # Check if detection data exists
    detections_path = "data/example/predictions/role_assignment/detections.json"
    
    if not os.path.exists(detections_path):
        print(f"Detection data not found at {detections_path}")
        print("Please ensure you have run the complete pipeline first.")
        return False
    
    # Load detection data
    print("Loading detection data...")
    try:
        with open(detections_path, "r") as f:
            detections = json.load(f)
        print(f"Loaded {len(detections)} frames of detection data")
    except Exception as e:
        print(f"Error loading detection data: {e}")
        return False
    
    # Define test frames and output directory
    test_frames = [212, 400, 460]
    output_dir = "data/example/images_for_paper/test_output"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Test 1: MinimapVisualizer
    print("\n" + "="*50)
    print("Testing MinimapVisualizer...")
    print("="*50)
    try:
        minimap_viz = MinimapVisualizer()
        minimap_viz.visualize_multiple_frames(detections, test_frames, output_dir)
        print("✓ MinimapVisualizer test completed successfully")
    except Exception as e:
        print(f"✗ MinimapVisualizer test failed: {e}")
        return False
    
    # Test 2: DecisionCriticalZoneVisualizer
    print("\n" + "="*50)
    print("Testing DecisionCriticalZoneVisualizer...")
    print("="*50)
    try:
        critical_zone_viz = DecisionCriticalZoneVisualizer()
        critical_zone_viz.visualize_multiple_frames(detections, test_frames, output_dir, include_heatmap=True)
        print("✓ DecisionCriticalZoneVisualizer test completed successfully")
    except Exception as e:
        print(f"✗ DecisionCriticalZoneVisualizer test failed: {e}")
        return False
    
    # Test 3: AngleDuelVisualizer
    print("\n" + "="*50)
    print("Testing AngleDuelVisualizer...")
    print("="*50)
    try:
        duel_viz = AngleDuelVisualizer()
        duel_viz.visualize_multiple_frames(detections, test_frames, output_dir, include_heatmap=True)
        print("✓ AngleDuelVisualizer test completed successfully")
    except Exception as e:
        print(f"✗ AngleDuelVisualizer test failed: {e}")
        return False
    
    # Test 4: GeneralPositionVisualizer
    print("\n" + "="*50)
    print("Testing GeneralPositionVisualizer...")
    print("="*50)
    try:
        general_viz = GeneralPositionVisualizer()
        # Test both original and symmetrical versions
        general_viz.visualize_multiple_frames(detections, test_frames, output_dir, use_symmetrical=False)
        general_viz.visualize_multiple_frames(detections, test_frames, output_dir, use_symmetrical=True)
        print("✓ GeneralPositionVisualizer test completed successfully")
    except Exception as e:
        print(f"✗ GeneralPositionVisualizer test failed: {e}")
        print(f"Note: This visualizer requires plots/ground_truth/referee_positions.json to exist")
        return False
    
    # Test 5: HeatmapSuperimposer (NEW)
    print("\n" + "="*50)
    print("Testing HeatmapSuperimposer...")
    print("="*50)
    try:
        superimposer = HeatmapSuperimposer()
        superimposed_output_dir = os.path.join(output_dir, "superimposed")
        superimposer.visualize_multiple_frames(detections, test_frames, superimposed_output_dir)
        print("✓ HeatmapSuperimposer test completed successfully")
        print("✓ Created S_general_position(x,y), S_duel_angle(x,y), and S_distance(x,y) functions")
        print("✓ Computed S_overall(x,y) = S_general_position * S_duel_angle * S_distance (normalized)")
        print("✓ Generated superimposed heatmaps with fine granularity (20 points per meter)")
    except Exception as e:
        print(f"✗ HeatmapSuperimposer test failed: {e}")
        print(f"Note: This requires plots/ground_truth/referee_positions.json to exist for full functionality")
        return False
    
    print("\n" + "="*50)
    print("All tests completed successfully!")
    print(f"Output images saved to: {output_dir}")
    print(f"Superimposed heatmaps saved to: {superimposed_output_dir}")
    print("="*50)
    
    return True


def list_generated_files():
    """List all generated files"""
    output_dir = "data/example/images_for_paper/test_output"
    
    if not os.path.exists(output_dir):
        print("No output directory found")
        return
    
    files = os.listdir(output_dir)
    if not files:
        print("No files generated")
        return
    
    print(f"\nGenerated files in {output_dir}:")
    for file in sorted(files):
        file_path = os.path.join(output_dir, file)
        file_size = os.path.getsize(file_path)
        print(f"  - {file} ({file_size} bytes)")


if __name__ == "__main__":
    print("Football Referee Evaluation - Visualizer Test Suite")
    print("="*60)
    
    success = test_visualizers()
    
    if success:
        list_generated_files()
        print("\n🎉 All visualizers are working correctly!")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1) 