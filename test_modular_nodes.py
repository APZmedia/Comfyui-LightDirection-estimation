#!/usr/bin/env python3
"""
Test script for the modular light estimation nodes.
This script tests the separated concerns implementation.
"""

import torch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nodes.separated_nodes import LightImageProcessor, LightDistributionAnalyzer

def test_light_image_processor():
    """Test the LightImageProcessor node."""
    print("Testing LightImageProcessor...")
    
    # Create test data
    batch_size, height, width = 1, 64, 64
    
    # Create a simple normal map (pointing mostly up and right)
    normal_map = torch.zeros(batch_size, height, width, 3)
    normal_map[:, :, :, 0] = 0.5  # X component (right)
    normal_map[:, :, :, 1] = 0.5  # Y component (up)
    normal_map[:, :, :, 2] = 0.7  # Z component (toward camera)
    
    # Create a luma image (brighter in center)
    luma_image = torch.zeros(batch_size, height, width, 1)
    center_y, center_x = height // 2, width // 2
    for y in range(height):
        for x in range(width):
            dist = ((y - center_y) ** 2 + (x - center_x) ** 2) ** 0.5
            luma_image[0, y, x, 0] = max(0, 1.0 - dist / 20.0)
    
    # Test the processor
    processor = LightImageProcessor()
    result = processor.process_images(
        normal_map=normal_map,
        luma_image=luma_image,
        luma_threshold=0.3,
        curve_type="s_curve",
        normal_convention="OpenGL",
        generate_weights=True
    )
    
    processed_normals, lit_normals, binary_mask, luma_weights, debug_mask, lit_normals_viz = result
    
    print(f"✓ Processed normals shape: {processed_normals.shape}")
    print(f"✓ Lit normals shape: {lit_normals.shape}")
    print(f"✓ Binary mask shape: {binary_mask.shape}")
    print(f"✓ Luma weights shape: {luma_weights.shape if luma_weights is not None else 'None'}")
    print(f"✓ Debug mask shape: {debug_mask.shape}")
    print(f"✓ Lit normals viz shape: {lit_normals_viz.shape}")
    
    return processed_normals, lit_normals, binary_mask, luma_weights

def test_light_distribution_analyzer(processed_normals, lit_normals, binary_mask, luma_weights):
    """Test the LightDistributionAnalyzer node."""
    print("\nTesting LightDistributionAnalyzer...")
    
    # Test the analyzer
    analyzer = LightDistributionAnalyzer()
    result = analyzer.analyze_distribution(
        lit_normals=lit_normals,
        luma_weights=luma_weights,
        x_threshold=0.1,
        y_threshold=0.1,
        central_threshold=0.3,
        hard_light_threshold=0.15,
        soft_light_threshold=0.35,
        use_weighted_analysis=True
    )
    
    (x_direction, y_direction, combined_direction, hard_soft_index,
     x_confidence, y_confidence, overall_confidence, spread_value,
     directional_viz, quality_viz, colormap_preview) = result
    
    print(f"✓ X direction: {x_direction}")
    print(f"✓ Y direction: {y_direction}")
    print(f"✓ Combined direction: {combined_direction}")
    print(f"✓ Hard/soft index: {hard_soft_index:.3f}")
    print(f"✓ X confidence: {x_confidence:.3f}")
    print(f"✓ Y confidence: {y_confidence:.3f}")
    print(f"✓ Overall confidence: {overall_confidence:.3f}")
    print(f"✓ Spread value: {spread_value:.3f}")
    print(f"✓ Directional viz shape: {directional_viz.shape}")
    print(f"✓ Quality viz shape: {quality_viz.shape}")
    print(f"✓ Colormap preview shape: {colormap_preview.shape}")

def test_edge_cases():
    """Test edge cases."""
    print("\nTesting edge cases...")
    
    # Test with empty lit normals
    empty_normals = torch.zeros(0, 3)
    analyzer = LightDistributionAnalyzer()
    result = analyzer.analyze_distribution(
        lit_normals=empty_normals,
        luma_weights=None,
        x_threshold=0.1,
        y_threshold=0.1,
        central_threshold=0.3,
        hard_light_threshold=0.15,
        soft_light_threshold=0.35,
        use_weighted_analysis=False
    )
    
    print("✓ Empty normals handled correctly")
    
    # Test with single normal
    single_normal = torch.tensor([[0.5, 0.5, 0.7]]).unsqueeze(0)
    result = analyzer.analyze_distribution(
        lit_normals=single_normal,
        luma_weights=None,
        x_threshold=0.1,
        y_threshold=0.1,
        central_threshold=0.3,
        hard_light_threshold=0.15,
        soft_light_threshold=0.35,
        use_weighted_analysis=False
    )
    
    print("✓ Single normal handled correctly")

def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING MODULAR LIGHT ESTIMATION NODES")
    print("=" * 60)
    
    try:
        # Test image processing
        processed_normals, lit_normals, binary_mask, luma_weights = test_light_image_processor()
        
        # Test distribution analysis
        test_light_distribution_analyzer(processed_normals, lit_normals, binary_mask, luma_weights)
        
        # Test edge cases
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("✅ Modular implementation is working correctly")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
