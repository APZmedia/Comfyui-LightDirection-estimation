#!/usr/bin/env python3
"""
Test script for IRE Shadow Analysis functionality.
Demonstrates the false color IRE stops approach for shadow detection in sRGB images.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.ire_shadow_analyzer import IREShadowAnalyzer
from utils.debug_visualizer import DebugVisualizer

def create_test_image():
    """
    Create a test sRGB image with various shadow characteristics.
    """
    # Create a 512x512 test image
    height, width = 512, 512
    image = torch.zeros((1, height, width, 3))
    
    # Create a gradient from left to right (simulating lighting)
    x_coords = torch.linspace(0, 1, width).unsqueeze(0).repeat(height, 1)
    y_coords = torch.linspace(0, 1, height).unsqueeze(1).repeat(1, width)
    
    # Base lighting gradient
    base_light = 0.3 + 0.7 * x_coords  # Left side darker, right side brighter
    
    # Add some objects with shadows
    # Object 1: Hard shadow (sharp transition)
    obj1_x, obj1_y = 150, 200
    obj1_size = 80
    obj1_mask = ((torch.abs(torch.arange(width) - obj1_x) < obj1_size) & 
                 (torch.abs(torch.arange(height).unsqueeze(1) - obj1_y) < obj1_size))
    
    # Hard shadow: sharp cutoff
    hard_shadow = torch.where(
        (x_coords < 0.4) & (y_coords > 0.3) & (y_coords < 0.7),
        torch.tensor(0.1),  # Very dark shadow
        base_light
    )
    
    # Object 2: Soft shadow (gradual transition)
    obj2_x, obj2_y = 350, 300
    obj2_size = 60
    
    # Soft shadow: gradual transition using Gaussian-like falloff
    shadow_center_x, shadow_center_y = 0.6, 0.5
    shadow_distance = torch.sqrt((x_coords - shadow_center_x)**2 + (y_coords - shadow_center_y)**2)
    soft_shadow_factor = torch.exp(-shadow_distance * 3)  # Gradual falloff
    
    soft_shadow = base_light * (1 - 0.6 * soft_shadow_factor)
    
    # Combine both shadow types
    final_image = torch.minimum(hard_shadow, soft_shadow)
    
    # Add some noise for realism
    noise = torch.randn_like(final_image) * 0.02
    final_image = torch.clamp(final_image + noise, 0, 1)
    
    # Convert to RGB
    image[0, :, :, 0] = final_image  # Red channel
    image[0, :, :, 1] = final_image  # Green channel  
    image[0, :, :, 2] = final_image  # Blue channel
    
    return image

def test_ire_analysis():
    """
    Test the IRE shadow analysis functionality.
    """
    print("=== IRE Shadow Analysis Test ===")
    
    # Create test image
    print("Creating test image with hard and soft shadows...")
    test_image = create_test_image()
    
    # Initialize IRE analyzer
    analyzer = IREShadowAnalyzer(
        shadow_ire_threshold=20.0,
        transition_sensitivity=0.1
    )
    
    # Perform analysis
    print("Performing IRE analysis...")
    results = analyzer.generate_ire_analysis_report(test_image)
    
    # Extract results
    ire_values = results['ire_values']
    false_color = results['false_color_visualization']
    transition_analysis = results['transition_analysis']
    shadow_classification = results['shadow_classification']
    ire_stats = results['ire_statistics']
    
    # Print results
    print("\n=== IRE Analysis Results ===")
    print(f"Shadow Character: {shadow_classification['shadow_character']}")
    print(f"Transition Quality: {shadow_classification['transition_quality']}")
    print(f"Soft Shadow Ratio: {shadow_classification['soft_ratio']}")
    print(f"Hard Shadow Ratio: {shadow_classification['hard_ratio']}")
    print(f"Mean Gradient: {shadow_classification['mean_gradient']}")
    
    print(f"\n=== IRE Statistics ===")
    print(f"Min IRE: {ire_stats['min_ire']:.1f}")
    print(f"Max IRE: {ire_stats['max_ire']:.1f}")
    print(f"Mean IRE: {ire_stats['mean_ire']:.1f}")
    print(f"Shadow Percentage: {ire_stats['shadow_percentage']:.1f}%")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Convert tensors to numpy for visualization
    original_np = test_image[0].numpy()
    ire_np = ire_values[0].numpy()
    false_color_np = false_color[0].numpy()
    shadow_mask_np = transition_analysis['shadow_mask'][0].numpy()
    soft_shadow_np = transition_analysis['soft_shadow_mask'][0].numpy()
    hard_shadow_np = transition_analysis['hard_shadow_mask'][0].numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('IRE Shadow Analysis Results', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(original_np)
    axes[0, 0].set_title('Original sRGB Image')
    axes[0, 0].axis('off')
    
    # IRE values (grayscale)
    im1 = axes[0, 1].imshow(ire_np, cmap='gray', vmin=0, vmax=100)
    axes[0, 1].set_title('IRE Values (0-100)')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # False color visualization
    axes[0, 2].imshow(false_color_np)
    axes[0, 2].set_title('False Color IRE Visualization')
    axes[0, 2].axis('off')
    
    # Shadow mask
    axes[1, 0].imshow(shadow_mask_np, cmap='gray')
    axes[1, 0].set_title('Shadow Mask')
    axes[1, 0].axis('off')
    
    # Soft shadows
    axes[1, 1].imshow(soft_shadow_np, cmap='Blues')
    axes[1, 1].set_title('Soft Shadow Areas')
    axes[1, 1].axis('off')
    
    # Hard shadows
    axes[1, 2].imshow(hard_shadow_np, cmap='Reds')
    axes[1, 2].set_title('Hard Shadow Areas')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('ire_analysis_results.png', dpi=150, bbox_inches='tight')
    print("Results saved to 'ire_analysis_results.png'")
    
    # Create IRE legend
    legend = analyzer.create_ire_legend()
    legend_np = legend[0].numpy()
    
    plt.figure(figsize=(8, 4))
    plt.imshow(legend_np)
    plt.title('IRE Color Mapping Legend')
    plt.axis('off')
    
    # Add IRE range labels
    ire_ranges = ['0-10', '10-20', '20-30', '30-50', '50-70', '70-90', '90-100']
    y_positions = np.linspace(legend_np.shape[0] - 20, 20, len(ire_ranges))
    for i, (label, y_pos) in enumerate(zip(ire_ranges, y_positions)):
        plt.text(legend_np.shape[1] + 10, y_pos, f'{label} IRE', 
                verticalalignment='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('ire_legend.png', dpi=150, bbox_inches='tight')
    print("Legend saved to 'ire_legend.png'")
    
    return results

def test_shadow_transition_analysis():
    """
    Test shadow transition analysis specifically.
    """
    print("\n=== Shadow Transition Analysis Test ===")
    
    # Create test image with known shadow characteristics
    test_image = create_test_image()
    
    analyzer = IREShadowAnalyzer(shadow_ire_threshold=20.0, transition_sensitivity=0.1)
    
    # Convert to IRE
    ire_values = analyzer.srgb_to_ire(test_image)
    
    # Analyze transitions
    transition_analysis = analyzer.analyze_shadow_transitions(ire_values)
    
    print("Transition Analysis Results:")
    print(f"Mean Shadow Gradient: {transition_analysis['mean_shadow_gradient']:.2f}")
    print(f"Max Shadow Gradient: {transition_analysis['max_shadow_gradient']:.2f}")
    print(f"Soft Shadow Ratio: {transition_analysis['soft_shadow_ratio']:.3f}")
    print(f"Hard Shadow Ratio: {transition_analysis['hard_shadow_ratio']:.3f}")
    
    # Classify shadow characteristics
    classification = analyzer.classify_shadow_characteristics(transition_analysis)
    
    print(f"\nShadow Classification:")
    print(f"Shadow Character: {classification['shadow_character']}")
    print(f"Transition Quality: {classification['transition_quality']}")
    
    return transition_analysis, classification

def demonstrate_ire_color_mapping():
    """
    Demonstrate the IRE color mapping system.
    """
    print("\n=== IRE Color Mapping Demonstration ===")
    
    analyzer = IREShadowAnalyzer()
    
    # Create a test image with known IRE values
    test_ire_values = torch.linspace(0, 100, 100).unsqueeze(0).unsqueeze(0).repeat(1, 50, 1)
    test_ire_values = test_ire_values.unsqueeze(0)  # Add batch dimension
    
    # Create false color visualization
    false_color = analyzer.create_false_color_visualization(test_ire_values)
    
    # Convert to numpy for visualization
    false_color_np = false_color[0].numpy()
    
    plt.figure(figsize=(12, 4))
    plt.imshow(false_color_np)
    plt.title('IRE Color Mapping Demonstration (0-100 IRE)')
    plt.xlabel('IRE Value')
    plt.ylabel('')
    plt.xticks(range(0, 100, 10), range(0, 100, 10))
    plt.yticks([])
    
    # Add color zone labels
    zone_boundaries = [0, 10, 20, 30, 50, 70, 90, 100]
    zone_labels = ['Deep Shadow', 'Shadow', 'Mid-Shadow', 'Low-Midtone', 
                   'Midtone', 'High-Midtone', 'Highlight', 'Bright Highlight']
    
    for i, (boundary, label) in enumerate(zip(zone_boundaries[:-1], zone_labels)):
        plt.axvline(x=boundary, color='white', linestyle='--', alpha=0.7)
        plt.text(boundary + 5, 25, label, rotation=90, verticalalignment='bottom', 
                fontsize=8, color='white', weight='bold')
    
    plt.tight_layout()
    plt.savefig('ire_color_mapping_demo.png', dpi=150, bbox_inches='tight')
    print("IRE color mapping demonstration saved to 'ire_color_mapping_demo.png'")

if __name__ == "__main__":
    print("Starting IRE Shadow Analysis Tests...")
    
    try:
        # Test basic IRE analysis
        results = test_ire_analysis()
        
        # Test shadow transition analysis
        transition_results, classification = test_shadow_transition_analysis()
        
        # Demonstrate IRE color mapping
        demonstrate_ire_color_mapping()
        
        print("\n=== All Tests Completed Successfully ===")
        print("Generated files:")
        print("- ire_analysis_results.png: Complete analysis visualization")
        print("- ire_legend.png: IRE color mapping legend")
        print("- ire_color_mapping_demo.png: IRE color mapping demonstration")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
