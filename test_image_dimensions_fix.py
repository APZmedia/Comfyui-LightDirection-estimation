#!/usr/bin/env python3
"""
Test script to verify that image dimension fixes work correctly.
This script simulates the ComfyUI image output validation.
"""

import torch
import numpy as np
from PIL import Image

def test_image_dimensions():
    """Test that all image outputs have valid dimensions for ComfyUI"""
    
    print("Testing image dimension fixes...")
    
    # Simulate input image dimensions
    luma_image = torch.randn(1, 1280, 728, 3)
    print(f"Input image shape: {luma_image.shape}")
    
    # Simulate problematic outputs that might cause the error
    test_cases = [
        ("Valid 3D tensor", torch.randn(1280, 728, 3)),
        ("Valid 2D tensor", torch.randn(1280, 728)),
        ("Invalid 1x1x1 tensor", torch.randn(1, 1, 1)),
        ("Invalid single pixel", torch.randn(1, 1)),
        ("Wrong dimensions", torch.randn(512, 512, 3)),
    ]
    
    def ensure_valid_image(img_tensor, reference_shape):
        """Ensure image tensor has valid dimensions and data type for ComfyUI"""
        if img_tensor.dim() == 2:
            img_tensor = img_tensor.unsqueeze(-1)
        if img_tensor.shape[:2] != reference_shape[:2]:
            img_tensor = torch.nn.functional.interpolate(
                img_tensor.permute(2, 0, 1).unsqueeze(0),
                size=(reference_shape[1], reference_shape[2]),
                mode='nearest'
            ).squeeze(0).permute(1, 2, 0)
        return img_tensor.clamp(0, 1).float()
    
    for test_name, test_tensor in test_cases:
        print(f"\nTesting: {test_name}")
        print(f"  Input shape: {test_tensor.shape}")
        print(f"  Input dtype: {test_tensor.dtype}")
        
        try:
            # Apply the fix
            fixed_tensor = ensure_valid_image(test_tensor, luma_image.shape)
            print(f"  Fixed shape: {fixed_tensor.shape}")
            print(f"  Fixed dtype: {fixed_tensor.dtype}")
            
            # Test PIL conversion (this is where the error occurred)
            numpy_array = fixed_tensor.numpy()
            if numpy_array.ndim == 3 and numpy_array.shape[2] == 1:
                numpy_array = numpy_array.squeeze(2)
            
            # Convert to uint8 for PIL
            numpy_array = np.clip(numpy_array * 255, 0, 255).astype(np.uint8)
            
            # Test PIL Image creation
            pil_image = Image.fromarray(numpy_array)
            print(f"  PIL conversion: SUCCESS - {pil_image.size}")
            
        except Exception as e:
            print(f"  PIL conversion: FAILED - {e}")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    test_image_dimensions()
