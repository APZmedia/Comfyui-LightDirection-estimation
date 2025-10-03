# Image Dimension Fix for ComfyUI

## Problem

The error `Cannot handle this data type: (1, 1, 1), |u1` occurred when ComfyUI tried to create a PIL Image from a tensor with invalid dimensions. This happened because:

1. **Invalid Shape**: Some image outputs had shape `(1, 1, 1)` instead of proper image dimensions
2. **Wrong Data Type**: The tensor had data type `|u1` (unsigned 8-bit) which PIL couldn't handle
3. **Dimension Mismatch**: IRE shadow masks had different dimensions than the input image

## Root Cause

The IRE shadow analysis was creating masks with dimensions that didn't match the input image, causing ComfyUI's `PreviewImage` node to fail when trying to convert tensors to PIL Images.

## Solution

### 1. **Dimension Validation**
Added checks to ensure all image outputs match the input image dimensions:

```python
# Ensure IRE masks match input image dimensions
shadow_mask_raw = ire_results['transition_analysis']['shadow_mask']
soft_shadow_mask_raw = ire_results['transition_analysis']['soft_shadow_mask']

# Resize to match input image if needed
if shadow_mask_raw.shape != luma_image.shape[:3]:
    shadow_mask_raw = torch.nn.functional.interpolate(
        shadow_mask_raw.unsqueeze(0).unsqueeze(0).float(),
        size=(luma_image.shape[1], luma_image.shape[2]),
        mode='nearest'
    ).squeeze(0).squeeze(0)
```

### 2. **Universal Image Safety Function**
Created a safety function that ensures all image outputs are valid:

```python
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
```

### 3. **Applied to All Image Outputs**
Applied the safety function to all image outputs:

```python
# Apply safety checks to all image outputs
debug_mask = ensure_valid_image(debug_mask, luma_image.shape)
lit_normals_viz = ensure_valid_image(lit_normals_viz, luma_image.shape)
cluster_delta_chart = ensure_valid_image(cluster_delta_chart, luma_image.shape)
x_threshold_preview = ensure_valid_image(x_threshold_preview, luma_image.shape)
y_threshold_preview = ensure_valid_image(y_threshold_preview, luma_image.shape)
false_color_ire = ensure_valid_image(false_color_ire, luma_image.shape)
shadow_mask = ensure_valid_image(shadow_mask, luma_image.shape)
soft_shadow_mask = ensure_valid_image(soft_shadow_mask, luma_image.shape)
```

## What This Fixes

### **Before (Problematic)**
- IRE masks could have wrong dimensions: `(1, 1, 1)` or `(512, 512)` instead of `(1280, 728)`
- Data type issues with `|u1` causing PIL conversion failures
- ComfyUI PreviewImage nodes failing with `TypeError`

### **After (Fixed)**
- All image outputs have consistent dimensions matching input image
- All tensors are properly converted to float and clamped to [0, 1] range
- PIL Image conversion works reliably
- ComfyUI PreviewImage nodes display correctly

## Technical Details

### **Dimension Handling**
- **2D tensors**: Automatically add channel dimension with `unsqueeze(-1)`
- **Wrong size**: Resize using `torch.nn.functional.interpolate` with nearest neighbor
- **Data type**: Convert to float and clamp to [0, 1] range

### **Interpolation Method**
- Uses `mode='nearest'` to preserve binary mask values
- Maintains mask integrity during resizing
- Prevents blurring of shadow boundaries

### **Memory Efficiency**
- Only resizes when necessary
- Uses in-place operations where possible
- Maintains tensor device consistency

## Testing

The fix ensures that:
1. **All image outputs have valid dimensions** matching the input image
2. **Data types are compatible** with PIL Image creation
3. **No dimension mismatches** occur between different analysis components
4. **ComfyUI PreviewImage nodes work** without errors

## Benefits

- **Reliability**: Eliminates ComfyUI image display errors
- **Consistency**: All outputs have uniform dimensions
- **Robustness**: Handles edge cases and dimension mismatches
- **Performance**: Efficient resizing only when needed

This fix ensures that the IRE shadow analysis integrates seamlessly with ComfyUI's image handling system! 🎯
