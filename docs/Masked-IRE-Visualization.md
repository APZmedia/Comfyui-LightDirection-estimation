# Masked IRE False Color Visualization

## Overview

The IRE false color visualization now includes **masking** to show only the areas being analyzed, similar to other preview outputs like `debug_mask` and `lit_normals_viz`.

## What Changed

### Before
- `false_color_ire` showed IRE false color for the **entire image**
- Areas not being analyzed still showed IRE colors
- Could be confusing as it included background/unlit areas

### After  
- `false_color_ire` shows IRE false color **only in analyzed areas**
- Uses the same mask as `debug_mask` and `lit_normals_viz`
- Consistent with other preview outputs
- More accurate representation of actual analysis

## Technical Implementation

```python
# Get raw IRE false color visualization
false_color_ire_raw = ire_results['false_color_visualization']

# Apply the same mask used for normal map analysis
false_color_ire = false_color_ire_raw * mask.unsqueeze(-1).float()
```

## Visual Result

### Masked IRE Visualization
- **White areas**: IRE false color analysis (same as `debug_mask`)
- **Black areas**: Masked out (not analyzed)
- **Color coding**: IRE values only in analyzed regions

### IRE Color Zones (in analyzed areas only)
| IRE Range | Color | Description |
|-----------|-------|-------------|
| 0-10 IRE | Purple/Blue | Deep shadows |
| 10-20 IRE | Blue | Shadows |
| 20-30 IRE | Cyan | Mid-shadows |
| 30-50 IRE | Green | Low-midtones |
| 50-70 IRE | Yellow | Midtones |
| 70-90 IRE | Orange | High-midtones |
| 90-100 IRE | Red | Bright highlights |

## Benefits

### 1. **Consistency**
- Matches other preview outputs (`debug_mask`, `lit_normals_viz`)
- Only shows analysis in relevant areas
- Eliminates confusion from background areas

### 2. **Accuracy**
- IRE analysis only applies to lit/analyzed areas
- False color represents actual analysis scope
- More meaningful for debugging and validation

### 3. **Professional Workflow**
- Aligns with broadcast/video industry standards
- IRE analysis focused on subject areas
- Cleaner visual feedback

## Usage

The masked IRE visualization is automatically applied when using:
- `analysis_method="combined"` (default)
- `analysis_method="ire_only"`

The masking uses the same logic as:
- `debug_mask`: Shows which pixels are analyzed
- `lit_normals_viz`: Shows normal analysis areas
- `luma_mask`: Shows luminance analysis areas

## Comparison

### Unmasked (Previous)
```
[Entire Image IRE Colors]
- Background: IRE colors (misleading)
- Subject: IRE colors (accurate)
- Shadows: IRE colors (accurate)
```

### Masked (Current)
```
[Only Analyzed Areas IRE Colors]
- Background: Black (masked out)
- Subject: IRE colors (accurate)
- Shadows: IRE colors (accurate)
```

This change makes the IRE false color visualization more accurate and consistent with the overall analysis workflow.
