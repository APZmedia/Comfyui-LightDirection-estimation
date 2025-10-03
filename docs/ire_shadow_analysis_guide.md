# IRE Shadow Analysis Guide

## Overview

This guide explains the new IRE (Institute of Radio Engineers) stops false color approach for shadow detection in sRGB/display-referred images. This method provides precise analysis of shadow characteristics including soft vs hard shadow transitions.

## What is IRE Analysis?

IRE (Institute of Radio Engineers) is a standardized scale used in video and broadcast to measure luminance levels. The scale ranges from 0 IRE (pure black) to 100 IRE (pure white), with specific ranges corresponding to different exposure zones.

### IRE Color Mapping for sRGB Images

| IRE Range | Color | Description | Use Case |
|----------|-------|-------------|----------|
| 0-10 IRE | Purple/Blue | Deep shadows | Very dark areas, crushed blacks |
| 10-20 IRE | Blue | Shadows | Main shadow areas |
| 20-30 IRE | Cyan | Mid-shadows | Shadow transitions |
| 30-50 IRE | Green | Low-midtones | Properly exposed areas |
| 50-70 IRE | Yellow | Midtones | Well-lit areas |
| 70-90 IRE | Orange | High-midtones | Bright areas |
| 90-100 IRE | Red | Highlights | Very bright areas, potential clipping |

## Key Features

### 1. False Color Visualization
- Maps IRE values to distinct colors for easy visual analysis
- Helps identify exposure zones and shadow characteristics
- Provides immediate feedback on lighting distribution

### 2. Shadow Transition Analysis
- Analyzes gradient patterns in shadow areas
- Detects soft vs hard shadow transitions
- Measures transition smoothness using gradient magnitude

### 3. Soft vs Hard Shadow Classification
- **Soft Shadows**: Gradual transitions (low gradients)
- **Hard Shadows**: Abrupt transitions (high gradients)
- Provides quantitative ratios and qualitative descriptions

## ComfyUI Nodes

### IREShadowAnalyzer Node

**Purpose**: Analyze shadow characteristics using IRE false color approach

**Inputs**:
- `image`: sRGB image to analyze
- `shadow_ire_threshold`: IRE value below which pixels are considered shadows (default: 20.0)
- `transition_sensitivity`: Sensitivity for detecting shadow transitions (default: 0.1)
- `analysis_mode`: Analysis scope (full, shadows_only, transitions_only)

**Outputs**:
- `false_color_ire`: False color visualization of IRE values
- `shadow_mask`: Binary mask of shadow areas
- `soft_shadow_mask`: Areas with soft shadow transitions
- `hard_shadow_mask`: Areas with hard shadow transitions
- `ire_legend`: Color mapping legend
- `shadow_character`: Classification (Very Soft, Soft, Medium, Hard)
- `transition_quality`: Description of transition smoothness
- `ire_range`: IRE value range in the image
- `shadow_coverage`: Percentage of shadow coverage
- `gradient_analysis`: Description of gradient characteristics

### IREShadowComparison Node

**Purpose**: Compare shadow characteristics between two images

**Inputs**:
- `image_a`, `image_b`: Images to compare
- `shadow_ire_threshold`: IRE threshold for shadow detection
- `comparison_mode`: Comparison visualization method

**Outputs**:
- `ire_comparison`: Side-by-side or difference visualization
- `shadow_difference`: Difference in shadow areas
- `transition_comparison`: Difference in transition characteristics
- Comparative statistics and classifications

### EnhancedLightEstimator Node

**Purpose**: Combined analysis using both normal maps and IRE shadow analysis

**Inputs**:
- All inputs from NormalMapLightEstimator
- `shadow_ire_threshold`: IRE threshold for shadow detection
- `transition_sensitivity`: Transition analysis sensitivity
- `ire_analysis_weight`: Weight for IRE analysis in combined results (0.0-1.0)

**Outputs**:
- All outputs from NormalMapLightEstimator
- All outputs from IREShadowAnalyzer
- `final_shadow_character`: Combined shadow classification
- `final_light_quality`: Combined light quality assessment
- `combined_confidence`: Overall analysis confidence
- `analysis_consistency`: Consistency between normal map and IRE analysis

## Usage Examples

### Basic Shadow Analysis

```python
# Initialize analyzer
analyzer = IREShadowAnalyzer(
    shadow_ire_threshold=20.0,
    transition_sensitivity=0.1
)

# Analyze image
results = analyzer.generate_ire_analysis_report(srgb_image)

# Extract shadow characteristics
shadow_character = results['shadow_classification']['shadow_character']
transition_quality = results['shadow_classification']['transition_quality']
soft_ratio = results['transition_analysis']['soft_shadow_ratio']
```

### Shadow Transition Analysis

```python
# Convert sRGB to IRE values
ire_values = analyzer.srgb_to_ire(srgb_image)

# Analyze transitions
transition_analysis = analyzer.analyze_shadow_transitions(ire_values)

# Classify shadow characteristics
classification = analyzer.classify_shadow_characteristics(transition_analysis)
```

### False Color Visualization

```python
# Create false color visualization
false_color = analyzer.create_false_color_visualization(ire_values)

# Create IRE legend
legend = analyzer.create_ire_legend()
```

## Shadow Classification System

### Shadow Character Classifications

| Classification | Soft Ratio | Description |
|----------------|------------|-------------|
| Very Soft | > 0.7 | Predominantly gradual transitions |
| Soft | 0.5 - 0.7 | Mostly gradual transitions |
| Medium | 0.3 - 0.5 | Mixed transition types |
| Hard | < 0.3 | Predominantly sharp transitions |

### Transition Quality Descriptions

| Quality | Mean Gradient | Description |
|---------|--------------|-------------|
| Very Gradual | < 5.0 | Very smooth transitions |
| Gradual | 5.0 - 15.0 | Smooth transitions |
| Moderate | 15.0 - 30.0 | Moderate transitions |
| Sharp | 30.0 - 50.0 | Sharp transitions |
| Very Sharp | > 50.0 | Very abrupt transitions |

## Best Practices

### 1. IRE Threshold Selection
- **Low threshold (10-15 IRE)**: Detects only very dark shadows
- **Medium threshold (15-25 IRE)**: Balanced shadow detection
- **High threshold (25-35 IRE)**: Includes mid-tone shadows

### 2. Transition Sensitivity
- **Low sensitivity (0.05-0.1)**: Detects only very obvious transitions
- **Medium sensitivity (0.1-0.2)**: Balanced transition detection
- **High sensitivity (0.2-0.5)**: Detects subtle transitions

### 3. Analysis Mode Selection
- **Full**: Complete analysis with all visualizations
- **Shadows Only**: Focus on shadow areas only
- **Transitions Only**: Focus on transition analysis

### 4. Combined Analysis
- Use `EnhancedLightEstimator` for comprehensive analysis
- Adjust `ire_analysis_weight` based on image characteristics
- Monitor `analysis_consistency` for result reliability

## Troubleshooting

### Common Issues

**"No shadows detected"**
- Lower the `shadow_ire_threshold`
- Check if image has sufficient contrast
- Verify image is in sRGB format

**"Incorrect shadow classification"**
- Adjust `transition_sensitivity`
- Check IRE value distribution
- Verify shadow areas are properly exposed

**"Poor transition analysis"**
- Increase image resolution
- Check for noise in shadow areas
- Adjust gradient analysis parameters

### Performance Optimization

- Use appropriate image resolution (512x512 to 1024x1024 recommended)
- Process images in batches for efficiency
- Cache IRE color mappings for repeated analysis
- Use GPU acceleration when available

## Technical Details

### IRE Conversion Formula
```python
# For sRGB display-referred images
ire_value = luma * 100.0
```

### Gradient Analysis
```python
# Sobel gradient calculation
grad_x = F.conv2d(ire_values, sobel_x_filter)
grad_y = F.conv2d(ire_values, sobel_y_filter)
gradient_magnitude = sqrt(grad_x² + grad_y²)
```

### Shadow Classification
```python
# Soft shadow detection
soft_shadow_mask = (gradient_magnitude < soft_threshold) & shadow_mask
soft_ratio = soft_shadow_mask.sum() / shadow_mask.sum()
```

## Integration with Existing System

The IRE shadow analysis integrates seamlessly with the existing light estimation system:

1. **Complementary Analysis**: IRE analysis provides luminance-based insights while normal map analysis provides geometric insights
2. **Combined Results**: Enhanced estimator combines both approaches for comprehensive analysis
3. **Consistency Checking**: System can detect when analyses agree or disagree
4. **Weighted Results**: User can control the influence of each analysis method

This approach provides a robust, professional-grade shadow analysis system suitable for cinematography, photography, and computer graphics applications.
