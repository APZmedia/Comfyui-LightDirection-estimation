# NormalMapLightEstimator - Design Notes

## 🎯 Project Overview

A ComfyUI custom node for estimating light direction and quality from normal maps using luma masking. The system analyzes surface normals to infer lighting information for downstream tasks like adaptive relighting, directional masking, or stylized effects.

## 🧠 Core Algorithm

### Physics-Based Approach
- **Normal Map (RGB)** → Surface orientation (XYZ)
- **Luma Mask** → Which areas are actually lit
- **Filtered Normals** → Only consider lit surface orientations
- **Statistical Analysis** → Infer light direction and quality

### Key Insight
Lit surface normals point **TOWARD** the light source. By averaging these directions, we get the light direction.

## 📦 Functional Requirements

### Node Name
`NormalMapLightEstimator`

### Inputs
| Name | Type | Description |
|------|------|-------------|
| `normal_map` | IMAGE | RGB normal map in tangent space (0–255) |
| `luma_image` | IMAGE | Grayscale or RGB image to extract luminance |
| `luma_threshold` | FLOAT | Threshold to mask out non-lit areas (default 0.5) |
| `curve_type` | COMBO | Luma processing curve: linear, s_curve, exponential, logarithmic |
| `x_threshold` | FLOAT | Sensitivity for left/right detection (default 0.1) |
| `y_threshold` | FLOAT | Sensitivity for top/bottom detection (default 0.1) |
| `central_threshold` | FLOAT | Threshold for central lighting detection (default 0.3) |
| `hard_light_threshold` | FLOAT | Below this = hard light (default 0.15) |
| `soft_light_threshold` | FLOAT | Above this = soft light (default 0.35) |
| `use_weighted` | BOOLEAN | Use weighted averaging instead of binary masking |
| `custom_curve_points` | STRING | Custom curve points for advanced luma processing |

### Outputs
| Name | Type | Description |
|------|------|-------------|
| `x_direction` | STRING | "left", "central", "right" |
| `y_direction` | STRING | "top", "central", "bottom" |
| `combined_direction` | STRING | "top-left", "top-central", "top-right", etc. |
| `hard_soft_index` | FLOAT | 0.0 (hard) to 1.0 (soft) |
| `x_confidence` | FLOAT | Confidence in X direction classification |
| `y_confidence` | FLOAT | Confidence in Y direction classification |
| `overall_confidence` | FLOAT | Combined confidence score |
| `spread_value` | FLOAT | Raw numerical spread value |
| `debug_mask` | IMAGE | Binary mask showing which normals were used |
| `directional_viz` | IMAGE | Color-coded visualization of results |

## ⚙️ Processing Pipeline

### 1. Normal Map Processing
- Convert RGB to XYZ: `normals = ((RGB / 255) * 2.0) - 1.0`
- Apply axis inversions for different normal conventions
- Support for OpenGL vs DirectX normal maps

### 2. Luma Mask Processing with Curves Strategy
- **Linear**: Simple threshold-like behavior
- **S-curve**: Enhance mid-tones, compress highlights/shadows
- **Exponential**: Emphasize bright areas
- **Logarithmic**: Compress highlights, expand shadows
- **Custom**: User-defined curve points

### 3. Directional Analysis
- **X Direction**: Count normals in left/right/central quadrants
- **Y Direction**: Count normals in top/bottom/central quadrants
- **Central Detection**: Identify omnidirectional lighting patterns

### 4. Hard/Soft Classification
- **Spread Analysis**: Use covariance analysis of lit normals
- **Hard/Soft Index**: Continuous value from 0.0 (hard) to 1.0 (soft)
- **Threshold Control**: User-adjustable hard/soft boundaries

## 🎛️ Threshold Controls

### Directional Thresholds
- **X Threshold**: Controls left/right detection sensitivity
- **Y Threshold**: Controls top/bottom detection sensitivity
- **Central Threshold**: Detects central lighting patterns

### Quality Thresholds
- **Hard Light Threshold**: Below this = hard light (index 0.0)
- **Soft Light Threshold**: Above this = soft light (index 1.0)
- **Between**: Linear interpolation from 0.0 to 1.0

## 🎯 Output Categories

### X Direction
- **"left"**: Light coming from the left
- **"central"**: Central/omnidirectional lighting
- **"right"**: Light coming from the right

### Y Direction
- **"top"**: Light coming from above
- **"central"**: Central/omnidirectional lighting
- **"bottom"**: Light coming from below

### Combined Directions
- **"top-left"**, **"top-central"**, **"top-right"**
- **"central-left"**, **"central-central"**, **"central-right"**
- **"bottom-left"**, **"bottom-central"**, **"bottom-right"**

### Hard/Soft Index
- **0.0**: Hard light (tight cluster of normals)
- **0.5**: Intermediate light
- **1.0**: Soft light (wide distribution of normals)

## 🔧 Technical Implementation

### Modular Architecture
- **`normal_map_processor.py`**: Normal map processing (leverage existing ComfyUI nodes)
- **`luma_mask_processor.py`**: Luma processing with curves strategy
- **`light_estimator.py`**: Core light estimation algorithm
- **`debug_visualizer.py`**: Debug visualization tools
- **`nodes.py`**: Main ComfyUI node class

### Key Algorithms
1. **Directional Analysis**: Quadrant-based normal distribution analysis
2. **Spread Calculation**: Covariance analysis of lit normals
3. **Confidence Scoring**: Statistical confidence in classifications
4. **Curves Processing**: Advanced luma masking with user control

## 📊 Usage Examples

### Basic Directional Detection
```
X Threshold: 0.1 (sensitive to left/right)
Y Threshold: 0.1 (sensitive to top/bottom)
Central Threshold: 0.3 (detect central lighting)
```

### Hard Light Detection
```
Hard Threshold: 0.1 (very sensitive)
Soft Threshold: 0.2 (narrow range)
Result: Index 0.0-0.5 range
```

### Soft Light Detection
```
Hard Threshold: 0.2 (less sensitive)
Soft Threshold: 0.5 (wider range)
Result: Index 0.5-1.0 range
```

## 🎨 Visualization System

### Debug Mask
- White pixels: Lit areas used in analysis
- Black pixels: Masked out areas

### Directional Visualization
- **Red Channel**: X direction (left=255, central=192, right=128)
- **Green Channel**: Y direction (top=255, central=192, bottom=128)
- **Blue Channel**: Hard/soft index (hard=255, soft=128)
- **Confidence**: Applied as transparency effect

## 🚀 Future Enhancements

### Phase 2 Features
- Weighted averaging using luma intensity
- Multi-scale processing for different detail levels
- Polar histogram output for normal distribution
- Integration with existing ComfyUI lighting nodes

### Phase 3 Features
- Real-time threshold adjustment
- Batch processing optimization
- Advanced curve presets
- Export/import threshold configurations

## 📝 Implementation Notes

### ComfyUI Integration
- **Category**: "Custom/Lighting"
- **Dependencies**: torch, numpy, Pillow
- **Auto-installation**: Dependencies install automatically
- **Error Handling**: Graceful fallbacks for edge cases

### Performance Considerations
- Optimized for ComfyUI tensor operations
- Batch processing support
- Memory-efficient covariance calculations
- GPU acceleration where possible

### Edge Cases Handled
- No lit areas: Returns default central direction
- Single normal: Uses that direction
- Collinear normals: Handles degenerate cases gracefully
- Low confidence: Provides reliability indicators
