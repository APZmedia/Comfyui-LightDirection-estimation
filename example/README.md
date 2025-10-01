# Example Workflows

This directory contains example workflows demonstrating the NormalMapLightEstimator node.

## Files

- `example_workflow.json` - Basic workflow showing node usage
- `README.md` - This file

## Usage

1. Load the example workflow in ComfyUI
2. Connect your normal map and luma image
3. Adjust parameters as needed
4. Run the workflow to see results

## Example Parameters

### Basic Setup
- `luma_threshold`: 0.5 (default)
- `curve_type`: "s_curve" (enhances mid-tones)
- `x_threshold`: 0.1 (sensitive to left/right)
- `y_threshold`: 0.1 (sensitive to top/bottom)

### Hard Light Detection
- `hard_light_threshold`: 0.1 (very sensitive)
- `soft_light_threshold`: 0.2 (narrow range)

### Soft Light Detection  
- `hard_light_threshold`: 0.2 (less sensitive)
- `soft_light_threshold`: 0.5 (wider range)

## Outputs

The node provides:
- Categorical directions (X, Y, combined)
- Hard/soft index (0.0 to 1.0)
- Confidence scores
- Debug visualizations
