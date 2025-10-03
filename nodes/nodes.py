import torch
from ..utils.light_estimator import CategoricalLightEstimator
from ..utils.luma_mask_processor import LumaMaskProcessor
from ..utils.debug_visualizer import DebugVisualizer
from .separated_nodes import LightImageProcessor, LightDistributionAnalyzer

class NormalMapLightEstimator:
    """
    ComfyUI custom node for estimating light direction and quality from normal maps.
    """
    
    CATEGORY = "Custom/Lighting"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal_map": ("IMAGE",),
                "luma_image": ("IMAGE",),
                "luma_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "curve_type": (["linear", "s_curve", "exponential", "logarithmic"], {"default": "s_curve"}),
                "x_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "y_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "central_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "hard_light_threshold": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "soft_light_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "format_mode": (["auto", "manual"], {"default": "auto"}),
                "normal_standard": (["OpenGL", "DirectX", "World_Space", "Object_Space"], {"default": "OpenGL"}),
            }
        }
    
    RETURN_TYPES = (
        "STRING", "STRING", "STRING", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT",
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
    )

    RETURN_NAMES = (
        "x_direction", "y_direction", "combined_direction", "hard_soft_index",
        "x_confidence", "y_confidence", "overall_confidence", "spread_value",
        "debug_mask", "directional_viz", "lit_normals_viz", "colormap_preview",
        "histogram_before", "histogram_after", "color_histogram",
    )
    
    FUNCTION = "estimate_lighting"
    
    def estimate_lighting(self, normal_map, luma_image, luma_threshold, curve_type,
                        x_threshold, y_threshold, central_threshold,
                        hard_light_threshold, soft_light_threshold,
                        format_mode="auto", normal_standard="OpenGL"):
        """
        Enhanced categorical light direction estimation.
        """
        luma_processor = LumaMaskProcessor()
        light_estimator = CategoricalLightEstimator(
            x_threshold, y_threshold, central_threshold,
            hard_light_threshold, soft_light_threshold,
            format_mode, normal_standard
        )
        
        mask = luma_processor.process_with_curves(
            luma_image, curve_type, None, luma_threshold
        )
        
        # Extract lit normals for histogram generation
        current_mask = mask.bool()
        lit_normals = normal_map[current_mask]

        results = light_estimator.analyze_directional_categories(normal_map, mask)

        x_direction = results['x_category']
        y_direction = results['y_category']
        combined_direction = f"{y_direction}-{x_direction}"
        hard_soft_index = results['hard_soft_index']
        x_confidence = results['confidence']['x_confidence']
        y_confidence = results['confidence']['y_confidence']
        overall_confidence = results['confidence']['overall_confidence']
        spread_value = results['quality_analysis']['spread']

        debug_mask = DebugVisualizer.generate_debug_mask(mask)
        directional_viz = DebugVisualizer.generate_directional_visualization(normal_map, results)
        lit_normals_viz = DebugVisualizer.generate_lit_normals_visualization(normal_map, mask)
        colormap_preview = DebugVisualizer.generate_colormap_preview()

        # Generate separate histograms for before/after masking
        histogram_before = DebugVisualizer.generate_histogram_before(normal_map)
        # Ensure lit_normals is in correct format for histogram generation
        if lit_normals.dim() == 2:
            lit_normals_4d = lit_normals.unsqueeze(0)  # Add batch dimension
        else:
            lit_normals_4d = lit_normals
        histogram_after = DebugVisualizer.generate_histogram_after(lit_normals_4d)
        color_histogram = DebugVisualizer.generate_color_histogram(normal_map)  # Combined view

        return (
            x_direction, y_direction, combined_direction, hard_soft_index,
            x_confidence, y_confidence, overall_confidence, spread_value,
            debug_mask, directional_viz, lit_normals_viz, colormap_preview,
            histogram_before, histogram_after, color_histogram
        )


# ============================================================================
# MODULAR NODES - SEPARATED CONCERNS
# ============================================================================

# Import the separated nodes
from .separated_nodes import LightImageProcessor, LightDistributionAnalyzer

# Make them available for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "NormalMapLightEstimator": NormalMapLightEstimator,
    "LightImageProcessor": LightImageProcessor,
    "LightDistributionAnalyzer": LightDistributionAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NormalMapLightEstimator": "Normal Map Light Estimator (Monolithic)",
    "LightImageProcessor": "Light Image Processor",
    "LightDistributionAnalyzer": "Light Distribution Analyzer",
}
