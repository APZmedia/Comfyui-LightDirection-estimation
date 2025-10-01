import torch
import numpy as np
from ..utils.light_estimator import CategoricalLightEstimator
from ..utils.luma_mask_processor import LumaMaskProcessor
from ..utils.debug_visualizer import DebugVisualizer

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
                "luma_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "curve_type": (["linear", "s_curve", "exponential", "logarithmic"], {"default": "s_curve"}),
                # Directional thresholds
                "x_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "y_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "central_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                # Hard/Soft classification thresholds
                "hard_light_threshold": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "soft_light_threshold": ("FLOAT", {
                    "default": 0.35,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "use_weighted": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "custom_curve_points": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
            }
        }
    
    RETURN_TYPES = (
        "STRING",     # x_direction
        "STRING",     # y_direction
        "STRING",     # combined_direction
        "FLOAT",      # hard_soft_index
        "FLOAT",      # x_confidence
        "FLOAT",      # y_confidence
        "FLOAT",      # overall_confidence
        "FLOAT",      # spread_value
        "IMAGE",      # debug_mask
        "IMAGE",      # directional_visualization
    )
    
    RETURN_NAMES = (
        "x_direction",
        "y_direction",
        "combined_direction",
        "hard_soft_index",
        "x_confidence",
        "y_confidence",
        "overall_confidence",
        "spread_value",
        "debug_mask",
        "directional_viz"
    )
    
    FUNCTION = "estimate_lighting"
    
    def estimate_lighting(self, normal_map, luma_image, luma_threshold, curve_type,
                        x_threshold, y_threshold, central_threshold,
                        hard_light_threshold, soft_light_threshold, use_weighted,
                        custom_curve_points=""):
        """
        Enhanced categorical light direction estimation with hard/soft index.
        """
        # Initialize processors
        luma_processor = LumaMaskProcessor()
        light_estimator = CategoricalLightEstimator()
        
        # Set all thresholds
        light_estimator.x_threshold = x_threshold
        light_estimator.y_threshold = y_threshold
        light_estimator.central_threshold = central_threshold
        light_estimator.hard_light_threshold = hard_light_threshold
        light_estimator.soft_light_threshold = soft_light_threshold
        
        # Process luma mask with curves
        if curve_type == "custom" and custom_curve_points:
            curve_points = self._parse_curve_points(custom_curve_points)
            mask = luma_processor.process_with_curves(
                luma_image, "custom", curve_points, luma_threshold
            )
        else:
            mask = luma_processor.process_with_curves(
                luma_image, curve_type, None, luma_threshold
            )
        
        # Analyze directional categories with hard/soft index
        categorical_results = light_estimator.analyze_directional_categories(
            normal_map, mask
        )
        
        # Extract results for each batch
        x_directions = []
        y_directions = []
        combined_directions = []
        hard_soft_indices = []
        x_confidences = []
        y_confidences = []
        overall_confidences = []
        spread_values = []
        
        for result in categorical_results:
            x_directions.append(result['x_category'])
            y_directions.append(result['y_category'])
            combined_directions.append(result['combined_category'])
            hard_soft_indices.append(result['hard_soft_index'])
            x_confidences.append(result['confidence']['x_confidence'])
            y_confidences.append(result['confidence']['y_confidence'])
            overall_confidences.append(result['confidence']['overall_confidence'])
            spread_values.append(result['quality_analysis']['spread'])
        
        # Generate debug outputs
        debug_mask = DebugVisualizer.generate_debug_mask(mask)
        directional_viz = DebugVisualizer.generate_directional_visualization(
            normal_map, categorical_results
        )
        
        return (
            x_directions,
            y_directions,
            combined_directions,
            torch.tensor(hard_soft_indices),
            torch.tensor(x_confidences),
            torch.tensor(y_confidences),
            torch.tensor(overall_confidences),
            torch.tensor(spread_values),
            debug_mask,
            directional_viz
        )
    
    def _parse_curve_points(self, curve_points_str):
        """
        Parse custom curve points from string input.
        
        Args:
            curve_points_str: String with curve points in format "x,y" per line
        
        Returns:
            curve_points: List of (x, y) tuples
        """
        try:
            points = []
            for line in curve_points_str.strip().split('\n'):
                if line.strip():
                    x, y = map(float, line.strip().split(','))
                    points.append((x, y))
            return points
        except:
            # Return default linear curve if parsing fails
            return [(0.0, 0.0), (1.0, 1.0)]
