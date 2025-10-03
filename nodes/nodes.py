import torch
from ..utils.light_estimator import CategoricalLightEstimator
from ..utils.luma_mask_processor import LumaMaskProcessor
from ..utils.debug_visualizer import DebugVisualizer
from ..utils.ire_shadow_analyzer import IREShadowAnalyzer
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
                "x_threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "y_threshold_upper": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "y_threshold_lower": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "central_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "hard_light_threshold": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "soft_light_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shadow_ire_threshold": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 50.0, "step": 1.0}),
                "transition_sensitivity": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "format_mode": (["auto", "manual"], {"default": "auto"}),
                "normal_standard": (["OpenGL", "DirectX", "World_Space", "Object_Space"], {"default": "OpenGL"}),
                "analysis_method": (["advanced", "legacy", "combined"], {"default": "combined"}),
                "exclusion_mask": ("MASK",),
                "ire_analysis_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = (
        # Original outputs
        "STRING", "STRING", "STRING",  # x_dir, y_dir, combined_dir
        "STRING", "STRING", "STRING", "STRING", "STRING",  # hard_soft + confidences
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",  # debug_mask, lit_normals, delta_chart, x_preview, y_preview
        # IRE analysis outputs (simplified)
        "IMAGE", "IMAGE", "IMAGE",  # false_color_ire, shadow_mask, soft_shadow_mask
        "STRING", "STRING", "FLOAT", "FLOAT"  # shadow_character, transition_quality, soft_ratio, hard_ratio
    )

    RETURN_NAMES = (
        # Original outputs
        "x_direction", "y_direction", "combined_direction",
        "hard_soft_index", "x_confidence", "y_confidence", 
        "overall_confidence", "spread_value",
        "debug_mask", "lit_normals_viz", "cluster_delta_chart", "x_threshold_preview", "y_threshold_preview",
        # IRE analysis outputs (simplified)
        "false_color_ire", "shadow_mask", "soft_shadow_mask",
        "shadow_character", "transition_quality", "soft_ratio", "hard_ratio"
    )
    
    FUNCTION = "estimate_lighting"
    
    def estimate_lighting(self, normal_map, luma_image, luma_threshold, curve_type,
                        x_threshold, y_threshold_upper, y_threshold_lower, central_threshold,
                        hard_light_threshold, soft_light_threshold, shadow_ire_threshold, transition_sensitivity,
                        format_mode="auto", normal_standard="OpenGL", analysis_method="combined", exclusion_mask=None, ire_analysis_weight=0.5):
        """
        Enhanced categorical light direction estimation.
        """
        # Downscale the larger image to match the smaller one
        if normal_map.shape[1] > luma_image.shape[1] or normal_map.shape[2] > luma_image.shape[2]:
            normal_map = torch.nn.functional.interpolate(normal_map.permute(0, 3, 1, 2), size=(luma_image.shape[1], luma_image.shape[2]), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        elif luma_image.shape[1] > normal_map.shape[1] or luma_image.shape[2] > normal_map.shape[2]:
            luma_image = torch.nn.functional.interpolate(luma_image.permute(0, 3, 1, 2), size=(normal_map.shape[1], normal_map.shape[2]), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        luma_processor = LumaMaskProcessor()
        light_estimator = CategoricalLightEstimator(
            x_threshold, y_threshold_upper, y_threshold_lower, central_threshold,
            hard_light_threshold, soft_light_threshold,
            format_mode, normal_standard
        )
        
        # Process luma mask
        luma_mask = luma_processor.process_with_curves(
            luma_image, curve_type, None, luma_threshold
        )
        
        # Apply exclusion mask if provided
        if exclusion_mask is not None:
            print(f"Exclusion mask provided: {exclusion_mask.shape}")
            # Ensure exclusion mask is the same size as other images
            if exclusion_mask.shape[1] != luma_mask.shape[1] or exclusion_mask.shape[2] != luma_mask.shape[2]:
                # Handle different mask formats
                if exclusion_mask.dim() == 3:  # [B, H, W] format
                    exclusion_mask = exclusion_mask.unsqueeze(1)  # Add channel dimension: [B, 1, H, W]
                elif exclusion_mask.dim() == 2:  # [H, W] format
                    exclusion_mask = exclusion_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                
                # Now interpolate with proper 4D format
                exclusion_mask = torch.nn.functional.interpolate(
                    exclusion_mask, 
                    size=(luma_mask.shape[1], luma_mask.shape[2]), 
                    mode='nearest'
                )
                
                # Convert back to original format
                if exclusion_mask.shape[1] == 1:  # Remove channel dimension if it was added
                    exclusion_mask = exclusion_mask.squeeze(1)  # [B, H, W]
                if exclusion_mask.shape[0] == 1:  # Remove batch dimension if it was added
                    exclusion_mask = exclusion_mask.squeeze(0)  # [H, W]
            
            # Combine luma mask with exclusion mask (keep where exclusion_mask is True/white)
            mask = luma_mask & exclusion_mask.bool()
            included_pixels = exclusion_mask.sum().item()
            total_pixels = exclusion_mask.numel()
            print(f"Included {included_pixels}/{total_pixels} pixels from exclusion mask ({included_pixels/total_pixels*100:.1f}%)")
        else:
            mask = luma_mask
            print("No exclusion mask provided")
        
        # Extract lit normals for histogram generation
        current_mask = mask.bool()
        lit_normals = normal_map[current_mask]

        # Choose analysis method
        print(f"=== ComfyUI Light Estimation ===")
        print(f"Analysis method: {analysis_method}")
        print(f"Normal map shape: {normal_map.shape}")
        print(f"Luma image shape: {luma_image.shape}")
        print(f"Final mask coverage: {mask.sum().item()}/{mask.numel()} pixels ({mask.float().mean().item():.3f})")
        print(f"X threshold: {x_threshold}")
        print(f"Y thresholds: upper={y_threshold_upper}, lower={y_threshold_lower}")
        
        if analysis_method == "legacy":
            print("Using LEGACY analysis method (simple mean-based)")
            results = light_estimator.legacy_analysis(normal_map, mask)
        else:
            print("Using ADVANCED analysis method (clustering-based)")
            results = light_estimator.analyze_directional_categories(normal_map, mask)
        
        # Perform IRE shadow analysis if combined or IRE-only method
        ire_results = {}
        if analysis_method in ["combined", "ire_only"]:
            print("--- Performing IRE Shadow Analysis ---")
            ire_analyzer = IREShadowAnalyzer(
                shadow_ire_threshold=shadow_ire_threshold,
                transition_sensitivity=transition_sensitivity
            )
            ire_results = ire_analyzer.generate_ire_analysis_report(luma_image)
            print(f"IRE Shadow Character: {ire_results['shadow_classification']['shadow_character']}")
            print(f"IRE Transition Quality: {ire_results['shadow_classification']['transition_quality']}")

        # Ensure all outputs are strings
        x_direction = str(results['x_category'])
        y_direction = str(results['y_category'])
        combined_direction = f"{y_direction}-{x_direction}"
        
        # Log raw confidence values for debugging
        raw_x_conf = results['confidence']['x_confidence']
        raw_y_conf = results['confidence']['y_confidence']
        raw_overall_conf = results['confidence']['overall_confidence']
        raw_spread = results['quality_analysis']['spread']
        
        print(f"=== RAW CONFIDENCE VALUES ===")
        print(f"X confidence: {raw_x_conf:.4f}")
        print(f"Y confidence: {raw_y_conf:.4f}")
        print(f"Overall confidence: {raw_overall_conf:.4f}")
        print(f"Spread value: {raw_spread:.4f}")
        print(f"Hard/soft index: {results['hard_soft_index']:.4f}")
        
        # Convert numeric values to text descriptions
        hard_soft_index = self._hard_soft_to_text(results['hard_soft_index'])
        x_confidence = self._confidence_to_text(raw_x_conf)
        y_confidence = self._confidence_to_text(raw_y_conf)
        overall_confidence = self._confidence_to_text(raw_overall_conf)
        spread_value = self._spread_to_text(raw_spread)
        
        print(f"=== TEXT CONVERSIONS ===")
        print(f"Hard/soft index text: {hard_soft_index}")
        print(f"X confidence text: {x_confidence}")
        print(f"Y confidence text: {y_confidence}")
        print(f"Overall confidence text: {overall_confidence}")
        print(f"Spread text: {spread_value}")

        debug_mask = DebugVisualizer.generate_debug_mask(mask)
        lit_normals_viz = DebugVisualizer.generate_lit_normals_visualization(normal_map, mask)
        # Generate threshold-based classification chart
        cluster_delta_chart = DebugVisualizer.create_threshold_classification_chart(normal_map, x_threshold, y_threshold_upper, y_threshold_lower, mask)
        
        # Generate threshold preview images
        x_threshold_preview = self.create_x_threshold_preview(normal_map, x_threshold)
        y_threshold_preview = self.create_y_threshold_preview(normal_map, y_threshold_upper, y_threshold_lower)
        
        print(f"Generated X threshold preview with threshold: {x_threshold}")
        print(f"Generated Y threshold preview with upper: {y_threshold_upper}, lower: {y_threshold_lower}")
        
        # Prepare IRE analysis outputs
        if ire_results:
            # IRE visualizations - apply mask to false color IRE
            false_color_ire_raw = ire_results['false_color_visualization']
            # Apply the same mask used for normal map analysis to IRE visualization
            # This ensures IRE false color only shows in areas being analyzed (like debug_mask)
            false_color_ire = false_color_ire_raw * mask.unsqueeze(-1).float()
            
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
            
            if soft_shadow_mask_raw.shape != luma_image.shape[:3]:
                soft_shadow_mask_raw = torch.nn.functional.interpolate(
                    soft_shadow_mask_raw.unsqueeze(0).unsqueeze(0).float(),
                    size=(luma_image.shape[1], luma_image.shape[2]),
                    mode='nearest'
                ).squeeze(0).squeeze(0)
            
            shadow_mask = shadow_mask_raw.unsqueeze(-1).float()
            soft_shadow_mask = soft_shadow_mask_raw.unsqueeze(-1).float()
            
            # IRE text outputs
            shadow_character = ire_results['shadow_classification']['shadow_character']
            transition_quality = ire_results['shadow_classification']['transition_quality']
            
            # IRE numerical outputs
            soft_ratio = ire_results['transition_analysis']['soft_shadow_ratio'].item()
            hard_ratio = ire_results['transition_analysis']['hard_shadow_ratio'].item()
            
            # IRE analysis completed successfully
            print("--- IRE Analysis Completed ---")
        else:
            # Default values when no IRE analysis
            false_color_ire = torch.zeros_like(luma_image)
            shadow_mask = torch.zeros_like(luma_image[..., :1])
            soft_shadow_mask = torch.zeros_like(luma_image[..., :1])
            shadow_character = "N/A"
            transition_quality = "N/A"
            soft_ratio = 0.0
            hard_ratio = 0.0

        # Ensure all image outputs have valid dimensions and data types
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
        
        # Apply safety checks to all image outputs
        debug_mask = ensure_valid_image(debug_mask, luma_image.shape)
        lit_normals_viz = ensure_valid_image(lit_normals_viz, luma_image.shape)
        cluster_delta_chart = ensure_valid_image(cluster_delta_chart, luma_image.shape)
        x_threshold_preview = ensure_valid_image(x_threshold_preview, luma_image.shape)
        y_threshold_preview = ensure_valid_image(y_threshold_preview, luma_image.shape)
        false_color_ire = ensure_valid_image(false_color_ire, luma_image.shape)
        shadow_mask = ensure_valid_image(shadow_mask, luma_image.shape)
        soft_shadow_mask = ensure_valid_image(soft_shadow_mask, luma_image.shape)

        return (
            # Original outputs
            x_direction, y_direction, combined_direction,
            hard_soft_index, x_confidence, y_confidence, 
            overall_confidence, spread_value,
            debug_mask, lit_normals_viz, cluster_delta_chart, x_threshold_preview, y_threshold_preview,
            # IRE analysis outputs (simplified)
            false_color_ire, shadow_mask, soft_shadow_mask,
            shadow_character, transition_quality, soft_ratio, hard_ratio
        )
    
    @staticmethod
    def _confidence_to_text(confidence_value):
        """
        Convert numeric confidence value to descriptive text.
        """
        if confidence_value >= 0.8:
            return "Very High"
        elif confidence_value >= 0.6:
            return "High"
        elif confidence_value >= 0.4:
            return "Medium"
        elif confidence_value >= 0.2:
            return "Low"
        else:
            return "Very Low"
    
    @staticmethod
    def _spread_to_text(spread_value):
        """
        Convert numeric spread value to descriptive text.
        """
        if spread_value >= 0.8:
            return "Very Wide"
        elif spread_value >= 0.6:
            return "Wide"
        elif spread_value >= 0.4:
            return "Medium"
        elif spread_value >= 0.2:
            return "Narrow"
        else:
            return "Very Narrow"
    
    @staticmethod
    def _hard_soft_to_text(hard_soft_index):
        """
        Convert numeric hard/soft index to descriptive text.
        """
        if hard_soft_index >= 0.8:
            return "Very Soft"
        elif hard_soft_index >= 0.6:
            return "Soft"
        elif hard_soft_index >= 0.4:
            return "Medium"
        elif hard_soft_index >= 0.2:
            return "Hard"
        else:
            return "Very Hard"
    
    
    @staticmethod
    def create_x_threshold_preview(normal_map, x_threshold):
        """
        Create X threshold preview image showing left/center/right zones.
        Red = Left, Green = Center, Blue = Right
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        import io
        
        # Convert normal map to -1 to +1 range if needed
        if normal_map.max() <= 1.0:
            normals_x = (normal_map[:, :, :, 0] * 2.0) - 1.0  # Convert 0-1 to -1 to +1
        else:
            normals_x = normal_map[:, :, :, 0]
        
        # Create classification mask
        left_mask = normals_x < -x_threshold
        right_mask = normals_x > x_threshold
        center_mask = ~(left_mask | right_mask)
        
        # Create RGB image
        height, width = normals_x.shape[1], normals_x.shape[2]
        preview = torch.zeros((1, height, width, 3), dtype=torch.float32)
        
        # Red for left, Green for center, Blue for right
        preview[0, :, :, 0] = left_mask.float()  # Red channel
        preview[0, :, :, 1] = center_mask.float()  # Green channel  
        preview[0, :, :, 2] = right_mask.float()  # Blue channel
        
        # Add labels using matplotlib
        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(preview[0].numpy())
        ax.set_title(f'X Threshold Preview (threshold={x_threshold})', fontsize=12, pad=10)
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='red', label='Light from Right'),
            plt.Rectangle((0,0),1,1, facecolor='green', label='Center'),
            plt.Rectangle((0,0),1,1, facecolor='blue', label='Light from Left')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Convert to tensor
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        pil_img = Image.open(buf).convert('RGB')
        img_array = np.array(pil_img).astype(np.float32) / 255.0
        tensor_img = torch.from_numpy(img_array).unsqueeze(0)
        
        return tensor_img
    
    @staticmethod
    def create_y_threshold_preview(normal_map, y_threshold_upper, y_threshold_lower):
        """
        Create Y threshold preview image showing above/center/below zones.
        Red = Above, Green = Center, Blue = Below
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        import io
        
        # Convert normal map to -1 to +1 range if needed
        if normal_map.max() <= 1.0:
            normals_y = (normal_map[:, :, :, 1] * 2.0) - 1.0  # Convert 0-1 to -1 to +1
        else:
            normals_y = normal_map[:, :, :, 1]
        
        # Create classification mask using separate thresholds
        above_mask = normals_y > y_threshold_upper  # Positive Y = surfaces pointing up = light from above
        below_mask = normals_y < -y_threshold_lower  # Negative Y = surfaces pointing down = light from below
        center_mask = ~(above_mask | below_mask)
        
        # Create RGB image
        height, width = normals_y.shape[1], normals_y.shape[2]
        preview = torch.zeros((1, height, width, 3), dtype=torch.float32)
        
        # Red for above, Green for center, Blue for below
        preview[0, :, :, 0] = above_mask.float()  # Red channel (Light from Above)
        preview[0, :, :, 1] = center_mask.float()  # Green channel (Center)
        preview[0, :, :, 2] = below_mask.float()  # Blue channel (Light from Below)
        
        # Add labels using matplotlib
        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(preview[0].numpy())
        ax.set_title(f'Y Threshold Preview (upper={y_threshold_upper}, lower={y_threshold_lower})', fontsize=12, pad=10)
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='red', label='Light from Above'),
            plt.Rectangle((0,0),1,1, facecolor='green', label='Center'),
            plt.Rectangle((0,0),1,1, facecolor='blue', label='Light from Below')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Convert to tensor
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        pil_img = Image.open(buf).convert('RGB')
        img_array = np.array(pil_img).astype(np.float32) / 255.0
        tensor_img = torch.from_numpy(img_array).unsqueeze(0)
        
        return tensor_img


# ============================================================================
# MODULAR NODES - SEPARATED CONCERNS
# ============================================================================

# Import the separated nodes
from .separated_nodes import LightImageProcessor, LightDistributionAnalyzer
from .ire_shadow_nodes import IREShadowAnalyzerNode, IREShadowComparisonNode
from .enhanced_light_estimator import EnhancedLightEstimator

# Make them available for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "NormalMapLightEstimator": NormalMapLightEstimator,
    "LightImageProcessor": LightImageProcessor,
    "LightDistributionAnalyzer": LightDistributionAnalyzer,
    "IREShadowAnalyzer": IREShadowAnalyzerNode,
    "IREShadowComparison": IREShadowComparisonNode,
    "EnhancedLightEstimator": EnhancedLightEstimator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NormalMapLightEstimator": "Normal Map Light Estimator (Monolithic)",
    "LightImageProcessor": "Light Image Processor",
    "LightDistributionAnalyzer": "Light Distribution Analyzer",
    "IREShadowAnalyzer": "IRE Shadow Analyzer",
    "IREShadowComparison": "IRE Shadow Comparison",
    "EnhancedLightEstimator": "Enhanced Light Estimator (IRE + Normal)",
}
