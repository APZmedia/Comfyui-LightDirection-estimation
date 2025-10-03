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
                "x_threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "y_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "central_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "hard_light_threshold": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "soft_light_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "format_mode": (["auto", "manual"], {"default": "auto"}),
                "normal_standard": (["OpenGL", "DirectX", "World_Space", "Object_Space"], {"default": "OpenGL"}),
                "analysis_method": (["advanced", "legacy"], {"default": "advanced"}),
            }
        }
    
    RETURN_TYPES = (
        "STRING", "STRING", "STRING",  # x_dir, y_dir, combined_dir
        "STRING", "STRING", "STRING", "STRING", "STRING",  # hard_soft + confidences
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE"  # debug_mask, lit_normals, delta_chart, x_preview, y_preview
    )

    RETURN_NAMES = (
        "x_direction", "y_direction", "combined_direction",
        "hard_soft_index", "x_confidence", "y_confidence", 
        "overall_confidence", "spread_value",
        "debug_mask", "lit_normals_viz", "cluster_delta_chart", "x_threshold_preview", "y_threshold_preview"
    )
    
    FUNCTION = "estimate_lighting"
    
    def estimate_lighting(self, normal_map, luma_image, luma_threshold, curve_type,
                        x_threshold, y_threshold, central_threshold,
                        hard_light_threshold, soft_light_threshold,
                        format_mode="auto", normal_standard="OpenGL", analysis_method="advanced"):
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

        # Choose analysis method
        print(f"=== ComfyUI Light Estimation ===")
        print(f"Analysis method: {analysis_method}")
        print(f"Normal map shape: {normal_map.shape}")
        print(f"Luma image shape: {luma_image.shape}")
        print(f"Mask coverage: {mask.sum().item()}/{mask.numel()} pixels ({mask.float().mean().item():.3f})")
        print(f"X threshold: {x_threshold}")
        print(f"Y threshold: {y_threshold}")
        
        if analysis_method == "legacy":
            print("Using LEGACY analysis method (simple mean-based)")
            results = light_estimator.legacy_analysis(normal_map, mask)
        else:
            print("Using ADVANCED analysis method (clustering-based)")
            results = light_estimator.analyze_directional_categories(normal_map, mask)

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
        # Generate cluster delta visualization
        cluster_delta_chart = DebugVisualizer.create_cluster_delta_chart(results)
        
        # Generate threshold preview images
        x_threshold_preview = self.create_x_threshold_preview(normal_map, x_threshold)
        y_threshold_preview = self.create_y_threshold_preview(normal_map, y_threshold)
        
        print(f"Generated X threshold preview with threshold: {x_threshold}")
        print(f"Generated Y threshold preview with threshold: {y_threshold}")

        return (
            x_direction, y_direction, combined_direction,
            hard_soft_index, x_confidence, y_confidence, 
            overall_confidence, spread_value,
            debug_mask, lit_normals_viz, cluster_delta_chart, x_threshold_preview, y_threshold_preview
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
    def create_y_threshold_preview(normal_map, y_threshold):
        """
        Create Y threshold preview image showing down/center/up zones.
        Red = Down, Green = Center, Blue = Up
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
        
        # Create classification mask
        down_mask = normals_y < -y_threshold
        up_mask = normals_y > y_threshold
        center_mask = ~(down_mask | up_mask)
        
        # Create RGB image
        height, width = normals_y.shape[1], normals_y.shape[2]
        preview = torch.zeros((1, height, width, 3), dtype=torch.float32)
        
        # Red for down, Green for center, Blue for up
        preview[0, :, :, 0] = down_mask.float()  # Red channel
        preview[0, :, :, 1] = center_mask.float()  # Green channel
        preview[0, :, :, 2] = up_mask.float()  # Blue channel
        
        # Add labels using matplotlib
        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(preview[0].numpy())
        ax.set_title(f'Y Threshold Preview (threshold={y_threshold})', fontsize=12, pad=10)
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='red', label='Down'),
            plt.Rectangle((0,0),1,1, facecolor='green', label='Center'),
            plt.Rectangle((0,0),1,1, facecolor='blue', label='Up')
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
