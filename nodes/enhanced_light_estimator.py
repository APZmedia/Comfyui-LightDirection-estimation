import torch
import torch.nn.functional as F
from ..utils.light_estimator import CategoricalLightEstimator
from ..utils.luma_mask_processor import LumaMaskProcessor
from ..utils.ire_shadow_analyzer import IREShadowAnalyzer
from ..utils.debug_visualizer import DebugVisualizer

class EnhancedLightEstimator:
    """
    Enhanced light estimator that combines normal map analysis with IRE shadow analysis.
    Provides comprehensive lighting analysis using both geometric and luminance-based approaches.
    """
    
    CATEGORY = "Custom/Lighting/Enhanced"
    
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
        "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING",
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        # IRE analysis outputs
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
        "STRING", "STRING", "STRING", "STRING", "STRING",
        "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT",
        # Combined analysis
        "STRING", "STRING", "FLOAT", "FLOAT"
    )
    
    RETURN_NAMES = (
        # Original outputs
        "x_direction", "y_direction", "combined_direction", "hard_soft_index", 
        "x_confidence", "y_confidence", "overall_confidence", "spread_value",
        "debug_mask", "lit_normals_viz", "cluster_delta_chart", "x_threshold_preview", "y_threshold_preview",
        # IRE analysis outputs
        "false_color_ire", "shadow_mask", "soft_shadow_mask", "hard_shadow_mask", "ire_legend",
        "shadow_character", "transition_quality", "ire_range", "shadow_coverage", "gradient_analysis",
        "mean_ire", "shadow_percentage", "soft_ratio", "hard_ratio", "mean_gradient",
        # Combined analysis
        "final_shadow_character", "final_light_quality", "combined_confidence", "analysis_consistency"
    )
    
    FUNCTION = "enhanced_light_analysis"
    
    def enhanced_light_analysis(self, normal_map, luma_image, luma_threshold, curve_type,
                              x_threshold, y_threshold_upper, y_threshold_lower, central_threshold,
                              hard_light_threshold, soft_light_threshold, shadow_ire_threshold, transition_sensitivity,
                              format_mode="auto", normal_standard="OpenGL", analysis_method="combined",
                              exclusion_mask=None, ire_analysis_weight=0.5):
        """
        Perform enhanced light analysis combining normal map and IRE shadow analysis.
        """
        print(f"=== ENHANCED LIGHT ANALYSIS ===")
        print(f"Analysis method: {analysis_method}")
        print(f"IRE analysis weight: {ire_analysis_weight}")
        
        # Resize images to match
        if normal_map.shape[1] > luma_image.shape[1] or normal_map.shape[2] > luma_image.shape[2]:
            normal_map = torch.nn.functional.interpolate(normal_map.permute(0, 3, 1, 2), 
                                                       size=(luma_image.shape[1], luma_image.shape[2]), 
                                                       mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        elif luma_image.shape[1] > normal_map.shape[1] or luma_image.shape[2] > normal_map.shape[2]:
            luma_image = torch.nn.functional.interpolate(luma_image.permute(0, 3, 1, 2), 
                                                       size=(normal_map.shape[1], normal_map.shape[2]), 
                                                       mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        
        # Initialize processors
        luma_processor = LumaMaskProcessor()
        light_estimator = CategoricalLightEstimator(
            x_threshold, y_threshold_upper, y_threshold_lower, central_threshold,
            hard_light_threshold, soft_light_threshold, format_mode, normal_standard
        )
        ire_analyzer = IREShadowAnalyzer(shadow_ire_threshold=shadow_ire_threshold, 
                                       transition_sensitivity=transition_sensitivity)
        
        # Process luma mask
        luma_mask = luma_processor.process_with_curves(luma_image, curve_type, None, luma_threshold)
        
        # Apply exclusion mask if provided
        if exclusion_mask is not None:
            if exclusion_mask.shape[1] != luma_mask.shape[1] or exclusion_mask.shape[2] != luma_mask.shape[2]:
                if exclusion_mask.dim() == 3:
                    exclusion_mask = exclusion_mask.unsqueeze(1)
                elif exclusion_mask.dim() == 2:
                    exclusion_mask = exclusion_mask.unsqueeze(0).unsqueeze(0)
                
                exclusion_mask = torch.nn.functional.interpolate(
                    exclusion_mask, 
                    size=(luma_mask.shape[1], luma_mask.shape[2]), 
                    mode='nearest'
                )
                
                if exclusion_mask.shape[1] == 1:
                    exclusion_mask = exclusion_mask.squeeze(1)
                if exclusion_mask.shape[0] == 1:
                    exclusion_mask = exclusion_mask.squeeze(0)
            
            mask = luma_mask & exclusion_mask.bool()
        else:
            mask = luma_mask
        
        # Perform normal map analysis
        print("--- Performing Normal Map Analysis ---")
        if analysis_method == "legacy":
            normal_results = light_estimator.legacy_analysis(normal_map, mask)
        else:
            normal_results = light_estimator.analyze_directional_categories(normal_map, mask)
        
        # Perform IRE shadow analysis
        print("--- Performing IRE Shadow Analysis ---")
        ire_results = ire_analyzer.generate_ire_analysis_report(luma_image)
        
        # Extract IRE analysis results
        ire_values = ire_results['ire_values']
        false_color = ire_results['false_color_visualization']
        transition_analysis = ire_results['transition_analysis']
        shadow_classification = ire_results['shadow_classification']
        ire_stats = ire_results['ire_statistics']
        
        # Create IRE visualization masks
        shadow_mask = transition_analysis['shadow_mask'].unsqueeze(-1).float()
        soft_shadow_mask = transition_analysis['soft_shadow_mask'].unsqueeze(-1).float()
        hard_shadow_mask = transition_analysis['hard_shadow_mask'].unsqueeze(-1).float()
        ire_legend = ire_analyzer.create_ire_legend()
        
        # Combine analyses if requested
        if analysis_method == "combined":
            print("--- Combining Analysis Results ---")
            combined_results = self._combine_analyses(normal_results, ire_results, ire_analysis_weight)
        else:
            combined_results = {
                'final_shadow_character': shadow_classification['shadow_character'],
                'final_light_quality': normal_results.get('hard_soft_index', 0.5),
                'combined_confidence': (normal_results['confidence']['overall_confidence'] + 
                                      (1.0 - transition_analysis['mean_shadow_gradient'].item() / 50.0)) / 2,
                'analysis_consistency': 0.5  # Default consistency
            }
        
        # Generate original visualizations
        debug_mask = DebugVisualizer.generate_debug_mask(mask)
        lit_normals_viz = DebugVisualizer.generate_lit_normals_visualization(normal_map, mask)
        cluster_delta_chart = DebugVisualizer.create_threshold_classification_chart(
            normal_map, x_threshold, y_threshold_upper, y_threshold_lower, mask)
        x_threshold_preview = self.create_x_threshold_preview(normal_map, x_threshold)
        y_threshold_preview = self.create_y_threshold_preview(normal_map, y_threshold_upper, y_threshold_lower)
        
        # Convert numeric values to text
        hard_soft_index = self._hard_soft_to_text(normal_results.get('hard_soft_index', 0.5))
        x_confidence = self._confidence_to_text(normal_results['confidence']['x_confidence'])
        y_confidence = self._confidence_to_text(normal_results['confidence']['y_confidence'])
        overall_confidence = self._confidence_to_text(normal_results['confidence']['overall_confidence'])
        spread_value = self._spread_to_text(normal_results['quality_analysis']['spread'])
        
        # IRE text classifications
        shadow_character = shadow_classification['shadow_character']
        transition_quality = shadow_classification['transition_quality']
        
        # IRE range and coverage
        min_ire = ire_stats['min_ire']
        max_ire = ire_stats['max_ire']
        ire_range = f"{min_ire:.1f}-{max_ire:.1f} IRE"
        
        shadow_pct = ire_stats['shadow_percentage']
        if shadow_pct < 10:
            shadow_coverage = "Minimal Shadows"
        elif shadow_pct < 25:
            shadow_coverage = "Light Shadows"
        elif shadow_pct < 50:
            shadow_coverage = "Moderate Shadows"
        elif shadow_pct < 75:
            shadow_coverage = "Heavy Shadows"
        else:
            shadow_coverage = "Dominant Shadows"
        
        # Gradient analysis
        mean_gradient = transition_analysis['mean_shadow_gradient'].item()
        if mean_gradient < 5:
            gradient_analysis = "Very Gradual Transitions"
        elif mean_gradient < 15:
            gradient_analysis = "Gradual Transitions"
        elif mean_gradient < 30:
            gradient_analysis = "Moderate Transitions"
        elif mean_gradient < 50:
            gradient_analysis = "Sharp Transitions"
        else:
            gradient_analysis = "Very Sharp Transitions"
        
        print(f"=== ANALYSIS COMPLETE ===")
        print(f"Normal map direction: {normal_results['x_category']}-{normal_results['y_category']}")
        print(f"IRE shadow character: {shadow_character}")
        print(f"Combined shadow character: {combined_results['final_shadow_character']}")
        
        return (
            # Original outputs
            str(normal_results['x_category']), str(normal_results['y_category']), 
            f"{normal_results['y_category']}-{normal_results['x_category']}",
            hard_soft_index, x_confidence, y_confidence, overall_confidence, spread_value,
            debug_mask, lit_normals_viz, cluster_delta_chart, x_threshold_preview, y_threshold_preview,
            # IRE analysis outputs
            false_color, shadow_mask, soft_shadow_mask, hard_shadow_mask, ire_legend,
            shadow_character, transition_quality, ire_range, shadow_coverage, gradient_analysis,
            ire_stats['mean_ire'], shadow_pct, transition_analysis['soft_shadow_ratio'].item(),
            transition_analysis['hard_shadow_ratio'].item(), mean_gradient,
            # Combined analysis
            combined_results['final_shadow_character'], str(combined_results['final_light_quality']),
            combined_results['combined_confidence'], combined_results['analysis_consistency']
        )
    
    def _combine_analyses(self, normal_results, ire_results, ire_weight):
        """
        Combine normal map and IRE shadow analysis results.
        """
        # Extract key values
        normal_hard_soft = normal_results.get('hard_soft_index', 0.5)
        ire_soft_ratio = ire_results['transition_analysis']['soft_shadow_ratio'].item()
        ire_hard_ratio = ire_results['transition_analysis']['hard_shadow_ratio'].item()
        ire_gradient = ire_results['transition_analysis']['mean_shadow_gradient'].item()
        
        # Weight the analyses
        combined_softness = (1 - ire_weight) * normal_hard_soft + ire_weight * ire_soft_ratio
        
        # Determine final shadow character
        if combined_softness > 0.7:
            final_shadow_character = "Very Soft"
        elif combined_softness > 0.5:
            final_shadow_character = "Soft"
        elif combined_softness > 0.3:
            final_shadow_character = "Medium"
        else:
            final_shadow_character = "Hard"
        
        # Calculate consistency between analyses
        normal_confidence = normal_results['confidence']['overall_confidence']
        ire_confidence = 1.0 - min(1.0, ire_gradient / 50.0)  # Convert gradient to confidence
        consistency = 1.0 - abs(normal_hard_soft - ire_soft_ratio)
        
        return {
            'final_shadow_character': final_shadow_character,
            'final_light_quality': combined_softness,
            'combined_confidence': (normal_confidence + ire_confidence) / 2,
            'analysis_consistency': consistency
        }
    
    @staticmethod
    def _confidence_to_text(confidence_value):
        """Convert numeric confidence to text."""
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
        """Convert numeric spread to text."""
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
        """Convert hard/soft index to text."""
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
        """Create X threshold preview image."""
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
        import io
        
        if normal_map.max() <= 1.0:
            normals_x = (normal_map[:, :, :, 0] * 2.0) - 1.0
        else:
            normals_x = normal_map[:, :, :, 0]
        
        left_mask = normals_x < -x_threshold
        right_mask = normals_x > x_threshold
        center_mask = ~(left_mask | right_mask)
        
        height, width = normals_x.shape[1], normals_x.shape[2]
        preview = torch.zeros((1, height, width, 3), dtype=torch.float32)
        
        preview[0, :, :, 0] = left_mask.float()
        preview[0, :, :, 1] = center_mask.float()
        preview[0, :, :, 2] = right_mask.float()
        
        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(preview[0].numpy())
        ax.set_title(f'X Threshold Preview (threshold={x_threshold})', fontsize=12, pad=10)
        
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='red', label='Light from Right'),
            plt.Rectangle((0,0),1,1, facecolor='green', label='Center'),
            plt.Rectangle((0,0),1,1, facecolor='blue', label='Light from Left')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        ax.set_xticks([])
        ax.set_yticks([])
        
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
        """Create Y threshold preview image."""
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
        import io
        
        if normal_map.max() <= 1.0:
            normals_y = (normal_map[:, :, :, 1] * 2.0) - 1.0
        else:
            normals_y = normal_map[:, :, :, 1]
        
        above_mask = normals_y > y_threshold_upper
        below_mask = normals_y < -y_threshold_lower
        center_mask = ~(above_mask | below_mask)
        
        height, width = normals_y.shape[1], normals_y.shape[2]
        preview = torch.zeros((1, height, width, 3), dtype=torch.float32)
        
        preview[0, :, :, 0] = above_mask.float()
        preview[0, :, :, 1] = center_mask.float()
        preview[0, :, :, 2] = below_mask.float()
        
        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(preview[0].numpy())
        ax.set_title(f'Y Threshold Preview (upper={y_threshold_upper}, lower={y_threshold_lower})', fontsize=12, pad=10)
        
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='red', label='Light from Above'),
            plt.Rectangle((0,0),1,1, facecolor='green', label='Center'),
            plt.Rectangle((0,0),1,1, facecolor='blue', label='Light from Below')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        pil_img = Image.open(buf).convert('RGB')
        img_array = np.array(pil_img).astype(np.float32) / 255.0
        tensor_img = torch.from_numpy(img_array).unsqueeze(0)
        
        return tensor_img
