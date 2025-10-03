import torch
import torch.nn.functional as F
from ..utils.ire_shadow_analyzer import IREShadowAnalyzer
from ..utils.debug_visualizer import DebugVisualizer

class IREShadowAnalyzerNode:
    """
    ComfyUI node for IRE-based shadow analysis using false color approach.
    Designed for sRGB/display-referred images.
    """
    
    CATEGORY = "Custom/Lighting/IRE Analysis"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "shadow_ire_threshold": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 50.0, "step": 1.0}),
                "transition_sensitivity": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "analysis_mode": (["full", "shadows_only", "transitions_only"], {"default": "full"}),
            },
            "optional": {
                "ire_range_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 1.0}),
                "ire_range_max": ("FLOAT", {"default": 100.0, "min": 50.0, "max": 100.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = (
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",  # Visualizations
        "STRING", "STRING", "STRING", "STRING", "STRING",  # Classifications
        "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT"  # Statistics
    )
    
    RETURN_NAMES = (
        "false_color_ire", "shadow_mask", "soft_shadow_mask", "hard_shadow_mask", "ire_legend",
        "shadow_character", "transition_quality", "ire_range", "shadow_coverage", "gradient_analysis",
        "mean_ire", "shadow_percentage", "soft_ratio", "hard_ratio", "mean_gradient"
    )
    
    FUNCTION = "analyze_ire_shadows"
    
    def analyze_ire_shadows(self, image, shadow_ire_threshold, transition_sensitivity, 
                          analysis_mode="full", ire_range_min=0.0, ire_range_max=100.0):
        """
        Perform IRE-based shadow analysis on sRGB image.
        """
        # Initialize IRE analyzer
        analyzer = IREShadowAnalyzer(
            ire_range=(ire_range_min, ire_range_max),
            shadow_ire_threshold=shadow_ire_threshold,
            transition_sensitivity=transition_sensitivity
        )
        
        # Generate comprehensive analysis
        analysis_results = analyzer.generate_ire_analysis_report(image)
        
        # Extract results
        ire_values = analysis_results['ire_values']
        false_color = analysis_results['false_color_visualization']
        transition_analysis = analysis_results['transition_analysis']
        shadow_classification = analysis_results['shadow_classification']
        ire_stats = analysis_results['ire_statistics']
        
        # Create visualization masks
        shadow_mask = transition_analysis['shadow_mask'].unsqueeze(-1).float()
        soft_shadow_mask = transition_analysis['soft_shadow_mask'].unsqueeze(-1).float()
        hard_shadow_mask = transition_analysis['hard_shadow_mask'].unsqueeze(-1).float()
        
        # Create IRE legend
        ire_legend = analyzer.create_ire_legend()
        
        # Generate text classifications
        shadow_character = shadow_classification['shadow_character']
        transition_quality = shadow_classification['transition_quality']
        
        # Calculate IRE range description
        min_ire = ire_stats['min_ire']
        max_ire = ire_stats['max_ire']
        ire_range = f"{min_ire:.1f}-{max_ire:.1f} IRE"
        
        # Shadow coverage description
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
        
        # Gradient analysis description
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
        
        # Extract numerical values
        mean_ire = ire_stats['mean_ire']
        shadow_percentage = shadow_pct
        soft_ratio = transition_analysis['soft_shadow_ratio'].item()
        hard_ratio = transition_analysis['hard_shadow_ratio'].item()
        mean_gradient_val = mean_gradient
        
        return (
            false_color, shadow_mask, soft_shadow_mask, hard_shadow_mask, ire_legend,
            shadow_character, transition_quality, ire_range, shadow_coverage, gradient_analysis,
            mean_ire, shadow_percentage, soft_ratio, hard_ratio, mean_gradient_val
        )

class IREShadowComparisonNode:
    """
    ComfyUI node for comparing shadow characteristics between two images using IRE analysis.
    """
    
    CATEGORY = "Custom/Lighting/IRE Analysis"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "shadow_ire_threshold": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 50.0, "step": 1.0}),
                "comparison_mode": (["side_by_side", "difference", "ratio"], {"default": "side_by_side"}),
            }
        }
    
    RETURN_TYPES = (
        "IMAGE", "IMAGE", "IMAGE", "IMAGE",  # Visualizations
        "STRING", "STRING", "STRING", "STRING",  # Comparisons
        "FLOAT", "FLOAT", "FLOAT", "FLOAT"  # Statistics
    )
    
    RETURN_NAMES = (
        "ire_comparison", "shadow_difference", "transition_comparison", "ire_legend",
        "shadow_character_a", "shadow_character_b", "transition_quality_a", "transition_quality_b",
        "shadow_ratio_a", "shadow_ratio_b", "gradient_ratio_a", "gradient_ratio_b"
    )
    
    FUNCTION = "compare_ire_shadows"
    
    def compare_ire_shadows(self, image_a, image_b, shadow_ire_threshold, comparison_mode):
        """
        Compare shadow characteristics between two images using IRE analysis.
        """
        # Initialize analyzer
        analyzer = IREShadowAnalyzer(shadow_ire_threshold=shadow_ire_threshold)
        
        # Analyze both images
        analysis_a = analyzer.generate_ire_analysis_report(image_a)
        analysis_b = analyzer.generate_ire_analysis_report(image_b)
        
        # Extract key results
        ire_a = analysis_a['ire_values']
        ire_b = analysis_b['ire_values']
        false_color_a = analysis_a['false_color_visualization']
        false_color_b = analysis_b['false_color_visualization']
        
        transition_a = analysis_a['transition_analysis']
        transition_b = analysis_b['transition_analysis']
        
        classification_a = analysis_a['shadow_classification']
        classification_b = analysis_b['shadow_classification']
        
        # Create comparison visualizations
        if comparison_mode == "side_by_side":
            # Concatenate images side by side
            ire_comparison = torch.cat([false_color_a, false_color_b], dim=2)
        elif comparison_mode == "difference":
            # Show IRE difference
            ire_diff = torch.abs(ire_a - ire_b)
            ire_comparison = analyzer.create_false_color_visualization(ire_diff)
        else:  # ratio
            # Show IRE ratio (avoid division by zero)
            ire_ratio = ire_a / (ire_b + 1e-8)
            ire_ratio = torch.clamp(ire_ratio, 0, 2.0)  # Clamp to reasonable range
            ire_comparison = analyzer.create_false_color_visualization(ire_ratio * 50)  # Scale to IRE range
        
        # Shadow difference analysis
        shadow_mask_a = transition_a['shadow_mask']
        shadow_mask_b = transition_b['shadow_mask']
        shadow_difference = torch.abs(shadow_mask_a.float() - shadow_mask_b.float()).unsqueeze(-1)
        
        # Transition comparison
        grad_a = transition_a['gradient_magnitude']
        grad_b = transition_b['gradient_magnitude']
        transition_comparison = torch.abs(grad_a - grad_b).unsqueeze(-1)
        
        # Create legend
        ire_legend = analyzer.create_ire_legend()
        
        # Extract classifications
        shadow_character_a = classification_a['shadow_character']
        shadow_character_b = classification_b['shadow_character']
        transition_quality_a = classification_a['transition_quality']
        transition_quality_b = classification_b['transition_quality']
        
        # Calculate ratios
        shadow_ratio_a = transition_a['soft_shadow_ratio'].item()
        shadow_ratio_b = transition_b['soft_shadow_ratio'].item()
        gradient_ratio_a = transition_a['mean_shadow_gradient'].item()
        gradient_ratio_b = transition_b['mean_shadow_gradient'].item()
        
        return (
            ire_comparison, shadow_difference, transition_comparison, ire_legend,
            shadow_character_a, shadow_character_b, transition_quality_a, transition_quality_b,
            shadow_ratio_a, shadow_ratio_b, gradient_ratio_a, gradient_ratio_b
        )

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "IREShadowAnalyzer": IREShadowAnalyzerNode,
    "IREShadowComparison": IREShadowComparisonNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IREShadowAnalyzer": "IRE Shadow Analyzer",
    "IREShadowComparison": "IRE Shadow Comparison",
}
