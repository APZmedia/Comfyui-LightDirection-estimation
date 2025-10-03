import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, Tuple, List, Optional

class IREShadowAnalyzer:
    """
    IRE stops false color analysis for shadow detection in sRGB/display-referred images.
    
    This analyzer uses IRE (Institute of Radio Engineers) values to map luminance levels
    to false colors, enabling precise analysis of shadow characteristics including
    soft vs hard shadow transitions.
    """
    
    def __init__(self, 
                 ire_range: Tuple[float, float] = (0.0, 100.0),
                 shadow_ire_threshold: float = 20.0,
                 transition_sensitivity: float = 0.1):
        """
        Initialize IRE Shadow Analyzer for sRGB images.
        
        Args:
            ire_range: IRE value range (min, max) - typically (0, 100)
            shadow_ire_threshold: IRE value below which pixels are considered shadows
            transition_sensitivity: Sensitivity for detecting shadow transitions
        """
        self.ire_range = ire_range
        self.shadow_ire_threshold = shadow_ire_threshold
        self.transition_sensitivity = transition_sensitivity
        
        # Define IRE color mapping for display-referred sRGB
        self.ire_color_map = self._create_ire_color_map()
        
        # Sobel filters for gradient analysis
        self.sobel_x = torch.tensor([[[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype=torch.float32)
        self.sobel_y = torch.tensor([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype=torch.float32)
    
    def _create_ire_color_map(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Create IRE to false color mapping for sRGB display-referred images.
        
        Returns:
            Dictionary mapping IRE ranges to RGB colors
        """
        return {
            # Deep shadows (0-10 IRE) - Purple/Blue
            'deep_shadow': (0.2, 0.0, 0.4),      # Dark purple
            'shadow': (0.1, 0.0, 0.6),           # Purple
            
            # Mid-shadows (10-20 IRE) - Blue
            'mid_shadow': (0.0, 0.0, 0.8),       # Blue
            
            # Low-midtones (20-30 IRE) - Cyan
            'low_midtone': (0.0, 0.4, 0.8),      # Cyan
            
            # Midtones (30-50 IRE) - Green
            'midtone': (0.0, 0.6, 0.0),          # Green
            
            # High-midtones (50-70 IRE) - Yellow
            'high_midtone': (0.8, 0.8, 0.0),      # Yellow
            
            # Highlights (70-90 IRE) - Orange
            'highlight': (1.0, 0.4, 0.0),        # Orange
            
            # Bright highlights (90-100 IRE) - Red
            'bright_highlight': (1.0, 0.0, 0.0)  # Red
        }
    
    def srgb_to_ire(self, srgb_image: torch.Tensor) -> torch.Tensor:
        """
        Convert sRGB values to IRE values for display-referred analysis.
        
        Args:
            srgb_image: sRGB image tensor (B, H, W, C) with values 0-1
            
        Returns:
            IRE values tensor (B, H, W) with values 0-100
        """
        # Convert to grayscale using Rec. 709 coefficients for sRGB
        if srgb_image.shape[-1] == 3:
            luma = 0.2126 * srgb_image[..., 0] + 0.7152 * srgb_image[..., 1] + 0.0722 * srgb_image[..., 2]
        else:
            luma = srgb_image[..., 0] if srgb_image.shape[-1] == 1 else srgb_image
        
        # Convert sRGB luma to IRE (0-100 scale)
        # For display-referred, we use a linear mapping from 0-1 to 0-100 IRE
        ire_values = luma * 100.0
        
        return ire_values
    
    def create_false_color_visualization(self, ire_values: torch.Tensor) -> torch.Tensor:
        """
        Create false color visualization from IRE values.
        
        Args:
            ire_values: IRE values tensor (B, H, W)
            
        Returns:
            False color RGB image tensor (B, H, W, 3)
        """
        batch_size, height, width = ire_values.shape
        false_color = torch.zeros((batch_size, height, width, 3), dtype=torch.float32)
        
        # Create masks for each IRE range
        ire_np = ire_values.detach().cpu().numpy()
        
        # Deep shadows (0-10 IRE)
        deep_shadow_mask = (ire_np >= 0) & (ire_np < 10)
        false_color[deep_shadow_mask] = torch.tensor(self.ire_color_map['deep_shadow'])
        
        # Shadows (10-20 IRE)
        shadow_mask = (ire_np >= 10) & (ire_np < 20)
        false_color[shadow_mask] = torch.tensor(self.ire_color_map['shadow'])
        
        # Mid-shadows (20-30 IRE)
        mid_shadow_mask = (ire_np >= 20) & (ire_np < 30)
        false_color[mid_shadow_mask] = torch.tensor(self.ire_color_map['mid_shadow'])
        
        # Low-midtones (30-50 IRE)
        low_midtone_mask = (ire_np >= 30) & (ire_np < 50)
        false_color[low_midtone_mask] = torch.tensor(self.ire_color_map['low_midtone'])
        
        # Midtones (50-70 IRE)
        midtone_mask = (ire_np >= 50) & (ire_np < 70)
        false_color[midtone_mask] = torch.tensor(self.ire_color_map['midtone'])
        
        # High-midtones (70-90 IRE)
        high_midtone_mask = (ire_np >= 70) & (ire_np < 90)
        false_color[high_midtone_mask] = torch.tensor(self.ire_color_map['high_midtone'])
        
        # Highlights (90-100 IRE)
        highlight_mask = (ire_np >= 90) & (ire_np <= 100)
        false_color[highlight_mask] = torch.tensor(self.ire_color_map['bright_highlight'])
        
        return false_color
    
    def analyze_shadow_transitions(self, ire_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze shadow transitions to determine soft vs hard shadows.
        
        Args:
            ire_values: IRE values tensor (B, H, W)
            
        Returns:
            Dictionary containing transition analysis results
        """
        # Create shadow mask
        shadow_mask = ire_values < self.shadow_ire_threshold
        
        # Calculate gradients for transition analysis
        ire_expanded = ire_values.unsqueeze(1)  # Add channel dimension for conv2d
        
        grad_x = F.conv2d(ire_expanded, self.sobel_x.unsqueeze(1), padding=1)
        grad_y = F.conv2d(ire_expanded, self.sobel_y.unsqueeze(1), padding=1)
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Remove channel dimension
        grad_magnitude = grad_magnitude.squeeze(1)
        
        # Analyze transitions in shadow areas
        shadow_gradients = grad_magnitude * shadow_mask.float()
        
        # Calculate transition characteristics
        mean_shadow_gradient = shadow_gradients[shadow_mask].mean() if shadow_mask.any() else torch.tensor(0.0)
        max_shadow_gradient = shadow_gradients[shadow_mask].max() if shadow_mask.any() else torch.tensor(0.0)
        
        # Soft shadow detection: gradual transitions (low gradients)
        # Hard shadow detection: abrupt transitions (high gradients)
        soft_shadow_threshold = self.transition_sensitivity * 10.0  # IRE units
        hard_shadow_threshold = self.transition_sensitivity * 30.0  # IRE units
        
        soft_shadow_mask = (shadow_gradients < soft_shadow_threshold) & shadow_mask
        hard_shadow_mask = (shadow_gradients > hard_shadow_threshold) & shadow_mask
        
        return {
            'shadow_mask': shadow_mask,
            'gradient_magnitude': grad_magnitude,
            'shadow_gradients': shadow_gradients,
            'soft_shadow_mask': soft_shadow_mask,
            'hard_shadow_mask': hard_shadow_mask,
            'mean_shadow_gradient': mean_shadow_gradient,
            'max_shadow_gradient': max_shadow_gradient,
            'soft_shadow_ratio': soft_shadow_mask.float().sum() / (shadow_mask.float().sum() + 1e-8),
            'hard_shadow_ratio': hard_shadow_mask.float().sum() / (shadow_mask.float().sum() + 1e-8)
        }
    
    def classify_shadow_characteristics(self, transition_analysis: Dict[str, torch.Tensor]) -> Dict[str, str]:
        """
        Classify shadow characteristics based on IRE transition analysis.
        
        Args:
            transition_analysis: Results from analyze_shadow_transitions
            
        Returns:
            Dictionary with shadow classification results
        """
        soft_ratio = transition_analysis['soft_shadow_ratio'].item()
        hard_ratio = transition_analysis['hard_shadow_ratio'].item()
        mean_gradient = transition_analysis['mean_shadow_gradient'].item()
        
        # Classify overall shadow character
        if soft_ratio > 0.7:
            shadow_character = "Very Soft"
        elif soft_ratio > 0.5:
            shadow_character = "Soft"
        elif hard_ratio > 0.5:
            shadow_character = "Hard"
        elif hard_ratio > 0.3:
            shadow_character = "Medium-Hard"
        else:
            shadow_character = "Mixed"
        
        # Classify transition quality
        if mean_gradient < 5.0:
            transition_quality = "Very Gradual"
        elif mean_gradient < 15.0:
            transition_quality = "Gradual"
        elif mean_gradient < 30.0:
            transition_quality = "Moderate"
        elif mean_gradient < 50.0:
            transition_quality = "Sharp"
        else:
            transition_quality = "Very Sharp"
        
        return {
            'shadow_character': shadow_character,
            'transition_quality': transition_quality,
            'soft_ratio': f"{soft_ratio:.2f}",
            'hard_ratio': f"{hard_ratio:.2f}",
            'mean_gradient': f"{mean_gradient:.2f}"
        }
    
    def generate_ire_analysis_report(self, srgb_image: torch.Tensor) -> Dict:
        """
        Generate comprehensive IRE analysis report for shadow detection.
        
        Args:
            srgb_image: sRGB image tensor (B, H, W, C)
            
        Returns:
            Dictionary containing complete IRE analysis results
        """
        # Convert to IRE values
        ire_values = self.srgb_to_ire(srgb_image)
        
        # Create false color visualization
        false_color = self.create_false_color_visualization(ire_values)
        
        # Analyze shadow transitions
        transition_analysis = self.analyze_shadow_transitions(ire_values)
        
        # Classify shadow characteristics
        shadow_classification = self.classify_shadow_characteristics(transition_analysis)
        
        # Calculate IRE statistics
        ire_stats = {
            'min_ire': ire_values.min().item(),
            'max_ire': ire_values.max().item(),
            'mean_ire': ire_values.mean().item(),
            'std_ire': ire_values.std().item(),
            'shadow_pixel_count': transition_analysis['shadow_mask'].sum().item(),
            'total_pixel_count': ire_values.numel(),
            'shadow_percentage': (transition_analysis['shadow_mask'].sum().item() / ire_values.numel()) * 100
        }
        
        return {
            'ire_values': ire_values,
            'false_color_visualization': false_color,
            'transition_analysis': transition_analysis,
            'shadow_classification': shadow_classification,
            'ire_statistics': ire_stats,
            'ire_color_map': self.ire_color_map
        }
    
    def create_ire_legend(self) -> torch.Tensor:
        """
        Create a legend showing IRE color mapping.
        
        Returns:
            Legend image tensor (1, H, W, 3)
        """
        # Create legend image
        legend_height = 200
        legend_width = 400
        legend = torch.zeros((1, legend_height, legend_width, 3))
        
        # Create color bars for each IRE range
        bar_height = legend_height // len(self.ire_color_map)
        
        ire_ranges = [
            (0, 10, 'deep_shadow'),
            (10, 20, 'shadow'),
            (20, 30, 'mid_shadow'),
            (30, 50, 'low_midtone'),
            (50, 70, 'midtone'),
            (70, 90, 'high_midtone'),
            (90, 100, 'bright_highlight')
        ]
        
        for i, (start_ire, end_ire, color_key) in enumerate(ire_ranges):
            y_start = i * bar_height
            y_end = min((i + 1) * bar_height, legend_height)
            
            color = torch.tensor(self.ire_color_map[color_key])
            legend[0, y_start:y_end, :] = color
        
        return legend
