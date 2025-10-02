import torch
from ..utils.light_estimator import CategoricalLightEstimator
from ..utils.luma_mask_processor import LumaMaskProcessor
from ..utils.debug_visualizer import DebugVisualizer

# ============================================================================
# 1. IMAGE PROCESSING NODE
# ============================================================================

class LightImageProcessor:
    """
    Processes normal maps and luma images to extract lit normals and weights.
    Handles all image processing concerns: normal conversion, luma masking, filtering.
    """
    
    CATEGORY = "Custom/Lighting/Processing"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal_map": ("IMAGE",),
                "luma_image": ("IMAGE",),
                "luma_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "curve_type": (["linear", "s_curve", "exponential", "logarithmic"], {"default": "s_curve"}),
                "normal_convention": (["OpenGL", "DirectX"], {"default": "OpenGL"}),
                "generate_weights": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("TENSOR", "TENSOR", "MASK", "WEIGHTS", "IMAGE", "IMAGE")
    RETURN_NAMES = ("processed_normals", "lit_normals", "binary_mask", "luma_weights", "debug_mask", "lit_normals_viz")
    FUNCTION = "process_images"
    
    def process_images(self, normal_map, luma_image, luma_threshold, curve_type, normal_convention, generate_weights):
        """
        Process normal map and luma image to extract lit normals and weights.
        """
        # 1. Process normal map: Convert RGB to XYZ
        processed_normals = self._process_normal_map(normal_map, normal_convention)
        
        # 2. Process luma image: Generate mask and weights
        luma_processor = LumaMaskProcessor()
        binary_mask = luma_processor.process_with_curves(luma_image, curve_type, None, luma_threshold)
        
        # 3. Generate weights if requested
        luma_weights = None
        if generate_weights:
            luma_weights = luma_processor.weighted_luma_mask(luma_image, luma_threshold)
        
        # 4. Extract lit normals
        lit_normals = self._extract_lit_normals(processed_normals, binary_mask)
        
        # 5. Generate debug visualizations
        debug_mask = DebugVisualizer.generate_debug_mask(binary_mask)
        lit_normals_viz = DebugVisualizer.generate_lit_normals_visualization(processed_normals, binary_mask)
        
        return processed_normals, lit_normals, binary_mask, luma_weights, debug_mask, lit_normals_viz
    
    def _process_normal_map(self, normal_map, convention):
        """Convert RGB normal map to XYZ coordinates."""
        # Convert to XYZ: ((RGB / 255) * 2.0) - 1.0
        xyz_normals = ((normal_map / 255.0) * 2.0) - 1.0
        
        # Apply convention-specific flips
        if convention == "DirectX":
            xyz_normals[..., 1] = -xyz_normals[..., 1]  # Flip Y for DirectX
        
        # Normalize
        norm = torch.norm(xyz_normals, dim=-1, keepdim=True)
        xyz_normals = xyz_normals / (norm + 1e-8)
        
        return xyz_normals
    
    def _extract_lit_normals(self, normals, mask):
        """Extract normals where mask is True."""
        current_mask = mask.bool()
        lit_normals = normals[current_mask]
        
        if lit_normals.numel() == 0:
            return torch.zeros(0, 3, device=normals.device)
        
        if lit_normals.dim() == 1:
            lit_normals = lit_normals.unsqueeze(0)
        
        return lit_normals

# ============================================================================
# 2. DISTRIBUTION ANALYSIS NODE
# ============================================================================

class LightDistributionAnalyzer:
    """
    Analyzes the distribution of lit normals to determine light direction and quality.
    Handles all statistical analysis concerns: directional analysis, quality assessment, classification.
    """
    
    CATEGORY = "Custom/Lighting/Analysis"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lit_normals": ("TENSOR",),
                "luma_weights": ("WEIGHTS", {"default": None}),
                "x_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "y_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "central_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "hard_light_threshold": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "soft_light_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "use_weighted_analysis": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = (
        "STRING", "STRING", "STRING", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT",
        "IMAGE", "IMAGE", "IMAGE"
    )
    RETURN_NAMES = (
        "x_direction", "y_direction", "combined_direction", "hard_soft_index",
        "x_confidence", "y_confidence", "overall_confidence", "spread_value",
        "directional_viz", "quality_viz", "colormap_preview"
    )
    FUNCTION = "analyze_distribution"
    
    def analyze_distribution(self, lit_normals, luma_weights, x_threshold, y_threshold, central_threshold,
                           hard_light_threshold, soft_light_threshold, use_weighted_analysis):
        """
        Analyze the distribution of lit normals to determine light characteristics.
        """
        if lit_normals.numel() == 0:
            return self._empty_results()
        
        # Extract XY components for directional analysis
        xy_normals = lit_normals[:, :2]
        
        # 1. Directional Analysis
        x_analysis = self._analyze_x_direction(xy_normals, x_threshold, central_threshold, luma_weights, use_weighted_analysis)
        y_analysis = self._analyze_y_direction(xy_normals, y_threshold, central_threshold, luma_weights, use_weighted_analysis)
        
        # 2. Quality Analysis
        quality_analysis = self._analyze_light_quality(xy_normals, luma_weights, use_weighted_analysis)
        
        # 3. Classification
        x_direction = self._classify_x_direction(x_analysis)
        y_direction = self._classify_y_direction(y_analysis)
        combined_direction = f"{y_direction}-{x_direction}"
        
        # 4. Hard/Soft Index
        hard_soft_index = self._calculate_hard_soft_index(quality_analysis, hard_light_threshold, soft_light_threshold)
        
        # 5. Confidence Scores
        x_confidence = x_analysis['confidence']
        y_confidence = y_analysis['confidence']
        overall_confidence = (x_confidence + y_confidence) / 2
        spread_value = quality_analysis['spread']
        
        # 6. Generate Visualizations
        directional_viz = self._generate_directional_visualization(x_direction, y_direction, hard_soft_index, overall_confidence)
        quality_viz = self._generate_quality_visualization(quality_analysis)
        colormap_preview = DebugVisualizer.generate_colormap_preview()
        
        return (
            x_direction, y_direction, combined_direction, hard_soft_index,
            x_confidence, y_confidence, overall_confidence, spread_value,
            directional_viz, quality_viz, colormap_preview
        )
    
    def _analyze_x_direction(self, xy_normals, x_threshold, central_threshold, luma_weights, use_weighted):
        """Analyze X-direction distribution with optional weighting."""
        x_components = xy_normals[:, 0]
        total_count = x_components.shape[0]
        
        if use_weighted and luma_weights is not None:
            # Weighted analysis - this would need proper weight extraction
            # For now, use simple analysis
            pass
        
        left_pct = torch.sum(x_components < -x_threshold).item() / total_count
        right_pct = torch.sum(x_components > x_threshold).item() / total_count
        central_pct = torch.sum(torch.abs(x_components) <= central_threshold).item() / total_count
        
        mean_x = x_components.mean().item()
        max_pct = max(left_pct, right_pct, central_pct)
        confidence = max_pct if max_pct > 0.3 else 0.0
        
        return {
            'left_pct': left_pct,
            'right_pct': right_pct,
            'central_pct': central_pct,
            'mean_x': mean_x,
            'confidence': confidence
        }
    
    def _analyze_y_direction(self, xy_normals, y_threshold, central_threshold, luma_weights, use_weighted):
        """Analyze Y-direction distribution with optional weighting."""
        y_components = xy_normals[:, 1]
        total_count = y_components.shape[0]
        
        if use_weighted and luma_weights is not None:
            # Weighted analysis - this would need proper weight extraction
            # For now, use simple analysis
            pass
        
        top_pct = torch.sum(y_components > y_threshold).item() / total_count
        bottom_pct = torch.sum(y_components < -y_threshold).item() / total_count
        central_pct = torch.sum(torch.abs(y_components) <= central_threshold).item() / total_count
        
        mean_y = y_components.mean().item()
        max_pct = max(top_pct, bottom_pct, central_pct)
        confidence = max_pct if max_pct > 0.3 else 0.0
        
        return {
            'top_pct': top_pct,
            'bottom_pct': bottom_pct,
            'central_pct': central_pct,
            'mean_y': mean_y,
            'confidence': confidence
        }
    
    def _analyze_light_quality(self, xy_normals, luma_weights, use_weighted):
        """Analyze light quality based on normal spread."""
        if xy_normals.shape[0] < 2:
            return {'spread': 0.0, 'confidence': 0.0}
        
        # Calculate spread using covariance matrix
        cov_matrix = torch.cov(xy_normals.T)
        eigenvalues = torch.linalg.eigvals(cov_matrix).real
        spread = torch.sqrt(torch.max(eigenvalues))
        
        # Calculate confidence
        confidence = torch.clamp(spread / 0.5, 0.0, 1.0)
        
        return {
            'spread': spread.item(),
            'confidence': confidence.item()
        }
    
    def _classify_x_direction(self, x_analysis):
        """Classify X direction."""
        if x_analysis['central_pct'] > max(x_analysis['left_pct'], x_analysis['right_pct']):
            return "central"
        elif x_analysis['left_pct'] > x_analysis['right_pct']:
            return "left"
        else:
            return "right"
    
    def _classify_y_direction(self, y_analysis):
        """Classify Y direction."""
        if y_analysis['central_pct'] > max(y_analysis['top_pct'], y_analysis['bottom_pct']):
            return "central"
        elif y_analysis['top_pct'] > y_analysis['bottom_pct']:
            return "top"
        else:
            return "bottom"
    
    def _calculate_hard_soft_index(self, quality_analysis, hard_threshold, soft_threshold):
        """Calculate hard/soft index from 0 (hard) to 1 (soft)."""
        spread = quality_analysis['spread']
        
        if spread <= hard_threshold:
            return 0.0
        elif spread >= soft_threshold:
            return 1.0
        else:
            return (spread - hard_threshold) / (soft_threshold - hard_threshold)
    
    def _generate_directional_visualization(self, x_direction, y_direction, hard_soft_index, overall_confidence):
        """Generate directional visualization."""
        # Create a simple color-coded visualization
        viz = torch.zeros(256, 256, 3)
        
        # Base colors for directions
        if x_direction == "left":
            base_color = torch.tensor([255, 100, 100])
        elif x_direction == "right":
            base_color = torch.tensor([200, 80, 80])
        else:
            base_color = torch.tensor([150, 150, 150])
        
        # Adjust for Y direction
        if y_direction == "top":
            base_color += torch.tensor([0, 100, 50])
        elif y_direction == "bottom":
            base_color += torch.tensor([50, -50, 0])
        
        # Apply softness and confidence
        softness_factor = 1.0 - hard_soft_index
        final_color = base_color * (0.7 + 0.3 * softness_factor) * overall_confidence
        final_color = torch.clamp(final_color, 0, 255)
        
        viz[:, :] = final_color.byte()
        return viz.float() / 255.0
    
    def _generate_quality_visualization(self, quality_analysis):
        """Generate quality visualization."""
        # Create a visualization showing the spread/quality
        viz = torch.zeros(256, 256, 3)
        spread = quality_analysis['spread']
        
        # Color based on spread (red = hard, blue = soft)
        intensity = min(spread * 2, 1.0)
        color = torch.tensor([255 * (1 - intensity), 0, 255 * intensity])
        
        viz[:, :] = color
        return viz.float() / 255.0
    
    def _empty_results(self):
        """Return empty results for edge cases."""
        empty_viz = torch.zeros(256, 256, 3)
        return (
            "central", "central", "central-central", 0.5,
            0.0, 0.0, 0.0, 0.0,
            empty_viz, empty_viz, empty_viz
        )
