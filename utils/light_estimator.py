import torch

class CategoricalLightEstimator:
    """
    Light estimator with categorical directional outputs and simplified hard/soft classification.
    """
    
    def __init__(self, x_threshold=0.1, y_threshold=0.1, central_threshold=0.3,
                 hard_light_threshold=0.15, soft_light_threshold=0.35,
                 format_mode="auto", normal_standard="OpenGL"):
        # Directional thresholds
        self.x_threshold = x_threshold
        self.y_threshold = y_threshold
        self.central_threshold = central_threshold

        # Hard/Soft classification thresholds
        self.hard_light_threshold = hard_light_threshold
        self.soft_light_threshold = soft_light_threshold

        # Normal map format settings
        self.format_mode = format_mode
        self.normal_standard = normal_standard
        self.detected_format = None
        
    def analyze_directional_categories(self, normals, mask):
        """
        Analyze normals and classify into directional categories with hard/soft index.

        Args:
            normals: Normal vectors tensor (B, H, W, 3)
            mask: Binary mask tensor (B, H, W)

        Returns:
            results: Dictionary of analysis results for the batch
        """
        batch_size, height, width, _ = normals.shape

        # Ensure mask is boolean and 3D
        if mask.dim() == 4:
            mask = mask.squeeze(-1)
        current_mask = mask.bool()

        # Detect and validate normal map format
        detected_format = self._detect_normal_format(normals) if self.format_mode == "auto" else self.normal_standard
        self.detected_format = detected_format

        # Analyze color distribution before masking
        color_analysis_before = self._analyze_color_distribution(normals)

        # Apply mask to normals
        lit_normals = normals[current_mask]

        if lit_normals.numel() == 0:
            return self._empty_categories_with_color_analysis(batch_size, normals.device, color_analysis_before)

        if lit_normals.dim() == 1:
            lit_normals = lit_normals.unsqueeze(0)

        # Analyze color distribution after masking
        color_analysis_after = self._analyze_color_distribution(lit_normals)

        # DEBUG: Print key information for pipeline verification
        print("=== Pipeline Verification ===")
        print(f"Input normals shape: {normals.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Lit normals count: {lit_normals.shape[0]}")
        print(f"Total pixels: {normals.shape[1] * normals.shape[2]}")
        print(f"Color before - Mean: {color_analysis_before['mean_color']}")
        print(f"Color before - Dominant: {color_analysis_before['dominant_direction']}")
        print(f"Color after - Mean: {color_analysis_after['mean_color']}")
        print(f"Color after - Dominant: {color_analysis_after['dominant_direction']}")
        print("==========================")

        # Extract XY components
        xy_normals = lit_normals[:, :2]

        # Analyze directional distribution
        x_analysis = self._analyze_x_direction(xy_normals)
        y_analysis = self._analyze_y_direction(xy_normals)

        # Analyze light quality
        quality_analysis = self._analyze_light_quality(xy_normals)

        # Determine categories
        x_category = self._classify_x_direction(x_analysis)
        y_category = self._classify_y_direction(y_analysis)

        # Calculate hard/soft index
        hard_soft_index = self._calculate_hard_soft_index(quality_analysis)

        # Calculate confidence scores
        confidence = {
            'x_confidence': x_analysis['confidence'],
            'y_confidence': y_analysis['confidence'],
            'quality_confidence': quality_analysis['confidence'],
            'overall_confidence': (x_analysis['confidence'] + y_analysis['confidence'] + quality_analysis['confidence']) / 3
        }

        return {
            'x_category': x_category,
            'y_category': y_category,
            'hard_soft_index': hard_soft_index,
            'confidence': confidence,
            'quality_analysis': quality_analysis,
            'color_analysis_before': color_analysis_before,
            'color_analysis_after': color_analysis_after,
            'lit_pixel_count': lit_normals.shape[0],
            'total_pixel_count': normals.shape[1] * normals.shape[2]
        }
    
    def _analyze_light_quality(self, xy_normals):
        """
        Analyze light quality based on spread of normals.
        """
        if xy_normals.shape[0] < 2:
            return {'spread': 0.0, 'confidence': 0.0}
        
        # Calculate spread using covariance matrix
        cov_matrix = torch.cov(xy_normals.T)
        eigenvalues = torch.linalg.eigvals(cov_matrix).real
        spread = torch.sqrt(torch.max(eigenvalues))
        
        # Normalize confidence
        confidence = torch.clamp(spread / 0.5, 0.0, 1.0)
        
        return {'spread': spread.item(), 'confidence': confidence.item()}
    
    def _calculate_hard_soft_index(self, quality_analysis):
        """
        Calculate hard/soft index from 0 (hard) to 1 (soft).
        """
        spread = quality_analysis['spread']
        
        if spread <= self.hard_light_threshold:
            return 0.0
        elif spread >= self.soft_light_threshold:
            return 1.0
        else:
            return (spread - self.hard_light_threshold) / (self.soft_light_threshold - self.hard_light_threshold)
    
    def _analyze_x_direction(self, xy_normals):
        """
        Analyze X-direction distribution.
        """
        x_components = xy_normals[:, 0]
        total_count = x_components.shape[0]
        
        left_pct = torch.sum(x_components < -self.x_threshold).item() / total_count
        right_pct = torch.sum(x_components > self.x_threshold).item() / total_count
        central_pct = torch.sum(torch.abs(x_components) <= self.central_threshold).item() / total_count
        
        max_pct = max(left_pct, right_pct, central_pct)
        confidence = max_pct if max_pct > 0.3 else 0.0
        
        return {'left_pct': left_pct, 'right_pct': right_pct, 'central_pct': central_pct, 'confidence': confidence}
    
    def _analyze_y_direction(self, xy_normals):
        """
        Analyze Y-direction distribution.
        """
        y_components = xy_normals[:, 1]
        total_count = y_components.shape[0]

        top_pct = torch.sum(y_components > self.y_threshold).item() / total_count
        bottom_pct = torch.sum(y_components < -self.y_threshold).item() / total_count
        central_pct = torch.sum(torch.abs(y_components) <= self.central_threshold).item() / total_count
        
        max_pct = max(top_pct, bottom_pct, central_pct)
        confidence = max_pct if max_pct > 0.3 else 0.0
        
        return {'top_pct': top_pct, 'bottom_pct': bottom_pct, 'central_pct': central_pct, 'confidence': confidence}
    
    def _classify_x_direction(self, x_analysis):
        """
        Classify X direction.
        """
        if x_analysis['central_pct'] > max(x_analysis['left_pct'], x_analysis['right_pct']):
            return "central"
        elif x_analysis['left_pct'] > x_analysis['right_pct']:
            return "left"
        else:
            return "right"
    
    def _classify_y_direction(self, y_analysis):
        """
        Classify Y direction.
        """
        if y_analysis['central_pct'] > max(y_analysis['top_pct'], y_analysis['bottom_pct']):
            return "central"
        elif y_analysis['top_pct'] > y_analysis['bottom_pct']:
            return "top"
        else:
            return "bottom"
    
    def _analyze_color_distribution(self, normals):
        """
        Analyze the color distribution of normal vectors.

        Args:
            normals: Normal vectors tensor (B, H, W, 3) or (N, 3)

        Returns:
            color_stats: Dictionary with color statistics
        """
        if normals.dim() == 4:
            # Flatten to (N, 3) for analysis
            flat_normals = normals.view(-1, 3)
        else:
            flat_normals = normals

        if flat_normals.shape[0] == 0:
            return {
                'mean_color': torch.zeros(3),
                'std_color': torch.zeros(3),
                'dominant_direction': 'central',
                'color_variance': 0.0
            }

        # Calculate color statistics
        mean_color = torch.mean(flat_normals, dim=0)
        # Use unbiased=False to avoid degrees of freedom issues with small samples
        std_color = torch.std(flat_normals, dim=0, unbiased=False)

        # Determine dominant direction based on color values
        x_val = mean_color[0].item()
        y_val = mean_color[1].item()

        if abs(x_val) <= self.central_threshold and abs(y_val) <= self.central_threshold:
            dominant_direction = 'central'
        elif abs(x_val) > abs(y_val):
            dominant_direction = 'left' if x_val < 0 else 'right'
        else:
            dominant_direction = 'bottom' if y_val < 0 else 'top'

        # Calculate overall color variance
        color_variance = torch.mean(std_color).item()

        return {
            'mean_color': mean_color,
            'std_color': std_color,
            'dominant_direction': dominant_direction,
            'color_variance': color_variance
        }

    def _empty_categories_with_color_analysis(self, batch_size, device, color_analysis_before):
        """
        Return empty categories for cases with no lit normals, including color analysis.
        """
        return {
            'x_category': "central",
            'y_category': "central",
            'hard_soft_index': 0.5,
            'confidence': {'x_confidence': 0.0, 'y_confidence': 0.0, 'quality_confidence': 0.0, 'overall_confidence': 0.0},
            'quality_analysis': {'spread': 0.0, 'confidence': 0.0},
            'color_analysis_before': color_analysis_before,
            'color_analysis_after': color_analysis_before,  # Same as before if no lit pixels
            'lit_pixel_count': 0,
            'total_pixel_count': 0
        }

    def _empty_categories(self, batch_size, device):
        """
        Return empty categories for cases with no lit normals.
        """
        return {
            'x_category': "central",
            'y_category': "central",
            'hard_soft_index': 0.5,
            'confidence': {'x_confidence': 0.0, 'y_confidence': 0.0, 'quality_confidence': 0.0, 'overall_confidence': 0.0},
            'quality_analysis': {'spread': 0.0, 'confidence': 0.0}
        }

    def _detect_normal_format(self, normals):
        """
        Auto-detect the normal map format by analyzing color distribution patterns.

        Args:
            normals: Normal vectors tensor (B, H, W, 3)

        Returns:
            detected_format: String indicating the likely format
        """
        # Flatten to analyze color distribution
        flat_normals = normals.view(-1, 3)

        # Calculate mean colors
        mean_colors = torch.mean(flat_normals, dim=0)

        # Check for DirectX vs OpenGL pattern
        # DirectX typically has more variation in green channel if Y is inverted
        green_std = torch.std(flat_normals[:, 1]).item()
        blue_std = torch.std(flat_normals[:, 2]).item()

        # Simple heuristic: if green channel has significantly more variation than blue,
        # it might be DirectX format (where Y is inverted)
        if green_std > blue_std * 1.5:
            return "DirectX"
        else:
            return "OpenGL"

    def _interpret_normal_colors(self, x_val, y_val, z_val):
        """
        Interpret normal vector values based on the detected format.

        Args:
            x_val, y_val, z_val: Color components from normal map

        Returns:
            interpreted_x, interpreted_y: Properly oriented normal components
        """
        if self.detected_format == "DirectX":
            # DirectX inverts Y axis
            return x_val, -y_val
        else:
            # OpenGL, World Space, Object Space (standard orientation)
            return x_val, y_val
