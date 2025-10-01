import torch

class CategoricalLightEstimator:
    """
    Light estimator with categorical directional outputs and simplified hard/soft classification.
    """
    
    def __init__(self, x_threshold=0.1, y_threshold=0.1, central_threshold=0.3,
                 hard_light_threshold=0.15, soft_light_threshold=0.35):
        # Directional thresholds
        self.x_threshold = x_threshold
        self.y_threshold = y_threshold
        self.central_threshold = central_threshold
        
        # Hard/Soft classification thresholds
        self.hard_light_threshold = hard_light_threshold
        self.soft_light_threshold = soft_light_threshold
        
    def analyze_directional_categories(self, normals, mask):
        """
        Analyze normals and classify into directional categories with hard/soft index.
        
        Args:
            normals: Normal vectors tensor (B, H, W, 3)
            mask: Binary mask tensor (B, H, W)
        
        Returns:
            results: Dictionary of analysis results for the batch
        """
        batch_size, _, _, _ = normals.shape
        
        # Ensure mask is boolean
        current_mask = mask.bool()

        # Apply mask to normals
        lit_normals = normals[current_mask]

        if lit_normals.numel() == 0:
            return self._empty_categories(batch_size, normals.device)

        if lit_normals.dim() == 1:
            lit_normals = lit_normals.unsqueeze(0)

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
            'quality_analysis': quality_analysis
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
