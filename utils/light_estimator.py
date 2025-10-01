import torch
import numpy as np

class CategoricalLightEstimator:
    """
    Light estimator with categorical directional outputs and simplified hard/soft classification.
    """
    
    def __init__(self):
        # Directional thresholds
        self.x_threshold = 0.1
        self.y_threshold = 0.1
        self.central_threshold = 0.3
        
        # Hard/Soft classification thresholds
        self.hard_light_threshold = 0.15  # Below this = hard light
        self.soft_light_threshold = 0.35  # Above this = soft light
        
    def analyze_directional_categories(self, normals, mask):
        """
        Analyze normals and classify into directional categories with hard/soft index.
        
        Args:
            normals: Normal vectors tensor (B, H, W, 3)
            mask: Binary mask tensor (B, H, W)
        
        Returns:
            results: List of analysis results for each batch
        """
        batch_size = normals.shape[0]
        results = []
        
        for b in range(batch_size):
            # Get mask for current batch and ensure it's boolean
            current_mask = mask[b].bool()

            # Apply mask to normals while preserving dimensions
            lit_normals = normals[b][current_mask]

            # Reshape to ensure 2D structure [N, 3] where N is number of lit pixels
            if lit_normals.dim() == 1:
                lit_normals = lit_normals.unsqueeze(0) if lit_normals.numel() > 0 else torch.empty(0, 3, device=normals.device, dtype=normals.dtype)

            if lit_normals.shape[0] == 0:
                results.append(self._empty_categories())
                continue

            # Extract XY components
            xy_normals = lit_normals[:, :2]
            
            # Analyze directional distribution
            x_analysis = self._analyze_x_direction(xy_normals)
            y_analysis = self._analyze_y_direction(xy_normals)
            
            # Analyze light quality (hard/soft index)
            quality_analysis = self._analyze_light_quality(xy_normals)
            
            # Determine categories
            x_category = self._classify_x_direction(x_analysis)
            y_category = self._classify_y_direction(y_analysis)
            combined_category = f"{y_category}-{x_category}"
            
            # Calculate hard/soft index (0 = hard, 1 = soft)
            hard_soft_index = self._calculate_hard_soft_index(quality_analysis)
            
            # Calculate confidence scores
            confidence = {
                'x_confidence': x_analysis['confidence'],
                'y_confidence': y_analysis['confidence'],
                'quality_confidence': quality_analysis['confidence'],
                'overall_confidence': (x_analysis['confidence'] + y_analysis['confidence'] + quality_analysis['confidence']) / 3
            }
            
            results.append({
                'x_category': x_category,
                'y_category': y_category,
                'combined_category': combined_category,
                'hard_soft_index': hard_soft_index,
                'confidence': confidence,
                'x_analysis': x_analysis,
                'y_analysis': y_analysis,
                'quality_analysis': quality_analysis
            })
        
        return results
    
    def _analyze_light_quality(self, xy_normals):
        """
        Analyze light quality based on spread of normals.
        
        Args:
            xy_normals: XY components of normal vectors (N, 2)
        
        Returns:
            quality_analysis: Dict with spread and confidence
        """
        if len(xy_normals) < 2:
            return {
                'spread': 0.0,
                'confidence': 0.0
            }
        
        # Calculate spread using covariance analysis
        xy_np = xy_normals.detach().cpu().numpy()
        cov_matrix = np.cov(xy_np.T)
        eigenvalues = np.linalg.eigvals(cov_matrix)
        spread = np.sqrt(np.max(eigenvalues))
        
        # Calculate confidence based on how clear the spread is
        confidence = min(1.0, spread / 0.5)  # Normalize confidence
        
        return {
            'spread': spread,
            'confidence': confidence
        }
    
    def _calculate_hard_soft_index(self, quality_analysis):
        """
        Calculate hard/soft index from 0 (hard) to 1 (soft).
        
        Args:
            quality_analysis: Analysis results with spread value
        
        Returns:
            hard_soft_index: Float from 0.0 (hard) to 1.0 (soft)
        """
        spread = quality_analysis['spread']
        
        # Linear interpolation between thresholds
        if spread <= self.hard_light_threshold:
            # Hard light range
            return 0.0
        elif spread >= self.soft_light_threshold:
            # Soft light range
            return 1.0
        else:
            # Intermediate range - linear interpolation
            ratio = (spread - self.hard_light_threshold) / (self.soft_light_threshold - self.hard_light_threshold)
            return ratio
    
    def _analyze_x_direction(self, xy_normals):
        """
        Analyze X-direction distribution and calculate confidence.
        
        Args:
            xy_normals: XY components of normal vectors (N, 2)
        
        Returns:
            x_analysis: Dict with percentages and confidence
        """
        x_components = xy_normals[:, 0]
        
        # Count normals in each X category
        left_count = torch.sum(x_components < -self.x_threshold).item()
        right_count = torch.sum(x_components > self.x_threshold).item()
        central_count = torch.sum(torch.abs(x_components) <= self.central_threshold).item()
        
        total_count = len(x_components)
        
        # Calculate percentages
        left_pct = left_count / total_count if total_count > 0 else 0
        right_pct = right_count / total_count if total_count > 0 else 0
        central_pct = central_count / total_count if total_count > 0 else 0
        
        # Calculate confidence
        max_pct = max(left_pct, right_pct, central_pct)
        confidence = max_pct if max_pct > 0.3 else 0.0
        
        return {
            'left_pct': left_pct,
            'right_pct': right_pct,
            'central_pct': central_pct,
            'confidence': confidence,
            'total_normals': total_count
        }
    
    def _analyze_y_direction(self, xy_normals):
        """
        Analyze Y-direction distribution and calculate confidence.
        
        Args:
            xy_normals: XY components of normal vectors (N, 2)
        
        Returns:
            y_analysis: Dict with percentages and confidence
        """
        y_components = xy_normals[:, 1]
        
        # Count normals in each Y category
        top_count = torch.sum(y_components > self.y_threshold).item()
        bottom_count = torch.sum(y_components < -self.y_threshold).item()
        central_count = torch.sum(torch.abs(y_components) <= self.central_threshold).item()
        
        total_count = len(y_components)
        
        # Calculate percentages
        top_pct = top_count / total_count if total_count > 0 else 0
        bottom_pct = bottom_count / total_count if total_count > 0 else 0
        central_pct = central_count / total_count if total_count > 0 else 0
        
        # Calculate confidence
        max_pct = max(top_pct, bottom_pct, central_pct)
        confidence = max_pct if max_pct > 0.3 else 0.0
        
        return {
            'top_pct': top_pct,
            'bottom_pct': bottom_pct,
            'central_pct': central_pct,
            'confidence': confidence,
            'total_normals': total_count
        }
    
    def _classify_x_direction(self, x_analysis):
        """
        Classify X direction based on analysis.
        
        Args:
            x_analysis: X direction analysis results
        
        Returns:
            x_category: "left", "central", or "right"
        """
        left_pct = x_analysis['left_pct']
        right_pct = x_analysis['right_pct']
        central_pct = x_analysis['central_pct']
        
        if central_pct > max(left_pct, right_pct):
            return "central"
        elif left_pct > right_pct:
            return "left"
        else:
            return "right"
    
    def _classify_y_direction(self, y_analysis):
        """
        Classify Y direction based on analysis.
        
        Args:
            y_analysis: Y direction analysis results
        
        Returns:
            y_category: "top", "central", or "bottom"
        """
        top_pct = y_analysis['top_pct']
        bottom_pct = y_analysis['bottom_pct']
        central_pct = y_analysis['central_pct']
        
        if central_pct > max(top_pct, bottom_pct):
            return "central"
        elif top_pct > bottom_pct:
            return "top"
        else:
            return "bottom"
    
    def _empty_categories(self):
        """
        Return empty categories for cases with no lit normals.
        
        Returns:
            empty_result: Dict with default values
        """
        return {
            'x_category': "central",
            'y_category': "central",
            'combined_category': "central-central",
            'hard_soft_index': 0.5,  # Default to intermediate
            'confidence': {
                'x_confidence': 0.0, 
                'y_confidence': 0.0, 
                'quality_confidence': 0.0,
                'overall_confidence': 0.0
            },
            'x_analysis': {'left_pct': 0.0, 'right_pct': 0.0, 'central_pct': 0.0, 'confidence': 0.0, 'total_normals': 0},
            'y_analysis': {'top_pct': 0.0, 'bottom_pct': 0.0, 'central_pct': 0.0, 'confidence': 0.0, 'total_normals': 0},
            'quality_analysis': {'spread': 0.0, 'confidence': 0.0}
        }
