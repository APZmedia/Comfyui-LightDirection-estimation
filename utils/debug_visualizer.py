import torch
import numpy as np

class DebugVisualizer:
    """
    Debug visualization tools for light estimation analysis.
    """
    
    @staticmethod
    def generate_debug_mask(mask):
        """
        Generate debug visualization of the binary mask.

        Args:
            mask: Binary mask tensor (B, H, W) or (B, H, W, 1)

        Returns:
            debug_mask: RGB debug image (B, H, W, 3)
        """
        # Handle different tensor shapes - squeeze extra dimensions if present
        if mask.dim() == 4 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        elif mask.dim() > 3:
            # If more than 3 dimensions, take first 3 and squeeze the rest
            mask = mask.flatten(0, mask.dim() - 3).squeeze()

        batch_size, height, width = mask.shape
        debug_masks = []
        
        for b in range(batch_size):
            # Convert binary mask to RGB
            mask_np = mask[b].detach().cpu().numpy()
            debug_rgb = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Set lit areas to white, unlit to black
            debug_rgb[mask_np] = [255, 255, 255]
            
            debug_masks.append(torch.from_numpy(debug_rgb).float() / 255.0)
        
        return torch.stack(debug_masks)
    
    @staticmethod
    def generate_normal_visualization(normals, mask=None):
        """
        Generate RGB visualization of normal vectors.
        
        Args:
            normals: Normal vectors (B, H, W, 3)
            mask: Optional binary mask to highlight specific areas
        
        Returns:
            normal_viz: RGB visualization (B, H, W, 3)
        """
        # Convert normals from [-1, 1] to [0, 1] for RGB display
        normal_viz = (normals + 1.0) / 2.0
        
        if mask is not None:
            # Dim unlit areas
            normal_viz = normal_viz * mask.unsqueeze(-1)
        
        return normal_viz
    
    @staticmethod
    def generate_spread_histogram(normals, mask, bins=36):
        """
        Generate polar histogram of lit normal distribution.
        
        Args:
            normals: Normal vectors (B, H, W, 3)
            mask: Binary mask (B, H, W)
            bins: Number of histogram bins
        
        Returns:
            histogram: Polar histogram data
        """
        batch_size = normals.shape[0]
        histograms = []
        
        for b in range(batch_size):
            lit_normals = normals[b][mask[b]]
            
            if len(lit_normals) == 0:
                histograms.append(np.zeros(bins))
                continue
            
            # Extract XY components and convert to polar coordinates
            xy_normals = lit_normals[:, :2]
            angles = torch.atan2(xy_normals[:, 1], xy_normals[:, 0])
            
            # Convert to degrees and create histogram
            angles_deg = torch.rad2deg(angles).detach().cpu().numpy()
            hist, _ = np.histogram(angles_deg, bins=bins, range=(-180, 180))
            histograms.append(hist)
        
        return histograms
    
    @staticmethod
    def generate_directional_visualization(normals, categorical_results):
        """
        Generate visualization showing categorical analysis results.
        
        Args:
            normals: Normal vectors (B, H, W, 3)
            categorical_results: Results from light estimator
        
        Returns:
            directional_viz: Color-coded visualization (B, H, W, 3)
        """
        batch_size = normals.shape[0]
        visualizations = []
        
        for b in range(batch_size):
            result = categorical_results[b]
            
            # Create color-coded visualization
            height, width = normals.shape[1], normals.shape[2]
            viz = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Get categories and index
            x_category = result['x_category']
            y_category = result['y_category']
            hard_soft_index = result['hard_soft_index']
            overall_confidence = result['confidence']['overall_confidence']
            
            # X direction colors (Red channel)
            if x_category == "left":
                red = 255
            elif x_category == "right":
                red = 128
            else:  # central
                red = 192
            
            # Y direction colors (Green channel)
            if y_category == "top":
                green = 255
            elif y_category == "bottom":
                green = 128
            else:  # central
                green = 192
            
            # Hard/Soft index colors (Blue channel)
            # 0.0 (hard) = 255 (bright blue)
            # 1.0 (soft) = 128 (medium blue)
            blue = int(255 - (hard_soft_index * 127))
            
            # Apply confidence as alpha/transparency effect
            confidence_factor = overall_confidence
            red = int(red * confidence_factor)
            green = int(green * confidence_factor)
            blue = int(blue * confidence_factor)
            
            viz[:, :] = [red, green, blue]
            visualizations.append(torch.from_numpy(viz).float() / 255.0)
        
        return torch.stack(visualizations)
    
    @staticmethod
    def generate_confidence_heatmap(normals, categorical_results):
        """
        Generate confidence heatmap visualization.
        
        Args:
            normals: Normal vectors (B, H, W, 3)
            categorical_results: Results from light estimator
        
        Returns:
            confidence_viz: Confidence heatmap (B, H, W, 3)
        """
        batch_size = normals.shape[0]
        visualizations = []
        
        for b in range(batch_size):
            result = categorical_results[b]
            
            # Create confidence heatmap
            height, width = normals.shape[1], normals.shape[2]
            viz = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Get confidence values
            x_confidence = result['confidence']['x_confidence']
            y_confidence = result['confidence']['y_confidence']
            quality_confidence = result['confidence']['quality_confidence']
            overall_confidence = result['confidence']['overall_confidence']
            
            # Color code based on confidence levels
            # Red = X confidence, Green = Y confidence, Blue = Quality confidence
            red = int(255 * x_confidence)
            green = int(255 * y_confidence)
            blue = int(255 * quality_confidence)
            
            viz[:, :] = [red, green, blue]
            visualizations.append(torch.from_numpy(viz).float() / 255.0)
        
        return torch.stack(visualizations)
    
    @staticmethod
    def generate_quadrant_visualization(normals, mask):
        """
        Generate visualization showing directional quadrants.
        
        Args:
            normals: Normal vectors (B, H, W, 3)
            mask: Binary mask (B, H, W)
        
        Returns:
            quadrant_viz: Quadrant visualization (B, H, W, 3)
        """
        batch_size = normals.shape[0]
        visualizations = []
        
        for b in range(batch_size):
            # Create quadrant visualization
            height, width = normals.shape[1], normals.shape[2]
            viz = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Get lit normals
            lit_normals = normals[b][mask[b]]
            
            if len(lit_normals) == 0:
                visualizations.append(torch.from_numpy(viz).float() / 255.0)
                continue
            
            # Extract XY components
            xy_normals = lit_normals[:, :2]
            
            # Color code based on quadrants
            for i, (x, y) in enumerate(xy_normals):
                if x > 0 and y > 0:  # Top-right
                    color = [255, 255, 0]  # Yellow
                elif x < 0 and y > 0:  # Top-left
                    color = [255, 0, 255]  # Magenta
                elif x < 0 and y < 0:  # Bottom-left
                    color = [0, 255, 255]  # Cyan
                else:  # Bottom-right
                    color = [255, 0, 0]  # Red
                
                # Find pixel position (simplified - would need proper mapping)
                # This is a placeholder for the actual implementation
                pass
            
            visualizations.append(torch.from_numpy(viz).float() / 255.0)
        
        return torch.stack(visualizations)
