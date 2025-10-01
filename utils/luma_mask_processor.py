import torch
import numpy as np
from scipy.interpolate import interp1d

class LumaMaskProcessor:
    """
    Enhanced luma mask processor with curves strategy and multiple processing modes.
    """
    
    @staticmethod
    def process_with_curves(luma_image, curve_type="linear", curve_points=None, threshold=0.5):
        """
        Process luma with various curve strategies.
        
        Args:
            luma_image: Input luma values tensor (B, H, W, C)
            curve_type: "linear", "s_curve", "exponential", "logarithmic", "custom"
            curve_points: Custom curve points for "custom" type
            threshold: Final threshold after curve processing
        
        Returns:
            mask: Binary mask tensor (B, H, W) where True indicates lit areas
        """
        # Ensure grayscale by averaging channels if necessary
        if luma_image.shape[-1] > 1:
            luma_image = luma_image.mean(dim=-1, keepdim=True)

        # Convert to numpy for curve processing
        luma_np = luma_image.detach().cpu().numpy()

        if curve_type == "linear":
            # Simple linear mapping
            processed_luma = luma_np
            
        elif curve_type == "s_curve":
            # S-curve: enhance mid-tones
            processed_luma = 3 * luma_np**2 - 2 * luma_np**3
            
        elif curve_type == "exponential":
            # Exponential: emphasize bright areas
            processed_luma = luma_np**2
            
        elif curve_type == "logarithmic":
            # Logarithmic: compress highlights
            processed_luma = np.log(1 + luma_np * 9) / np.log(10)
            
        elif curve_type == "custom" and curve_points:
            # Custom curve interpolation
            x_points, y_points = zip(*curve_points)
            curve_func = interp1d(x_points, y_points, kind='cubic', 
                                bounds_error=False, fill_value='extrapolate')
            processed_luma = curve_func(luma_np)
            
        else:
            processed_luma = luma_np
        
        # Apply threshold
        mask = processed_luma > threshold
        
        return torch.from_numpy(mask).to(luma_image.device)
    
    @staticmethod
    def weighted_luma_mask(luma_image, threshold=0.5):
        """
        Create weighted mask using luma intensity as weights.
        
        Args:
            luma_image: RGB or grayscale image tensor (B, H, W, C)
            threshold: Minimum luma threshold
        
        Returns:
            weights: Weight tensor (B, H, W) with luma values as weights
        """
        if luma_image.shape[-1] == 3:  # RGB image
            luma = 0.2126 * luma_image[..., 0] + 0.7152 * luma_image[..., 1] + 0.0722 * luma_image[..., 2]
        else:  # Already grayscale
            luma = luma_image[..., 0] if luma_image.shape[-1] == 1 else luma_image
        
        # Use luma values as weights, zero out below threshold
        weights = torch.where(luma > threshold, luma, torch.zeros_like(luma))
        
        return weights
    
    @staticmethod
    def adaptive_threshold(luma_image, method="otsu"):
        """
        Apply adaptive thresholding methods.
        
        Args:
            luma_image: Input luma values tensor (B, H, W, C)
            method: "otsu", "mean", "median"
        
        Returns:
            mask: Binary mask tensor (B, H, W)
        """
        luma_np = luma_image.detach().cpu().numpy()
        
        if method == "otsu":
            # Otsu's method for optimal threshold
            try:
                from skimage.filters import threshold_otsu
                threshold = threshold_otsu(luma_np)
            except ImportError:
                # Fallback to mean if skimage not available
                threshold = np.mean(luma_np)
            
        elif method == "mean":
            # Mean-based threshold
            threshold = np.mean(luma_np)
            
        elif method == "median":
            # Median-based threshold
            threshold = np.median(luma_np)
            
        else:
            threshold = 0.5
        
        mask = luma_np > threshold
        return torch.from_numpy(mask).to(luma_image.device)
    
    @staticmethod
    def multi_scale_processing(luma_image, scales=[1.0, 0.5, 2.0], weights=[0.5, 0.3, 0.2]):
        """
        Process luma at multiple scales and combine results.
        
        Args:
            luma_image: Input luma values tensor (B, H, W, C)
            scales: List of scale factors
            weights: Weights for combining scales
        
        Returns:
            combined_mask: Combined binary mask tensor (B, H, W)
        """
        results = []
        
        for scale, weight in zip(scales, weights):
            # Resize image
            if scale != 1.0:
                resized = torch.nn.functional.interpolate(
                    luma_image.permute(0, 3, 1, 2),
                    scale_factor=scale,
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
            else:
                resized = luma_image
            
            # Process at this scale
            processed = LumaMaskProcessor.process_with_curves(resized, "s_curve")
            
            # Resize back to original size
            if scale != 1.0:
                processed = torch.nn.functional.interpolate(
                    processed.unsqueeze(1).float(),
                    size=luma_image.shape[1:3],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
            
            results.append(processed * weight)
        
        # Combine results
        combined = torch.sum(torch.stack(results), dim=0)
        return combined > 0.5
