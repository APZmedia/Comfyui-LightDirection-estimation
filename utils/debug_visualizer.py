import torch

class DebugVisualizer:
    """
    Debug visualization tools for light estimation analysis.
    """
    
    @staticmethod
    def generate_debug_mask(mask):
        """
        Generate debug visualization of the binary mask.
        """
        # Ensure the mask is 3D (B, H, W) before unsqueezing
        if mask.dim() == 4 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        
        # Add channel dimension and repeat for RGB
        debug_mask = mask.float().unsqueeze(-1).repeat(1, 1, 1, 3)
        return debug_mask

    @staticmethod
    def generate_lit_normals_visualization(normals, mask):
        """
        Generate visualization showing only the lit normal vectors.
        """
        normal_rgb = (normals + 1.0) / 2.0
        lit_mask = mask.unsqueeze(-1)
        lit_normals_viz = normal_rgb * lit_mask
        return lit_normals_viz

    @staticmethod
    def generate_directional_visualization(normals, results):
        """
        Generate color-coded visualization of directional analysis.
        """
        batch_size, height, width, _ = normals.shape
        viz = torch.zeros(batch_size, height, width, 3, device=normals.device)

        x_category = results['x_category']
        y_category = results['y_category']
        hard_soft_index = results['hard_soft_index']
        overall_confidence = results['confidence']['overall_confidence']

        # Base colors
        if x_category == "left":
            base_color = torch.tensor([255, 100, 100], device=normals.device)
        elif x_category == "right":
            base_color = torch.tensor([200, 80, 80], device=normals.device)
        else:
            base_color = torch.tensor([150, 150, 150], device=normals.device)

        # Adjust for Y direction
        if y_category == "top":
            base_color += torch.tensor([0, 100, 50], device=normals.device)
        elif y_category == "bottom":
            base_color += torch.tensor([50, -50, 0], device=normals.device)

        # Apply softness and confidence
        softness_factor = 1.0 - hard_soft_index
        final_color = base_color * (0.7 + 0.3 * softness_factor) * overall_confidence
        final_color = torch.clamp(final_color, 0, 255)

        viz[:, :, :] = final_color.byte()
        return viz.float() / 255.0
    
    @staticmethod
    def generate_colormap_preview(height=256, width=256):
        """
        Generate a visual preview of the colormap used for analysis.
        """
        x = torch.linspace(-1, 1, width)
        y = torch.linspace(-1, 1, height)
        xx, yy = torch.meshgrid(y, x, indexing='ij')
        
        # Mock results for visualization
        mock_normals = torch.stack([xx, yy, torch.zeros_like(xx)], dim=-1).unsqueeze(0)
        
        # Generate a color for each point in the grid
        colormap = torch.zeros(height, width, 3)
        for i in range(height):
            for j in range(width):
                # Pseudo-analysis for color mapping
                x_val, y_val = xx[i, j], yy[i, j]
                
                # Determine color based on position
                red = 128 + 127 * x_val
                green = 128 + 127 * y_val
                blue = 128 - 127 * abs(x_val)
                
                colormap[i, j] = torch.tensor([red, green, blue])
        
        return colormap.byte().float() / 255.0
