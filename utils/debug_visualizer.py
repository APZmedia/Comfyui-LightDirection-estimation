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
        # Ensure mask is 3D (B, H, W)
        if mask.dim() == 4 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)

        # Convert to RGB visualization: white for lit areas, black for unlit
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
        Generate a clear reference guide for normal map color encoding.

        This shows how 3D surface orientations are encoded as RGB colors
        in normal maps, making it easy to understand what each color means.
        """
        # Create reference visualization
        viz = torch.zeros(height, width, 3)

        # Color legend for normal map encoding
        normal_colors = {
            'right': torch.tensor([1.0, 0.5, 0.5]),      # Red - Right facing
            'left': torch.tensor([0.5, 1.0, 1.0]),       # Cyan - Left facing
            'up': torch.tensor([0.5, 1.0, 0.5]),         # Green - Up facing
            'down': torch.tensor([1.0, 0.5, 1.0]),       # Purple - Down facing
            'out': torch.tensor([0.5, 0.5, 1.0]),        # Blue - Outward facing
            'flat': torch.tensor([0.5, 0.5, 0.5]),       # Gray - Flat/neutral
        }

        # Create visual layout
        section_height = height // 3
        section_width = width // 3

        sections = [
            ('X+ (Right)', normal_colors['right'], 0, 0),
            ('X- (Left)', normal_colors['left'], 0, 1),
            ('Y+ (Up)', normal_colors['up'], 1, 0),
            ('Y- (Down)', normal_colors['down'], 1, 1),
            ('Z+ (Out)', normal_colors['out'], 2, 0),
            ('Z=0 (Flat)', normal_colors['flat'], 2, 1),
        ]

        # Draw color sections with labels
        for label, color, row, col in sections:
            y_start, x_start = row * section_height, col * section_width

            # Fill section with color
            viz[y_start:y_start+section_height, x_start:x_start+section_width] = color

            # Add simple label area (could be enhanced with text rendering)
            label_y = y_start + section_height // 2
            label_x = x_start + 10

        # Add title and explanation
        title_area_height = height // 6
        viz[:title_area_height, :] = 0.8  # Light gray background for title

        return viz.unsqueeze(0)

    @staticmethod
    def generate_color_histogram(normals, title="Normal Distribution"):
        """
        Generate a clear, informative histogram of normal map color distribution.

        Args:
            normals: Normal vectors tensor (B, H, W, 3) in [-1, 1] range
            title: Title for the histogram

        Returns:
            histogram_viz: Clear visualization with labels and reference (1, 512, 512, 3)
        """
        return DebugVisualizer._create_enhanced_histogram(normals, title, show_reference=True)

    @staticmethod
    def generate_histogram_before(normals, title="Before Masking"):
        """
        Generate histogram of full normal map distribution.

        Args:
            normals: Normal vectors tensor (B, H, W, 3) for entire image
            title: Title for the histogram

        Returns:
            histogram_viz: RGB histogram of full distribution (1, 256, 256, 3)
        """
        return DebugVisualizer._create_color_histogram(normals, title)

    @staticmethod
    def generate_histogram_after(lit_normals, title="After Masking (Lit Areas)"):
        """
        Generate histogram of lit areas normal distribution.

        Args:
            lit_normals: Normal vectors tensor (N, 3) for lit areas only
            title: Title for the histogram

        Returns:
            histogram_viz: RGB histogram of lit areas distribution (1, 256, 256, 3)
        """
        # Add batch dimension if needed for consistent processing
        if lit_normals.dim() == 2:
            lit_normals = lit_normals.unsqueeze(0)

        return DebugVisualizer._create_color_histogram(lit_normals, title)

    @staticmethod
    def _create_color_histogram(normals, title="Color Distribution"):
        """
        Internal method to create color histogram from normal vectors.

        Args:
            normals: Normal vectors tensor (B, H, W, 3) or (N, 3)
            title: Title for the histogram

        Returns:
            histogram_viz: RGB visualization (1, 256, 256, 3)
        """
        # Flatten normals to analyze distribution
        flat_normals = normals.view(-1, 3)

        if flat_normals.shape[0] == 0:
            return torch.zeros(1, 256, 256, 3)

        # Scale normal vectors from [-1, 1] to [0, 255] for histogram bins
        scaled_normals = (flat_normals + 1.0) * 127.5
        scaled_normals = torch.clamp(scaled_normals, 0, 255)

        # Create 2D histogram using Red and Green channels as coordinates
        hist_size = 256
        histogram = torch.zeros(hist_size, hist_size)

        # Sample for performance
        sample_size = min(flat_normals.shape[0], 100000)
        indices = torch.randperm(flat_normals.shape[0])[:sample_size]

        for i in indices:
            normal = scaled_normals[i]
            r, g, b = normal[0].item(), normal[1].item(), normal[2].item()

            # Use red and green as histogram coordinates
            r_bin = int(r)
            g_bin = int(g)

            if 0 <= r_bin < hist_size and 0 <= g_bin < hist_size:
                histogram[r_bin, g_bin] += 1.0

        # Normalize for visualization
        if histogram.max() > 0:
            histogram = histogram / histogram.max()

        # Create RGB visualization
        rgb_histogram = torch.zeros(hist_size, hist_size, 3)

        for i in range(hist_size):
            for j in range(hist_size):
                density = histogram[i, j].item()

                if density > 0:
                    # Color by position and density
                    # X direction (red) + Y direction (green) + density (blue)
                    x_intensity = i / 255.0  # Red channel shows X position
                    y_intensity = j / 255.0  # Green channel shows Y position
                    z_intensity = density    # Blue channel shows density

                    rgb_histogram[i, j] = torch.tensor([
                        x_intensity,  # X direction indicator
                        y_intensity,  # Y direction indicator
                        z_intensity   # Density indicator
                    ])

        return rgb_histogram.unsqueeze(0)

    @staticmethod
    def _create_enhanced_histogram(normals, title="Normal Distribution", show_reference=True):
        """
        Create an enhanced histogram with labels, legends, and clear visual design.

        Args:
            normals: Normal vectors tensor (B, H, W, 3) or (N, 3)
            title: Title for the histogram
            show_reference: Whether to show reference information

        Returns:
            enhanced_viz: Complete visualization with labels (1, 512, 512, 3)
        """
        # Get base histogram (256x256)
        base_histogram = DebugVisualizer._create_color_histogram_internal(normals)

        if base_histogram is None:
            return torch.zeros(1, 512, 512, 3)

        # Create enhanced visualization with labels and reference
        viz_size = 512
        hist_size = 256
        enhanced = torch.zeros(viz_size, viz_size, 3)

        # Place histogram in top-left area (256x256)
        enhanced[:hist_size, :hist_size] = base_histogram[0]

        # Add reference information (right side)
        if show_reference:
            enhanced = DebugVisualizer._add_histogram_labels(enhanced, title)

        return enhanced.unsqueeze(0)

    @staticmethod
    def _create_color_histogram_internal(normals):
        """Internal histogram creation without labels"""
        flat_normals = normals.view(-1, 3)
        if flat_normals.shape[0] == 0:
            return None

        # Scale to [0, 255] for histogram bins
        scaled_normals = (flat_normals + 1.0) * 127.5
        scaled_normals = torch.clamp(scaled_normals, 0, 255)

        # Create histogram
        hist_size = 256
        histogram = torch.zeros(hist_size, hist_size)

        sample_size = min(flat_normals.shape[0], 50000)
        indices = torch.randperm(flat_normals.shape[0])[:sample_size]

        for i in indices:
            normal = scaled_normals[i]
            r, g, b = normal[0].item(), normal[1].item(), normal[2].item()
            r_bin, g_bin = int(r), int(g)

            if 0 <= r_bin < hist_size and 0 <= g_bin < hist_size:
                histogram[r_bin, g_bin] += 1.0

        # Normalize
        if histogram.max() > 0:
            histogram = histogram / histogram.max()

        # Create visualization with clear color mapping
        rgb_histogram = torch.zeros(hist_size, hist_size, 3)

        # Add coordinate grid lines
        grid_color = torch.tensor([0.3, 0.3, 0.3])
        for i in range(0, hist_size, 32):
            rgb_histogram[i, :, :] = grid_color
            rgb_histogram[:, i, :] = grid_color

        # Add density visualization
        for i in range(hist_size):
            for j in range(hist_size):
                density = histogram[i, j].item()
                if density > 0.01:  # Threshold to reduce noise
                    # Heatmap: blue (low) to red (high)
                    if density < 0.33:
                        intensity = density * 3
                        rgb_histogram[i, j] = torch.tensor([0.0, 0.0, intensity])
                    elif density < 0.66:
                        intensity = (density - 0.33) * 3
                        rgb_histogram[i, j] = torch.tensor([intensity, 0.0, 0.0])
                    else:
                        intensity = (density - 0.66) * 3
                        rgb_histogram[i, j] = torch.tensor([1.0, intensity, 0.0])

        return rgb_histogram.unsqueeze(0)

    @staticmethod
    def _add_histogram_labels(enhanced_viz, title):
        """Add labels and reference information to histogram"""
        viz_size = enhanced_viz.shape[0]

        # Add title (top area)
        title_y = 20
        # Simple title for now - could be enhanced with text rendering

        # Add axis labels
        # X-axis: "Left ← X Direction → Right"
        # Y-axis: "Bottom ← Y Direction → Top"

        # Add color legend (right side)
        legend_x = int(viz_size * 0.8)
        legend_width = viz_size - legend_x

        # Color legend: show what colors mean
        legend_colors = [
            torch.tensor([0.0, 0.0, 1.0]),  # Blue = low density
            torch.tensor([0.0, 1.0, 0.0]),  # Green = medium density
            torch.tensor([1.0, 0.0, 0.0]),  # Red = high density
        ]

        for i, color in enumerate(legend_colors):
            y = int(viz_size * 0.3) + i * 30
            enhanced_viz[y:y+20, legend_x:legend_x+20] = color

        return enhanced_viz
