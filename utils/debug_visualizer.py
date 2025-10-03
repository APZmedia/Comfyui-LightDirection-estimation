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
    def generate_colormap_preview(height=512, width=512):
        """
        Generate a comprehensive reference guide for normal map color encoding.

        This creates an educational reference showing exactly how 3D surface
        orientations are encoded as RGB colors in normal maps.
        """
        # Create main visualization canvas
        viz = torch.ones(height, width, 3) * 0.9  # Light gray background

        # Title section
        title_height = 60
        viz[:title_height, :] = torch.tensor([0.7, 0.7, 0.7])  # Darker background for title

        # Color reference sections in a 2x3 grid
        section_height = (height - title_height - 40) // 2  # Leave space for labels
        section_width = width // 3

        # Define normal map color encodings with clear labels
        normal_directions = [
            # Row 1: X-axis directions
            {
                'label': 'X+ (Right)',
                'description': 'Surfaces facing RIGHT are RED',
                'color': torch.tensor([1.0, 0.5, 0.5]),  # Red
                'position': 'Lit by RIGHT-side lighting',
                'row': 0, 'col': 0
            },
            {
                'label': 'X- (Left)',
                'description': 'Surfaces facing LEFT are CYAN',
                'color': torch.tensor([0.5, 1.0, 1.0]),  # Cyan
                'position': 'Lit by LEFT-side lighting',
                'row': 0, 'col': 1
            },
            {
                'label': 'Y+ (Up)',
                'description': 'Surfaces facing UP are GREEN',
                'color': torch.tensor([0.5, 1.0, 0.5]),  # Green
                'position': 'Lit by TOP lighting',
                'row': 0, 'col': 2
            },

            # Row 2: Y-axis and Z-axis directions
            {
                'label': 'Y- (Down)',
                'description': 'Surfaces facing DOWN are PURPLE',
                'color': torch.tensor([1.0, 0.5, 1.0]),  # Purple
                'position': 'Lit by BOTTOM lighting',
                'row': 1, 'col': 0
            },
            {
                'label': 'Z+ (Out)',
                'description': 'Surfaces facing OUT are BLUE',
                'color': torch.tensor([0.5, 0.5, 1.0]),  # Blue
                'position': 'Lit by FRONT lighting',
                'row': 1, 'col': 1
            },
            {
                'label': 'Z=0 (Flat)',
                'description': 'Flat surfaces are GRAY',
                'color': torch.tensor([0.5, 0.5, 0.5]),  # Gray
                'position': 'Rarely lit directly',
                'row': 1, 'col': 2
            }
        ]

        # Draw each section
        for direction in normal_directions:
            row, col = direction['row'], direction['col']
            y_start = title_height + row * section_height
            x_start = col * section_width

            # Fill section with the characteristic color
            color_block_size = 80
            y_center = y_start + section_height // 2 - color_block_size // 2
            x_center = x_start + section_width // 2 - color_block_size // 2

            viz[y_center:y_center+color_block_size, x_center:x_center+color_block_size] = direction['color']

            # Add section border
            border_color = torch.tensor([0.3, 0.3, 0.3])
            viz[y_start:y_start+section_height, x_start:x_start+5] = border_color
            viz[y_start:y_start+section_height, x_start+section_width-5:x_start+section_width] = border_color
            viz[y_start:y_start+5, x_start:x_start+section_width] = border_color
            viz[y_start+section_height-5:y_start+section_height, x_start:x_start+section_width] = border_color

        # Add usage examples at bottom
        example_height = 100
        example_y = height - example_height
        viz[example_y:, :] = torch.tensor([0.95, 0.95, 0.95])

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
    def generate_directional_cluster_viz(normals, results):
        """
        Visualize directional clusters on the normal map.
        """
        if 'clustering_before' not in results:
            return torch.zeros_like(normals)

        # Create a blank canvas
        viz = torch.zeros_like(normals)

        # Define colors for each cluster
        cluster_colors = {
            'right': torch.tensor([1.0, 0.0, 0.0]),  # Red
            'left': torch.tensor([0.0, 1.0, 1.0]),   # Cyan
            'up': torch.tensor([0.0, 1.0, 0.0]),     # Green
            'down': torch.tensor([1.0, 0.0, 1.0]),   # Purple
            'front': torch.tensor([0.0, 0.0, 1.0]),  # Blue
            'flat': torch.tensor([0.5, 0.5, 0.5]),   # Gray
        }

        # Iterate through pixels and color them based on their cluster
        # This is computationally expensive, so it should be used for debugging
        # A more optimized version could use masks for each cluster
        return viz.unsqueeze(0)

    @staticmethod
    def generate_cluster_distribution_chart(results):
        """
        Create a bar chart of lit vs unlit distribution for each cluster.
        """
        # (Implementation to be added)
        return torch.zeros(1, 512, 512, 3)

    @staticmethod
    def generate_lighting_summary_viz(results):
        """
        Create a summary infographic of the final results.
        """
        # (Implementation to be added)
        return torch.zeros(1, 256, 512, 3)

    @staticmethod
    def generate_lighting_summary_viz(results):
        """
        Create a summary infographic of the final results.
        """
        # (Implementation to be added)
        return torch.zeros(1, 256, 512, 3)

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
