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

        # Analyze true geometry (all normals)
        geometry_analysis = self._analyze_directional_clustering(normals, None)
        
        # Analyze perceived geometry through lighting mask
        perceived_analysis = self._analyze_directional_clustering(normals, current_mask)

        # Enhanced analysis using directional clustering
        enhanced_results = self._analyze_directional_clustering_enhanced(
            normals, current_mask, geometry_analysis, perceived_analysis
        )

        # Analyze red channel intensities
        red_analysis = {
            'full_red': normals[..., 0].mean().item(),
            'lit_red': normals[current_mask][..., 0].mean().item() if current_mask.any() else 0.0,
            'red_impact_ratio': 0.0
        }
        if red_analysis['full_red'] > 0:
            red_analysis['red_impact_ratio'] = red_analysis['lit_red'] / red_analysis['full_red']

        # DEBUG: Print key information for pipeline verification
        print("\nRed Channel Analysis:")
        print(f"Full image red: {red_analysis['full_red']:.3f}")
        print(f"Lit areas red: {red_analysis['lit_red']:.3f}")
        print(f"Impact ratio: {red_analysis['red_impact_ratio']:.2f}x")
        print("=== Pipeline Verification ===")
        print(f"Input normals shape: {normals.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Lit pixels analyzed: {perceived_analysis['total_normals']}")
        print(f"Total pixels: {geometry_analysis['total_normals']}")

        # Show clustering analysis
        print("=== Geometry vs Lighting Perception ===")
        print("True Geometry Distribution:")
        for cluster, data in geometry_analysis['clusters'].items():
            print(f"- {cluster}: {data['percentage']:.1%}")
        
        print("\nLighting Perception Analysis:")
        # Show all directions with explicit vertical labels
        clusters_to_show = ['right', 'left', 'up', 'down', 'front', 'flat']
        for cluster in clusters_to_show:
            geo_pct = geometry_analysis['clusters'][cluster]['percentage']
            perc_pct = perceived_analysis['clusters'][cluster]['percentage']
            direction_label = {'up': 'top', 'down': 'bottom'}.get(cluster, cluster)
            print(f"- {direction_label}: {perc_pct:.1%} (Δ{perc_pct-geo_pct:+.1%})")
        print(f"Enhanced analysis: {enhanced_results['primary_direction']}")
        print("==========================")

        # Extract XY components
        xy_normals = normals[current_mask][:, :2]

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
            'true_geometry': geometry_analysis,
            'perceived_geometry': perceived_analysis,
            'enhanced_analysis': enhanced_results,
            'red_channel_analysis': red_analysis,
            'lit_pixel_count': perceived_analysis['total_normals'],
            'total_pixel_count': geometry_analysis['total_normals']
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

    def _analyze_directional_clustering(self, normals, mask=None):
        """
        Analyze normal vectors by clustering them into directional categories.

        Args:
            normals: Normal vectors tensor (B, H, W, 3)
            mask: Optional binary mask to limit analysis

        Returns:
            clustering_results: Dictionary with directional cluster analysis
        """
        if normals.dim() == 4:
            flat_normals = normals.view(-1, 3)
            height, width = normals.shape[1], normals.shape[2]
        else:
            flat_normals = normals
            height, width = 1, normals.shape[0]

        if mask is not None:
            if mask.dim() == 4:
                mask = mask.squeeze(-1)
            mask_flat = mask.view(-1)
            flat_normals = flat_normals[mask_flat.bool()]

        if flat_normals.shape[0] == 0:
            return self._empty_directional_clustering()

        # Define directional color clusters
        clusters = self._define_directional_clusters()

        # Analyze each cluster
        cluster_results = {}
        for cluster_name, color_ranges in clusters.items():
            cluster_normals = self._find_normals_in_clusters(flat_normals, color_ranges)
            cluster_results[cluster_name] = {
                'count': cluster_normals.shape[0],
                'percentage': cluster_normals.shape[0] / flat_normals.shape[0],
                'mean_normal': torch.mean(cluster_normals, dim=0) if cluster_normals.shape[0] > 0 else torch.zeros(3)
            }

        return {
            'clusters': cluster_results,
            'total_normals': flat_normals.shape[0],
            'dominant_cluster': max(cluster_results.keys(), key=lambda k: cluster_results[k]['count'])
        }

    def _define_directional_clusters(self):
        """
        Define color ranges for each directional cluster.

        Returns:
            clusters: Dictionary mapping direction names to color ranges
        """
        # Define color ranges for each direction (in RGB space)
        clusters = {
            'right': {  # Red-dominant colors (X+)
                'x_range': [0.4, 1.0],    # Red channel high
                'y_range': [0.3, 0.7],    # Green channel mid
                'z_range': [0.3, 0.7]     # Blue channel mid
            },
            'left': {   # Cyan-dominant colors (X-)
                'x_range': [0.0, 0.6],    # Red channel low
                'y_range': [0.4, 1.0],    # Green channel high
                'z_range': [0.4, 1.0]     # Blue channel high
            },
            'up': {     # Green-dominant colors (Y+)
                'x_range': [0.3, 0.7],    # Red channel mid
                'y_range': [0.4, 1.0],    # Green channel high
                'z_range': [0.3, 0.7]     # Blue channel mid
            },
            'down': {   # Purple-dominant colors (Y-)
                'x_range': [0.4, 1.0],    # Red channel high
                'y_range': [0.0, 0.6],    # Green channel low
                'z_range': [0.4, 1.0]     # Blue channel high
            },
            'front': {  # Blue-dominant colors (Z+)
                'x_range': [0.3, 0.7],    # Red channel mid
                'y_range': [0.3, 0.7],    # Green channel mid
                'z_range': [0.4, 1.0]     # Blue channel high
            },
            'flat': {   # Gray/neutral colors (Z≈0)
                'x_range': [0.3, 0.7],    # Red channel mid
                'y_range': [0.3, 0.7],    # Green channel mid
                'z_range': [0.0, 0.6]     # Blue channel low
            }
        }

        return clusters

    def _find_normals_in_clusters(self, normals, color_ranges):
        """
        Find normal vectors that fall within specified color ranges.

        Args:
            normals: Normal vectors tensor (N, 3)
            color_ranges: Dictionary with x_range, y_range, z_range

        Returns:
            filtered_normals: Normals within the color ranges
        """
        x_min, x_max = color_ranges['x_range']
        y_min, y_max = color_ranges['y_range']
        z_min, z_max = color_ranges['z_range']

        # Filter normals based on color ranges
        mask = (normals[:, 0] >= x_min) & (normals[:, 0] <= x_max) & \
               (normals[:, 1] >= y_min) & (normals[:, 1] <= y_max) & \
               (normals[:, 2] >= z_min) & (normals[:, 2] <= z_max)

        return normals[mask]

    def _analyze_directional_clustering_enhanced(self, normals, mask, clustering_before, clustering_after):
        """
        Enhanced analysis using directional clustering with luminance correlation.

        Args:
            normals: Full normal map tensor
            mask: Binary mask tensor
            clustering_before: Clustering results for full image
            clustering_after: Clustering results for lit areas

        Returns:
            enhanced_results: Advanced lighting analysis
        """
        # Calculate lighting effect on each directional cluster
        lighting_effects = {}

        for cluster_name in clustering_before['clusters'].keys():
            before_pct = clustering_before['clusters'][cluster_name]['percentage']
            after_pct = clustering_after['clusters'][cluster_name]['percentage']

            # Calculate how much this cluster is affected by lighting
            lighting_effect = after_pct - before_pct
            lighting_effects[cluster_name] = {
                'before_percentage': before_pct,
                'after_percentage': after_pct,
                'lighting_effect': lighting_effect,
                'is_illuminated': lighting_effect > 0.05  # Threshold for significant illumination
            }

        # Determine primary lighting direction based on cluster effects
        illuminated_clusters = [name for name, effect in lighting_effects.items() if effect['is_illuminated']]

        # Map cluster illumination to lighting direction
        direction_mapping = {
            'right': 'right',
            'left': 'left',
            'up': 'top',
            'down': 'bottom',
            'front': 'front'
        }

        detected_directions = [direction_mapping.get(cluster, 'unknown') for cluster in illuminated_clusters]

        return {
            'lighting_effects': lighting_effects,
            'illuminated_clusters': illuminated_clusters,
            'detected_directions': detected_directions,
            'primary_direction': max(detected_directions, default='unknown') if detected_directions else 'diffuse'
        }

    def _empty_directional_clustering(self):
        """Return empty clustering results"""
        clusters = {name: {'count': 0, 'percentage': 0.0, 'mean_normal': torch.zeros(3)}
                   for name in ['right', 'left', 'up', 'down', 'front', 'flat']}

        return {
            'clusters': clusters,
            'total_normals': 0,
            'dominant_cluster': 'none'
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
