import torch
import torch.nn.functional as F
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CategoricalLightEstimator:
    def __init__(self, x_threshold=0.4, y_threshold=0.1, central_threshold=0.3,
                 hard_light_threshold=0.15, soft_light_threshold=0.35,
                 format_mode="auto", normal_standard="OpenGL"):
        # Initialize thresholds
        self.x_threshold = x_threshold
        self.y_threshold = y_threshold
        self.central_threshold = central_threshold
        self.hard_light_threshold = hard_light_threshold
        self.soft_light_threshold = soft_light_threshold
        self.format_mode = format_mode
        self.normal_standard = normal_standard
        self.detected_format = None

        # Initialize Sobel filters for gradient analysis
        self.sobel_x = torch.tensor([[[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype=torch.float32)
        self.sobel_y = torch.tensor([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype=torch.float32)

    def full_analysis(self, normals, mask, luma_before=None, luma_after=None):
        """
        Perform comprehensive light direction analysis with verbose logging.
        """
        logger.info("=== STARTING FULL LIGHT ANALYSIS ===")
        logger.info(f"Input normals shape: {normals.shape}")
        logger.info(f"Input mask shape: {mask.shape}")
        logger.info(f"Mask coverage: {mask.sum().item()}/{mask.numel()} pixels ({mask.float().mean().item():.3f})")
        
        # Perform original clustering analysis
        logger.info("--- Analyzing geometric clustering (all normals) ---")
        geo = self._analyze_directional_clustering(normals, None)
        logger.info(f"Geometric clusters: {geo['clusters']}")
        logger.info(f"Dominant geometric cluster: {geo['dominant_cluster']}")
        
        logger.info("--- Analyzing perceived clustering (lit normals only) ---")
        perceived = self._analyze_directional_clustering(normals, mask.bool())
        logger.info(f"Perceived clusters: {perceived['clusters']}")
        logger.info(f"Dominant perceived cluster: {perceived['dominant_cluster']}")
        
        logger.info("--- Enhanced clustering analysis ---")
        enhanced = self._analyze_directional_clustering_enhanced(normals, mask, geo, perceived)
        logger.info(f"Enhanced analysis completed with {len(enhanced['enhanced_clusters'])} clusters")
        
        # Add regression analysis if luma data provided
        regression_data = {}
        if luma_before is not None and luma_after is not None:
            logger.info("--- Performing regression analysis ---")
            regression_data = self._regression_analysis(normals, mask, luma_before, luma_after)
            logger.info(f"Regression coefficients: {regression_data.get('regression_coefficients', 'N/A')}")
        
        # Perform legacy analysis
        logger.info("--- Performing legacy analysis ---")
        legacy_results = self._legacy_analysis(normals, mask, geo, perceived, enhanced)
        
        # Combine all results
        final_results = {
            **legacy_results,
            **regression_data
        }
        
        logger.info("=== ANALYSIS COMPLETE ===")
        logger.info(f"Final results keys: {list(final_results.keys())}")
        logger.info(f"X category: {final_results.get('x_category', 'N/A')}")
        logger.info(f"Y category: {final_results.get('y_category', 'N/A')}")
        logger.info(f"Hard/soft index: {final_results.get('hard_soft_index', 'N/A')}")
        
        return final_results

    def _regression_analysis(self, normals, mask, luma_before, luma_after):
        delta_y = luma_after - luma_before
        X = torch.stack([normals[...,0], normals[...,1], torch.ones_like(normals[...,0])], dim=-1)
        coeffs = torch.linalg.lstsq(X, delta_y.unsqueeze(-1)).solution.squeeze()
        
        grad_x = F.conv2d(delta_y.unsqueeze(1), self.sobel_x.unsqueeze(1))
        grad_y = F.conv2d(delta_y.unsqueeze(1), self.sobel_y.unsqueeze(1))
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        
        return {
            'regression_vector': coeffs[:2].tolist(),
            'gradient_softness': self._calc_gradient_softness(grad_mag, delta_y),
            'regression_coefficients': coeffs.tolist()
        }

    def _calc_gradient_softness(self, grad_mag, delta_y):
        grad_energy = torch.mean(grad_mag)
        signal_energy = torch.mean(torch.abs(delta_y))
        return (grad_energy / (signal_energy + 1e-8)).item()

    def _legacy_analysis(self, normals, mask, geo, perceived, enhanced):
        batch_size, height, width, _ = normals.shape
        current_mask = mask.bool()
        xy_normals = normals[current_mask][:, :2]
        
        x_analysis = self._analyze_x_direction(xy_normals)
        y_analysis = self._analyze_y_direction(xy_normals)
        quality_analysis = self._analyze_light_quality(xy_normals)
        
        return {
            'x_category': self._classify_x_direction(x_analysis),
            'y_category': self._classify_y_direction(y_analysis),
            'hard_soft_index': self._calculate_hard_soft_index(quality_analysis),
            'confidence': {
                'x_confidence': x_analysis['confidence'],
                'y_confidence': y_analysis['confidence'],
                'quality_confidence': quality_analysis['confidence'],
                'overall_confidence': (x_analysis['confidence'] + y_analysis['confidence'] + quality_analysis['confidence']) / 3
            },
            'quality_analysis': {
                'spread': (quality_analysis['x_spread'] + quality_analysis['y_spread']) / 2,
                'hard_soft_ratio': quality_analysis['hard_soft_ratio'],
                'confidence': quality_analysis['confidence']
            },
            'true_geometry': geo,
            'perceived_geometry': perceived,
            'enhanced_analysis': enhanced,
            'lit_pixel_count': perceived['total_normals'],
            'total_pixel_count': geo['total_normals']
        }

    def _analyze_directional_clustering(self, normals, mask=None):
        """
        Analyze directional clustering with detailed logging.
        """
        logger.info(f"--- Directional Clustering Analysis ---")
        logger.info(f"Input normals shape: {normals.shape}")
        logger.info(f"Mask provided: {mask is not None}")
        
        if normals.dim() == 4:
            flat_normals = normals.view(-1, 3)
            logger.info(f"Flattened normals from 4D to 2D: {flat_normals.shape}")
        else:
            flat_normals = normals
            logger.info(f"Normals already 2D: {flat_normals.shape}")
        
        if mask is not None:
            mask_flat = mask.view(-1).bool()
            original_count = len(flat_normals)
            flat_normals = flat_normals[mask_flat]
            logger.info(f"Applied mask: {original_count} -> {len(flat_normals)} normals")
            logger.info(f"Mask coverage: {mask_flat.sum().item()}/{mask_flat.numel()} ({mask_flat.float().mean().item():.3f})")
        else:
            logger.info("No mask applied - analyzing all normals")
        
        clusters = self._define_directional_clusters()
        logger.info(f"Defined {len(clusters)} directional clusters: {list(clusters.keys())}")
        
        cluster_results = {}
        total_analyzed = len(flat_normals)
        logger.info(f"Analyzing {total_analyzed} normal vectors")
        
        for name, ranges in clusters.items():
            logger.info(f"--- Analyzing cluster '{name}' ---")
            logger.info(f"  X range: {ranges['x_range']}")
            logger.info(f"  Y range: {ranges['y_range']}")
            logger.info(f"  Z range: {ranges['z_range']}")
            
            # Create mask for this cluster
            cluster_mask = (
                (flat_normals[:,0] >= ranges['x_range'][0]) & 
                (flat_normals[:,0] <= ranges['x_range'][1]) &
                (flat_normals[:,1] >= ranges['y_range'][0]) &
                (flat_normals[:,1] <= ranges['y_range'][1]) &
                (flat_normals[:,2] >= ranges['z_range'][0]) &
                (flat_normals[:,2] <= ranges['z_range'][1])
            )
            
            count = cluster_mask.sum().item()
            percentage = count / total_analyzed if total_analyzed > 0 else 0.0
            mean_normal = flat_normals[cluster_mask].mean(dim=0) if count > 0 else torch.zeros(3)
            
            logger.info(f"  Found {count} normals ({percentage:.3f}%)")
            logger.info(f"  Mean normal: [{mean_normal[0]:.3f}, {mean_normal[1]:.3f}, {mean_normal[2]:.3f}]")
            
            cluster_results[name] = {
                'count': count,
                'percentage': percentage,
                'mean_normal': mean_normal
            }
        
        # Find dominant cluster
        dominant_cluster = max(cluster_results, key=lambda k: cluster_results[k]['count'])
        logger.info(f"Dominant cluster: '{dominant_cluster}' with {cluster_results[dominant_cluster]['count']} normals")
        
        # Log summary
        logger.info("--- Clustering Summary ---")
        for name, data in cluster_results.items():
            logger.info(f"  {name}: {data['count']} normals ({data['percentage']:.3f}%)")
        
        return {
            'clusters': cluster_results,
            'total_normals': total_analyzed,
            'dominant_cluster': dominant_cluster
        }

    def analyze_directional_categories(self, normals, mask):
        """
        Main method for analyzing directional categories from normal maps.
        This is the primary interface method called by the ComfyUI node.
        """
        return self.full_analysis(normals, mask)
    
    def legacy_analysis(self, normals, mask):
        """
        Legacy analysis method - simpler approach for comparison.
        """
        logger.info("=== STARTING LEGACY LIGHT ANALYSIS ===")
        logger.info(f"Input normals shape: {normals.shape}")
        logger.info(f"Input mask shape: {mask.shape}")
        
        # Simple approach: just analyze the lit normals directly
        current_mask = mask.bool()
        lit_normals = normals[current_mask]
        
        logger.info(f"Lit normals count: {len(lit_normals)}")
        
        if len(lit_normals) == 0:
            logger.warning("No lit normals found!")
            return {
                'x_category': 'center',
                'y_category': 'center', 
                'hard_soft_index': 0.5,
                'confidence': {'x_confidence': 0.0, 'y_confidence': 0.0, 'overall_confidence': 0.0},
                'quality_analysis': {'spread': 0.0, 'hard_soft_ratio': 0.5, 'confidence': 0.0}
            }
        
        # Simple mean-based analysis
        mean_normal = lit_normals.mean(dim=0)
        logger.info(f"Mean normal: [{mean_normal[0]:.4f}, {mean_normal[1]:.4f}, {mean_normal[2]:.4f}]")
        
        # X direction classification
        x_mean = mean_normal[0].item()
        if x_mean > self.x_threshold:
            x_category = 'left'  # Light from left (surfaces point right)
        elif x_mean < -self.x_threshold:
            x_category = 'right'  # Light from right (surfaces point left)
        else:
            x_category = 'center'
        
        # Y direction classification  
        y_mean = mean_normal[1].item()
        if y_mean > self.y_threshold:
            y_category = 'above'  # Light from above (surfaces point up)
        elif y_mean < -self.y_threshold:
            y_category = 'below'  # Light from below (surfaces point down)
        else:
            y_category = 'center'
        
        # Simple spread calculation
        x_spread = lit_normals[:, 0].std().item()
        y_spread = lit_normals[:, 1].std().item()
        total_spread = (x_spread + y_spread) / 2
        
        # Simple confidence calculation
        x_confidence = min(1.0, abs(x_mean) * 2)
        y_confidence = min(1.0, abs(y_mean) * 2)
        overall_confidence = (x_confidence + y_confidence) / 2
        
        # Hard/soft calculation
        hard_soft_ratio = min(1.0, total_spread / self.soft_light_threshold)
        hard_soft_index = hard_soft_ratio
        
        logger.info(f"Legacy results:")
        logger.info(f"  X category: {x_category} (mean: {x_mean:.4f})")
        logger.info(f"  Y category: {y_category} (mean: {y_mean:.4f})")
        logger.info(f"  Hard/soft index: {hard_soft_index:.4f}")
        logger.info(f"  Overall confidence: {overall_confidence:.4f}")
        
        return {
            'x_category': x_category,
            'y_category': y_category,
            'hard_soft_index': hard_soft_index,
            'confidence': {
                'x_confidence': x_confidence,
                'y_confidence': y_confidence,
                'overall_confidence': overall_confidence
            },
            'quality_analysis': {
                'spread': total_spread,
                'hard_soft_ratio': hard_soft_ratio,
                'confidence': overall_confidence
            }
        }

    def _analyze_directional_clustering_enhanced(self, normals, mask, geo, perceived):
        """
        Enhanced clustering analysis that combines geometric and perceived data.
        """
        enhanced_clusters = {}
        
        # Combine geometric and perceived cluster data
        for cluster_name in geo['clusters'].keys():
            geo_data = geo['clusters'][cluster_name]
            perceived_data = perceived['clusters'].get(cluster_name, {'count': 0, 'percentage': 0.0})
            
            enhanced_clusters[cluster_name] = {
                'geometric_count': geo_data['count'],
                'perceived_count': perceived_data['count'],
                'geometric_percentage': geo_data['percentage'],
                'perceived_percentage': perceived_data['percentage'],
                'visibility_ratio': perceived_data['percentage'] / (geo_data['percentage'] + 1e-8),
                'mean_normal_geo': geo_data['mean_normal'],
                'mean_normal_perceived': perceived_data.get('mean_normal', torch.zeros(3))
            }
        
        return {
            'enhanced_clusters': enhanced_clusters,
            'total_geometric_normals': geo['total_normals'],
            'total_perceived_normals': perceived['total_normals'],
            'dominant_geometric_cluster': geo['dominant_cluster'],
            'dominant_perceived_cluster': perceived['dominant_cluster']
        }

    def _analyze_x_direction(self, xy_normals):
        """
        Analyze X-direction characteristics of normal vectors.
        """
        logger.info(f"--- X-Direction Analysis ---")
        logger.info(f"Input XY normals shape: {xy_normals.shape}")
        
        if len(xy_normals) == 0:
            logger.warning("No XY normals provided for X-direction analysis")
            return {'mean_x': 0.0, 'x_spread': 0.0, 'confidence': 0.0}
        
        x_values = xy_normals[:, 0]
        mean_x = x_values.mean().item()
        x_spread = x_values.std().item()
        
        # Calculate confidence based on spread and magnitude
        confidence = max(0.0, 1.0 - x_spread) * min(1.0, abs(mean_x) * 2)
        
        logger.info(f"X-direction stats:")
        logger.info(f"  Mean X: {mean_x:.4f}")
        logger.info(f"  X Spread (std): {x_spread:.4f}")
        logger.info(f"  Confidence: {confidence:.4f}")
        logger.info(f"  X range: [{x_values.min().item():.4f}, {x_values.max().item():.4f}]")
        
        return {
            'mean_x': mean_x,
            'x_spread': x_spread,
            'confidence': confidence
        }

    def _analyze_y_direction(self, xy_normals):
        """
        Analyze Y-direction characteristics of normal vectors.
        """
        logger.info(f"--- Y-Direction Analysis ---")
        logger.info(f"Input XY normals shape: {xy_normals.shape}")
        
        if len(xy_normals) == 0:
            logger.warning("No XY normals provided for Y-direction analysis")
            return {'mean_y': 0.0, 'y_spread': 0.0, 'confidence': 0.0}
        
        y_values = xy_normals[:, 1]
        mean_y = y_values.mean().item()
        y_spread = y_values.std().item()
        
        # Calculate confidence based on spread and magnitude
        confidence = max(0.0, 1.0 - y_spread) * min(1.0, abs(mean_y) * 2)
        
        logger.info(f"Y-direction stats:")
        logger.info(f"  Mean Y: {mean_y:.4f}")
        logger.info(f"  Y Spread (std): {y_spread:.4f}")
        logger.info(f"  Confidence: {confidence:.4f}")
        logger.info(f"  Y range: [{y_values.min().item():.4f}, {y_values.max().item():.4f}]")
        
        return {
            'mean_y': mean_y,
            'y_spread': y_spread,
            'confidence': confidence
        }

    def _analyze_light_quality(self, xy_normals):
        """
        Analyze light quality (hard vs soft) based on normal distribution.
        """
        if len(xy_normals) == 0:
            return {'hard_soft_ratio': 0.0, 'confidence': 0.0}
        
        # Calculate the spread of normals as an indicator of light softness
        x_spread = xy_normals[:, 0].std().item()
        y_spread = xy_normals[:, 1].std().item()
        total_spread = (x_spread + y_spread) / 2
        
        # Higher spread indicates softer light
        hard_soft_ratio = min(1.0, total_spread / self.soft_light_threshold)
        
        # Confidence based on how clear the distinction is
        confidence = min(1.0, abs(hard_soft_ratio - 0.5) * 2)
        
        return {
            'hard_soft_ratio': hard_soft_ratio,
            'confidence': confidence,
            'x_spread': x_spread,
            'y_spread': y_spread
        }

    def _classify_x_direction(self, x_analysis):
        """
        Classify X-direction into categorical labels.
        Returns the direction the light is coming FROM.
        """
        mean_x = x_analysis['mean_x']
        
        if mean_x > self.x_threshold:
            return "left"  # Light from left (surfaces point right)
        elif mean_x < -self.x_threshold:
            return "right"  # Light from right (surfaces point left)
        else:
            return "center"

    def _classify_y_direction(self, y_analysis):
        """
        Classify Y-direction into categorical labels.
        Returns the direction the light is coming FROM.
        """
        mean_y = y_analysis['mean_y']
        
        if mean_y > self.y_threshold:
            return "above"  # Light from above (surfaces point up)
        elif mean_y < -self.y_threshold:
            return "below"  # Light from below (surfaces point down)
        else:
            return "center"

    def _calculate_hard_soft_index(self, quality_analysis):
        """
        Calculate hard/soft light index based on quality analysis.
        """
        ratio = quality_analysis['hard_soft_ratio']
        
        if ratio < self.hard_light_threshold:
            return 0.0  # Hard light
        elif ratio > self.soft_light_threshold:
            return 1.0  # Soft light
        else:
            # Linear interpolation between hard and soft
            return (ratio - self.hard_light_threshold) / (self.soft_light_threshold - self.hard_light_threshold)

    def _define_directional_clusters(self):
        """
        Define directional clusters for normal vector analysis.
        """
        return {
            'front': {
                'x_range': [-0.5, 0.5],
                'y_range': [-0.5, 0.5],
                'z_range': [0.5, 1.0]
            },
            'back': {
                'x_range': [-0.5, 0.5],
                'y_range': [-0.5, 0.5],
                'z_range': [-1.0, -0.5]
            },
            'left': {
                'x_range': [-1.0, -0.5],
                'y_range': [-0.5, 0.5],
                'z_range': [-0.5, 0.5]
            },
            'right': {
                'x_range': [0.5, 1.0],
                'y_range': [-0.5, 0.5],
                'z_range': [-0.5, 0.5]
            },
            'up': {
                'x_range': [-0.5, 0.5],
                'y_range': [0.5, 1.0],
                'z_range': [-0.5, 0.5]
            },
            'down': {
                'x_range': [-0.5, 0.5],
                'y_range': [-1.0, -0.5],
                'z_range': [-0.5, 0.5]
            }
        }
