import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

class DebugVisualizer:
    """
    Simplified visualization tools focused on quantitative analysis
    """
    
    @staticmethod
    def generate_debug_mask(mask):
        """
        Generate debug visualization of the binary mask.
        """
        if mask.dim() == 4 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        viz = mask.float().unsqueeze(-1).repeat(1, 1, 1, 3)
        return viz.clamp(0.0, 1.0)  # Ensure valid pixel values

    @staticmethod
    def generate_lit_normals_visualization(normals, mask):
        """
        Generate visualization showing only the lit normal vectors.
        """
        normal_rgb = (normals + 1.0) / 2.0
        lit_mask = mask.unsqueeze(-1)
        return normal_rgb * lit_mask

    @staticmethod
    def create_cluster_delta_chart(cluster_results):
        """
        Create matplotlib bar chart comparing original vs filtered cluster distributions
        """
        plt.switch_backend('Agg')  # Required for headless environments
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Use the actual cluster names defined in CategoricalLightEstimator
        cluster_names = ['front', 'back', 'left', 'right', 'up', 'down']
        cluster_labels = ['Front', 'Back', 'Left', 'Right', 'Up', 'Down']
        
        # Safely get cluster percentages, defaulting to 0.0 if cluster doesn't exist
        def safe_get_percentage(clusters_dict, cluster_name):
            return clusters_dict.get(cluster_name, {}).get('percentage', 0.0)
        
        before = [safe_get_percentage(cluster_results['true_geometry']['clusters'], c) for c in cluster_names]
        after = [safe_get_percentage(cluster_results['perceived_geometry']['clusters'], c) for c in cluster_names]

        x = np.arange(len(cluster_labels))
        width = 0.35

        rects1 = ax.bar(x - width/2, before, width, label='Original', color='#1f77b4')
        rects2 = ax.bar(x + width/2, after, width, label='Filtered', color='#ff7f0e')

        ax.set_ylabel('Percentage of Normals')
        ax.set_title('True Geometry vs Lighting Perception Clusters', fontsize=14, pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(cluster_labels)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        # Convert to tensor
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, pad_inches=0.1)
        # Set explicit image dimensions to ensure consistent output size
        plt.gcf().set_size_inches(12, 6)  # Force 12"x6" at 100dpi = 1200x600 pixels
        plt.close(fig)
        buf.seek(0)
        
        pil_img = Image.open(buf).convert('RGB')
        img_array = np.array(pil_img).astype(np.float32) / 255.0
        tensor_img = torch.from_numpy(img_array).unsqueeze(0)  # Shape: [1, H, W, 3]
        
        # Ensure proper format for ComfyUI
        tensor_img = tensor_img.clamp(0.0, 1.0).type(torch.float32)
        return tensor_img
    
    @staticmethod
    def create_threshold_classification_chart(normal_map, x_threshold, y_threshold, mask=None):
        """
        Create a chart showing threshold-based classification results.
        Shows original normal values vs lit-only masked values comparison.
        """
        import torch
        import numpy as np
        
        plt.switch_backend('Agg')
        
        # Convert normal map to -1 to +1 range if needed
        if normal_map.max() <= 1.0:
            normals_x = (normal_map[:, :, :, 0] * 2.0) - 1.0
            normals_y = (normal_map[:, :, :, 1] * 2.0) - 1.0
        else:
            normals_x = normal_map[:, :, :, 0]
            normals_y = normal_map[:, :, :, 1]
        
        # Flatten for analysis
        flat_x = normals_x.view(-1)
        flat_y = normals_y.view(-1)
        
        # Calculate overall distribution (all pixels)
        x_left_all = (flat_x < -x_threshold).sum().item()
        x_center_all = ((flat_x >= -x_threshold) & (flat_x <= x_threshold)).sum().item()
        x_right_all = (flat_x > x_threshold).sum().item()
        x_total_all = len(flat_x)
        
        y_above_all = (flat_y < -y_threshold).sum().item()
        y_center_all = ((flat_y >= -y_threshold) & (flat_y <= y_threshold)).sum().item()
        y_below_all = (flat_y > y_threshold).sum().item()
        y_total_all = len(flat_y)
        
        # Calculate lit-only distribution if mask provided
        if mask is not None:
            # Ensure mask is same size
            if mask.shape[1] != normals_x.shape[1] or mask.shape[2] != normals_x.shape[2]:
                # Handle different mask formats
                if mask.dim() == 3:  # [B, H, W] format
                    mask = mask.unsqueeze(1)  # Add channel dimension: [B, 1, H, W]
                elif mask.dim() == 2:  # [H, W] format
                    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                
                # Now interpolate with proper 4D format
                mask = torch.nn.functional.interpolate(
                    mask, 
                    size=(normals_x.shape[1], normals_x.shape[2]), 
                    mode='nearest'
                )
                
                # Convert back to original format
                if mask.shape[1] == 1:  # Remove channel dimension if it was added
                    mask = mask.squeeze(1)  # [B, H, W]
                if mask.shape[0] == 1:  # Remove batch dimension if it was added
                    mask = mask.squeeze(0)  # [H, W]
            
            flat_mask = mask.view(-1).bool()
            lit_x = flat_x[flat_mask]
            lit_y = flat_y[flat_mask]
            
            x_left_lit = (lit_x < -x_threshold).sum().item()
            x_center_lit = ((lit_x >= -x_threshold) & (lit_x <= x_threshold)).sum().item()
            x_right_lit = (lit_x > x_threshold).sum().item()
            x_total_lit = len(lit_x)
            
            y_above_lit = (lit_y < -y_threshold).sum().item()
            y_center_lit = ((lit_y >= -y_threshold) & (lit_y <= y_threshold)).sum().item()
            y_below_lit = (lit_y > y_threshold).sum().item()
            y_total_lit = len(lit_y)
            
            # Create 2x2 subplot layout
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # X Direction - Original Normal Values
            x_categories = ['Light from Right', 'Center', 'Light from Left']
            x_counts_all = [x_left_all, x_center_all, x_right_all]
            x_percentages_all = [count / x_total_all * 100 for count in x_counts_all]
            
            bars1 = ax1.bar(x_categories, x_percentages_all, color=['red', 'green', 'blue'])
            ax1.set_title(f'X Direction - Original Normal Values (threshold={x_threshold})', fontsize=12)
            ax1.set_ylabel('Percentage of All Pixels')
            ax1.set_ylim(0, 100)
            
            for bar, pct in zip(bars1, x_percentages_all):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{pct:.1f}%', ha='center', va='bottom')
            
            # X Direction - Lit Masked Values
            x_counts_lit = [x_left_lit, x_center_lit, x_right_lit]
            x_percentages_lit = [count / x_total_lit * 100 for count in x_counts_lit] if x_total_lit > 0 else [0, 0, 0]
            
            bars2 = ax2.bar(x_categories, x_percentages_lit, color=['red', 'green', 'blue'])
            ax2.set_title(f'X Direction - Lit Masked Values (threshold={x_threshold})', fontsize=12)
            ax2.set_ylabel('Percentage of Lit Pixels')
            ax2.set_ylim(0, 100)
            
            for bar, pct in zip(bars2, x_percentages_lit):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{pct:.1f}%', ha='center', va='bottom')
            
            # Y Direction - Original Normal Values
            y_categories = ['Light from Above', 'Center', 'Light from Below']
            y_counts_all = [y_above_all, y_center_all, y_below_all]
            y_percentages_all = [count / y_total_all * 100 for count in y_counts_all]
            
            bars3 = ax3.bar(y_categories, y_percentages_all, color=['red', 'green', 'blue'])
            ax3.set_title(f'Y Direction - Original Normal Values (threshold={y_threshold})', fontsize=12)
            ax3.set_ylabel('Percentage of All Pixels')
            ax3.set_ylim(0, 100)
            
            for bar, pct in zip(bars3, y_percentages_all):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{pct:.1f}%', ha='center', va='bottom')
            
            # Y Direction - Lit Masked Values
            y_counts_lit = [y_above_lit, y_center_lit, y_below_lit]
            y_percentages_lit = [count / y_total_lit * 100 for count in y_counts_lit] if y_total_lit > 0 else [0, 0, 0]
            
            bars4 = ax4.bar(y_categories, y_percentages_lit, color=['red', 'green', 'blue'])
            ax4.set_title(f'Y Direction - Lit Masked Values (threshold={y_threshold})', fontsize=12)
            ax4.set_ylabel('Percentage of Lit Pixels')
            ax4.set_ylim(0, 100)
            
            for bar, pct in zip(bars4, y_percentages_lit):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{pct:.1f}%', ha='center', va='bottom')
        
        else:
            # No mask - show only overall distribution
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # X Direction Classification
            x_categories = ['Light from Right', 'Center', 'Light from Left']
            x_counts = [x_left_all, x_center_all, x_right_all]
            x_percentages = [count / x_total_all * 100 for count in x_counts]
            
            bars1 = ax1.bar(x_categories, x_percentages, color=['red', 'green', 'blue'])
            ax1.set_title(f'X Direction - Original Normal Values (threshold={x_threshold})', fontsize=12)
            ax1.set_ylabel('Percentage of Pixels')
            ax1.set_ylim(0, 100)
            
            for bar, pct in zip(bars1, x_percentages):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{pct:.1f}%', ha='center', va='bottom')
            
            # Y Direction Classification
            y_categories = ['Light from Above', 'Center', 'Light from Below']
            y_counts = [y_above_all, y_center_all, y_below_all]
            y_percentages = [count / y_total_all * 100 for count in y_counts]
            
            bars2 = ax2.bar(y_categories, y_percentages, color=['red', 'green', 'blue'])
            ax2.set_title(f'Y Direction - Original Normal Values (threshold={y_threshold})', fontsize=12)
            ax2.set_ylabel('Percentage of Pixels')
            ax2.set_ylim(0, 100)
            
            for bar, pct in zip(bars2, y_percentages):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{pct:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Convert to tensor
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        pil_img = Image.open(buf).convert('RGB')
        img_array = np.array(pil_img).astype(np.float32) / 255.0
        tensor_img = torch.from_numpy(img_array).unsqueeze(0)
        
        return tensor_img
