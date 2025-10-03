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
        
        clusters = ['Right', 'Left', 'Up', 'Down', 'Front', 'Flat']
        before = [cluster_results['clustering_before']['clusters'][c]['percentage'] for c in ['right', 'left', 'up', 'down', 'front', 'flat']]
        after = [cluster_results['clustering_after']['clusters'][c]['percentage'] for c in ['right', 'left', 'up', 'down', 'front', 'flat']]

        x = np.arange(len(clusters))
        width = 0.35

        rects1 = ax.bar(x - width/2, before, width, label='Original', color='#1f77b4')
        rects2 = ax.bar(x + width/2, after, width, label='Filtered', color='#ff7f0e')

        ax.set_ylabel('Percentage of Normals')
        ax.set_title('Cluster Distribution: Original vs Filtered')
        ax.set_xticks(x)
        ax.set_xticklabels(clusters)
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
