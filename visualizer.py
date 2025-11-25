# Author: Yifan Wang
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class ClusterVisualizer:
    def __init__(self, n_clusters=20):
        self.n_clusters = n_clusters
    
    def plot_clusters(self, data, labels, save_dir='plots/'):
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = labels == cluster_id
            cluster_data = data[cluster_mask]
            
            if len(cluster_data) == 0:
                continue
            
            centroid = cluster_data.mean(axis=0)
            std = cluster_data.std(axis=0)
            
# Plotting code generated with AI assistance
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(centroid))
            ax.plot(x, centroid, linewidth=2.5, color='blue', label='Centroid')
            ax.fill_between(x, centroid - std, centroid + std, alpha=0.3, color='lightblue')
            
            ax.set_title(f'Cluster {cluster_id} (Count: {len(cluster_data)})', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Steps', fontsize=12)
            ax.set_ylabel('Normalized Close Price', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            fig.savefig(save_path / f'cluster_{cluster_id}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Saved cluster_{cluster_id}.png (n={len(cluster_data)})")
        
        self._plot_grid_summary(data, labels, save_path)
    
    def _plot_grid_summary(self, data, labels, save_path):
        # Plotting code generated with AI assistance
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.flatten()
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = labels == cluster_id
            cluster_data = data[cluster_mask]
            
            if len(cluster_data) == 0:
                axes[cluster_id].set_visible(False)
                continue
            
            centroid = cluster_data.mean(axis=0)
            std = cluster_data.std(axis=0)
            
            x = np.arange(len(centroid))
            axes[cluster_id].plot(x, centroid, linewidth=2, color='blue')
            axes[cluster_id].fill_between(x, centroid - std, centroid + std, 
                                         alpha=0.3, color='lightblue')
            
            axes[cluster_id].set_title(f'Cluster {cluster_id} (n={len(cluster_data)})', 
                                      fontsize=10, fontweight='bold')
            axes[cluster_id].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(save_path / 'all_clusters.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved all_clusters.png")
