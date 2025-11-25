# Author: Yifan Wang
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import Config


def main():
    cfg = Config()
    
    print("=" * 60)
    print("Normalized Latent Space Visualization")
    print("=" * 60)
    
    print("\n[1/3] Loading Normalized Data...")
    train_data = np.load(Path(cfg.DATA_DIR) / 'train_data.npy')
    cluster_labels = np.load(Path(cfg.DATA_DIR) / 'cluster_labels.npy')
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Cluster labels shape: {cluster_labels.shape}")
    
    print("\n[2/3] Extracting Cluster Sequences...")
    cluster_0_mask = cluster_labels == 0
    cluster_8_mask = cluster_labels == 8
    
    cluster_0_sequences = train_data[cluster_0_mask][:, :, 0]
    cluster_8_sequences = train_data[cluster_8_mask][:, :, 0]
    
    print(f"Cluster 0: {cluster_0_sequences.shape[0]} sequences")
    print(f"Cluster 8: {cluster_8_sequences.shape[0]} sequences")
    
    cluster_0_centroid = cluster_0_sequences.mean(axis=0)
    cluster_8_centroid = cluster_8_sequences.mean(axis=0)
    
    print("\n[3/3] Generating Visualization...")
    
        # Plotting code generated with AI assistance
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(cluster_0_centroid))
    
    ax.plot(x, cluster_0_centroid, linewidth=3.5, color='darkred', 
            label='Cluster 0 (Bearish): -0.32% forward return', alpha=0.9)
    ax.plot(x, cluster_8_centroid, linewidth=3.5, color='darkgreen', 
            label='Cluster 8 (Bullish): +0.69% forward return', alpha=0.9)
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Mean (Z=0)')
    
    tail_start = 50
    
    ax.annotate('Exhaustion\n(Flat/Declining)', 
                xy=(55, cluster_0_centroid[55]), 
                xytext=(48, cluster_0_centroid[55] - 0.4),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                fontsize=11, color='darkred', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='mistyrose', alpha=0.8))
    
    ax.annotate('Momentum\n(Rising Slope)', 
                xy=(55, cluster_8_centroid[55]), 
                xytext=(48, cluster_8_centroid[55] + 0.4),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
                fontsize=11, color='darkgreen', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    slope_0 = cluster_0_centroid[-1] - cluster_0_centroid[-10]
    slope_8 = cluster_8_centroid[-1] - cluster_8_centroid[-10]
    
    ax.text(0.02, 0.98, f'Last 10-Day Slope:\nCluster 0: {slope_0:.3f}\nCluster 8: {slope_8:.3f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title("The AI's View: Momentum (Cluster 8) vs Exhaustion (Cluster 0)", 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Days (60-day window)', fontsize=13)
    ax.set_ylabel('Normalized Price (Z-Score)', fontsize=13)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ax.set_ylim(cluster_0_centroid.min() - 0.3, cluster_8_centroid.max() + 0.3)
    
    output_path = Path('plots') / 'normalized_comparison.png'
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nSaved: {output_path}")
    
    print("\n" + "=" * 60)
    print("Latent Pattern Analysis:")
    print("=" * 60)
    print(f"Cluster 0 (Bearish):")
    print(f"  Mean Z-Score: {cluster_0_centroid.mean():.3f}")
    print(f"  Final Value: {cluster_0_centroid[-1]:.3f}")
    print(f"  Trend (last 10 days): {slope_0:.3f}")
    print(f"\nCluster 8 (Bullish):")
    print(f"  Mean Z-Score: {cluster_8_centroid.mean():.3f}")
    print(f"  Final Value: {cluster_8_centroid[-1]:.3f}")
    print(f"  Trend (last 10 days): {slope_8:.3f}")
    print(f"\nDifferential: {slope_8 - slope_0:.3f} Z-score units")
    print("=" * 60)


if __name__ == '__main__':
    main()

