# Author: Yifan Wang
import numpy as np
from pathlib import Path
from config import Config
from visualizer import ClusterVisualizer


def main():
    cfg = Config()
    
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    
    print("Loading data...")
    train_data = np.load(Path(cfg.DATA_DIR) / 'train_data.npy')
    cluster_labels = np.load(Path(cfg.DATA_DIR) / 'cluster_labels.npy')
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Cluster labels shape: {cluster_labels.shape}")
    
    close_prices = train_data[:, :, 0]
    print(f"Close prices shape: {close_prices.shape}")
    
    print("\nGenerating cluster visualizations...")
    visualizer = ClusterVisualizer(n_clusters=20)
    visualizer.plot_clusters(close_prices, cluster_labels, save_dir='plots/')
    
    print("\nPlots saved to plots/ folder.")


if __name__ == '__main__':
    main()

