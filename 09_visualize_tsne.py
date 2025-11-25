# Author: Yifan Wang
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from config import Config
from models import CnnAutoencoder


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    cfg = Config()
    
    print("=" * 60)
    print("t-SNE Manifold Visualization")
    print("=" * 60)
    
    print("\n[1/5] Loading Data...")
    train_data = np.load(Path(cfg.DATA_DIR) / 'train_data.npy')
    cluster_labels = np.load(Path(cfg.DATA_DIR) / 'cluster_labels.npy')
    
    print(f"Total samples: {len(train_data)}")
    
    print("\n[2/5] Sampling 5,000 points...")
    np.random.seed(42)
    sample_indices = np.random.choice(len(train_data), size=5000, replace=False)
    sampled_data = train_data[sample_indices]
    sampled_labels = cluster_labels[sample_indices]
    
    print(f"Sampled data: {sampled_data.shape}")
    print(f"Cluster 0 samples: {(sampled_labels == 0).sum()}")
    print(f"Cluster 8 samples: {(sampled_labels == 8).sum()}")
    
    print("\n[3/5] Loading Model and Extracting Embeddings...")
    device = get_device()
    model = CnnAutoencoder(
        input_dim=cfg.INPUT_DIM,
        seq_len=cfg.SEQ_LEN,
        latent_dim=cfg.LATENT_DIM
    ).to(device)
    
    checkpoint = torch.load(cfg.MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    embeddings = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(sampled_data), batch_size):
            batch = sampled_data[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).permute(0, 2, 1).to(device)
            z = model.encode(batch_tensor)
            embeddings.append(z.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    
    print("\n[4/5] Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    print(f"t-SNE output: {embeddings_2d.shape}")
    
    print("\n[5/5] Creating Visualization...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = np.array(['lightgrey'] * len(sampled_labels))
    colors[sampled_labels == 0] = 'red'
    colors[sampled_labels == 8] = 'green'
    
    alphas = np.array([0.3] * len(sampled_labels))
    alphas[sampled_labels == 0] = 0.8
    alphas[sampled_labels == 8] = 0.8
    
    for i in range(len(embeddings_2d)):
        ax.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], 
                  c=colors[i], s=20, alpha=alphas[i], edgecolors='none')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.8, label=f'Cluster 0 (Bearish, n={(sampled_labels == 0).sum()})'),
        Patch(facecolor='green', alpha=0.8, label=f'Cluster 8 (Bullish, n={(sampled_labels == 8).sum()})'),
        Patch(facecolor='lightgrey', alpha=0.3, label='Other Clusters')
    ]
    ax.legend(handles=legend_elements, fontsize=12, loc='upper right')
    
    ax.set_title('t-SNE Visualization of Learned Embeddings (32D → 2D)', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=13)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=13)
    ax.grid(True, alpha=0.2)
    
    output_path = Path('plots') / 'tsne_manifold.png'
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nSaved: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()

