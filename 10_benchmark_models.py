# Author: Yifan Wang
import numpy as np
import torch
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from config import Config
from models import CnnAutoencoder


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def method_a_raw_kmeans(data, n_clusters=20):
    flattened = data.reshape(len(data), -1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(flattened)
    score = silhouette_score(flattened, labels)
    return labels, score


def method_b_pca_kmeans(data, n_clusters=20, n_components=32):
    flattened = data.reshape(len(data), -1)
    pca = PCA(n_components=n_components, random_state=42)
    embeddings = pca.fit_transform(flattened)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    return labels, score, pca.explained_variance_ratio_.sum()


def method_c_cnn_kmeans(data, model, device, n_clusters=20):
    embeddings = []
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).permute(0, 2, 1).to(device)
            z = model.encode(batch_tensor)
            embeddings.append(z.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    return labels, score


def main():
    cfg = Config()
    
    print("=" * 70)
    print("  BENCHMARK: Comparing Embedding Methods")
    print("=" * 70)
    
    print("\n[1/4] Loading Data...")
    train_data = np.load(Path(cfg.DATA_DIR) / 'train_data.npy')
    
    print(f"Total samples: {len(train_data)}")
    
    print("\n[2/4] Sampling 10,000 subset for comparison...")
    np.random.seed(42)
    sample_indices = np.random.choice(len(train_data), size=10000, replace=False)
    sampled_data = train_data[sample_indices]
    print(f"Subset: {sampled_data.shape}")
    
    print("\n[3/4] Running Method A: Raw Data + KMeans...")
    labels_a, score_a = method_a_raw_kmeans(sampled_data)
    print(f"  Silhouette Score: {score_a:.4f}")
    
    print("\n[4/4] Running Method B: PCA + KMeans...")
    labels_b, score_b, variance_explained = method_b_pca_kmeans(sampled_data)
    print(f"  PCA Variance Explained: {variance_explained:.2%}")
    print(f"  Silhouette Score: {score_b:.4f}")
    
    print("\n[5/5] Running Method C: CNN Embeddings (Ours) + KMeans...")
    device = get_device()
    model = CnnAutoencoder(
        input_dim=cfg.INPUT_DIM,
        seq_len=cfg.SEQ_LEN,
        latent_dim=cfg.LATENT_DIM
    ).to(device)
    
    checkpoint = torch.load(cfg.MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    labels_c, score_c = method_c_cnn_kmeans(sampled_data, model, device)
    print(f"  Silhouette Score: {score_c:.4f}")
    
    print("\n" + "=" * 70)
    print("  RESULTS COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Method':<30} {'Silhouette Score':<20} {'Notes'}")
    print("-" * 70)
    print(f"{'A: Raw Data + KMeans':<30} {score_a:<20.4f} {'Baseline'}")
    print(f"{'B: PCA (32D) + KMeans':<30} {score_b:<20.4f} {f'Var: {variance_explained:.1%}'}")
    print(f"{'C: CNN Embeddings + KMeans':<30} {score_c:<20.4f} {'OURS (Supervised)'}")
    print("-" * 70)
    
    best_method = max([('A', score_a), ('B', score_b), ('C', score_c)], key=lambda x: x[1])
    improvement_vs_raw = ((score_c - score_a) / abs(score_a)) * 100
    improvement_vs_pca = ((score_c - score_b) / abs(score_b)) * 100
    
    print(f"\nBest Method: {best_method[0]} (Score: {best_method[1]:.4f})")
    print(f"CNN vs Raw: {improvement_vs_raw:+.1f}%")
    print(f"CNN vs PCA: {improvement_vs_pca:+.1f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()

