import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from config import Config
from dataset import TimeSeriesDataset
from inference import InferenceEngine
from clustering import PatternClusterer
from data_loader import YahooDownloader


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def extract_dates_from_data():
    cfg = Config()
    downloader = YahooDownloader()
    
    print("Reconstructing dates from original data...")
    data_dict = downloader.fetch_data(cfg.TRAIN_SYMBOLS, cfg.START_DATE, cfg.END_DATE)
    
    qqq_data = data_dict[cfg.TARGET_SYMBOL]
    val_split_idx = int(len(qqq_data) * 0.8)
    qqq_train = qqq_data.iloc[:val_split_idx]
    
    all_dates = []
    for symbol in cfg.TRAIN_SYMBOLS:
        if symbol not in data_dict:
            continue
        
        if symbol == cfg.TARGET_SYMBOL:
            df = qqq_train
        else:
            df = data_dict[symbol]
        
        if len(df) >= cfg.SEQ_LEN:
            dates = df.index[cfg.SEQ_LEN - 1:]
            all_dates.extend(dates)
    
    return all_dates


def main():
    cfg = Config()
    device = get_device()
    print(f"Using device: {device}")
    
    train_path = Path(cfg.DATA_DIR) / 'train_data.npy'
    train_dataset = TimeSeriesDataset(train_path)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    print(f"\nLoaded {len(train_dataset)} training samples")
    
    print("\nInitializing InferenceEngine...")
    engine = InferenceEngine(
        model_path=cfg.MODEL_PATH,
        input_dim=cfg.INPUT_DIM,
        seq_len=cfg.SEQ_LEN,
        latent_dim=cfg.LATENT_DIM,
        device=device
    )
    
    print("Generating embeddings...")
    embeddings, original_inputs = engine.get_embeddings(train_loader)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Original inputs shape: {original_inputs.shape}")
    
    print("\nInitializing PatternClusterer...")
    clusterer = PatternClusterer(n_clusters=20)
    
    print("Fitting clusters...")
    cluster_labels, kmeans_model = clusterer.fit_predict(embeddings)
    
    print("\n=== Cluster Distribution ===")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        percentage = (count / len(cluster_labels)) * 100
        print(f"Cluster {cluster_id:2d}: {count:6d} samples ({percentage:5.2f}%)")
    
    dates = extract_dates_from_data()
    
    if len(dates) != len(cluster_labels):
        print(f"\nWarning: Date mismatch ({len(dates)} dates vs {len(cluster_labels)} labels)")
        dates = pd.date_range(start=cfg.START_DATE, periods=len(cluster_labels))
        print(f"Using synthetic dates instead")
    
    np.save(Path(cfg.DATA_DIR) / 'embeddings.npy', embeddings)
    np.save(Path(cfg.DATA_DIR) / 'cluster_labels.npy', cluster_labels)
    
    clusterer.save_clusters(dates, cluster_labels, Path(cfg.DATA_DIR) / 'clusters.csv')
    
    print(f"\nSaved:")
    print(f"  - {cfg.DATA_DIR}/embeddings.npy")
    print(f"  - {cfg.DATA_DIR}/cluster_labels.npy")
    print(f"  - {cfg.DATA_DIR}/clusters.csv")


if __name__ == '__main__':
    main()

