# Author: Yifan Wang
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import config
from models import TimeSeriesEmbeddingModel
from utils import set_seed, get_device


def load_data():
    data_path = Path(config.OUTPUT_DIR) / f"{config.SYMBOLS[0]}_sequences.npy"
    sequences = np.load(data_path)
    sequences = torch.FloatTensor(sequences).permute(0, 2, 1)
    return sequences


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        x = batch[0].to(device)
        optimizer.zero_grad()
        x_recon, z = model(x)
        loss = criterion(x_recon, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    set_seed(42)
    device = get_device()
    
    print(f"Loading data from {config.OUTPUT_DIR}")
    sequences = load_data()
    n_samples, n_features, window_size = sequences.shape
    print(f"Data shape: {sequences.shape}")
    
    dataset = TensorDataset(sequences)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = TimeSeriesEmbeddingModel(
        input_dim=n_features,
        window_size=window_size,
        latent_dim=64,
        hidden_dim=128
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    best_loss = float('inf')
    epochs = 100
    
    for epoch in range(epochs):
        loss = train_epoch(model, loader, criterion, optimizer, device)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
        
        if loss < best_loss:
            best_loss = loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_dir / 'best_model.pth')
    
    print(f"Training complete. Best loss: {best_loss:.6f}")


if __name__ == '__main__':
    main()

