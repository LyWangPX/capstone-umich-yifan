# Author: Yifan Wang
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from config import Config
from models_vae import CnnVAE
from dataset import TimeSeriesDataset


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in loader:
        x = batch.to(device)
        
        optimizer.zero_grad()
        reconstructed, mu, logvar = model(x)
        loss = loss_function(reconstructed, x, mu, logvar)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            x = batch.to(device)
            reconstructed, mu, logvar = model(x)
            loss = loss_function(reconstructed, x, mu, logvar)
            total_loss += loss.item()
    
    return total_loss / len(loader.dataset)


def main():
    cfg = Config()
    device = get_device()
    print(f"Using device: {device}")
    
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    print(f"\nLoading datasets...")
    train_dataset = TimeSeriesDataset(Path(cfg.DATA_DIR) / 'train_data.npy')
    val_dataset = TimeSeriesDataset(Path(cfg.DATA_DIR) / 'val_data.npy')
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    model = CnnVAE(
        input_dim=cfg.INPUT_DIM,
        seq_len=cfg.SEQ_LEN,
        latent_dim=cfg.LATENT_DIM
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    
    best_val_loss = float('inf')
    
    print(f"\nStarting VAE training for {cfg.EPOCHS} epochs...")
    
    for epoch in range(cfg.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{cfg.EPOCHS} - Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_dir / 'best_vae.pth')
            print(f"  → Model saved (best val loss: {best_val_loss:.2f})")
    
    print(f"\nVAE Training Complete. Best Validation Loss: {best_val_loss:.2f}")


if __name__ == '__main__':
    main()

