# Author: Yifan Wang
import torch
import torch.nn as nn


class CnnAutoencoder(nn.Module):
    def __init__(self, input_dim, seq_len, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * (seq_len // 4)),
            nn.ReLU(),
            nn.Unflatten(1, (128, seq_len // 4)),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, input_dim, kernel_size=3, stride=1, padding=1)
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        latent_vector = self.encode(x)
        reconstructed = self.decode(latent_vector)
        return reconstructed, latent_vector
