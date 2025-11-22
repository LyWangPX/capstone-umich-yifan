import torch
import numpy as np
from models import CnnAutoencoder


class InferenceEngine:
    def __init__(self, model_path, input_dim, seq_len, latent_dim, device):
        self.device = device
        self.model = CnnAutoencoder(
            input_dim=input_dim,
            seq_len=seq_len,
            latent_dim=latent_dim
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def get_embeddings(self, loader):
        embeddings = []
        original_inputs = []
        
        with torch.no_grad():
            for batch in loader:
                x = batch.to(self.device)
                latent = self.model.encode(x)
                
                embeddings.append(latent.cpu().numpy())
                original_inputs.append(x.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        original_inputs = np.vstack(original_inputs)
        
        return embeddings, original_inputs
