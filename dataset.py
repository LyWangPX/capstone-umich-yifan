# Author: Yifan Wang
import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.FloatTensor(sample).permute(1, 0)

