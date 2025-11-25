# Author: Yifan Wang
import numpy as np


class DataProcessor:
    def __init__(self, seq_len):
        self.seq_len = seq_len
    
    def make_sequences(self, df_dict, features=['Close', 'Volume']):
        all_sequences = []
        all_raw_prices = []
        
        for symbol, df in df_dict.items():
            if len(df) < self.seq_len:
                continue
            
            data = df[features].values
            close_prices = df['Close'].values
            sequences = []
            raw_prices = []
            
            for i in range(len(data) - self.seq_len + 1):
                window = data[i:i + self.seq_len]
                sequences.append(window)
                raw_prices.append(close_prices[i + self.seq_len - 1])
            
            if sequences:
                sequences = np.array(sequences)
                raw_prices = np.array(raw_prices)
                all_sequences.append(sequences)
                all_raw_prices.append(raw_prices)
                print(f"{symbol}: {sequences.shape[0]} sequences")
        
        if all_sequences:
            return np.vstack(all_sequences), np.concatenate(all_raw_prices)
        return np.array([]), np.array([])
    
    def normalize_window(self, sequences):
        normalized = []
        
        for seq in sequences:
            mean = seq.mean(axis=0, keepdims=True)
            std = seq.std(axis=0, keepdims=True)
            std = np.where(std == 0, 1e-6, std)
            normalized_seq = (seq - mean) / std
            normalized.append(normalized_seq)
        
        return np.array(normalized)
