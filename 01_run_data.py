import numpy as np
from pathlib import Path
from config import Config
from data_loader import YahooDownloader
from processor import DataProcessor


def main():
    cfg = Config()
    output_dir = Path(cfg.DATA_DIR)
    output_dir.mkdir(exist_ok=True)
    
    downloader = YahooDownloader()
    processor = DataProcessor(seq_len=cfg.SEQ_LEN)
    
    print(f"Downloading data for {len(cfg.TRAIN_SYMBOLS)} symbols...")
    data_dict = downloader.fetch_data(cfg.TRAIN_SYMBOLS, cfg.START_DATE, cfg.END_DATE)
    
    print(f"\nSuccessfully downloaded {len(data_dict)} symbols")
    
    qqq_data = data_dict.pop(cfg.TARGET_SYMBOL)
    val_split_idx = int(len(qqq_data) * 0.8)
    qqq_train = qqq_data.iloc[:val_split_idx]
    qqq_val = qqq_data.iloc[val_split_idx:]
    
    train_dict = {**data_dict, cfg.TARGET_SYMBOL: qqq_train}
    
    print("\nCreating training sequences from all symbols...")
    train_sequences, train_raw_prices = processor.make_sequences(train_dict)
    
    print("\nCreating validation sequences from QQQ (last 20%)...")
    val_sequences, val_raw_prices = processor.make_sequences({cfg.TARGET_SYMBOL: qqq_val})
    
    print("\nNormalizing sequences...")
    train_normalized = processor.normalize_window(train_sequences)
    val_normalized = processor.normalize_window(val_sequences)
    
    train_path = output_dir / 'train_data.npy'
    val_path = output_dir / 'val_data.npy'
    train_prices_path = output_dir / 'train_prices.npy'
    val_prices_path = output_dir / 'val_prices.npy'
    
    np.save(train_path, train_normalized)
    np.save(val_path, val_normalized)
    np.save(train_prices_path, train_raw_prices)
    np.save(val_prices_path, val_raw_prices)
    
    print(f"\nSaved training data to {train_path}")
    print(f"Saved validation data to {val_path}")
    print(f"Saved raw prices to {train_prices_path}")
    print(f"\nTraining Shape: {train_normalized.shape}")
    print(f"Validation Shape: {val_normalized.shape}")
    print(f"Saved Raw Prices shape: {train_raw_prices.shape}")


if __name__ == '__main__':
    main()

