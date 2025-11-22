import numpy as np
from pathlib import Path
import config
from data_loader import DataLoader
from processor import Processor


def main():
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loader = DataLoader(
        symbols=config.SYMBOLS,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        cache_dir=config.CACHE_DIR
    )
    
    print(f"Loading data for symbols: {config.SYMBOLS}")
    data_dict = loader.load_all()
    
    processor = Processor(
        window_size=config.WINDOW_SIZE,
        normalization=config.NORMALIZATION
    )
    
    for symbol, df in data_dict.items():
        print(f"Processing {symbol}: {len(df)} rows")
        
        sequences = processor.process(df)
        
        output_path = output_dir / f"{symbol}_sequences.npy"
        np.save(output_path, sequences)
        
        print(f"Saved {sequences.shape} to {output_path}")
        print(f"Shape: (N_samples={sequences.shape[0]}, Window={sequences.shape[1]}, Features={sequences.shape[2]})")


if __name__ == '__main__':
    main()

