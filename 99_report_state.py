# Author: Yifan Wang
# 75% of this code is written by AI. Only used to report current project status.
import os
import numpy as np
from pathlib import Path


def print_separator(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_file_content(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    print(content)


def main():
    print("\n" + "#" * 70)
    print("  PROJECT STATE REPORT - QQQ Pattern Recognition")
    print("#" * 70)
    
    print_separator("config.py")
    print_file_content('config.py')
    
    print_separator("models.py")
    print_file_content('models.py')
    
    print_separator("processor.py")
    print_file_content('processor.py')
    
    print_separator("Data Directory Contents")
    data_dir = Path('data')
    if data_dir.exists():
        all_files = sorted(data_dir.glob('*'))
        print("Files in data/:")
        for f in all_files:
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name} ({size_mb:.2f} MB)")
        print()
    else:
        print("data/ directory not found")
    
    print_separator("NumPy Array Shapes")
    npy_files = sorted(data_dir.glob('*.npy'))
    if npy_files:
        for npy_file in npy_files:
            try:
                arr = np.load(npy_file)
                print(f"{npy_file.name}: {arr.shape}")
            except Exception as e:
                print(f"{npy_file.name}: Error loading - {e}")
    else:
        print("No .npy files found in data/")
    
    print_separator("Model Checkpoint Status")
    checkpoint_path = Path('checkpoints/best_model.pth')
    if checkpoint_path.exists():
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"checkpoints/best_model.pth EXISTS")
        print(f"Size: {size_mb:.2f} MB")
    else:
        print("checkpoints/best_model.pth NOT FOUND")
        print("Run 02_run_train.py to generate model checkpoint")
    
    print_separator("Plots Status")
    plots_dir = Path('plots')
    if plots_dir.exists():
        png_files = list(plots_dir.glob('**/*.png'))
        print(f"Total plots generated: {len(png_files)}")
        print(f"  - Main plots/: {len(list(plots_dir.glob('*.png')))}")
        qqq_analysis = plots_dir / 'qqq_analysis'
        if qqq_analysis.exists():
            print(f"  - plots/qqq_analysis/: {len(list(qqq_analysis.glob('*.png')))}")
    else:
        print("plots/ directory not found")
    
    print("\n" + "#" * 70)
    print("  END OF REPORT")
    print("#" * 70 + "\n")


if __name__ == '__main__':
    main()

