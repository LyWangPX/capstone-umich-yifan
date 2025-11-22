# QQQ Pattern Recognition with Deep Learning

Capstone project exploring latent market patterns in QQQ using CNN autoencoders and unsupervised clustering.

## Overview

This project trains a 1D CNN autoencoder on 18 tech stocks (1999-2025) to learn compressed representations of 60-day price windows. The model encodes sequences into 32-dimensional vectors, which are then clustered to identify recurring market patterns.

**Key Finding**: The model discovered 20 distinct patterns with measurable predictive power. Cluster 8 (bullish momentum) showed +0.69% forward returns with 66% win rate, while Cluster 0 (exhaustion pattern) showed -0.32% returns.

## Structure

```
01_run_data.py          # Multi-stock data pipeline
02_run_train.py         # Train CNN autoencoder
03_run_analysis.py      # Generate embeddings and cluster
04_run_viz.py           # Visualize all 20 patterns
05_run_backtest.py      # Basic backtest framework
06_run_qqq_inference.py # QQQ-specific evaluation
07_visualize_qqq.py     # Context plots (2021-2023)
08_visualize_normalized.py  # Latent space comparison
```

## Usage

```bash
conda activate homepage2

# Generate training data
python 01_run_data.py

# Train model (20 epochs, ~5 min on GPU)
python 02_run_train.py

# Run analysis
python 03_run_analysis.py

# Visualizations
python 04_run_viz.py
python 06_run_qqq_inference.py
python 07_visualize_qqq.py
python 08_visualize_normalized.py
```

## Results

**QQQ Backtest (1999-2025)**:
- Cluster Strategy: 608% return (107 trades)
- Buy & Hold: 1268% return
- Validation Loss: 0.25 (explains 75% variance)

The strategy underperformed buy-and-hold but successfully identified patterns with statistical significance. Top 3 clusters averaged +0.58% 5-day returns vs -0.21% for bottom 3.

## Data

Training: 111,012 sequences from 18 NASDAQ stocks  
Validation: 1,285 sequences (QQQ only, last 20%)  
Features: Close price, Volume (Z-score normalized per window)

Large files (`.npy`, `.pth`) excluded from repo. Run pipeline scripts to regenerate.

## Architecture

- Encoder: 3-layer 1D CNN → 32-dim latent space
- Decoder: 3-layer transposed CNN
- Clustering: K-Means (k=20)
- Loss: MSE reconstruction

## Plots

Generated visualizations in `plots/`:
- `all_clusters.png` - 4x5 grid of all patterns
- `normalized_comparison.png` - Bullish vs bearish latent shapes
- `qqq_analysis/` - Context plots showing when patterns occurred

---

UMich Capstone 2025

