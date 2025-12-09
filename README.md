# QQQ Pattern Recognition with Deep Learning

Discovering latent market regimes in equity time series using convolutional autoencoders.

## Project Summary

This project applies unsupervised deep learning to identify recurring patterns in financial markets. A 1D CNN autoencoder compresses 60-day price-volume windows into 32-dimensional embeddings, which are then clustered to reveal 20 distinct market regimes. The learned representations demonstrate predictive power when combined with signal magnitude filtering.

**Key Results (Test Period 2020-2025)**:
- Leverage-augmented strategy: 176% return vs 125% QQQ benchmark (+51pp alpha)
- Risk reduction: -15% max drawdown vs -35% for QQQ
- Statistical significance: 62.1% win rate (p=0.099, marginally significant)

## Architecture

**Model**: 1D CNN Autoencoder  
**Latent Space**: 32 dimensions  
**Clustering**: K-Means (k=20)  
**Training**: 111,012 sequences from 18 NASDAQ stocks (1999-2025)  
**Validation**: 0.25 MSE loss (75% variance explained)  

**Embedding Quality**:
- CNN embeddings: Silhouette 0.0719
- PCA baseline: Silhouette 0.0509 (+41% improvement)
- Raw data: Silhouette 0.0301 (+139% improvement)

## Data Access & Pipeline

Since large data files (.npy, .pth) are excluded from the repository, you must generate the data locally before training.

### 1. Data Generation
The project uses `yfinance` to download historical stock data. Run the data generation script to:
- Download daily Close and Volume data for 18 NASDAQ stocks (1999-2025).
- Preprocess the data into 60-day rolling windows.
- Apply Z-score normalization per window.
- Save the processed sequences to `data/`.

```bash
python 01_run_data.py
```

### 2. Training & Analysis Pipeline

Once the data is generated, you can proceed with the core pipeline:

```bash
conda activate homepage2

# Core Pipeline
python 02_run_train.py         # Train autoencoder
python 03_run_analysis.py      # Generate embeddings + cluster

# Analysis & Visualization
python 06_run_qqq_inference.py # QQQ-specific evaluation
python 07_visualize_qqq.py     # Context plots
python 08_visualize_normalized.py  # Latent space analysis
python 09_visualize_tsne.py    # 2D manifold projection
python 10_benchmark_models.py  # Compare embedding methods

# Advanced Experiments
python 11_train_vae.py         # Variational autoencoder
python 12_sensitivity_analysis.py  # Risk shield strategy
python 13_run_uncertainty_filter.py  # Reconstruction error analysis
python 14_run_centroid_filter.py    # Distance-based filtering
python 15_run_magnitude_filter.py   # Signal intensity analysis
python 16_run_final_optimization.py # Grid search (P30-P75 optimal)
python 17_run_final_backtest.py     # Final metrics
python 18_run_leverage_switch.py    # TQQQ leverage strategy
python 19_run_bootstrap.py          # Statistical significance test
```

## Key Findings

**Pattern Discovery**:
- 20 distinct market regimes identified without supervision
- Cluster 8 (bullish): +0.69% avg return, 66% win rate
- Cluster 0 (bearish): -0.32% avg return, 51% win rate

**Signal Calibration**:
- Moderate magnitude signals (P30-P75) outperform extremes
- Inverted-U relationship: weak signals underperform, extreme signals mean-revert
- Optimal band: 62.1% win rate vs 55.4% baseline (+6.7pp)

**Strategy Performance**:
- Conservative (cash-heavy): 5.4% return, -15% max DD
- Leverage-augmented (hybrid): 176% return, beats QQQ by 51pp
- Statistical validation: p=0.099 (marginally significant)

## Data

**Training**: 111,012 sequences (18 stocks, 1999-2025)  
**Test**: 1,333 days (QQQ only, 2020-2025)  
**Features**: Close price, Volume (Z-score normalized per window)  
**Train/Test Split**: 80/20 temporal (no look-ahead bias)

Large files (.npy, .pth) excluded. Run pipeline to regenerate.

## Methodology

**Preprocessing**: 60-day rolling windows with per-window Z-score normalization

**Encoder**: Conv1D(2→32→64→128) + MaxPool → 32-dim latent  
**Decoder**: Linear + ConvTranspose1D (128→64→32→2)  
**Loss**: MSE reconstruction + KL divergence (VAE variant)

**Clustering**: K-Means on frozen embeddings  
**Filtering**: Magnitude band-pass (P30-P75) for signal quality

## Visualizations

- `tsne_manifold.png` - 2D projection showing cluster separation
- `normalized_comparison.png` - Bearish vs bullish pattern centroids
- `magnitude_analysis.png` - Inverted-U calibration curve
- `final_optimization.png` - Grid search heatmap
- `final_backtest.png` - Equity curves and drawdown
- `leverage_switch_backtest.png` - Hybrid strategy performance
- `bootstrap_significance.png` - Statistical validation

## Citation

UMich Capstone 2025  
Repository: https://github.com/LyWangPX/capstone-umich-yifan

This readme is written by Yifan Wang, about 15% grammar fixed by LLM.