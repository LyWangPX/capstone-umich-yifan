import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config import Config
from models import CnnAutoencoder
from data_loader import YahooDownloader
from processor import DataProcessor
from clustering import PatternClusterer


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def calculate_reconstruction_errors(model, data, device, batch_size=64):
    model.eval()
    reconstruction_errors = []
    criterion = nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).permute(0, 2, 1).to(device)
            
            reconstructed, _ = model(batch_tensor)
            
            mse_per_sample = criterion(reconstructed, batch_tensor)
            mse_per_sample = mse_per_sample.mean(dim=[1, 2])
            
            reconstruction_errors.append(mse_per_sample.cpu().numpy())
    
    return np.concatenate(reconstruction_errors)


def calculate_forward_returns(prices, lookahead=5):
    returns = []
    for i in range(len(prices)):
        if i + lookahead < len(prices):
            ret = (prices[i + lookahead] - prices[i]) / prices[i]
        else:
            ret = np.nan
        returns.append(ret)
    return np.array(returns)


def filter_trades_by_threshold(errors, returns, clusters, target_clusters, threshold):
    mask = (errors <= threshold) & np.isin(clusters, target_clusters) & ~np.isnan(returns)
    
    if mask.sum() == 0:
        return 0, 0.0, 0
    
    filtered_returns = returns[mask]
    win_rate = (filtered_returns > 0).sum() / len(filtered_returns)
    avg_return = filtered_returns.mean()
    count = len(filtered_returns)
    
    return win_rate, avg_return, count


def main():
    cfg = Config()
    device = get_device()
    
    print("=" * 80)
    print("  Step 13: Reconstruction Error Filtering (Uncertainty-Aware Trading)")
    print("=" * 80)
    
    print("\n[1/6] Loading QQQ Data...")
    downloader = YahooDownloader()
    data_dict = downloader.fetch_data([cfg.TARGET_SYMBOL], cfg.START_DATE, cfg.END_DATE)
    qqq_df = data_dict[cfg.TARGET_SYMBOL]
    
    processor = DataProcessor(seq_len=cfg.SEQ_LEN)
    sequences, prices = processor.make_sequences({cfg.TARGET_SYMBOL: qqq_df})
    prices = prices.flatten()
    normalized_sequences = processor.normalize_window(sequences)
    
    print(f"QQQ sequences: {sequences.shape}")
    print(f"QQQ prices: {prices.shape}")
    
    print("\n[2/6] Loading Trained Model...")
    model = CnnAutoencoder(
        input_dim=cfg.INPUT_DIM,
        seq_len=cfg.SEQ_LEN,
        latent_dim=cfg.LATENT_DIM
    ).to(device)
    
    checkpoint = torch.load(cfg.MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded: {cfg.MODEL_PATH}")
    
    print("\n[3/6] Calculating Reconstruction Errors...")
    reconstruction_errors = calculate_reconstruction_errors(model, normalized_sequences, device)
    print(f"Reconstruction errors shape: {reconstruction_errors.shape}")
    print(f"Error range: [{reconstruction_errors.min():.4f}, {reconstruction_errors.max():.4f}]")
    print(f"Mean error: {reconstruction_errors.mean():.4f}")
    print(f"Median error: {np.median(reconstruction_errors):.4f}")
    
    print("\n[4/6] Generating Embeddings and Clustering...")
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(normalized_sequences), 64):
            batch = normalized_sequences[i:i+64]
            batch_tensor = torch.FloatTensor(batch).permute(0, 2, 1).to(device)
            _, z = model(batch_tensor)
            embeddings.append(z.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    
    clusterer = PatternClusterer(n_clusters=20)
    cluster_labels, _ = clusterer.fit_predict(embeddings)
    
    print(f"Embeddings: {embeddings.shape}")
    print(f"Clusters: {len(np.unique(cluster_labels))}")
    
    print("\n[5/6] Calculating Forward Returns...")
    forward_returns = calculate_forward_returns(prices, lookahead=5)
    
    valid_mask = ~np.isnan(forward_returns)
    errors_valid = reconstruction_errors[valid_mask]
    returns_valid = forward_returns[valid_mask]
    clusters_valid = cluster_labels[valid_mask]
    
    print(f"Valid samples: {valid_mask.sum()}")
    
    print("\n[6/6] Creating Visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    scatter = ax1.scatter(errors_valid, returns_valid * 100, 
                         c=clusters_valid, cmap='tab20', 
                         alpha=0.4, s=20, edgecolors='none')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Reconstruction Error (MSE)', fontsize=12)
    ax1.set_ylabel('5-Day Forward Return (%)', fontsize=12)
    ax1.set_title('Reconstruction Error vs Forward Returns', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Cluster ID', fontsize=11)
    
    percentiles = [50, 60, 70, 80, 90, 95]
    error_percentile_values = np.percentile(errors_valid, percentiles)
    for p, val in zip(percentiles, error_percentile_values):
        ax1.axvline(x=val, color='gray', linestyle=':', alpha=0.5)
        ax1.text(val, ax1.get_ylim()[1] * 0.95, f'P{p}', 
                fontsize=8, rotation=90, va='top')
    
    ax2.hist(errors_valid, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Reconstruction Error (MSE)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Reconstruction Errors', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for p, val in zip(percentiles, error_percentile_values):
        ax2.axvline(x=val, color='red', linestyle='--', alpha=0.5)
        ax2.text(val, ax2.get_ylim()[1] * 0.95, f'P{p}', 
                fontsize=8, rotation=90, va='top')
    
    plt.tight_layout()
    output_path = Path('plots') / 'reconstruction_error_analysis.png'
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nVisualization saved: {output_path}")
    
    print("\n" + "=" * 80)
    print("  OPTIMIZATION: Testing Error Thresholds")
    print("=" * 80)
    
    qqq_stats = pd.read_csv(Path(cfg.DATA_DIR) / 'qqq_cluster_stats.csv')
    top_3_clusters = qqq_stats.nlargest(3, 'Avg_Return')['Cluster_ID'].values
    
    print(f"\nTarget Clusters (Top 3 Bullish): {top_3_clusters}")
    
    baseline_mask = np.isin(clusters_valid, top_3_clusters)
    baseline_returns = returns_valid[baseline_mask]
    baseline_win_rate = (baseline_returns > 0).sum() / len(baseline_returns)
    baseline_avg_return = baseline_returns.mean()
    baseline_count = len(baseline_returns)
    
    print(f"\nBaseline (No Filtering):")
    print(f"  Trades: {baseline_count}")
    print(f"  Win Rate: {baseline_win_rate:.2%}")
    print(f"  Avg Return: {baseline_avg_return:.4f} ({baseline_avg_return*100:.2f}%)")
    
    thresholds_percentiles = [50, 60, 70, 80, 90, 95, 100]
    results = []
    
    print("\n" + "-" * 80)
    print(f"{'Threshold':<20} {'Trades':<10} {'Win Rate':<15} {'Avg Return':<15} {'Delta'}")
    print("-" * 80)
    
    for p in thresholds_percentiles:
        if p == 100:
            threshold = errors_valid.max()
            label = "No Filter"
        else:
            threshold = np.percentile(errors_valid, p)
            label = f"Keep P{p}% (MSE≤{threshold:.3f})"
        
        win_rate, avg_return, count = filter_trades_by_threshold(
            errors_valid, returns_valid, clusters_valid, top_3_clusters, threshold
        )
        
        if count > 0:
            win_rate_delta = win_rate - baseline_win_rate
            avg_return_delta = avg_return - baseline_avg_return
            
            results.append({
                'Threshold': label,
                'MSE_Threshold': threshold,
                'Trades': count,
                'Win_Rate': win_rate,
                'Avg_Return': avg_return,
                'Win_Rate_Delta': win_rate_delta,
                'Avg_Return_Delta': avg_return_delta
            })
            
            print(f"{label:<20} {count:<10} {win_rate:<15.2%} {avg_return*100:<15.2f}% "
                  f"{avg_return_delta*100:+.2f}%")
    
    print("-" * 80)
    
    results_df = pd.DataFrame(results)
    best_return = results_df.loc[results_df['Avg_Return'].idxmax()]
    best_winrate = results_df.loc[results_df['Win_Rate'].idxmax()]
    
    print("\n" + "=" * 80)
    print("  KEY FINDINGS")
    print("=" * 80)
    
    print(f"\nBest Average Return:")
    print(f"  Threshold: {best_return['Threshold']}")
    print(f"  Avg Return: {best_return['Avg_Return']*100:.2f}% "
          f"({best_return['Avg_Return_Delta']*100:+.2f}% vs baseline)")
    print(f"  Win Rate: {best_return['Win_Rate']:.2%}")
    print(f"  Trades: {best_return['Trades']}")
    
    print(f"\nBest Win Rate:")
    print(f"  Threshold: {best_winrate['Threshold']}")
    print(f"  Win Rate: {best_winrate['Win_Rate']:.2%} "
          f"({best_winrate['Win_Rate_Delta']*100:+.2f}% vs baseline)")
    print(f"  Avg Return: {best_winrate['Avg_Return']*100:.2f}%")
    print(f"  Trades: {best_winrate['Trades']}")
    
    results_df.to_csv(Path(cfg.DATA_DIR) / 'uncertainty_filter_results.csv', index=False)
    print(f"\nResults saved to {cfg.DATA_DIR}/uncertainty_filter_results.csv")
    
    print("\n" + "=" * 80)
    print("  CONCLUSION")
    print("=" * 80)
    
    improvement = best_return['Avg_Return_Delta'] * 100
    if improvement > 0:
        print(f"\n✓ Reconstruction error filtering IMPROVES performance by {improvement:.2f}%")
        print(f"  Recommendation: Filter out samples with MSE > {best_return['MSE_Threshold']:.3f}")
    else:
        print(f"\n✗ Reconstruction error filtering does NOT improve performance")
        print(f"  The model's reconstruction quality may not correlate with prediction accuracy")
    
    print("=" * 80)


if __name__ == '__main__':
    main()

