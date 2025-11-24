import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def generate_embeddings(model, data, device, batch_size=64):
    embeddings = []
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).permute(0, 2, 1).to(device)
            _, z = model(batch_tensor)
            embeddings.append(z.cpu().numpy())
    
    return np.vstack(embeddings)


def calculate_l2_norms(embeddings):
    return np.linalg.norm(embeddings, axis=1)


def calculate_forward_returns(prices, lookahead=5):
    returns = []
    for i in range(len(prices)):
        if i + lookahead < len(prices):
            ret = (prices[i + lookahead] - prices[i]) / prices[i]
        else:
            ret = np.nan
        returns.append(ret)
    return np.array(returns)


def analyze_magnitude_buckets(magnitudes, returns, quantiles=[25, 50, 75, 100]):
    valid_mask = ~np.isnan(returns)
    magnitudes_valid = magnitudes[valid_mask]
    returns_valid = returns[valid_mask]
    
    results = []
    
    for i, percentile in enumerate(quantiles):
        if i == 0:
            threshold = np.percentile(magnitudes_valid, percentile)
            bucket_mask = magnitudes_valid <= threshold
            label = f"Bottom {percentile}% (Weakest)"
        elif i == len(quantiles) - 1:
            prev_percentile = quantiles[i-1]
            lower_threshold = np.percentile(magnitudes_valid, prev_percentile)
            bucket_mask = magnitudes_valid > lower_threshold
            label = f"Top {100-prev_percentile}% (Strongest)"
        else:
            prev_percentile = quantiles[i-1]
            lower_threshold = np.percentile(magnitudes_valid, prev_percentile)
            upper_threshold = np.percentile(magnitudes_valid, percentile)
            bucket_mask = (magnitudes_valid > lower_threshold) & (magnitudes_valid <= upper_threshold)
            label = f"{prev_percentile}-{percentile}%"
        
        if bucket_mask.sum() > 0:
            bucket_returns = returns_valid[bucket_mask]
            win_rate = (bucket_returns > 0).sum() / len(bucket_returns)
            avg_return = bucket_returns.mean()
            count = len(bucket_returns)
            median_magnitude = np.median(magnitudes_valid[bucket_mask])
            
            results.append({
                'Bucket': label,
                'Count': count,
                'Median_Magnitude': median_magnitude,
                'Win_Rate': win_rate,
                'Avg_Return': avg_return
            })
    
    return pd.DataFrame(results)


def main():
    cfg = Config()
    device = get_device()
    
    print("=" * 80)
    print("  Step 15: Signal Magnitude Analysis (Loud vs Quiet Activations)")
    print("=" * 80)
    
    print("\n[1/7] Loading QQQ Data...")
    downloader = YahooDownloader()
    data_dict = downloader.fetch_data([cfg.TARGET_SYMBOL], cfg.START_DATE, cfg.END_DATE)
    qqq_df = data_dict[cfg.TARGET_SYMBOL]
    
    processor = DataProcessor(seq_len=cfg.SEQ_LEN)
    sequences, prices = processor.make_sequences({cfg.TARGET_SYMBOL: qqq_df})
    prices = prices.flatten()
    normalized_sequences = processor.normalize_window(sequences)
    
    print(f"QQQ sequences: {sequences.shape}")
    
    print("\n[2/7] Splitting Train (80%) / Test (20%)...")
    split_idx = int(len(normalized_sequences) * 0.8)
    
    data_train = normalized_sequences[:split_idx]
    data_test = normalized_sequences[split_idx:]
    prices_train = prices[:split_idx]
    prices_test = prices[split_idx:]
    
    print(f"Train: {data_train.shape}")
    print(f"Test: {data_test.shape}")
    
    print("\n[3/7] Loading Model and Generating Embeddings...")
    model = CnnAutoencoder(
        input_dim=cfg.INPUT_DIM,
        seq_len=cfg.SEQ_LEN,
        latent_dim=cfg.LATENT_DIM
    ).to(device)
    
    checkpoint = torch.load(cfg.MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    embeddings_train = generate_embeddings(model, data_train, device)
    embeddings_test = generate_embeddings(model, data_test, device)
    
    print(f"Train embeddings: {embeddings_train.shape}")
    print(f"Test embeddings: {embeddings_test.shape}")
    
    print("\n[4/7] Calculating L2 Norms (Signal Magnitude)...")
    magnitudes_test = calculate_l2_norms(embeddings_test)
    
    print(f"\nMagnitude Statistics (Test Set):")
    print(f"  Min: {magnitudes_test.min():.4f}")
    print(f"  Median: {np.median(magnitudes_test):.4f}")
    print(f"  Mean: {magnitudes_test.mean():.4f}")
    print(f"  Max: {magnitudes_test.max():.4f}")
    print(f"  Std: {magnitudes_test.std():.4f}")
    
    print("\n[5/7] Clustering and Identifying Top 3 Bullish Clusters...")
    clusterer = PatternClusterer(n_clusters=20)
    labels_train, kmeans_model = clusterer.fit_predict(embeddings_train)
    
    from evaluator import ClusterEvaluator
    evaluator = ClusterEvaluator()
    
    stats_df = evaluator.calculate_forward_returns(prices_train, labels_train, lookahead=5)
    stats_df_sorted = stats_df.sort_values('Avg_Return', ascending=False)
    
    top_3_clusters = stats_df_sorted.head(3)['Cluster_ID'].values
    print(f"\nTop 3 Bullish Clusters: {top_3_clusters}")
    
    labels_test = kmeans_model.predict(embeddings_test)
    
    print("\n[6/7] Calculating Forward Returns...")
    forward_returns = calculate_forward_returns(prices_test, lookahead=5)
    
    print("\n[7/7] Analyzing Magnitude Buckets...")
    
    buckets_results = analyze_magnitude_buckets(magnitudes_test, forward_returns, 
                                               quantiles=[25, 50, 75, 100])
    
    print("\n" + "=" * 80)
    print("  RESULTS: Signal Magnitude vs Performance")
    print("=" * 80)
    print()
    print(buckets_results.to_string(index=False))
    print("=" * 80)
    
    valid_mask = ~np.isnan(forward_returns)
    baseline_returns = forward_returns[valid_mask]
    baseline_win_rate = (baseline_returns > 0).sum() / len(baseline_returns)
    baseline_avg_return = baseline_returns.mean()
    
    print(f"\nBaseline (All Test Samples):")
    print(f"  Win Rate: {baseline_win_rate:.2%}")
    print(f"  Avg Return: {baseline_avg_return*100:.2f}%")
    print(f"  Count: {len(baseline_returns)}")
    
    print("\n" + "=" * 80)
    print("  INTERSECTION ANALYSIS: Top Clusters + High Magnitude")
    print("=" * 80)
    
    test_in_top_clusters = np.isin(labels_test, top_3_clusters)
    
    high_magnitude_threshold = np.percentile(magnitudes_test, 75)
    high_magnitude_mask = magnitudes_test > high_magnitude_threshold
    
    scenarios = [
        ("All Test Samples", valid_mask),
        ("Top 3 Clusters Only", test_in_top_clusters & valid_mask),
        ("High Magnitude Only (Top 25%)", high_magnitude_mask & valid_mask),
        ("SUPER SIGNAL (Both)", test_in_top_clusters & high_magnitude_mask & valid_mask)
    ]
    
    print()
    print(f"{'Strategy':<35} {'Count':<10} {'Win Rate':<15} {'Avg Return'}")
    print("-" * 80)
    
    super_signal_result = None
    
    for name, mask in scenarios:
        if mask.sum() > 0:
            scenario_returns = forward_returns[mask]
            win_rate = (scenario_returns > 0).sum() / len(scenario_returns)
            avg_return = scenario_returns.mean()
            count = len(scenario_returns)
            
            print(f"{name:<35} {count:<10} {win_rate:<15.2%} {avg_return*100:+.2f}%")
            
            if name == "SUPER SIGNAL (Both)":
                super_signal_result = (win_rate, avg_return, count)
        else:
            print(f"{name:<35} {'0':<10} {'N/A':<15} {'N/A'}")
    
    print("-" * 80)
    
    print("\n[8/8] Creating Visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax1 = axes[0, 0]
    valid_mask_plot = ~np.isnan(forward_returns)
    magnitudes_plot = magnitudes_test[valid_mask_plot]
    returns_plot = forward_returns[valid_mask_plot]
    labels_plot = labels_test[valid_mask_plot]
    
    in_top_3 = np.isin(labels_plot, top_3_clusters)
    
    ax1.scatter(magnitudes_plot[~in_top_3], returns_plot[~in_top_3] * 100,
               alpha=0.3, s=20, color='gray', label='Other Clusters')
    ax1.scatter(magnitudes_plot[in_top_3], returns_plot[in_top_3] * 100,
               alpha=0.6, s=30, color='green', label='Top 3 Bullish Clusters')
    
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(x=high_magnitude_threshold, color='blue', linestyle='--', 
               linewidth=2, alpha=0.5, label=f'High Magnitude (P75={high_magnitude_threshold:.2f})')
    
    ax1.set_xlabel('L2 Norm (Signal Magnitude)', fontsize=12)
    ax1.set_ylabel('5-Day Forward Return (%)', fontsize=12)
    ax1.set_title('Signal Magnitude vs Forward Returns', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.hist(magnitudes_test, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.set_xlabel('L2 Norm (Signal Magnitude)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Signal Magnitudes', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    percentiles = [25, 50, 75]
    for p in percentiles:
        val = np.percentile(magnitudes_test, p)
        ax2.axvline(x=val, color='red', linestyle='--', alpha=0.5)
        ax2.text(val, ax2.get_ylim()[1] * 0.95, f'P{p}',
                fontsize=9, rotation=90, va='top')
    
    ax3 = axes[1, 0]
    bucket_labels = buckets_results['Bucket'].values
    avg_returns = buckets_results['Avg_Return'].values * 100
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(bucket_labels)))
    
    bars = ax3.bar(range(len(bucket_labels)), avg_returns, color=colors,
                   alpha=0.8, edgecolor='black')
    ax3.axhline(y=baseline_avg_return * 100, color='blue', linestyle='--',
               linewidth=2, label=f'Baseline: {baseline_avg_return*100:.2f}%')
    ax3.set_xticks(range(len(bucket_labels)))
    ax3.set_xticklabels(bucket_labels, rotation=45, ha='right')
    ax3.set_ylabel('Average Return (%)', fontsize=12)
    ax3.set_title('Performance by Signal Magnitude', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    for i, ret in enumerate(avg_returns):
        ax3.text(i, ret + 0.05 if ret > 0 else ret - 0.1, f'{ret:.2f}%',
                ha='center', fontweight='bold', fontsize=10)
    
    ax4 = axes[1, 1]
    strategy_names = [s[0] for s in scenarios]
    strategy_returns = []
    strategy_counts = []
    
    for name, mask in scenarios:
        if mask.sum() > 0:
            scenario_returns = forward_returns[mask]
            avg_ret = scenario_returns.mean()
            strategy_returns.append(avg_ret * 100)
            strategy_counts.append(len(scenario_returns))
        else:
            strategy_returns.append(0)
            strategy_counts.append(0)
    
    colors_strategy = ['gray', 'orange', 'lightblue', 'darkgreen']
    bars = ax4.bar(range(len(strategy_names)), strategy_returns, 
                   color=colors_strategy, alpha=0.8, edgecolor='black')
    ax4.set_xticks(range(len(strategy_names)))
    ax4.set_xticklabels(strategy_names, rotation=45, ha='right', fontsize=10)
    ax4.set_ylabel('Average Return (%)', fontsize=12)
    ax4.set_title('Strategy Comparison: Intersection Analysis', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for i, (ret, count) in enumerate(zip(strategy_returns, strategy_counts)):
        ax4.text(i, ret + 0.1 if ret > 0 else ret - 0.15,
                f'{ret:.2f}%\n(n={count})', ha='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    output_path = Path('plots') / 'magnitude_analysis.png'
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nVisualization saved: {output_path}")
    
    buckets_results.to_csv(Path(cfg.DATA_DIR) / 'magnitude_filter_results.csv', index=False)
    print(f"Results saved: {cfg.DATA_DIR}/magnitude_filter_results.csv")
    
    print("\n" + "=" * 80)
    print("  CONCLUSION")
    print("=" * 80)
    
    strongest_bucket = buckets_results.iloc[-1]
    weakest_bucket = buckets_results.iloc[0]
    
    delta = (strongest_bucket['Avg_Return'] - weakest_bucket['Avg_Return']) * 100
    
    print(f"\nStrongest Signals (Top 25%): {strongest_bucket['Avg_Return']*100:+.2f}% avg return")
    print(f"Weakest Signals (Bottom 25%): {weakest_bucket['Avg_Return']*100:+.2f}% avg return")
    print(f"Delta: {delta:+.2f} percentage points")
    
    if super_signal_result:
        super_wr, super_ret, super_count = super_signal_result
        improvement = (super_ret - baseline_avg_return) * 100
        
        if improvement > 0.5:
            print(f"\n✓ SUPER SIGNAL works! {improvement:+.2f}% improvement over baseline")
            print(f"  Strategy: Trade only Top 3 Clusters + High Magnitude (>P75)")
            print(f"  Result: {super_ret*100:.2f}% avg return, {super_wr:.1%} win rate, {super_count} trades")
        else:
            print(f"\n→ SUPER SIGNAL neutral: {improvement:+.2f}% vs baseline")
    
    if delta > 1.0:
        print(f"\n✓ Signal Magnitude is PREDICTIVE!")
        print(f"  'Loud' activations ({strongest_bucket['Avg_Return']*100:+.2f}%) >> 'Quiet' ({weakest_bucket['Avg_Return']*100:+.2f}%)")
        print(f"  This validates Step 14: Far-from-centroid = Strong signal = Higher returns")
    else:
        print(f"\n→ Signal Magnitude has limited predictive power")
    
    print("=" * 80)


if __name__ == '__main__':
    main()

