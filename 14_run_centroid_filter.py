# Author: Yifan Wang
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics.pairwise import euclidean_distances
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


def calculate_forward_returns(prices, lookahead=5):
    returns = []
    for i in range(len(prices)):
        if i + lookahead < len(prices):
            ret = (prices[i + lookahead] - prices[i]) / prices[i]
        else:
            ret = np.nan
        returns.append(ret)
    return np.array(returns)


def calculate_centroid_distances(embeddings, labels, centroids):
    distances = np.zeros(len(embeddings))
    
    for i, (emb, label) in enumerate(zip(embeddings, labels)):
        if label in centroids:
            centroid = centroids[label]
            distances[i] = np.linalg.norm(emb - centroid)
        else:
            distances[i] = np.nan
    
    return distances

# This function has over 50% code that co-worked with AI
def analyze_distance_buckets(distances, returns, buckets=[25, 50, 75, 100]):
    valid_mask = ~np.isnan(distances) & ~np.isnan(returns)
    distances_valid = distances[valid_mask]
    returns_valid = returns[valid_mask]
    
    results = []
    
    for i, percentile in enumerate(buckets):
        if i == 0:
            threshold = np.percentile(distances_valid, percentile)
            bucket_mask = distances_valid <= threshold
            label = f"Top {percentile}% Closest"
        else:
            prev_percentile = buckets[i-1]
            lower_threshold = np.percentile(distances_valid, prev_percentile)
            upper_threshold = np.percentile(distances_valid, percentile)
            bucket_mask = (distances_valid > lower_threshold) & (distances_valid <= upper_threshold)
            label = f"{prev_percentile}-{percentile}% Closest"
        
        if bucket_mask.sum() > 0:
            bucket_returns = returns_valid[bucket_mask]
            win_rate = (bucket_returns > 0).sum() / len(bucket_returns)
            avg_return = bucket_returns.mean()
            count = len(bucket_returns)
            median_distance = np.median(distances_valid[bucket_mask])
            
            results.append({
                'Bucket': label,
                'Count': count,
                'Median_Distance': median_distance,
                'Win_Rate': win_rate,
                'Avg_Return': avg_return
            })
    
    return pd.DataFrame(results)


def main():
    cfg = Config()
    device = get_device()
    
    print("=" * 80)
    print("  Step 14: Centroid Distance Analysis (Textbook vs Edge Cases)")
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
    print(f"QQQ prices: {prices.shape}")
    
    print("\n[2/7] Splitting Train (80%) / Test (20%)...")
    split_idx = int(len(normalized_sequences) * 0.8)
    
    data_train = normalized_sequences[:split_idx]
    data_test = normalized_sequences[split_idx:]
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
    
    print("\n[4/7] Clustering on Training Data...")
    clusterer = PatternClusterer(n_clusters=20)
    labels_train, kmeans_model = clusterer.fit_predict(embeddings_train)
    
    from evaluator import ClusterEvaluator
    evaluator = ClusterEvaluator()
    
    train_prices = prices[:split_idx]
    stats_df = evaluator.calculate_forward_returns(train_prices, labels_train, lookahead=5)
    stats_df_sorted = stats_df.sort_values('Avg_Return', ascending=False)
    
    top_3_clusters = stats_df_sorted.head(3)['Cluster_ID'].values
    print(f"\nTop 3 Bullish Clusters (Training): {top_3_clusters}")
    
    for _, row in stats_df_sorted.head(3).iterrows():
        print(f"  Cluster {int(row['Cluster_ID'])}: {row['Avg_Return']*100:+.2f}% avg, "
              f"{row['Win_Rate']*100:.1f}% win rate, {int(row['Count'])} samples")
    
    print("\n[5/7] Calculating Centroids for Top 3 Clusters...")
    centroids = {}
    for cluster_id in top_3_clusters:
        cluster_mask = labels_train == cluster_id
        cluster_embeddings = embeddings_train[cluster_mask]
        centroid = cluster_embeddings.mean(axis=0)
        centroids[cluster_id] = centroid
        print(f"  Cluster {cluster_id}: Centroid calculated from {cluster_mask.sum()} training samples")
    
    print("\n[6/7] Predicting Test Labels and Calculating Distances...")
    labels_test = kmeans_model.predict(embeddings_test)
    
    test_in_top_clusters = np.isin(labels_test, top_3_clusters)
    print(f"Test samples in Top 3 clusters: {test_in_top_clusters.sum()}")
    
    distances = calculate_centroid_distances(embeddings_test, labels_test, centroids)
    
    valid_distances = distances[~np.isnan(distances)]
    print(f"\nDistance Statistics:")
    print(f"  Min: {valid_distances.min():.4f}")
    print(f"  Median: {np.median(valid_distances):.4f}")
    print(f"  Mean: {valid_distances.mean():.4f}")
    print(f"  Max: {valid_distances.max():.4f}")
    
    forward_returns = calculate_forward_returns(prices_test, lookahead=5)
    
    print("\n[7/7] Analyzing Distance Buckets...")
    
    buckets_results = analyze_distance_buckets(distances, forward_returns, buckets=[25, 50, 75, 100])
    
    print("\n" + "=" * 80)
    print("  RESULTS: Centroid Distance vs Performance")
    print("=" * 80)
    print()
    print(buckets_results.to_string(index=False))
    print("=" * 80)
    
    baseline_mask = test_in_top_clusters & ~np.isnan(forward_returns)
    baseline_returns = forward_returns[baseline_mask]
    baseline_win_rate = (baseline_returns > 0).sum() / len(baseline_returns)
    baseline_avg_return = baseline_returns.mean()
    
    print(f"\nBaseline (All Top 3 Cluster Trades):")
    print(f"  Win Rate: {baseline_win_rate:.2%}")
    print(f"  Avg Return: {baseline_avg_return*100:.2f}%")
    print(f"  Count: {len(baseline_returns)}")
    
    best_bucket = buckets_results.loc[buckets_results['Avg_Return'].idxmax()]
    print(f"\nBest Performing Bucket: {best_bucket['Bucket']}")
    print(f"  Win Rate: {best_bucket['Win_Rate']:.2%} "
          f"({(best_bucket['Win_Rate']-baseline_win_rate)*100:+.1f}% vs baseline)")
    print(f"  Avg Return: {best_bucket['Avg_Return']*100:.2f}% "
          f"({(best_bucket['Avg_Return']-baseline_avg_return)*100:+.2f}% vs baseline)")
    
    print("\n[8/8] Creating Visualizations...")
    
    # Plotting code generated with AI assistance
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    valid_mask = ~np.isnan(distances) & ~np.isnan(forward_returns)
    distances_plot = distances[valid_mask]
    returns_plot = forward_returns[valid_mask]
    labels_plot = labels_test[valid_mask]
    
    ax1 = axes[0, 0]
    for cluster_id in top_3_clusters:
        cluster_mask = labels_plot == cluster_id
        ax1.scatter(distances_plot[cluster_mask], returns_plot[cluster_mask] * 100,
                   alpha=0.5, s=30, label=f'Cluster {cluster_id}')
    
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Distance to Centroid', fontsize=12)
    ax1.set_ylabel('5-Day Forward Return (%)', fontsize=12)
    ax1.set_title('Distance vs Forward Returns (Test Set)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.hist(distances_plot, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Distance to Centroid', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Distances (Test Set)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    percentiles = [25, 50, 75]
    for p in percentiles:
        val = np.percentile(distances_plot, p)
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
    ax3.set_title('Performance by Distance Bucket', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    for i, (label, ret) in enumerate(zip(bucket_labels, avg_returns)):
        ax3.text(i, ret + 0.05, f'{ret:.2f}%', ha='center', fontweight='bold', fontsize=10)
    
    ax4 = axes[1, 1]
    win_rates = buckets_results['Win_Rate'].values * 100
    bars = ax4.bar(range(len(bucket_labels)), win_rates, color=colors,
                   alpha=0.8, edgecolor='black')
    ax4.axhline(y=baseline_win_rate * 100, color='blue', linestyle='--',
               linewidth=2, label=f'Baseline: {baseline_win_rate*100:.1f}%')
    ax4.set_xticks(range(len(bucket_labels)))
    ax4.set_xticklabels(bucket_labels, rotation=45, ha='right')
    ax4.set_ylabel('Win Rate (%)', fontsize=12)
    ax4.set_title('Win Rate by Distance Bucket', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    for i, (label, wr) in enumerate(zip(bucket_labels, win_rates)):
        ax4.text(i, wr + 1, f'{wr:.1f}%', ha='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    output_path = Path('plots') / 'centroid_distance_analysis.png'
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nVisualization saved: {output_path}")
    
    buckets_results.to_csv(Path(cfg.DATA_DIR) / 'centroid_filter_results.csv', index=False)
    print(f"Results saved: {cfg.DATA_DIR}/centroid_filter_results.csv")
    
    print("\n" + "=" * 80)
    print("  CONCLUSION")
    print("=" * 80)
    
    improvement = (best_bucket['Avg_Return'] - baseline_avg_return) * 100
    if improvement > 0.5:
        print(f"\n✓ Centroid filtering IMPROVES performance by {improvement:.2f}%")
        print(f"  Recommendation: Focus on '{best_bucket['Bucket']}' to centroid")
        print(f"  'Textbook patterns' outperform edge cases")
    elif improvement > -0.5:
        print(f"\n→ Centroid distance has NEUTRAL effect ({improvement:+.2f}%)")
        print(f"  Pattern quality is consistent across distance ranges")
    else:
        print(f"\n✗ Centroid filtering HURTS performance by {improvement:.2f}%")
        print(f"  'Edge cases' may capture emerging patterns not in training centroids")
    
    print("=" * 80)


if __name__ == '__main__':
    main()

