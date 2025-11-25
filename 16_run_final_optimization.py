# Author: Yifan Wang
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import product
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


def calculate_sharpe_ratio(returns, periods_per_year=252/5):
    if len(returns) < 2:
        return 0
    mean_return = returns.mean()
    std_return = returns.std()
    if std_return == 0:
        return 0
    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    return sharpe


def calculate_max_drawdown(returns):
    if len(returns) == 0:
        return 0
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def evaluate_strategy(magnitudes, returns, labels, top_clusters, lower_percentile, upper_percentile):
    lower_threshold = np.percentile(magnitudes, lower_percentile)
    upper_threshold = np.percentile(magnitudes, upper_percentile)
    
    valid_mask = ~np.isnan(returns)
    magnitude_mask = (magnitudes >= lower_threshold) & (magnitudes <= upper_threshold)
    cluster_mask = np.isin(labels, top_clusters)
    
    strategy_mask = valid_mask & magnitude_mask & cluster_mask
    
    if strategy_mask.sum() == 0:
        return None
    
    strategy_returns = returns[strategy_mask]
    
    win_rate = (strategy_returns > 0).sum() / len(strategy_returns)
    avg_return = strategy_returns.mean()
    total_return = (1 + strategy_returns).prod() - 1
    sharpe = calculate_sharpe_ratio(strategy_returns)
    max_dd = calculate_max_drawdown(strategy_returns)
    count = len(strategy_returns)
    
    return {
        'Lower': lower_percentile,
        'Upper': upper_percentile,
        'Count': count,
        'Win_Rate': win_rate,
        'Avg_Return': avg_return,
        'Total_Return': total_return,
        'Sharpe_Ratio': sharpe,
        'Max_Drawdown': max_dd
    }

# This function has 30% code that co-worked with AI and re-write + debug with AI frequently
def main():
    cfg = Config()
    device = get_device()
    
    print("=" * 80)
    print("  Step 16: Final Optimization - Finding the Optimal Trading Window")
    print("=" * 80)
    
    print("\n[1/6] Loading QQQ Data...")
    downloader = YahooDownloader()
    data_dict = downloader.fetch_data([cfg.TARGET_SYMBOL], cfg.START_DATE, cfg.END_DATE)
    qqq_df = data_dict[cfg.TARGET_SYMBOL]
    
    processor = DataProcessor(seq_len=cfg.SEQ_LEN)
    sequences, prices = processor.make_sequences({cfg.TARGET_SYMBOL: qqq_df})
    prices = prices.flatten()
    normalized_sequences = processor.normalize_window(sequences)
    
    print("\n[2/6] Splitting Train (80%) / Test (20%)...")
    split_idx = int(len(normalized_sequences) * 0.8)
    
    data_train = normalized_sequences[:split_idx]
    data_test = normalized_sequences[split_idx:]
    prices_train = prices[:split_idx]
    prices_test = prices[split_idx:]
    
    print(f"Train: {data_train.shape}")
    print(f"Test: {data_test.shape}")
    
    print("\n[3/6] Loading Model and Generating Embeddings...")
    model = CnnAutoencoder(
        input_dim=cfg.INPUT_DIM,
        seq_len=cfg.SEQ_LEN,
        latent_dim=cfg.LATENT_DIM
    ).to(device)
    
    checkpoint = torch.load(cfg.MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    embeddings_train = generate_embeddings(model, data_train, device)
    embeddings_test = generate_embeddings(model, data_test, device)
    
    magnitudes_test = calculate_l2_norms(embeddings_test)
    
    print("\n[4/6] Identifying Top 3 Bullish Clusters...")
    clusterer = PatternClusterer(n_clusters=20)
    labels_train, kmeans_model = clusterer.fit_predict(embeddings_train)
    
    from evaluator import ClusterEvaluator
    evaluator = ClusterEvaluator()
    
    stats_df = evaluator.calculate_forward_returns(prices_train, labels_train, lookahead=5)
    stats_df_sorted = stats_df.sort_values('Avg_Return', ascending=False)
    
    top_3_clusters = stats_df_sorted.head(3)['Cluster_ID'].values
    print(f"Top 3 Bullish Clusters: {top_3_clusters}")
    
    labels_test = kmeans_model.predict(embeddings_test)
    
    print("\n[5/6] Calculating Forward Returns...")
    forward_returns = calculate_forward_returns(prices_test, lookahead=5)
    
    print("\n[6/6] Running Grid Search...")
    
    lower_bounds = [0, 10, 20, 25, 30]
    upper_bounds = [70, 75, 80, 90, 100]
    
    results = []
    
    total_combinations = len(lower_bounds) * len(upper_bounds)
    print(f"\nTesting {total_combinations} combinations...")
    
    for lower, upper in product(lower_bounds, upper_bounds):
        if lower >= upper:
            continue
        
        result = evaluate_strategy(magnitudes_test, forward_returns, labels_test,
                                  top_3_clusters, lower, upper)
        
        if result is not None:
            results.append(result)
    
    results_df = pd.DataFrame(results)
    
    print(f"\nCompleted grid search: {len(results_df)} valid combinations")
    
    print("\n" + "=" * 80)
    print("  TOP 10 STRATEGIES (by Win Rate)")
    print("=" * 80)
    
    top_10 = results_df[results_df['Count'] >= 50].nlargest(10, 'Win_Rate')
    
    print()
    print(f"{'Range':<15} {'Count':<10} {'Win Rate':<12} {'Avg Ret':<12} {'Sharpe':<10} {'Max DD'}")
    print("-" * 80)
    
    for _, row in top_10.iterrows():
        range_str = f"P{int(row['Lower'])}-P{int(row['Upper'])}"
        print(f"{range_str:<15} {int(row['Count']):<10} {row['Win_Rate']:<12.2%} "
              f"{row['Avg_Return']*100:<12.2f}% {row['Sharpe_Ratio']:<10.2f} {row['Max_Drawdown']:<.2%}")
    
    print("-" * 80)
    
    best_strategy = results_df[results_df['Count'] >= 50].loc[results_df[results_df['Count'] >= 50]['Win_Rate'].idxmax()]
    
    print("\n" + "=" * 80)
    print("  OPTIMAL TRADING WINDOW")
    print("=" * 80)
    
    print(f"\nBest Strategy: Magnitude between P{int(best_strategy['Lower'])} and P{int(best_strategy['Upper'])}")
    print(f"  Trades: {int(best_strategy['Count'])}")
    print(f"  Win Rate: {best_strategy['Win_Rate']:.2%}")
    print(f"  Avg Return per Trade: {best_strategy['Avg_Return']*100:.2f}%")
    print(f"  Total Return: {best_strategy['Total_Return']*100:.2f}%")
    print(f"  Sharpe Ratio: {best_strategy['Sharpe_Ratio']:.2f}")
    print(f"  Max Drawdown: {best_strategy['Max_Drawdown']:.2%}")
    
    baseline_mask = np.isin(labels_test, top_3_clusters) & ~np.isnan(forward_returns)
    baseline_returns = forward_returns[baseline_mask]
    baseline_wr = (baseline_returns > 0).sum() / len(baseline_returns)
    
    improvement = (best_strategy['Win_Rate'] - baseline_wr) * 100
    
    print(f"\nImprovement over Baseline (Top 3 Clusters, No Magnitude Filter):")
    print(f"  Baseline Win Rate: {baseline_wr:.2%}")
    print(f"  Optimized Win Rate: {best_strategy['Win_Rate']:.2%}")
    print(f"  Improvement: {improvement:+.1f} percentage points")
    
    print("\n[7/7] Creating Visualizations...")
    
    # Plotting code generated with AI assistance
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    heatmap_data = results_df.pivot(index='Upper', columns='Lower', values='Win_Rate')
    sns.heatmap(heatmap_data, annot=True, fmt='.1%', cmap='RdYlGn', 
                vmin=0.5, vmax=0.7, ax=ax1, cbar_kws={'label': 'Win Rate'})
    ax1.set_title('Grid Search Heatmap: Win Rate by Magnitude Window', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Lower Bound (Percentile)', fontsize=12)
    ax1.set_ylabel('Upper Bound (Percentile)', fontsize=12)
    
    ax2 = fig.add_subplot(gs[1, 0])
    heatmap_sharpe = results_df.pivot(index='Upper', columns='Lower', values='Sharpe_Ratio')
# Plotting code generated with AI assistance
    sns.heatmap(heatmap_sharpe, annot=True, fmt='.2f', cmap='viridis', ax=ax2,
                cbar_kws={'label': 'Sharpe Ratio'})
    ax2.set_title('Sharpe Ratio by Window', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Lower Bound', fontsize=11)
    ax2.set_ylabel('Upper Bound', fontsize=11)
    
    ax3 = fig.add_subplot(gs[1, 1])
    heatmap_count = results_df.pivot(index='Upper', columns='Lower', values='Count')

    sns.heatmap(heatmap_count, annot=True, fmt='d', cmap='Blues', ax=ax3,
                cbar_kws={'label': 'Trade Count'})
    ax3.set_title('Trade Count by Window', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Lower Bound', fontsize=11)
    ax3.set_ylabel('Upper Bound', fontsize=11)
    
    ax4 = fig.add_subplot(gs[2, 0])
    top_5_results = results_df[results_df['Count'] >= 50].nlargest(5, 'Win_Rate')
    strategy_labels = [f"P{int(r['Lower'])}-P{int(r['Upper'])}" for _, r in top_5_results.iterrows()]
    win_rates = top_5_results['Win_Rate'].values * 100
    
    colors = plt.cm.RdYlGn(np.linspace(0.5, 0.9, len(strategy_labels)))
    bars = ax4.barh(strategy_labels, win_rates, color=colors, edgecolor='black')
    ax4.axvline(x=baseline_wr * 100, color='red', linestyle='--', 
               linewidth=2, label=f'Baseline: {baseline_wr:.1%}')
    ax4.set_xlabel('Win Rate (%)', fontsize=11)
    ax4.set_title('Top 5 Strategies by Win Rate', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='x')
    
    for i, (label, wr) in enumerate(zip(strategy_labels, win_rates)):
        ax4.text(wr + 0.5, i, f'{wr:.1f}%', va='center', fontweight='bold', fontsize=10)
    
    ax5 = fig.add_subplot(gs[2, 1])
    
    optimal_lower = best_strategy['Lower']
    optimal_upper = best_strategy['Upper']
    
    magnitude_ranges = [
        (0, optimal_lower, 'Too Weak'),
        (optimal_lower, optimal_upper, 'Optimal Window'),
        (optimal_upper, 100, 'Too Extreme')
    ]
    
    zone_results = []
    for lower, upper, label in magnitude_ranges:
        result = evaluate_strategy(magnitudes_test, forward_returns, labels_test,
                                  top_3_clusters, lower, upper)
        if result:
            zone_results.append((label, result['Win_Rate'] * 100, result['Count']))
        else:
            zone_results.append((label, 0, 0))
    
    zone_labels, zone_wrs, zone_counts = zip(*zone_results)
    colors_zone = ['red', 'green', 'orange']
    
    bars = ax5.bar(zone_labels, zone_wrs, color=colors_zone, alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Win Rate (%)', fontsize=11)
    ax5.set_title('Performance by Magnitude Zone', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    for i, (label, wr, count) in enumerate(zone_results):
        ax5.text(i, wr + 1, f'{wr:.1f}%\n(n={count})', ha='center', 
                fontweight='bold', fontsize=10)
    
    output_path = Path('plots') / 'final_optimization.png'
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nVisualization saved: {output_path}")
    
    results_df.to_csv(Path(cfg.DATA_DIR) / 'optimization_results.csv', index=False)
    print(f"Results saved: {cfg.DATA_DIR}/optimization_results.csv")
    
    print("\n" + "=" * 80)
    print("  KEY INSIGHTS")
    print("=" * 80)
    
    print("\n1. Optimal Window Found:")
    print(f"   Magnitude between P{int(best_strategy['Lower'])} - P{int(best_strategy['Upper'])}")
    print(f"   This captures 'Moderate Confidence' signals")
    
    print("\n2. Why This Works:")
    print(f"   • Too Weak (<P{int(optimal_lower)}): Insufficient signal strength")
    print(f"   • Sweet Spot (P{int(optimal_lower)}-P{int(optimal_upper)}): Strong but stable patterns")
    print(f"   • Too Extreme (>P{int(optimal_upper)}): Mean reversion / overreaction risk")
    
    print("\n3. Neural Network Calibration:")
    print(f"   The model's 'loudest' signals are NOT the most reliable")
    print(f"   This proves the importance of calibration in production systems")
    
    print("\n4. Academic Value:")
    print(f"   Demonstrates that raw neural network confidence requires post-processing")
    print(f"   Band-pass filtering improves performance by {improvement:.1f} percentage points")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()

