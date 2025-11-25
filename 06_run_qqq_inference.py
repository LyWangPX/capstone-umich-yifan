# Author: Yifan Wang
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from config import Config
from data_loader import YahooDownloader
from processor import DataProcessor
from inference import InferenceEngine
from clustering import PatternClusterer
from evaluator import ClusterEvaluator
from backtester import SimpleBacktester
from dataset import TimeSeriesDataset
from torch.utils.data import DataLoader


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    cfg = Config()
    device = get_device()
    
    print("=" * 60)
    print("QQQ AI Pattern Recognition - Final Benchmark")
    print("=" * 60)
    
    print("\n[1/6] Downloading Fresh QQQ Data...")
    downloader = YahooDownloader()
    data_dict = downloader.fetch_data([cfg.TARGET_SYMBOL], cfg.START_DATE, cfg.END_DATE)
    qqq_df = data_dict[cfg.TARGET_SYMBOL]
    
    print(f"QQQ: {len(qqq_df)} trading days")
    print(f"Period: {qqq_df.index[0].date()} to {qqq_df.index[-1].date()}")
    print(f"Start Price: ${float(qqq_df['Close'].iloc[0]):.2f}")
    print(f"End Price: ${float(qqq_df['Close'].iloc[-1]):.2f}")
    
    print("\n[2/6] Processing Sequences...")
    processor = DataProcessor(seq_len=cfg.SEQ_LEN)
    sequences, raw_prices = processor.make_sequences({cfg.TARGET_SYMBOL: qqq_df})
    raw_prices = raw_prices.flatten()
    normalized_sequences = processor.normalize_window(sequences)
    
    print(f"Sequences: {normalized_sequences.shape}")
    print(f"Raw Prices: {raw_prices.shape}")
    print(f"Price Range: ${raw_prices.min():.2f} - ${raw_prices.max():.2f}")
    
    print("\n[3/6] Running Model Inference...")
    np.save('temp_qqq_inference.npy', normalized_sequences)
    dataset = TimeSeriesDataset('temp_qqq_inference.npy')
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    engine = InferenceEngine(
        model_path=cfg.MODEL_PATH,
        input_dim=cfg.INPUT_DIM,
        seq_len=cfg.SEQ_LEN,
        latent_dim=cfg.LATENT_DIM,
        device=device
    )
    
    embeddings, _ = engine.get_embeddings(loader)
    print(f"Embeddings Generated: {embeddings.shape}")
    
    print("\n[4/6] Clustering Patterns...")
    clusterer = PatternClusterer(n_clusters=20)
    cluster_labels, kmeans_model = clusterer.fit_predict(embeddings)
    
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"Clusters: {len(unique)}")
    print("Distribution:")
    for cid, cnt in zip(unique, counts):
        print(f"  Cluster {cid:2d}: {cnt:4d} days ({cnt/len(cluster_labels)*100:5.2f}%)")
    
    print("\n[5/6] Evaluating Cluster Performance (5-Day Lookahead)...")
    evaluator = ClusterEvaluator()
    stats_df = evaluator.calculate_forward_returns(raw_prices, cluster_labels, lookahead=5)
    stats_df_sorted = stats_df.sort_values('Avg_Return', ascending=False)
    
    print("\nCluster Performance:")
    print(stats_df_sorted.to_string(index=False))
    
    long_clusters = stats_df_sorted.head(3)['Cluster_ID'].values
    avoid_clusters = stats_df_sorted.tail(3)['Cluster_ID'].values
    
    print(f"\nBullish Clusters (Long): {long_clusters}")
    print(f"Bearish Clusters (Avoid): {avoid_clusters}")
    
    print("\n[6/6] Running Backtest...")
    backtester = SimpleBacktester()
    
    strategy_equity, total_trades = backtester.run_strategy(
        raw_prices, cluster_labels, long_clusters, avoid_clusters
    )
    
    initial_capital = 10000
    buy_hold_equity = initial_capital * (raw_prices / raw_prices[0])
    
    strategy_return = float((strategy_equity[-1] / strategy_equity[0]) - 1)
    buy_hold_return = float((buy_hold_equity[-1] / buy_hold_equity[0]) - 1)
    outperformance = strategy_return - buy_hold_return
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Initial Capital:        ${initial_capital:,.2f}")
    print(f"Total Trades:           {total_trades}")
    print(f"\nCluster Strategy:")
    print(f"  Final Value:          ${strategy_equity[-1]:,.2f}")
    print(f"  Total Return:         {strategy_return:.2%}")
    print(f"\nBuy & Hold Benchmark:")
    print(f"  Final Value:          ${buy_hold_equity[-1]:,.2f}")
    print(f"  Total Return:         {buy_hold_return:.2%}")
    print(f"\nOutperformance:         {outperformance:.2%}")
    print("=" * 60)
    
    results_df = pd.DataFrame({
        'Metric': ['Strategy_Return', 'BuyHold_Return', 'Outperformance', 'Total_Trades'],
        'Value': [strategy_return, buy_hold_return, outperformance, total_trades]
    })
    
    output_dir = Path(cfg.DATA_DIR)
    results_df.to_csv(output_dir / 'qqq_final_benchmark.csv', index=False)
    stats_df_sorted.to_csv(output_dir / 'qqq_cluster_stats.csv', index=False)
    
    dates = qqq_df.index[cfg.SEQ_LEN - 1:]
    clusterer.save_clusters(dates, cluster_labels, output_dir / 'qqq_clusters.csv')
    
    import os
    if os.path.exists('temp_qqq_inference.npy'):
        os.remove('temp_qqq_inference.npy')
    
    print(f"\nResults saved to {cfg.DATA_DIR}/")
    print("  - qqq_final_benchmark.csv")
    print("  - qqq_cluster_stats.csv")
    print("  - qqq_clusters.csv")


if __name__ == '__main__':
    main()

