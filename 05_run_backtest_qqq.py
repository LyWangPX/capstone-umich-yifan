import numpy as np
import pandas as pd
from pathlib import Path
from config import Config
from data_loader import YahooDownloader
from processor import DataProcessor
from inference import InferenceEngine
from evaluator import ClusterEvaluator
from backtester import SimpleBacktester
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    cfg = Config()
    
    print("Downloading QQQ data...")
    downloader = YahooDownloader()
    data_dict = downloader.fetch_data([cfg.TARGET_SYMBOL], cfg.START_DATE, cfg.END_DATE)
    qqq_df = data_dict[cfg.TARGET_SYMBOL]
    
    print(f"QQQ data: {len(qqq_df)} rows from {qqq_df.index[0]} to {qqq_df.index[-1]}")
    
    val_split_idx = int(len(qqq_df) * 0.8)
    qqq_train = qqq_df.iloc[:val_split_idx]
    
    processor = DataProcessor(seq_len=cfg.SEQ_LEN)
    sequences, prices = processor.make_sequences({cfg.TARGET_SYMBOL: qqq_train})
    prices = prices.flatten()
    normalized_sequences = processor.normalize_window(sequences)
    
    print(f"QQQ sequences: {sequences.shape}")
    print(f"QQQ prices: {prices.shape}")
    print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    
    from torch.utils.data import DataLoader
    from dataset import TimeSeriesDataset
    
    np.save('temp_qqq_data.npy', normalized_sequences)
    dataset = TimeSeriesDataset('temp_qqq_data.npy')
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    print("\nGenerating embeddings for QQQ...")
    device = get_device()
    engine = InferenceEngine(
        model_path=cfg.MODEL_PATH,
        input_dim=cfg.INPUT_DIM,
        seq_len=cfg.SEQ_LEN,
        latent_dim=cfg.LATENT_DIM,
        device=device
    )
    
    embeddings, _ = engine.get_embeddings(loader)
    print(f"Embeddings shape: {embeddings.shape}")
    
    from clustering import PatternClusterer
    clusterer = PatternClusterer(n_clusters=20)
    cluster_labels, _ = clusterer.fit_predict(embeddings)
    
    print("\n=== Step 1: Statistical Evaluation (QQQ Only) ===")
    evaluator = ClusterEvaluator()
    stats_df = evaluator.calculate_forward_returns(prices, cluster_labels, lookahead=5)
    
    stats_df_sorted = stats_df.sort_values('Avg_Return', ascending=False)
    print("\nCluster Performance (sorted by Avg_Return):")
    print(stats_df_sorted.to_string(index=False))
    
    print("\n=== Step 2: Cluster Selection ===")
    long_clusters = stats_df_sorted.head(3)['Cluster_ID'].values
    avoid_clusters = stats_df_sorted.tail(3)['Cluster_ID'].values
    
    print(f"Longing Clusters: {long_clusters}")
    print(f"Avoiding Clusters: {avoid_clusters}")
    
    print("\n=== Step 3: Backtesting (QQQ Only) ===")
    backtester = SimpleBacktester()
    
    strategy_equity, total_trades = backtester.run_strategy(prices, cluster_labels, long_clusters, avoid_clusters)
    
    initial_cash = 10000
    buy_hold_equity = initial_cash * (prices / prices[0])
    
    strategy_return = float((strategy_equity[-1] / strategy_equity[0]) - 1)
    buy_hold_return = float((buy_hold_equity[-1] / buy_hold_equity[0]) - 1)
    
    print(f"\nTotal Trades: {total_trades}")
    print(f"Strategy Final Return: {strategy_return:.2%}")
    print(f"Buy & Hold Final Return: {buy_hold_return:.2%}")
    print(f"Outperformance: {(strategy_return - buy_hold_return):.2%}")
    
    results_summary = {
        'Strategy': ['Cluster Strategy', 'Buy & Hold'],
        'Final_Return': [strategy_return, buy_hold_return],
        'Final_Value': [strategy_equity[-1], buy_hold_equity[-1]]
    }
    summary_df = pd.DataFrame(results_summary)
    
    summary_df.to_csv(Path(cfg.DATA_DIR) / 'backtest_summary_qqq.csv', index=False)
    stats_df_sorted.to_csv(Path(cfg.DATA_DIR) / 'cluster_stats_qqq.csv', index=False)
    
    import os
    if os.path.exists('temp_qqq_data.npy'):
        os.remove('temp_qqq_data.npy')
    
    print(f"\nResults saved to {cfg.DATA_DIR}/")


if __name__ == '__main__':
    main()

