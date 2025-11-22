import numpy as np
import pandas as pd
from pathlib import Path
from config import Config
from evaluator import ClusterEvaluator
from backtester import SimpleBacktester


def main():
    cfg = Config()
    
    print("Loading data...")
    cluster_labels = np.load(Path(cfg.DATA_DIR) / 'cluster_labels.npy')
    prices = np.load(Path(cfg.DATA_DIR) / 'train_prices.npy').flatten()
    
    print(f"Cluster labels shape: {cluster_labels.shape}")
    print(f"Raw prices shape: {prices.shape}")
    print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    
    print("\n=== Step 1: Statistical Evaluation ===")
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
    
    print("\n=== Step 3: Backtesting ===")
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
    
    summary_df.to_csv(Path(cfg.DATA_DIR) / 'backtest_summary.csv', index=False)
    stats_df_sorted.to_csv(Path(cfg.DATA_DIR) / 'cluster_stats.csv', index=False)
    
    print(f"\nResults saved to {cfg.DATA_DIR}/")


if __name__ == '__main__':
    main()

