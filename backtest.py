# Author: Yifan Wang
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import config


def get_top_clusters(performance_df, lookahead=5, top_n=3):
    subset = performance_df[performance_df['Lookahead'] == lookahead]
    top_clusters = subset.nlargest(top_n, 'Mean_Return')['Cluster_ID'].values
    return top_clusters


def simulate_strategy(cluster_results, top_clusters):
    signals = cluster_results['Cluster_ID'].isin(top_clusters).astype(int)
    
    daily_returns = cluster_results['Close_Price'].pct_change()
    strategy_returns = signals.shift(1) * daily_returns
    
    strategy_cumulative = (1 + strategy_returns.fillna(0)).cumprod()
    buy_hold_cumulative = (1 + daily_returns.fillna(0)).cumprod()
    
    return strategy_cumulative, buy_hold_cumulative, signals


def calculate_metrics(returns):
    total_return = returns.iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    sharpe = returns.pct_change().mean() / returns.pct_change().std() * np.sqrt(252)
    max_dd = (returns / returns.cummax() - 1).min()
    
    return {
        'Total_Return': total_return,
        'Annual_Return': annual_return,
        'Sharpe_Ratio': sharpe,
        'Max_Drawdown': max_dd
    }


def plot_backtest_results(dates, strategy_cumulative, buy_hold_cumulative, signals):
# Plotting code generated with AI assistance
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
    
    ax1.plot(dates, strategy_cumulative, label='Cluster Strategy', linewidth=2, color='steelblue')
    ax1.plot(dates, buy_hold_cumulative, label='Buy & Hold', linewidth=2, color='orange', alpha=0.7)
    ax1.set_title('Cumulative Returns: Cluster Strategy vs Buy & Hold', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Return', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.fill_between(dates, 0, signals, alpha=0.5, color='green', label='In Market')
    ax2.set_title('Strategy Position', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Signal', fontsize=12)
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    performance_df = pd.read_csv(Path(config.OUTPUT_DIR) / f"{config.SYMBOLS[0]}_cluster_performance.csv")
    cluster_results = pd.read_csv(Path(config.OUTPUT_DIR) / f"{config.SYMBOLS[0]}_clusters_enriched.csv")
    cluster_results['Date'] = pd.to_datetime(cluster_results['Date'])
    
    print("=== Backtesting Cluster-Based Strategy ===\n")
    
    lookahead = config.LOOKAHEAD
    top_clusters = get_top_clusters(performance_df, lookahead=lookahead, top_n=3)
    print(f"Top {len(top_clusters)} performing clusters (based on {lookahead}-day returns): {top_clusters}")
    
    strategy_cumulative, buy_hold_cumulative, signals = simulate_strategy(cluster_results, top_clusters)
    
    strategy_metrics = calculate_metrics(strategy_cumulative)
    buy_hold_metrics = calculate_metrics(buy_hold_cumulative)
    
    print("\n=== Performance Metrics ===")
    print("\nCluster Strategy:")
    for key, value in strategy_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nBuy & Hold:")
    for key, value in buy_hold_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nTime in Market: {signals.mean():.2%}")
    
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    
    fig = plot_backtest_results(cluster_results['Date'], strategy_cumulative, buy_hold_cumulative, signals)
    fig.savefig(plots_dir / 'backtest_results.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nBacktest plot saved to plots/backtest_results.png")
    
    results_summary = pd.DataFrame({
        'Strategy': ['Cluster Strategy', 'Buy & Hold'],
        'Total_Return': [strategy_metrics['Total_Return'], buy_hold_metrics['Total_Return']],
        'Annual_Return': [strategy_metrics['Annual_Return'], buy_hold_metrics['Annual_Return']],
        'Sharpe_Ratio': [strategy_metrics['Sharpe_Ratio'], buy_hold_metrics['Sharpe_Ratio']],
        'Max_Drawdown': [strategy_metrics['Max_Drawdown'], buy_hold_metrics['Max_Drawdown']]
    })
    
    results_summary.to_csv(Path(config.OUTPUT_DIR) / 'backtest_summary.csv', index=False)
    print(f"Summary saved to {config.OUTPUT_DIR}/backtest_summary.csv")


if __name__ == '__main__':
    main()

