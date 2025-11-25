# Author: Yifan Wang
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from data_loader import YahooDownloader


def run_bootstrap():
    print("=" * 80)
    print("  STEP 19: STATISTICAL SIGNIFICANCE TEST (Bootstrap)")
    print("=" * 80)
    
    observed_win_rate = 0.621
    n_trades = 127
    
    print(f"\nObserved AI Strategy:")
    print(f"  Win Rate: {observed_win_rate:.1%}")
    print(f"  Number of Trades: {n_trades}")
    
    print("\n[1/3] Loading QQQ Market Data (2020-2025)...")
    downloader = YahooDownloader()
    data_dict = downloader.fetch_data(['QQQ'], '2020-01-01', '2025-11-21')
    qqq_df = data_dict['QQQ']
    
    close_prices = qqq_df['Close'].values if isinstance(qqq_df['Close'], pd.Series) else qqq_df['Close'].iloc[:, 0].values
    daily_returns = pd.Series(close_prices).pct_change().dropna().values
    market_win_rate = float((daily_returns > 0).mean())
    
    print(f"Market Baseline:")
    print(f"  QQQ Win Rate (all days): {market_win_rate:.1%}")
    print(f"  Sample Size: {len(daily_returns)} days")
    
    print("\n[2/3] Running Bootstrap Simulation (n=10,000)...")
    
    np.random.seed(42)
    simulated_win_rates = []
    
    for iteration in range(10000):
        random_sample = np.random.choice(daily_returns, size=n_trades, replace=True)
        sim_win_rate = (random_sample > 0).mean()
        simulated_win_rates.append(sim_win_rate)
    
    simulated_win_rates = np.array(simulated_win_rates)
    
    p_value = (simulated_win_rates >= observed_win_rate).mean()
    
    print(f"\nBootstrap Results:")
    print(f"  Simulated Mean: {simulated_win_rates.mean():.1%}")
    print(f"  Simulated Std: {simulated_win_rates.std():.3f}")
    print(f"  95% CI: [{np.percentile(simulated_win_rates, 2.5):.1%}, {np.percentile(simulated_win_rates, 97.5):.1%}]")
    
    print("\n" + "=" * 80)
    print("  STATISTICAL SIGNIFICANCE")
    print("=" * 80)
    print(f"\nH0: AI strategy is no better than random sampling")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < 0.01:
        print(f"Result: HIGHLY SIGNIFICANT (p < 0.01) ✓✓✓")
        print(f"The AI strategy's 62.1% win rate is statistically superior to random chance.")
    elif p_value < 0.05:
        print(f"Result: SIGNIFICANT (p < 0.05) ✓✓")
        print(f"The AI strategy shows significant predictive power.")
    elif p_value < 0.10:
        print(f"Result: MARGINALLY SIGNIFICANT (p < 0.10) ✓")
    else:
        print(f"Result: NOT SIGNIFICANT (p >= 0.10)")
    
    print("=" * 80)
    
    print("\n[3/3] Creating Visualization...")
    
    # Plotting code generated with AI assistance
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.hist(simulated_win_rates, bins=50, alpha=0.7, color='lightgray', 
            edgecolor='black', label='Random Sampling Distribution')
    
    ax.axvline(market_win_rate, color='blue', linestyle='--', linewidth=2,
              label=f'Market Baseline: {market_win_rate:.1%}')
    ax.axvline(observed_win_rate, color='red', linestyle='-', linewidth=3,
              label=f'AI Strategy: {observed_win_rate:.1%} (p={p_value:.4f})')
    
    ax.axvline(np.percentile(simulated_win_rates, 95), color='green', 
              linestyle=':', linewidth=2, alpha=0.7, label='95th Percentile')
    
    ax.set_title('Statistical Significance: Win Rate vs Random Chance', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Win Rate', fontsize=13)
    ax.set_ylabel('Frequency (n=10,000 simulations)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    y_pos = ax.get_ylim()[1] * 0.7
    if p_value < 0.05:
        ax.text(observed_win_rate + 0.01, y_pos,
               f'Statistically\nSignificant\n(p={p_value:.4f})',
               fontsize=11, fontweight='bold', color='darkred',
               bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.9))
    
    output_path = Path('plots') / 'bootstrap_significance.png'
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nVisualization saved: {output_path}")
    
    results = pd.DataFrame({
        'Metric': ['Observed_Win_Rate', 'Market_Baseline', 'P_Value', 'N_Trades'],
        'Value': [observed_win_rate, market_win_rate, p_value, n_trades]
    })
    
    results.to_csv(Path('data') / 'bootstrap_results.csv', index=False)
    print(f"Results saved: data/bootstrap_results.csv")


if __name__ == '__main__':
    run_bootstrap()

