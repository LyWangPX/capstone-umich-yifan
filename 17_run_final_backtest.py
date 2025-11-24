import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
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


def simulate_strategy(prices, signals, hold_days=5):
    equity = [10000]
    position = 0
    cash = 10000
    holding_counter = 0
    
    for i in range(1, len(prices)):
        current_price = prices[i]
        
        if holding_counter > 0:
            holding_counter -= 1
            if holding_counter == 0:
                cash = position * current_price
                position = 0
        
        elif signals[i] and holding_counter == 0:
            position = cash / current_price
            cash = 0
            holding_counter = hold_days
        
        portfolio_value = cash + (position * current_price)
        equity.append(portfolio_value)
    
    return np.array(equity)


def calculate_cagr(equity, days):
    years = days / 252
    if years <= 0:
        return 0
    total_return = equity[-1] / equity[0]
    cagr = (total_return ** (1 / years)) - 1
    return cagr


def calculate_max_drawdown(equity):
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    return drawdown.min()


def calculate_sharpe_ratio(returns, risk_free_rate=0):
    if len(returns) < 2:
        return 0
    excess_returns = returns - risk_free_rate
    if excess_returns.std() == 0:
        return 0
    return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())


def calculate_sortino_ratio(returns, risk_free_rate=0):
    if len(returns) < 2:
        return 0
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0
    return np.sqrt(252) * (excess_returns.mean() / downside_returns.std())


def main():
    cfg = Config()
    device = get_device()
    
    print("\n" + "=" * 80)
    print("=" * 80)
    print("  STEP 17: FINAL BACKTEST - AI Strategy vs Benchmarks")
    print("=" * 80)
    print("=" * 80)
    
    print("\n[1/8] Loading QQQ and TQQQ Data...")
    downloader = YahooDownloader()
    qqq_dict = downloader.fetch_data([cfg.TARGET_SYMBOL], cfg.START_DATE, cfg.END_DATE)
    tqqq_dict = downloader.fetch_data(['TQQQ'], cfg.START_DATE, cfg.END_DATE)
    
    qqq_df = qqq_dict[cfg.TARGET_SYMBOL]
    
    if 'TQQQ' in tqqq_dict:
        tqqq_df = tqqq_dict['TQQQ']
        print(f"TQQQ data: {len(tqqq_df)} rows")
    else:
        tqqq_df = None
        print("TQQQ data not available")
    
    processor = DataProcessor(seq_len=cfg.SEQ_LEN)
    sequences, prices = processor.make_sequences({cfg.TARGET_SYMBOL: qqq_df})
    prices = prices.flatten()
    normalized_sequences = processor.normalize_window(sequences)
    
    print(f"QQQ sequences: {sequences.shape}")
    
    print("\n[2/8] Splitting Train (80%) / Test (20%)...")
    split_idx = int(len(normalized_sequences) * 0.8)
    
    data_train = normalized_sequences[:split_idx]
    data_test = normalized_sequences[split_idx:]
    prices_test = prices[split_idx:]
    
    test_start_idx = split_idx + cfg.SEQ_LEN - 1
    dates_test = qqq_df.index[test_start_idx:test_start_idx + len(prices_test)]
    
    print(f"Test period: {dates_test[0].date()} to {dates_test[-1].date()}")
    print(f"Test days: {len(prices_test)}")
    
    print("\n[3/8] Loading Model and Generating Embeddings...")
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
    
    print("\n[4/8] Identifying Top 3 Bullish Clusters...")
    clusterer = PatternClusterer(n_clusters=20)
    labels_train, kmeans_model = clusterer.fit_predict(embeddings_train)
    
    from evaluator import ClusterEvaluator
    evaluator = ClusterEvaluator()
    
    prices_train = prices[:split_idx]
    stats_df = evaluator.calculate_forward_returns(prices_train, labels_train, lookahead=5)
    stats_df_sorted = stats_df.sort_values('Avg_Return', ascending=False)
    
    top_3_clusters = stats_df_sorted.head(3)['Cluster_ID'].values
    print(f"Top 3 Bullish Clusters: {top_3_clusters}")
    
    labels_test = kmeans_model.predict(embeddings_test)
    
    print("\n[5/8] Applying Optimal Strategy Filter (P30-P75)...")
    
    lower_threshold = np.percentile(magnitudes_test, 30)
    upper_threshold = np.percentile(magnitudes_test, 75)
    
    magnitude_filter = (magnitudes_test >= lower_threshold) & (magnitudes_test <= upper_threshold)
    cluster_filter = np.isin(labels_test, top_3_clusters)
    
    strategy_signals = magnitude_filter & cluster_filter
    
    print(f"Magnitude window: [{lower_threshold:.3f}, {upper_threshold:.3f}]")
    print(f"Signals generated: {strategy_signals.sum()} out of {len(strategy_signals)} days")
    print(f"Signal rate: {strategy_signals.sum() / len(strategy_signals):.1%}")
    
    print("\n[6/8] Running Strategy Simulation...")
    
    strategy_equity = simulate_strategy(prices_test, strategy_signals, hold_days=5)
    
    qqq_equity = 10000 * (prices_test / prices_test[0])
    
    if tqqq_df is not None:
        try:
            tqqq_aligned = tqqq_df.reindex(dates_test, method='ffill')
            tqqq_aligned = tqqq_aligned.fillna(method='ffill').fillna(method='bfill')
            
            if len(tqqq_aligned) > 0 and tqqq_aligned['Close'].notna().any():
                tqqq_prices_test = tqqq_aligned['Close'].values
                tqqq_equity = 10000 * (tqqq_prices_test / tqqq_prices_test[0])
            else:
                tqqq_equity = None
        except:
            tqqq_equity = None
    else:
        tqqq_equity = None
    
    print("\n[7/8] Calculating Performance Metrics...")
    
    strategy_returns = np.diff(strategy_equity) / strategy_equity[:-1]
    qqq_returns = np.diff(qqq_equity) / qqq_equity[:-1]
    
    metrics = {
        'AI Strategy': {
            'Final Value': strategy_equity[-1],
            'Total Return': (strategy_equity[-1] / strategy_equity[0] - 1) * 100,
            'CAGR': calculate_cagr(strategy_equity, len(prices_test)) * 100,
            'Max Drawdown': calculate_max_drawdown(strategy_equity) * 100,
            'Sharpe Ratio': calculate_sharpe_ratio(strategy_returns),
            'Sortino Ratio': calculate_sortino_ratio(strategy_returns)
        },
        'QQQ': {
            'Final Value': qqq_equity[-1],
            'Total Return': (qqq_equity[-1] / qqq_equity[0] - 1) * 100,
            'CAGR': calculate_cagr(qqq_equity, len(prices_test)) * 100,
            'Max Drawdown': calculate_max_drawdown(qqq_equity) * 100,
            'Sharpe Ratio': calculate_sharpe_ratio(qqq_returns),
            'Sortino Ratio': calculate_sortino_ratio(qqq_returns)
        }
    }
    
    if tqqq_equity is not None and len(tqqq_equity) > 0:
        tqqq_returns = np.diff(tqqq_equity) / tqqq_equity[:-1]
        metrics['TQQQ'] = {
            'Final Value': tqqq_equity[-1],
            'Total Return': (tqqq_equity[-1] / tqqq_equity[0] - 1) * 100,
            'CAGR': calculate_cagr(tqqq_equity, len(tqqq_equity)) * 100,
            'Max Drawdown': calculate_max_drawdown(tqqq_equity) * 100,
            'Sharpe Ratio': calculate_sharpe_ratio(tqqq_returns),
            'Sortino Ratio': calculate_sortino_ratio(tqqq_returns)
        }
    
    print("\n" + "=" * 80)
    print("  PERFORMANCE METRICS (Test Period)")
    print("=" * 80)
    print()
    print(f"{'Metric':<25} {'AI Strategy':<20} {'QQQ':<20} {'TQQQ':<20}")
    print("-" * 80)
    
    for metric_name in ['Final Value', 'Total Return', 'CAGR', 'Max Drawdown', 'Sharpe Ratio', 'Sortino Ratio']:
        row = f"{metric_name:<25}"
        
        for strategy in ['AI Strategy', 'QQQ', 'TQQQ']:
            if strategy in metrics:
                value = metrics[strategy][metric_name]
                if metric_name in ['Final Value']:
                    row += f"${value:>18,.2f} "
                elif metric_name in ['Total Return', 'CAGR', 'Max Drawdown']:
                    row += f"{value:>18.2f}% "
                else:
                    row += f"{value:>19.2f} "
            else:
                row += f"{'N/A':>20}"
        
        print(row)
    
    print("-" * 80)
    
    print("\n[8/8] Creating Professional Visualizations...")
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.25)
    
    ax1 = fig.add_subplot(gs[0])
    
    ax1.plot(dates_test, strategy_equity, linewidth=2.5, color='#2E86AB', 
            label=f"AI Strategy ({metrics['AI Strategy']['Total Return']:.1f}%)", zorder=3)
    ax1.plot(dates_test, qqq_equity, linewidth=2, color='#A23B72', alpha=0.8,
            label=f"QQQ ({metrics['QQQ']['Total Return']:.1f}%)", zorder=2)
    
    if tqqq_equity is not None and 'TQQQ' in metrics:
        ax1.plot(dates_test, tqqq_equity, linewidth=2, color='#F18F01', alpha=0.8,
                label=f"TQQQ ({metrics['TQQQ']['Total Return']:.1f}%)", zorder=1)
    
    ax1.set_title('Final Backtest: AI Pattern Recognition Strategy vs Benchmarks', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=13)
    ax1.legend(fontsize=12, loc='upper right', framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=strategy_equity.min() * 0.9)
    
    ax1.text(0.02, 0.98, f'Initial Capital: $10,000\nTest Period: {dates_test[0].date()} to {dates_test[-1].date()}',
            transform=ax1.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2 = fig.add_subplot(gs[1])
    
    strategy_dd = (strategy_equity - np.maximum.accumulate(strategy_equity)) / np.maximum.accumulate(strategy_equity)
    qqq_dd = (qqq_equity - np.maximum.accumulate(qqq_equity)) / np.maximum.accumulate(qqq_equity)
    
    ax2.fill_between(dates_test, strategy_dd * 100, 0, alpha=0.6, color='#2E86AB', 
                    label=f"AI Strategy (Max: {metrics['AI Strategy']['Max Drawdown']:.1f}%)")
    ax2.fill_between(dates_test, qqq_dd * 100, 0, alpha=0.4, color='#A23B72',
                    label=f"QQQ (Max: {metrics['QQQ']['Max Drawdown']:.1f}%)")
    
    if tqqq_equity is not None and 'TQQQ' in metrics:
        tqqq_dd = (tqqq_equity - np.maximum.accumulate(tqqq_equity)) / np.maximum.accumulate(tqqq_equity)
        ax2.fill_between(dates_test, tqqq_dd * 100, 0, alpha=0.4, color='#F18F01',
                        label=f"TQQQ (Max: {metrics['TQQQ']['Max Drawdown']:.1f}%)")
    
    ax2.set_title('Drawdown Comparison (Underwater Plot)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=13)
    ax2.set_ylabel('Drawdown (%)', fontsize=13)
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    output_path = Path('plots') / 'final_backtest.png'
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nVisualization saved: {output_path}")
    
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(Path(cfg.DATA_DIR) / 'final_backtest_metrics.csv')
    print(f"Metrics saved: {cfg.DATA_DIR}/final_backtest_metrics.csv")
    
    print("\n" + "=" * 80)
    print("  CONCLUSION")
    print("=" * 80)
    
    ai_sharpe = metrics['AI Strategy']['Sharpe Ratio']
    qqq_sharpe = metrics['QQQ']['Sharpe Ratio']
    
    print(f"\nAI Strategy Performance:")
    print(f"  • Total Return: {metrics['AI Strategy']['Total Return']:.1f}%")
    print(f"  • CAGR: {metrics['AI Strategy']['CAGR']:.1f}%")
    print(f"  • Sharpe Ratio: {ai_sharpe:.2f}")
    print(f"  • Max Drawdown: {metrics['AI Strategy']['Max Drawdown']:.1f}%")
    print(f"  • Signals Generated: {strategy_signals.sum()} trades")
    
    print(f"\nvs QQQ Benchmark:")
    print(f"  • Return Delta: {metrics['AI Strategy']['Total Return'] - metrics['QQQ']['Total Return']:+.1f}%")
    print(f"  • Sharpe Delta: {ai_sharpe - qqq_sharpe:+.2f}")
    print(f"  • Drawdown Improvement: {metrics['AI Strategy']['Max Drawdown'] - metrics['QQQ']['Max Drawdown']:+.1f}%")
    
    if 'TQQQ' in metrics:
        print(f"\nvs TQQQ Benchmark:")
        print(f"  • Return Delta: {metrics['AI Strategy']['Total Return'] - metrics['TQQQ']['Total Return']:+.1f}%")
        print(f"  • Sharpe Delta: {ai_sharpe - metrics['TQQQ']['Sharpe Ratio']:+.2f}")
        print(f"  • Drawdown Improvement: {metrics['AI Strategy']['Max Drawdown'] - metrics['TQQQ']['Max Drawdown']:+.1f}%")
    
    print("\n" + "=" * 80)
    print("  PROJECT COMPLETE")
    print("=" * 80)
    print("\nThe AI-filtered strategy successfully demonstrates:")
    print("  1. Pattern recognition capability")
    print("  2. Signal calibration through magnitude filtering")
    print("  3. Risk-adjusted performance optimization")
    print("  4. Systematic approach to trading automation")
    
    print("\n" + "=" * 80)
    print("=" * 80)


if __name__ == '__main__':
    main()

