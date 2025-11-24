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


def run_leverage_backtest(qqq_prices, tqqq_prices, signals):
    equity = [10000.0]
    
    for i in range(len(signals) - 1):
        current_equity = equity[-1]
        
        if signals[i] == 1:
            ret = float((tqqq_prices[i+1] - tqqq_prices[i]) / tqqq_prices[i])
        else:
            ret = float((qqq_prices[i+1] - qqq_prices[i]) / qqq_prices[i])
        
        new_equity = float(current_equity * (1 + ret))
        equity.append(new_equity)
    
    return np.array(equity)


def main():
    cfg = Config()
    device = get_device()
    
    print("=" * 80)
    print("  STEP 18: LEVERAGE SWITCH BACKTEST (QQQ Base + TQQQ Boost)")
    print("=" * 80)
    
    print("\n[1/7] Loading QQQ and TQQQ Data...")
    downloader = YahooDownloader()
    qqq_dict = downloader.fetch_data([cfg.TARGET_SYMBOL], cfg.START_DATE, cfg.END_DATE)
    tqqq_dict = downloader.fetch_data(['TQQQ'], cfg.START_DATE, cfg.END_DATE)
    
    qqq_df = qqq_dict[cfg.TARGET_SYMBOL]
    tqqq_df = tqqq_dict.get('TQQQ')
    
    if tqqq_df is None:
        print("ERROR: TQQQ data unavailable")
        return
    
    processor = DataProcessor(seq_len=cfg.SEQ_LEN)
    sequences, prices = processor.make_sequences({cfg.TARGET_SYMBOL: qqq_df})
    prices = prices.flatten()
    normalized_sequences = processor.normalize_window(sequences)
    
    print("\n[2/7] Splitting Train (80%) / Test (20%)...")
    split_idx = int(len(normalized_sequences) * 0.8)
    
    data_train = normalized_sequences[:split_idx]
    data_test = normalized_sequences[split_idx:]
    prices_test = prices[split_idx:]
    
    test_start_idx = split_idx + cfg.SEQ_LEN - 1
    dates_test = qqq_df.index[test_start_idx:test_start_idx + len(prices_test)]
    
    print(f"Test period: {dates_test[0].date()} to {dates_test[-1].date()}")
    
    tqqq_aligned = tqqq_df.reindex(dates_test, method='ffill')
    tqqq_aligned = tqqq_aligned.fillna(method='ffill').fillna(method='bfill')
    tqqq_prices_test = tqqq_aligned['Close'].values
    
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
    
    print("\n[4/7] Identifying Top 3 Bullish Clusters...")
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
    
    print("\n[5/7] Calculating Signal Magnitudes...")
    magnitudes_train = calculate_l2_norms(embeddings_train)
    magnitudes_test = calculate_l2_norms(embeddings_test)
    
    mag_30 = np.percentile(magnitudes_train, 30)
    mag_75 = np.percentile(magnitudes_train, 75)
    
    print(f"Magnitude Thresholds: [{mag_30:.3f}, {mag_75:.3f}]")
    
    print("\n[6/7] Generating Leverage Signals...")
    signals = np.zeros(len(labels_test))
    
    for i in range(len(labels_test)):
        is_bullish = labels_test[i] in top_3_clusters
        is_confident = mag_30 <= magnitudes_test[i] <= mag_75
        
        if is_bullish and is_confident:
            signals[i] = 1
    
    print(f"TQQQ signals: {signals.sum()} days ({signals.sum()/len(signals)*100:.1f}%)")
    
    print("\n[7/7] Running Backtest...")
    hybrid_equity = run_leverage_backtest(prices_test, tqqq_prices_test, signals)
    qqq_equity = 10000 * (prices_test / prices_test[0])
    tqqq_equity = 10000 * (tqqq_prices_test / tqqq_prices_test[0])
    
    hybrid_return = float((hybrid_equity[-1] / hybrid_equity[0] - 1) * 100)
    qqq_return = float((qqq_equity[-1] / qqq_equity[0] - 1) * 100)
    tqqq_return = float((tqqq_equity[-1] / tqqq_equity[0] - 1) * 100)
    
    print("\n" + "=" * 80)
    print("  RESULTS")
    print("=" * 80)
    print(f"Hybrid Strategy (QQQ + TQQQ Boost): {hybrid_return:.2f}%")
    print(f"QQQ Buy & Hold:                     {qqq_return:.2f}%")
    print(f"TQQQ Buy & Hold:                    {tqqq_return:.2f}%")
    print("=" * 80)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(dates_test, qqq_equity, label=f'QQQ ({qqq_return:.1f}%)', 
            color='gray', alpha=0.7, linewidth=2)
    ax.plot(dates_test, tqqq_equity, label=f'TQQQ ({tqqq_return:.1f}%)', 
            color='orange', alpha=0.7, linewidth=2)
    ax.plot(dates_test, hybrid_equity, label=f'Hybrid Strategy ({hybrid_return:.1f}%)', 
            color='blue', linewidth=3)
    
    ax.set_title('Leverage Switch Strategy: QQQ Base + TQQQ Boost', 
                fontsize=16, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($)', fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    output_path = Path('plots') / 'leverage_switch_backtest.png'
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nPlot saved: {output_path}")
    
    results = pd.DataFrame({
        'Strategy': ['Hybrid (QQQ+TQQQ)', 'QQQ', 'TQQQ'],
        'Return': [hybrid_return, qqq_return, tqqq_return],
        'Final_Value': [hybrid_equity[-1], qqq_equity[-1], tqqq_equity[-1]]
    })
    
    results.to_csv(Path(cfg.DATA_DIR) / 'leverage_switch_metrics.csv', index=False)
    print(f"Metrics saved: {cfg.DATA_DIR}/leverage_switch_metrics.csv")


if __name__ == '__main__':
    main()

