# Author: Yifan Wang
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from config import Config
from models_vae import CnnVAE
from evaluator import ClusterEvaluator


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def generate_vae_embeddings(model, data, device, batch_size=64):
    embeddings = []
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).permute(0, 2, 1).to(device)
            mu, logvar = model.encode(batch_tensor)
            embeddings.append(mu.cpu().numpy())
    
    return np.vstack(embeddings)


def run_risk_shield_backtest(prices, labels, avoid_clusters):
    cash = 10000
    position = 0
    equity_curve = []
    trades = 0
    
    for i in range(len(prices)):
        current_price = prices[i]
        current_label = labels[i]
        
        if current_label in avoid_clusters:
            if position > 0:
                cash = position * current_price
                position = 0
                trades += 1
        else:
            if position == 0:
                position = cash / current_price
                cash = 0
                trades += 1
        
        portfolio_value = cash + (position * current_price)
        equity_curve.append(portfolio_value)
    
    return np.array(equity_curve), trades


def run_backtest_for_k(train_embeddings, train_prices, test_embeddings, test_prices, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(train_embeddings)
    
    train_labels = kmeans.labels_
    
    evaluator = ClusterEvaluator()
    stats_df = evaluator.calculate_forward_returns(train_prices, train_labels, lookahead=5)
    stats_df_sorted = stats_df.sort_values('Avg_Return', ascending=False)
    
    avoid_clusters = stats_df_sorted.tail(3)['Cluster_ID'].values
    
    test_labels = kmeans.predict(test_embeddings)
    
    strategy_equity, total_trades = run_risk_shield_backtest(test_prices, test_labels, avoid_clusters)
    
    strategy_return = float((strategy_equity[-1] / strategy_equity[0]) - 1)
    
    return strategy_return, total_trades


def main():
    cfg = Config()
    device = get_device()
    
    print("=" * 70)
    print("  Risk Shield Strategy: Avoid Bad Clusters, Else Stay Invested")
    print("=" * 70)
    
    print("\n[1/5] Loading Data...")
    train_data = np.load(Path(cfg.DATA_DIR) / 'train_data.npy')
    train_prices = np.load(Path(cfg.DATA_DIR) / 'train_prices.npy').flatten()
    
    print(f"Full dataset: {train_data.shape}")
    
    print("\n[2/5] Splitting into Train (80%) / Test (20%)...")
    split_idx = int(len(train_data) * 0.8)
    
    data_train = train_data[:split_idx]
    prices_train = train_prices[:split_idx]
    
    data_test = train_data[split_idx:]
    prices_test = train_prices[split_idx:]
    
    print(f"Train: {data_train.shape} (prices: {prices_train.shape})")
    print(f"Test:  {data_test.shape} (prices: {prices_test.shape})")
    
    print("\n[3/5] Loading VAE Model...")
    model = CnnVAE(
        input_dim=cfg.INPUT_DIM,
        seq_len=cfg.SEQ_LEN,
        latent_dim=cfg.LATENT_DIM
    ).to(device)
    
    checkpoint = torch.load('checkpoints/best_vae.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"VAE loaded from checkpoints/best_vae.pth")
    
    print("\n[4/5] Generating VAE Embeddings...")
    embeddings_train = generate_vae_embeddings(model, data_train, device)
    embeddings_test = generate_vae_embeddings(model, data_test, device)
    
    print(f"Train embeddings: {embeddings_train.shape}")
    print(f"Test embeddings: {embeddings_test.shape}")
    
    print("\n[5/5] Running Risk Shield Analysis (Test Set Only)...")
    k_values = [10, 15, 20, 25, 30]
    results = []
    
    buy_hold_return = float((prices_test[-1] / prices_test[0]) - 1)
    print(f"\nBuy & Hold Benchmark (Test Set): {buy_hold_return:.2%}")
    print("\nStrategy: Buy/Hold by default, Go to Cash only when in Bottom 3 clusters")
    print("-" * 70)
    
    for k in k_values:
        print(f"\nk = {k}...")
        strategy_return, total_trades = run_backtest_for_k(
            embeddings_train, prices_train, 
            embeddings_test, prices_test, 
            k
        )
        results.append({
            'k': k,
            'return': strategy_return,
            'trades': total_trades
        })
        outperformance = strategy_return - buy_hold_return
        print(f"  Test Set Return: {strategy_return:.2%}")
        print(f"  Outperformance: {outperformance:+.2%}")
        print(f"  Total Trades: {total_trades}")
    
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY (Risk Shield Strategy)")
    print("=" * 70)
    print(f"{'k':<10} {'Test Return':<20} {'vs Benchmark':<20} {'Trades':<10}")
    print("-" * 70)
    for r in results:
        outperf = r['return'] - buy_hold_return
        print(f"{r['k']:<10} {r['return']:<20.2%} {outperf:+20.2%} {r['trades']:<10}")
    print("-" * 70)
    print(f"{'Benchmark':<10} {buy_hold_return:<20.2%} {'—':<20} {'0':<10}")
    print("=" * 70)
    
    print("\n[6/6] Creating Visualization...")
# Plotting code generated with AI assistance
    fig, ax = plt.subplots(figsize=(12, 7))
    
    k_vals = [r['k'] for r in results]
    returns = [r['return'] * 100 for r in results]
    
    bars = ax.bar(k_vals, returns, width=3, color='steelblue', alpha=0.8, edgecolor='black')
    
    for i, (k_val, ret) in enumerate(zip(k_vals, returns)):
        outperf = ret - (buy_hold_return * 100)
        label_text = f"{ret:.1f}%\n({outperf:+.1f}%)"
        ax.text(k_val, ret + (max(returns) - min(returns)) * 0.02, 
                label_text, ha='center', fontweight='bold', fontsize=10)
    
    ax.axhline(y=buy_hold_return * 100, color='red', linestyle='--', linewidth=2, 
               label=f'Buy & Hold: {buy_hold_return:.1%}', alpha=0.7)
    
    ax.set_title('Risk Shield Strategy: Avoid Bad Clusters, Stay Invested Otherwise (Test Set)', 
                fontsize=13, fontweight='bold')
    ax.set_xlabel('Number of Clusters (k)', fontsize=13)
    ax.set_ylabel('Test Set Return (%)', fontsize=13)
    ax.set_xticks(k_vals)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    output_path = Path('plots') / 'sensitivity_k.png'
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nPlot saved to {output_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
