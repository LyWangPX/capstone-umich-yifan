# Author: Yifan Wang
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from config import Config
from data_loader import YahooDownloader
from processor import DataProcessor


def plot_cluster_centroid(sequences, cluster_id, title, save_path):
    centroid = sequences.mean(axis=0)
    std = sequences.std(axis=0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(centroid))
    
    ax.plot(x, centroid, linewidth=3, color='darkblue', label='Mean Pattern')
    ax.fill_between(x, centroid - std, centroid + std, alpha=0.3, color='lightblue', label='±1 Std Dev')
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Days (60-day window)', fontsize=13)
    ax.set_ylabel('Normalized Price', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_context_chart(qqq_df, cluster_df, target_cluster, year, save_path):
    year_mask = (qqq_df.index.year == year)
    qqq_year = qqq_df[year_mask]
    
    cluster_year = cluster_df[cluster_df['Date'].dt.year == year]
    cluster_dates = cluster_year[cluster_year['Cluster_ID'] == target_cluster]['Date']
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(qqq_year.index, qqq_year['Close'], linewidth=2, color='black', label='QQQ Close Price')
    
    for date in cluster_dates:
        if date in qqq_year.index:
            price = qqq_year.loc[date, 'Close']
            ax.scatter(date, price, color='red', s=50, alpha=0.6, zorder=5)
    
    ax.scatter([], [], color='red', s=50, label=f'Cluster {target_cluster} Signals')
    
    ax.set_title(f'QQQ {year} - Cluster {target_cluster} Occurrences', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=13)
    ax.set_ylabel('Price ($)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_comparison(bear_sequences, bull_sequences, save_path):
    bear_centroid = bear_sequences.mean(axis=0)
    bull_centroid = bull_sequences.mean(axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(bear_centroid))
    
    ax1.plot(x, bear_centroid, linewidth=3, color='red', label='Bearish Pattern')
    ax1.fill_between(x, bear_centroid - bear_sequences.std(axis=0), 
                     bear_centroid + bear_sequences.std(axis=0), alpha=0.3, color='lightcoral')
    ax1.set_title('Cluster 0 (Bearish): Avg Return -0.32%', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Days', fontsize=12)
    ax1.set_ylabel('Normalized Price', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(x, bull_centroid, linewidth=3, color='green', label='Bullish Pattern')
    ax2.fill_between(x, bull_centroid - bull_sequences.std(axis=0), 
                     bull_centroid + bull_sequences.std(axis=0), alpha=0.3, color='lightgreen')
    ax2.set_title('Cluster 8 (Bullish): Avg Return +0.69%, Win Rate 66%', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Days', fontsize=12)
    ax2.set_ylabel('Normalized Price', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    cfg = Config()
    
    print("=" * 60)
    print("QQQ Cluster Visualization for Report")
    print("=" * 60)
    
    output_dir = Path('plots/qqq_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/4] Loading QQQ Data...")
    downloader = YahooDownloader()
    data_dict = downloader.fetch_data([cfg.TARGET_SYMBOL], cfg.START_DATE, cfg.END_DATE)
    qqq_df = data_dict[cfg.TARGET_SYMBOL]
    
    print(f"QQQ: {len(qqq_df)} days loaded")
    
    print("\n[2/4] Processing Sequences...")
    processor = DataProcessor(seq_len=cfg.SEQ_LEN)
    sequences, raw_prices = processor.make_sequences({cfg.TARGET_SYMBOL: qqq_df})
    
    print(f"Sequences: {sequences.shape}")
    
    print("\n[3/4] Loading Cluster Assignments...")
    cluster_df = pd.read_csv(Path(cfg.DATA_DIR) / 'qqq_clusters.csv')
    cluster_df['Date'] = pd.to_datetime(cluster_df['Date'])
    
    cluster_labels = cluster_df['Cluster_ID'].values
    
    print(f"Cluster 0 occurrences: {(cluster_labels == 0).sum()}")
    print(f"Cluster 8 occurrences: {(cluster_labels == 8).sum()}")
    
    print("\n[4/4] Generating Visualizations...")
    
    cluster_0_mask = cluster_labels == 0
    cluster_8_mask = cluster_labels == 8
    
    cluster_0_sequences = sequences[cluster_0_mask][:, :, 0]
    cluster_8_sequences = sequences[cluster_8_mask][:, :, 0]
    
    plot_cluster_centroid(
        cluster_0_sequences, 
        0, 
        'Cluster 0 (Bearish): Average Pattern',
        output_dir / 'cluster_0_bearish.png'
    )
    
    plot_cluster_centroid(
        cluster_8_sequences, 
        8, 
        'Cluster 8 (Bullish): Average Pattern',
        output_dir / 'cluster_8_bullish.png'
    )
    
    plot_comparison(
        cluster_0_sequences,
        cluster_8_sequences,
        output_dir / 'bear_vs_bull_comparison.png'
    )
    
    plot_context_chart(
        qqq_df,
        cluster_df,
        target_cluster=0,
        year=2022,
        save_path=output_dir / 'qqq_2022_cluster0_context.png'
    )
    
    plot_context_chart(
        qqq_df,
        cluster_df,
        target_cluster=0,
        year=2021,
        save_path=output_dir / 'qqq_2021_cluster0_context.png'
    )
    
    plot_context_chart(
        qqq_df,
        cluster_df,
        target_cluster=8,
        year=2023,
        save_path=output_dir / 'qqq_2023_cluster8_context.png'
    )
    
    print("\n" + "=" * 60)
    print(f"All plots saved to {output_dir}/")
    print("=" * 60)
    
    print("\nGenerated Files:")
    print("  - cluster_0_bearish.png (Bearish pattern centroid)")
    print("  - cluster_8_bullish.png (Bullish pattern centroid)")
    print("  - bear_vs_bull_comparison.png (Side-by-side comparison)")
    print("  - qqq_2022_cluster0_context.png (Cluster 0 in 2022 bear market)")
    print("  - qqq_2021_cluster0_context.png (Cluster 0 in 2021)")
    print("  - qqq_2023_cluster8_context.png (Cluster 8 in 2023 recovery)")


if __name__ == '__main__':
    main()

