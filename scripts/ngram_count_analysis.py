import ijson
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import ast
import os

OUTPUT_DIR = "generated_data"

def strip_ngram_markers(ngram: str, datapath: str) -> str:
    """Strip pseudoquery markers from ngrams if using Minder data."""
    if "minder_output.json" in datapath:
        ngram = ngram.replace(" ||", "").strip()
    return ngram

def parse_ngrams(keys_str):
    if not keys_str: return []
    try:
        # Expected format: [(ngram, freq, score), ...]
        return ast.literal_eval(keys_str)
    except: return []

def analyze_ngram_distribution(datapath="data/seal_output.json"):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    results = []

    print(f"Loading {datapath}...")
    with open(datapath, 'r', encoding='utf-8') as f:
        for entry in ijson.items(f, 'item'):
            top_ctx = entry['ctxs'][0] if entry.get('ctxs') else None
            if not top_ctx: continue

            positive_ids = {ctx['passage_id'] for ctx in entry.get('positive_ctxs', [])}
            success_top1 = top_ctx['passage_id'] in positive_ids
            ngrams = parse_ngrams(top_ctx.get('keys', ''))
            
            count = len(ngrams)
            # Corpus Frequency is index 1 in the (ngram, freq, score) tuple
            avg_freq = np.mean([n[1] for n in ngrams]) if ngrams else 0

            results.append({
                'ngram_count': count,
                'avg_freq': avg_freq,
                'success_top1': success_top1,
            })

    df = pd.DataFrame(results)

    # ================================================================================
    # GLOBAL STATISTICS BLOCK
    # ================================================================================
    print("\n" + "=" * 80)
    print("EMPIRICAL ANALYSIS: N-GRAM COUNT VS. RETRIEVAL EFFECTIVENESS")
    print("=" * 80)
    
    print(f"\nSummary Statistics (N={len(df)}):")
    print(f"  Mean Count:   {df['ngram_count'].mean():.2f}")
    print(f"  Median:       {df['ngram_count'].median():.2f}")
    print(f"  Std. Dev:     {df['ngram_count'].std():.2f}")
    print(f"  Range:        [{df['ngram_count'].min()}, {df['ngram_count'].max()}]")

    # Define Intervals
    intervals = [
        (-np.inf, 30),
        (30, 38),
        (38, 44),
        (44, 52),
        (52, np.inf)
    ]

    print("\nRetrieval Performance by N-gram Count Interval:")
    print("-" * 75)
    header = f"{'Count Interval':18s} | {'N':8s} | {'Success@1':>10s} | {'Mean Corpus Freq'}"
    print(header)
    print("-" * 75)

    bin_data = []
    for lower, upper in intervals:
        mask = (df['ngram_count'] > lower) & (df['ngram_count'] <= upper)
        count_n = mask.sum()
        if count_n > 0:
            success = df.loc[mask, 'success_top1'].mean() * 100
            mean_f = df.loc[mask, 'avg_freq'].mean()
            
            low_val = int(lower) if lower != -np.inf else 0
            range_label = f">{low_val}" if upper == np.inf else f"{low_val + 1}–{int(upper)}"
                
            print(f" {range_label:17s} | {count_n:8d} | {success:9.2f}% | {mean_f:,.0f}")
            bin_data.append({'range': range_label, 'success': success, 'freq': mean_f})

    # Spearman Correlation
    rho_success, p_success = spearmanr(df['ngram_count'], df['success_top1'])
    rho_freq, p_freq = spearmanr(df['ngram_count'], df['avg_freq'])

    print("-" * 75)
    print(f"Spearman ρ (Count vs. Success): {rho_success:.3f} (p={p_success:.2e})")
    print(f"Spearman ρ (Count vs. Frequency): {rho_freq:.3f} (p={p_freq:.2e})")

    # ================================================================================
    # VISUALIZATION BLOCK
    # ================================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ranges = [b['range'] for b in bin_data]
    rates = [b['success'] for b in bin_data]
    freqs = [b['freq'] for b in bin_data]

    # Success Plot
    ax1.bar(ranges, rates, color='skyblue', edgecolor='navy')
    ax1.set_ylabel("Top-1 Success Rate (%)", fontweight='bold')
    ax1.set_xlabel("N-gram Count Interval", fontweight='bold')
    ax1.set_title("Retrieval Effectiveness by Identifier Count", fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Frequency Plot
    ax2.bar(ranges, freqs, color='salmon', edgecolor='darkred')
    ax2.set_ylabel(r"Mean Corpus Frequency ($\bar{F}_c$)", fontweight='bold')
    ax2.set_xlabel("N-gram Count Interval", fontweight='bold')
    ax2.set_title("Mean Specificity by Identifier Count", fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ngram_distribution_analysis.png", dpi=300)
    print(f"\nPlots saved to {OUTPUT_DIR}/ngram_distribution_analysis.png")
    plt.show()

if __name__ == "__main__":
    import sys
    datapath = sys.argv[1] if len(sys.argv) > 1 else 'data/seal_output.json'
    analyze_ngram_distribution(datapath)
