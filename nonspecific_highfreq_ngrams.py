import ijson
import ast
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
from scipy.stats import spearmanr
import sys


DATA_PATH = "data/seal_output.json"

def parse_ngrams(keys_str):
    if not keys_str:
        return []
    try:
        keys_list = ast.literal_eval(keys_str)
        return [(ngram, freq, score) for ngram, freq, score in keys_list]
    except:
        return []

def analyze_ngram_frequency():
    results = []

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for entry in ijson.items(f, 'item'):
            query = entry['question']
            positive_ids = {ctx['passage_id'] for ctx in entry.get('positive_ctxs', [])}

            ctxs = entry.get('ctxs', [])
            if not ctxs:
                continue

            # Collect top n queries
            top_passages = ctxs[:10]  # only consider top-10
            passage_ids = [ctx['passage_id'] for ctx in top_passages]

            # Success flags for top-1/top-2/top-10
            success_top1 = int(passage_ids[0] in positive_ids) if len(passage_ids) >= 1 else 0
            success_top2 = int(any(pid in positive_ids for pid in passage_ids[:2]))
            success_top10 = int(any(pid in positive_ids for pid in passage_ids[:10]))

            # N-grams from top-1 passage only
            ngrams = parse_ngrams(top_passages[0].get('keys', ''))
            if not ngrams:
                continue

            sorted_ngrams = sorted(ngrams, key=lambda x: x[2], reverse=True)
            top_5 = sorted_ngrams[:5]
            top_10 = sorted_ngrams[:10]

            frequencies = [freq for _, freq, _ in ngrams]
            top_5_freq = [ng[1] for ng in top_5]
            top_10_freq = [ng[1] for ng in top_10]

            results.append({
                'query': query,
                'success_top1': success_top1,
                'success_top2': success_top2,
                'success_top10': success_top10,
                'num_ngrams': len(ngrams),
                'avg_frequency_all': np.mean(frequencies),
                'median_frequency_all': np.median(frequencies),
                'avg_top5_frequency': np.mean(top_5_freq) if top_5_freq else 0,
                'avg_top10_frequency': np.mean(top_10_freq) if top_10_freq else 0,
            })

    df = pd.DataFrame(results)

    print(f"\nAnalyzed {len(df)} queries\n")

    print("Frequency Statistics (per-query averages):")
    print(f"  All n-grams mean frequency: {df['avg_frequency_all'].mean():.1f}")
    print(f"  Top-5 n-grams mean frequency: {df['avg_top5_frequency'].mean():.1f}")
    print(f"  Top-10 n-grams mean frequency: {df['avg_top10_frequency'].mean():.1f}")

    # Determine frequency bins dynamically
    freq_values = df['avg_top5_frequency']

    bins = {
        'low': freq_values < 500,
        'mid': (freq_values >= 500) & (freq_values < 5000),
        'high': freq_values >= 5000
    }

    for label, mask in bins.items():
        if mask.sum() == 0:
            continue  # skip empty bins
        bin_min = freq_values[mask].min()
        bin_max = freq_values[mask].max()
        print(f"\nSuccess Rates for {label} frequency (avg top-5 n-gram frequency: {bin_min:.0f}–{bin_max:.0f}):")
        print(f"  Top-1: {df.loc[mask, 'success_top1'].mean():.2%} ({mask.sum()} queries)")
        print(f"  Top-2: {df.loc[mask, 'success_top2'].mean():.2%} ({mask.sum()} queries)")
        print(f"  Top-10: {df.loc[mask, 'success_top10'].mean():.2%} ({mask.sum()} queries)")

    # Spearman correlation
    corr, p_val = spearmanr(df['avg_top5_frequency'], df['success_top1'])
    print(f"\nSpearman correlation (top-5 freq vs top-1 success): ρ={corr:.3f}, p={p_val:.4e}")

if __name__ == "__main__":
    analyze_ngram_frequency()