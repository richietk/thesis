import ijson
import ast
import pandas as pd
from collections import Counter
from scipy.stats import spearmanr
import sys

def strip_ngram_markers(ngram: str, datapath: str) -> str:
    """Strip pseudoquery markers from ngrams if using Minder data."""
    if "minder_output.json" in datapath:
        ngram = ngram.replace(" ||", "").strip()
    return ngram

def parse_ngrams(keys_str):
    if not keys_str:
        return []
    try:
        keys_list = ast.literal_eval(keys_str)
        return [(ngram, freq, score) for ngram, freq, score in keys_list]
    except:
        return []

def analyze_repetitive_generation(datapath="data/seal_output.json"):
    """Analyze token diversity in generated n-grams."""
    print("\n" + "="*80)
    print("ANALYSIS 5: REPETITIVE GENERATION")
    print("="*80)

    results = []

    with open(datapath, 'r', encoding='utf-8') as f:
        for entry in ijson.items(f, 'item'):
            query = entry['question']

            positive_ids = {ctx['passage_id'] for ctx in entry.get('positive_ctxs', [])}

            top_ctx = entry.get('ctxs', [None])[0]
            if not top_ctx:
                continue

            success = top_ctx['passage_id'] in positive_ids

            ngrams = parse_ngrams(top_ctx.get('keys', ''))
            if not ngrams:
                continue

            all_ngram_text = ' '.join([strip_ngram_markers(ng[0], datapath) for ng in ngrams])
            all_tokens = all_ngram_text.split()
            if not all_tokens:
                continue

            unique_tokens = set(all_tokens)
            diversity_ratio = len(unique_tokens) / len(all_tokens)

            token_counts = Counter(all_tokens)
            max_repetition = max(token_counts.values()) if token_counts else 0

            results.append({
                'query': query,
                'success': success,
                'num_ngrams': len(ngrams),
                'total_tokens': len(all_tokens),
                'unique_tokens': len(unique_tokens),
                'diversity_ratio': diversity_ratio,
                'max_repetition': max_repetition
            })

    df = pd.DataFrame(results)

    # Statistics by diversity
    low_div = df[df['diversity_ratio'] < 0.70]
    mid_div = df[(df['diversity_ratio'] >= 0.70) & (df['diversity_ratio'] < 0.85)]
    high_div = df[df['diversity_ratio'] >= 0.85]

    print(f"\nAnalyzed {len(df)} queries")
    print(f"Mean diversity: {df['diversity_ratio'].mean():.3f}")
    print(f"\nSuccess Rate by Diversity:")
    print(f"  Low (<0.70): {low_div['success'].mean():.2%} ({len(low_div)} queries)")
    print(f"  Mid (0.70-0.85): {mid_div['success'].mean():.2%} ({len(mid_div)} queries)")
    print(f"  High (>0.85): {high_div['success'].mean():.2%} ({len(high_div)} queries)")

    corr, p_val = spearmanr(df['diversity_ratio'], df['success'])
    print(f"\nSpearman correlation: ρ = {corr:.3f}, p = {p_val:.4e}")

if __name__ == "__main__":
    analyze_repetitive_generation()
