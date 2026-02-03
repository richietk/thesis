import ijson
import ast
import pandas as pd
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

def analyze_single_ngram_dominance(datapath="data/seal_output.json"):
    """Analyze score concentration in a single n-gram."""
    print("\n" + "="*80)
    print("ANALYSIS 6: SINGLE N-GRAM DOMINANCE")
    print("="*80)

    results = []

    with open(datapath, 'r', encoding='utf-8') as f:
        for entry in ijson.items(f, 'item'):
            query = entry['question']

            positive_ids = {ctx['passage_id'] for ctx in entry.get('positive_ctxs', [])}

            top_ctx = entry.get('ctxs', [None])[0]
            if not top_ctx:
                continue

            success_top1 = top_ctx['passage_id'] in positive_ids

            # N-grams from top-1 passage
            ngrams = parse_ngrams(top_ctx.get('keys', ''))
            if len(ngrams) <= 1:
                continue

            scores = [score for _, _, score in ngrams]
            total_score = sum(scores)
            if total_score <= 0:
                continue

            dominance = max(scores) / total_score

            results.append({
                'query': query,
                'dominance': dominance,
                'success_top1': success_top1,
                'num_ngrams': len(ngrams),
                'total_score': total_score,
                'max_score': max(scores)
            })

    df = pd.DataFrame(results)

    # Summary statistics
    mean_dom = df['dominance'].mean()
    median_dom = df['dominance'].median()
    print(f"\nAnalyzed {len(df)} queries")
    print(f"Mean dominance: {mean_dom:.3f}")
    print(f"Median dominance: {median_dom:.3f}")

    # Frequency bins
    low = df[df['dominance'] < 0.30]
    mid = df[(df['dominance'] >= 0.30) & (df['dominance'] <= 0.50)]
    high = df[df['dominance'] > 0.50]

    print("\nDominance bins:")
    print(f"  Low (<0.30): {len(low)} queries ({100*len(low)/len(df):.1f}%)")
    print(f"  Medium (0.30–0.50): {len(mid)} queries ({100*len(mid)/len(df):.1f}%)")
    print(f"  High (>0.50): {len(high)} queries ({100*len(high)/len(df):.1f}%)")

    # Decile analysis
    df_sorted = df.sort_values('dominance')
    num_bins = 10
    bin_size = len(df_sorted) // num_bins

    print("\nDominance range | Top-1 Success | Count")
    print("---------------------------------------")

    for i in range(num_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < num_bins - 1 else len(df_sorted)
        bin_data = df_sorted.iloc[start:end]
        if len(bin_data) == 0:
            continue

        t1 = 100 * bin_data['success_top1'].mean()
        d_min = bin_data['dominance'].min()
        d_max = bin_data['dominance'].max()

        print(f"{d_min:.2f}–{d_max:.2f}      | {t1:6.2f}% | {len(bin_data)}")

    # Spearman correlation
    corr, p_val = spearmanr(df['dominance'], df['success_top1'])
    print(f"\nSpearman correlation (dominance vs top-1 success): ρ = {corr:.3f}, p = {p_val:.4e}")


if __name__ == "__main__":
    datapath = sys.argv[1] if len(sys.argv) > 1 else 'data/seal_output.json'
    analyze_single_ngram_dominance(datapath)
