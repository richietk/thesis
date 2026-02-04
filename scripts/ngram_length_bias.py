import ijson
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import sys
import os
import matplotlib.pyplot as plt


def get_dataset_name(datapath: str) -> str:
    """Extract dataset name (seal or minder) from datapath."""
    if "minder" in datapath.lower():
        return "minder"
    elif "seal" in datapath.lower():
        return "seal"
    else:
        return os.path.splitext(os.path.basename(datapath))[0]


def strip_ngram_markers(ngram: str, datapath: str) -> str:
    """Strip pseudoquery markers from ngrams if using Minder data."""
    if "minder_output.json" in datapath:
        ngram = ngram.replace(" ||", "").strip()
    return ngram


def parse_ngrams(keys_str):
    """Parse n-gram keys from string format."""
    import ast
    if not keys_str:
        return []
    try:
        keys_list = ast.literal_eval(keys_str)
        return [(ngram, freq, score) for ngram, freq, score in keys_list]
    except:
        return []

def analyze_ngram_length_bias(datapath="data/seal_output.json"):
    """Analyze over-generation of unigrams vs multi-grams with bins and deciles."""
    script_name = "ngram_length_bias"
    print(f"running {script_name}")

    original_stdout = sys.stdout
    try:
        dataset_name = get_dataset_name(datapath)
        output_dir = f"generated_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Redirect stdout to log file
        log_file = os.path.join(output_dir, f"{script_name}_log.txt")
        sys.stdout = open(log_file, 'w', encoding='utf-8')

        print("\n" + "="*80)
        print("ANALYSIS: N-GRAM LENGTH BIAS")
        print("="*80)

        results = []

        with open(datapath, 'r', encoding='utf-8') as f:
            for entry in ijson.items(f, 'item'):
                query = entry['question']

                positive_ids = {ctx['passage_id'] for ctx in entry.get('positive_ctxs', [])}

                top_ctx = entry['ctxs'][0] if entry.get('ctxs') else None
                if not top_ctx:
                    continue

                success_top1 = top_ctx['passage_id'] in positive_ids

                ngrams = parse_ngrams(top_ctx.get('keys', ''))
                if not ngrams:
                    continue

                lengths = [len(strip_ngram_markers(ngram, datapath).strip().split()) for ngram, _, _ in ngrams]
                unigram_frac = sum(1 for l in lengths if l == 1) / len(lengths)
                avg_length = np.mean(lengths)

                results.append({
                    'query': query,
                    'unigram_frac': unigram_frac,
                    'success_top1': success_top1,
                    'num_ngrams': len(ngrams),
                    'avg_length': avg_length
                })

            df = pd.DataFrame(results)

        print(f"\nAnalyzed {len(df)} queries")
        print(f"Mean unigram fraction: {df['unigram_frac'].mean():.3f}")
        print(f"Median unigram fraction: {df['unigram_frac'].median():.3f}")
        print(f"Mean avg n-gram length: {df['avg_length'].mean():.2f}")
        print(f"Median avg n-gram length: {df['avg_length'].median():.2f}")

        # ------------------------------
        # Fixed-frequency bins
        # ------------------------------
        bins = {
            'Low (<0.7)': df['unigram_frac'] < 0.7,
            'Medium (0.7–0.9)': (df['unigram_frac'] >= 0.7) & (df['unigram_frac'] <= 0.9),
            'High (>0.9)': df['unigram_frac'] > 0.9
        }

        print("\nSuccess Rate by Unigram Fraction Bins (Top-1 passages):")
        for label, mask in bins.items():
            if mask.sum() == 0:
                continue
            mean_frac = df.loc[mask, 'unigram_frac'].mean()
            median_frac = df.loc[mask, 'unigram_frac'].median()
            success = df.loc[mask, 'success_top1'].mean() * 100
            print(f"  {label}: {mask.sum()} queries | mean frac: {mean_frac:.3f}, median frac: {median_frac:.3f} | success: {success:.2f}%")

        # ------------------------------
        # Decile analysis
        # ------------------------------
        df_sorted = df.sort_values('unigram_frac')
        num_bins = 10
        bin_size = len(df_sorted) // num_bins

        print("\nDecile Analysis of Unigram Fraction vs Success (Top-1):")
        print("Unigram frac range | Success % | Count")
        print("----------------------------------------")

        for i in range(num_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < num_bins - 1 else len(df_sorted)
            bin_data = df_sorted.iloc[start:end]
            t1 = 100 * bin_data['success_top1'].mean()
            u_min = bin_data['unigram_frac'].min()
            u_max = bin_data['unigram_frac'].max()
            print(f"{u_min:.2f}–{u_max:.2f}       | {t1:6.2f}% | {len(bin_data)}")

        # ------------------------------
        # Correlation
        # ------------------------------
        corr, p_val = spearmanr(df['unigram_frac'], df['success_top1'])
        print(f"\nSpearman correlation (unigram frac vs top-1 success): ρ = {corr:.3f}, p = {p_val:.4e}")

        # ------------------------------
        # Plot: Unigram fraction vs Top-1 success (deciles)
        # ------------------------------
        bin_centers = []
        success_rates = []

        for i in range(num_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < num_bins - 1 else len(df_sorted)
            bin_data = df_sorted.iloc[start:end]
            if len(bin_data) == 0:
                continue
            u_mean = bin_data['unigram_frac'].mean()
            t1 = 100 * bin_data['success_top1'].mean()
            bin_centers.append(u_mean)
            success_rates.append(t1)

        plt.figure(figsize=(8, 5))
        plt.plot(bin_centers, success_rates, marker='o', linestyle='-', linewidth=3, markersize=8)
        plt.xlabel("Mean Unigram Fraction (decile)", fontsize=16, fontweight='bold')
        plt.ylabel("Top-1 Success Rate (%)", fontsize=16, fontweight='bold')
        plt.title("Unigram Fraction vs Top-1 Success (Deciles)", fontsize=18, fontweight='bold')
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{script_name}_unigram_frac_vs_success.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Restore stdout
        sys.stdout.close()
        sys.stdout = original_stdout

        print(f"success running {script_name}")

    except Exception as e:
        # Restore stdout in case of error
        if sys.stdout != original_stdout:
            sys.stdout.close()
            sys.stdout = original_stdout
        print(f"error: running {script_name} {e}")
        raise

if __name__ == "__main__":
    datapath = sys.argv[1] if len(sys.argv) > 1 else 'data/seal_output.json'
    analyze_ngram_length_bias(datapath)



