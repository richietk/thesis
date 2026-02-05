import ijson
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import sys
import os
import matplotlib.pyplot as plt
from scripts.utils import get_dataset_name, strip_ngram_markers, parse_ngrams

def analyze_ngram_length_bias(datapath="data/seal_output.json"):
    """Analyze over-generation of unigrams vs multi-grams with bins and deciles."""
    script_name = "ngram_length_bias"
    print(f"running {script_name}")

    try:
        dataset_name = get_dataset_name(datapath)
        output_dir = f"generated_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

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

        # Check if dataframe is empty
        if len(df) == 0:
            print(f"success running {script_name}")
            return

        # Calculate statistics
        mean_unigram_frac = float(df['unigram_frac'].mean())
        median_unigram_frac = float(df['unigram_frac'].median())
        mean_avg_length = float(df['avg_length'].mean())
        median_avg_length = float(df['avg_length'].median())

        # Fixed-frequency bins
        bins_data = []
        bins = {
            'Low (<0.7)': df['unigram_frac'] < 0.7,
            'Medium (0.7–0.9)': (df['unigram_frac'] >= 0.7) & (df['unigram_frac'] <= 0.9),
            'High (>0.9)': df['unigram_frac'] > 0.9
        }

        for label, mask in bins.items():
            if mask.sum() == 0:
                continue
            bins_data.append({
                "bin": label,
                "count": int(mask.sum()),
                "mean_unigram_frac": float(df.loc[mask, 'unigram_frac'].mean()),
                "median_unigram_frac": float(df.loc[mask, 'unigram_frac'].median()),
                "success_top1_pct": float(df.loc[mask, 'success_top1'].mean() * 100)
            })

        # Decile analysis
        df_sorted = df.sort_values('unigram_frac')
        num_bins = 10
        bin_size = len(df_sorted) // num_bins

        deciles_data = []
        for i in range(num_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < num_bins - 1 else len(df_sorted)
            bin_data = df_sorted.iloc[start:end]
            deciles_data.append({
                "decile": i + 1,
                "unigram_frac_min": float(bin_data['unigram_frac'].min()),
                "unigram_frac_max": float(bin_data['unigram_frac'].max()),
                "success_top1_pct": float(bin_data['success_top1'].mean() * 100),
                "count": len(bin_data)
            })

        # Correlation
        corr, p_val = spearmanr(df['unigram_frac'], df['success_top1'])

        # Collect output data
        output_data = {
            "total_queries": len(df),
            "mean_unigram_frac": mean_unigram_frac,
            "median_unigram_frac": median_unigram_frac,
            "mean_avg_length": mean_avg_length,
            "median_avg_length": median_avg_length,
            "bins": bins_data,
            "deciles": deciles_data,
            "spearman_correlation": float(corr),
            "spearman_p_value": float(p_val)
        }

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

        # Write JSON output
        json_path = os.path.join(output_dir, f"{script_name}_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        print(f"success running {script_name}")

    except Exception as e:
        print(f"error: running {script_name} {e}")
        raise

if __name__ == "__main__":
    datapath = sys.argv[1] if len(sys.argv) > 1 else 'data/seal_output.json'
    analyze_ngram_length_bias(datapath)



