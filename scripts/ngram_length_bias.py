import ijson
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import sys
import os
import matplotlib.pyplot as plt
from utils.utils import get_dataset_name, strip_ngram_markers, parse_ngrams, calculate_retrieval_metrics

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

                hits_at_1 = top_ctx['passage_id'] in positive_ids

                # Hits@2 and Hits@10
                top2_ctxs = entry['ctxs'][:2] if len(entry.get('ctxs', [])) >= 2 else entry.get('ctxs', [])
                hits_at_2 = any(ctx['passage_id'] in positive_ids for ctx in top2_ctxs)

                top10_ctxs = entry['ctxs'][:10] if len(entry.get('ctxs', [])) >= 10 else entry.get('ctxs', [])
                hits_at_10 = any(ctx['passage_id'] in positive_ids for ctx in top10_ctxs)

                ngrams = parse_ngrams(top_ctx.get('keys', ''))
                if not ngrams:
                    continue

                lengths = [len(strip_ngram_markers(ngram, datapath).strip().split()) for ngram, _, _ in ngrams]
                unigram_frac = sum(1 for l in lengths if l == 1) / len(lengths)
                avg_length = np.mean(lengths)

                # Calculate retrieval metrics
                retrieved_ids = [ctx['passage_id'] for ctx in entry.get('ctxs', [])]
                metrics = calculate_retrieval_metrics(retrieved_ids, positive_ids)

                results.append({
                    'query': query,
                    'unigram_frac': unigram_frac,
                    'hits_at_1': hits_at_1,
                    'hits_at_2': hits_at_2,
                    'hits_at_10': hits_at_10,
                    'num_ngrams': len(ngrams),
                    'avg_length': avg_length,
                    'precision_at_1': metrics['precision_at_1'],
                    'r_precision': metrics['r_precision']
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
                "hits_at_1_pct": float(bin_data['hits_at_1'].mean() * 100),
                "hits_at_2_pct": float(bin_data['hits_at_2'].mean() * 100),
                "hits_at_10_pct": float(bin_data['hits_at_10'].mean() * 100),
                "count": len(bin_data)
            })

        # Correlations
        corr_hits1, p_val_hits1 = spearmanr(df['unigram_frac'], df['hits_at_1'])
        corr_hits10, p_val_hits10 = spearmanr(df['unigram_frac'], df['hits_at_10'])

        # Collect output data
        output_data = {
            "total_queries": len(df),
            "mean_unigram_frac": mean_unigram_frac,
            "median_unigram_frac": median_unigram_frac,
            "mean_avg_length": mean_avg_length,
            "median_avg_length": median_avg_length,
            "hits_at_1": float(df['hits_at_1'].mean()),
            "hits_at_2": float(df['hits_at_2'].mean()),
            "hits_at_10": float(df['hits_at_10'].mean()),
            "precision_at_1": float(df['precision_at_1'].mean()),
            "r_precision": float(df['r_precision'].mean()),
            "deciles": deciles_data,
            "spearman_correlation_hits_at_1": float(corr_hits1),
            "spearman_p_value_hits_at_1": float(p_val_hits1),
            "spearman_correlation_hits_at_10": float(corr_hits10),
            "spearman_p_value_hits_at_10": float(p_val_hits10)
        }

        # ------------------------------
        # Plot: Unigram fraction vs Hits@1 (deciles)
        # ------------------------------
        bin_centers = []
        hits_at_1_rates = []

        for i in range(num_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < num_bins - 1 else len(df_sorted)
            bin_data = df_sorted.iloc[start:end]
            if len(bin_data) == 0:
                continue
            u_mean = bin_data['unigram_frac'].mean()
            t1 = 100 * bin_data['hits_at_1'].mean()
            bin_centers.append(u_mean)
            hits_at_1_rates.append(t1)

        plt.figure(figsize=(8, 5))
        plt.plot(bin_centers, hits_at_1_rates, marker='o', linestyle='-', linewidth=3, markersize=8)
        plt.xlabel("Mean Unigram Fraction (decile)", fontsize=16, fontweight='bold')
        plt.ylabel("Hits@1 (%)", fontsize=16, fontweight='bold')
        plt.title("Unigram Fraction vs Hits@1 (Deciles)", fontsize=18, fontweight='bold')
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{script_name}_unigram_frac_vs_hits_at_1.png")
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



