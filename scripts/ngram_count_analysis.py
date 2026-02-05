import ijson
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import ast
import os
from scripts.utils.utils import strip_ngram_markers, get_dataset_name, parse_ngrams

# OUTPUT_DIR will be set dynamically based on dataset_name

def analyze_ngram_distribution(datapath="data/seal_output.json"):
    import sys

    script_name = "ngram_count_analysis"
    print(f"running {script_name}")

    try:
        dataset_name = get_dataset_name(datapath)
        output_dir = f"generated_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

        results = []

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

        # Calculate statistics
        mean_count = float(df['ngram_count'].mean())
        median_count = float(df['ngram_count'].median())
        std_count = float(df['ngram_count'].std())
        min_count = int(df['ngram_count'].min())
        max_count = int(df['ngram_count'].max())

        # Create decile bins
        df['ngram_decile'] = pd.qcut(df['ngram_count'], q=10, labels=False, duplicates='drop')

        deciles_data = []
        for decile in sorted(df['ngram_decile'].unique()):
            mask = df['ngram_decile'] == decile
            count_n = mask.sum()
            if count_n > 0:
                success = df.loc[mask, 'success_top1'].mean() * 100
                mean_f = df.loc[mask, 'avg_freq'].mean()

                count_min = int(df.loc[mask, 'ngram_count'].min())
                count_max = int(df.loc[mask, 'ngram_count'].max())

                deciles_data.append({
                    "decile": int(decile) + 1,
                    "count_min": count_min,
                    "count_max": count_max,
                    "success_top1_pct": float(success),
                    "mean_corpus_freq": float(mean_f),
                    "count": int(count_n)
                })

        # Drop the temporary column
        df = df.drop(columns=['ngram_decile'])

        # Spearman Correlation
        rho_success, p_success = spearmanr(df['ngram_count'], df['success_top1'])
        rho_freq, p_freq = spearmanr(df['ngram_count'], df['avg_freq'])

        # Collect output data
        output_data = {
            "total_queries": len(df),
            "mean_count": mean_count,
            "median_count": median_count,
            "std_count": std_count,
            "min_count": min_count,
            "max_count": max_count,
            "deciles": deciles_data,
            "spearman_count_vs_success": float(rho_success),
            "spearman_count_vs_success_p": float(p_success),
            "spearman_count_vs_freq": float(rho_freq),
            "spearman_count_vs_freq_p": float(p_freq)
        }

        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ranges = [f"{d['count_min']}–{d['count_max']}" for d in deciles_data]
        rates = [d['success_top1_pct'] for d in deciles_data]
        freqs = [d['mean_corpus_freq'] for d in deciles_data]

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
        output_file = f"{output_dir}/ngram_distribution_analysis_{dataset_name}.png"
        plt.savefig(output_file, dpi=300)
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
    import sys
    datapath = sys.argv[1] if len(sys.argv) > 1 else 'data/seal_output.json'
    analyze_ngram_distribution(datapath)
