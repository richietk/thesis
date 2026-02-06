import ijson
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
from scipy.stats import spearmanr
import sys
import os
from utils.utils import get_dataset_name, strip_ngram_markers, parse_ngrams, calculate_retrieval_metrics

def analyze_ngram_frequency(datapath="data/seal_output.json"):
    script_name = "nonspecific_highfreq_ngrams"
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

                # Calculate retrieval metrics
                retrieved_ids = [ctx['passage_id'] for ctx in ctxs]
                metrics = calculate_retrieval_metrics(retrieved_ids, positive_ids)

                results.append({
                    'query': query,
                    'hits@1': success_top1,
                    'hits@2': success_top2,
                    'hits@10': success_top10,
                    'num_ngrams': len(ngrams),
                    'avg_frequency_all': np.mean(frequencies),
                    'median_frequency_all': np.median(frequencies),
                    'avg_top5_frequency': np.mean(top_5_freq) if top_5_freq else 0,
                    'avg_top10_frequency': np.mean(top_10_freq) if top_10_freq else 0,
                    'precision_at_1': metrics['precision_at_1'],
                    'r_precision': metrics['r_precision']
                })

        df = pd.DataFrame(results)

        # Check if dataframe is empty
        if len(df) == 0:
            print(f"success running {script_name}")
            return

        # Calculate statistics
        avg_freq_all = float(df['avg_frequency_all'].mean())
        avg_freq_top5 = float(df['avg_top5_frequency'].mean())
        avg_freq_top10 = float(df['avg_top10_frequency'].mean())

        # Helper function for decile analysis
        def compute_deciles(df, freq_column, temp_col_name):
            freq_values = df[freq_column]
            df[temp_col_name] = pd.qcut(freq_values, q=10, labels=False, duplicates='drop')

            deciles_data = []
            for decile in sorted(df[temp_col_name].unique()):
                mask = df[temp_col_name] == decile
                if mask.sum() == 0:
                    continue
                bin_min = freq_values[mask].min()
                bin_max = freq_values[mask].max()
                top1_rate = df.loc[mask, 'hits@1'].mean()
                top2_rate = df.loc[mask, 'hits@2'].mean()
                top10_rate = df.loc[mask, 'hits@10'].mean()
                count = mask.sum()

                deciles_data.append({
                    "decile": int(decile) + 1,
                    "freq_min": float(bin_min),
                    "freq_max": float(bin_max),
                    "hits@1_pct": float(top1_rate * 100),
                    "hits@2_pct": float(top2_rate * 100),
                    "hits@10_pct": float(top10_rate * 100),
                    "count": int(count)
                })

            df = df.drop(columns=[temp_col_name])
            return deciles_data, df

        # Decile-based frequency analysis for top-5 n-grams
        deciles_top5, df = compute_deciles(df, 'avg_top5_frequency', 'freq_decile_top5')

        # Decile-based frequency analysis for all n-grams
        deciles_all, df = compute_deciles(df, 'avg_frequency_all', 'freq_decile_all')

        # Spearman correlations
        corr_top5, p_val_top5 = spearmanr(df['avg_top5_frequency'], df['hits@1'])
        corr_all, p_val_all = spearmanr(df['avg_frequency_all'], df['hits@1'])

        # Collect output data
        output_data = {
            "total_queries": len(df),
            "avg_frequency_all": avg_freq_all,
            "avg_frequency_top5": avg_freq_top5,
            "avg_frequency_top10": avg_freq_top10,
            "precision_at_1": float(df['precision_at_1'].mean()),
            "r_precision": float(df['r_precision'].mean()),
            "deciles_top5": deciles_top5,
            "deciles_all": deciles_all,
            "spearman_top5_vs_hit1": {
                "correlation": float(corr_top5),
                "p_value": float(p_val_top5)
            },
            "spearman_all_vs_hit1": {
                "correlation": float(corr_all),
                "p_value": float(p_val_all)
            }
        }

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
    analyze_ngram_frequency(datapath)