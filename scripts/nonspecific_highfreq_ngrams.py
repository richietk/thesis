import ijson
import ast
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
from scipy.stats import spearmanr
import sys
import os


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
    if not keys_str:
        return []
    try:
        keys_list = ast.literal_eval(keys_str)
        return [(ngram, freq, score) for ngram, freq, score in keys_list]
    except:
        return []

def analyze_ngram_frequency(datapath="data/seal_output.json"):
    script_name = "nonspecific_highfreq_ngrams"
    print(f"running {script_name}")

    try:
        dataset_name = get_dataset_name(datapath)
        output_dir = f"generated_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Redirect stdout to log file
        log_file = os.path.join(output_dir, f"{script_name}_log.txt")
        original_stdout = sys.stdout
        sys.stdout = open(log_file, 'w', encoding='utf-8')

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
    analyze_ngram_frequency(datapath)