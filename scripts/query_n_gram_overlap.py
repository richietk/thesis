import ijson
import ast
import numpy as np
import pandas as pd
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
    """Parse n-gram keys from string format."""
    if not keys_str:
        return []
    try:
        keys_list = ast.literal_eval(keys_str)
        return [(ngram, freq, score) for ngram, freq, score in keys_list]
    except:
        return []

def analyze_query_ngram_overlap_topk(datapath="data/seal_output.json"):
    """Analyze lexical overlap between query and generated n-grams for top-1, top-2, and top-10."""
    script_name = "query_n_gram_overlap"
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
        print("ANALYSIS 3: QUERY-N-GRAM OVERLAP (TOP-K)")
        print("="*80)

        results = []

        with open(datapath, 'r', encoding='utf-8') as f:
            for entry in ijson.items(f, 'item'):
                query = entry['question']
                query_tokens = set(query.lower().split())

                positive_ids = {ctx['passage_id'] for ctx in entry.get('positive_ctxs', [])}
                ctxs = entry.get('ctxs', [])
                if not ctxs:
                    continue

                topk = [1, 2, 10]
                topk_ctxs = {
                    k: ctxs[:k] if len(ctxs) >= k else ctxs[:]  # handle fewer than k retrieved
                    for k in topk
                }

                entry_data = {'query': query, 'num_query_tokens': len(query_tokens)}

                for k, passages in topk_ctxs.items():
                    # Combine n-grams from top-k passages
                    all_ngrams = []
                    for ctx in passages:
                        all_ngrams.extend(parse_ngrams(ctx.get('keys', '')))

                    ngram_text = ' '.join([strip_ngram_markers(ng[0], datapath).lower() for ng in all_ngrams])
                    ngram_tokens = set(ngram_text.split())

                    intersection = query_tokens & ngram_tokens
                    union = query_tokens | ngram_tokens

                    query_coverage = len(intersection) / len(query_tokens) if query_tokens else 0
                    jaccard = len(intersection) / len(union) if union else 0

                    # Success: at least one positive passage in top-k
                    success_topk = int(any(ctx['passage_id'] in positive_ids for ctx in passages))

                    entry_data.update({
                        f'query_coverage_top{k}': query_coverage,
                        f'jaccard_top{k}': jaccard,
                        f'success_top{k}': success_topk
                    })

                results.append(entry_data)

        df = pd.DataFrame(results)
        total_queries = len(df)
        print(f"\nAnalyzed {total_queries} queries\n")

        # Define coverage bins
        bins = {
            'low': (0, 0.3),
            'mid': (0.3, 0.6),
            'high': (0.6, 1.01)  # Changed to 1.01 to include 1.0
        }

        topk = [1, 2, 10]
        for k in topk:
            print(f"\nSuccess Rate by Query Coverage (Top-{k} passages):")
            for label, (min_c, max_c) in bins.items():
                mask = (df[f'query_coverage_top{k}'] >= min_c) & (df[f'query_coverage_top{k}'] < max_c)
                if mask.sum() == 0:
                    continue
                success_rate = df.loc[mask, f'success_top{k}'].mean()
                print(f"  {label.capitalize()} ({min_c:.1f}-{max_c:.1f}): {success_rate:.2%} ({mask.sum()} queries)")

            # Spearman correlation
            corr, p_val = spearmanr(df[f'query_coverage_top{k}'], df[f'success_top{k}'])
            print(f"  Spearman correlation: ρ = {corr:.3f}, p = {p_val:.4e}")

        # Print distribution table
        print("\n" + "="*80)
        print("DISTRIBUTION OF QUERIES ACROSS COVERAGE BINS")
        print("="*80)
        print(f"\n{'Coverage Bin':<15} {'Top-1':<20} {'Top-2':<20} {'Top-10':<20}")
        print("-" * 80)

        for label, (min_c, max_c) in bins.items():
            row = f"{label.capitalize()} ({min_c:.1f}-{max_c:.1f})"
            print(f"{row:<15}", end=" ")

            for k in topk:
                mask = (df[f'query_coverage_top{k}'] >= min_c) & (df[f'query_coverage_top{k}'] < max_c)
                count = mask.sum()
                percentage = (count / total_queries) * 100
                print(f"{count:>5} ({percentage:>5.2f}%)", end="    ")

            print()
    
        print("-" * 80)
        print(f"{'Total':<15}", end=" ")
        for k in topk:
            total_in_bins = sum(
                ((df[f'query_coverage_top{k}'] >= min_c) & (df[f'query_coverage_top{k}'] < max_c)).sum()
                for label, (min_c, max_c) in bins.items()
            )
            percentage = (total_in_bins / total_queries) * 100
            print(f"{total_in_bins:>5} ({percentage:>5.2f}%)", end="    ")
        print()
        print("NOTE TO SELF: top2 and top10 are deprecated and potentially incorrect, remove TODO")

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
    analyze_query_ngram_overlap_topk(datapath)