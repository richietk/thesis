import ijson
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import GPT2TokenizerFast
from utils.utils import get_dataset_name, strip_ngram_markers, parse_ngrams, calculate_retrieval_metrics

def analyze_query_ngram_overlap_topk(datapath="data/seal_output.json"):
    """Analyze token-level overlap between query and generated n-grams for top-1 and top-10."""
    script_name = "query_n_gram_overlap"
    print(f"running {script_name}")

    try:
        dataset_name = get_dataset_name(datapath)
        output_dir = f"generated_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Initialize tokenizer for token-level analysis
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

        results = []

        with open(datapath, 'r', encoding='utf-8') as f:
            for entry in ijson.items(f, 'item'):
                query = entry['question']
                query_tokens = set(tokenizer.encode(query.lower(), add_special_tokens=False))

                positive_ids = {ctx['passage_id'] for ctx in entry.get('positive_ctxs', [])}
                ctxs = entry.get('ctxs', [])
                if not ctxs:
                    continue

                topk = [1, 10]
                topk_ctxs = {
                    k: ctxs[:k] if len(ctxs) >= k else ctxs[:]  # handle fewer than k retrieved
                    for k in topk
                }

                # Calculate retrieval metrics
                retrieved_ids = [ctx['passage_id'] for ctx in ctxs]
                metrics = calculate_retrieval_metrics(retrieved_ids, positive_ids)

                entry_data = {
                    'query': query,
                    'num_query_tokens': len(query_tokens),
                    'precision_at_1': metrics['precision_at_1'],
                    'hits_at_10': metrics['hits_at_10'],
                    'r_precision': metrics['r_precision']
                }

                for k, passages in topk_ctxs.items():
                    # Combine n-grams from top-k passages
                    all_ngrams = []
                    for ctx in passages:
                        all_ngrams.extend(parse_ngrams(ctx.get('keys', '')))

                    ngram_text = ' '.join([strip_ngram_markers(ng[0], datapath).lower() for ng in all_ngrams])
                    ngram_tokens = set(tokenizer.encode(ngram_text, add_special_tokens=False))

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

        # Define equal-width coverage bins (0-20, 20-40, 40-60, 60-80, 80-100)
        topk = [1, 10]
        bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        # Collect output data
        output_data = {
            "total_queries": total_queries,
            "precision_at_1": float(df['precision_at_1'].mean()),
            "hits_at_10": float(df['hits_at_10'].mean()),
            "r_precision": float(df['r_precision'].mean())
        }

        # Bin by top1 coverage so both k=1 and k=10 metrics are computed over the same query groups
        df['coverage_bin'] = pd.cut(
            df['query_coverage_top1'],
            bins=bin_edges,
            labels=False,
            include_lowest=True
        )

        bins_data = []
        for bin_idx in range(len(bin_edges) - 1):
            mask = df['coverage_bin'] == bin_idx
            if mask.sum() == 0:
                continue
            count = mask.sum()
            bin_entry = {
                "bin": bin_idx + 1,
                "coverage_min": bin_edges[bin_idx],
                "coverage_max": bin_edges[bin_idx + 1],
                "count": int(count)
            }
            for k in topk:
                success_rate = df.loc[mask, f'success_top{k}'].mean()
                bin_entry[f"hits_at_{k}"] = float(success_rate * 100)
            bins_data.append(bin_entry)

        output_data["bins"] = bins_data

        for k in topk:
            # Spearman correlation
            corr, p_val = spearmanr(df[f'query_coverage_top{k}'], df[f'success_top{k}'])
            output_data[f"top{k}_spearman"] = {
                "spearman_correlation": float(corr),
                "spearman_p_value": float(p_val)
            }

        # Drop temporary column
        df = df.drop(columns=['coverage_bin'])

        # Write JSON output
        json_path = os.path.join(output_dir, f"{script_name}_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        print(f"success running {script_name}")

    except Exception as e:
        print(f"error: running {script_name} {e}")
        raise

DATASETS = [
    'data/seal_nq_output.json',
    'data/minder_nq_output.json',
    'data/minder_msmarco_output.json',
]

if __name__ == "__main__":
    for datapath in DATASETS:
        analyze_query_ngram_overlap_topk(datapath)