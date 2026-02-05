import ijson
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import sys
import os
from scripts.utils.utils import get_dataset_name, strip_ngram_markers, parse_ngrams

def analyze_query_ngram_overlap_topk(datapath="data/seal_output.json"):
    """Analyze lexical overlap between query and generated n-grams for top-1, top-2, and top-10."""
    script_name = "query_n_gram_overlap"
    print(f"running {script_name}")

    try:
        dataset_name = get_dataset_name(datapath)
        output_dir = f"generated_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

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

        # Define decile-based coverage bins
        topk = [1, 2, 10]

        # Collect output data
        output_data = {
            "total_queries": total_queries
        }

        for k in topk:
            # Create decile bins for this k
            df[f'coverage_decile_top{k}'] = pd.qcut(df[f'query_coverage_top{k}'], q=10, labels=False, duplicates='drop')

            deciles_data = []
            for decile in sorted(df[f'coverage_decile_top{k}'].unique()):
                mask = df[f'coverage_decile_top{k}'] == decile
                if mask.sum() == 0:
                    continue
                success_rate = df.loc[mask, f'success_top{k}'].mean()
                cov_min = df.loc[mask, f'query_coverage_top{k}'].min()
                cov_max = df.loc[mask, f'query_coverage_top{k}'].max()
                count = mask.sum()

                deciles_data.append({
                    "decile": int(decile) + 1,
                    "coverage_min": float(cov_min),
                    "coverage_max": float(cov_max),
                    "success_pct": float(success_rate * 100),
                    "count": int(count)
                })

            # Spearman correlation
            corr, p_val = spearmanr(df[f'query_coverage_top{k}'], df[f'success_top{k}'])

            output_data[f"top{k}"] = {
                "deciles": deciles_data,
                "spearman_correlation": float(corr),
                "spearman_p_value": float(p_val)
            }

        # Drop temporary columns
        for k in topk:
            if f'coverage_decile_top{k}' in df.columns:
                df = df.drop(columns=[f'coverage_decile_top{k}'])

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
    analyze_query_ngram_overlap_topk(datapath)