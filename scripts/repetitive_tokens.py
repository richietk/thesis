import ijson
import json
import pandas as pd
from collections import Counter
from scipy.stats import spearmanr
import sys
import os
from scripts.utils.utils import get_dataset_name, strip_ngram_markers, parse_ngrams, calculate_retrieval_metrics

def analyze_repetitive_generation(datapath="data/seal_output.json"):
    """Analyze token diversity in generated n-grams."""
    script_name = "repetitive_tokens"
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

                top_ctx = entry.get('ctxs', [None])[0]
                if not top_ctx:
                    continue

                success = top_ctx['passage_id'] in positive_ids

                ngrams = parse_ngrams(top_ctx.get('keys', ''))
                if not ngrams:
                    continue

                all_ngram_text = ' '.join([strip_ngram_markers(ng[0], datapath) for ng in ngrams])
                all_tokens = all_ngram_text.split()
                if not all_tokens:
                    continue

                unique_tokens = set(all_tokens)
                diversity_ratio = len(unique_tokens) / len(all_tokens)

                token_counts = Counter(all_tokens)
                max_repetition = max(token_counts.values()) if token_counts else 0

                # Calculate retrieval metrics
                retrieved_ids = [ctx['passage_id'] for ctx in entry.get('ctxs', [])]
                metrics = calculate_retrieval_metrics(retrieved_ids, positive_ids)

                results.append({
                    'query': query,
                    'success': success,
                    'num_ngrams': len(ngrams),
                    'total_tokens': len(all_tokens),
                    'unique_tokens': len(unique_tokens),
                    'diversity_ratio': diversity_ratio,
                    'max_repetition': max_repetition,
                    'precision_at_1': metrics['precision_at_1'],
                    'r_precision': metrics['r_precision']
                })

        df = pd.DataFrame(results)

        # Check if dataframe is empty
        if len(df) == 0:
            print(f"success running {script_name}")
            return

        # Calculate statistics
        mean_diversity = float(df['diversity_ratio'].mean())

        # Create decile bins
        df['diversity_decile'] = pd.qcut(df['diversity_ratio'], q=10, labels=False, duplicates='drop')

        deciles_data = []
        for decile in sorted(df['diversity_decile'].unique()):
            mask = df['diversity_decile'] == decile
            if mask.sum() == 0:
                continue
            success_rate = df.loc[mask, 'success'].mean()
            div_min = df.loc[mask, 'diversity_ratio'].min()
            div_max = df.loc[mask, 'diversity_ratio'].max()
            count = mask.sum()

            deciles_data.append({
                "decile": int(decile) + 1,
                "diversity_min": float(div_min),
                "diversity_max": float(div_max),
                "success_pct": float(success_rate * 100),
                "count": int(count)
            })

        # Drop the temporary column
        df = df.drop(columns=['diversity_decile'])

        # Correlation
        corr, p_val = spearmanr(df['diversity_ratio'], df['success'])

        # Collect output data
        output_data = {
            "total_queries": len(df),
            "mean_diversity": mean_diversity,
            "precision_at_1": float(df['precision_at_1'].mean()),
            "r_precision": float(df['r_precision'].mean()),
            "deciles": deciles_data,
            "spearman_correlation": float(corr),
            "spearman_p_value": float(p_val)
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
    analyze_repetitive_generation(datapath)
