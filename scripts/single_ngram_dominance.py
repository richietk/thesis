import ijson
import json
import pandas as pd
from scipy.stats import spearmanr
import sys
import os
from scripts.utils import get_dataset_name, strip_ngram_markers, parse_ngrams

def analyze_single_ngram_dominance(datapath="data/seal_output.json"):
    """Analyze score concentration in a single n-gram."""
    script_name = "single_ngram_dominance"
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

                success_top1 = top_ctx['passage_id'] in positive_ids

                # N-grams from top-1 passage
                ngrams = parse_ngrams(top_ctx.get('keys', ''))
                if len(ngrams) <= 1:
                    continue

                scores = [score for _, _, score in ngrams]
                total_score = sum(scores)
                if total_score <= 0:
                    continue

                dominance = max(scores) / total_score

                results.append({
                    'query': query,
                    'dominance': dominance,
                    'success_top1': success_top1,
                    'num_ngrams': len(ngrams),
                    'total_score': total_score,
                    'max_score': max(scores)
                })

        df = pd.DataFrame(results)

        # Check if dataframe is empty
        if len(df) == 0:
            print(f"success running {script_name}")
            return

        # Summary statistics
        mean_dom = float(df['dominance'].mean())
        median_dom = float(df['dominance'].median())

        # Create decile bins
        df['dominance_decile'] = pd.qcut(df['dominance'], q=10, labels=False, duplicates='drop')

        deciles_data = []
        for decile in sorted(df['dominance_decile'].unique()):
            mask = df['dominance_decile'] == decile
            if mask.sum() == 0:
                continue

            success_rate = df.loc[mask, 'success_top1'].mean()
            dom_min = df.loc[mask, 'dominance'].min()
            dom_max = df.loc[mask, 'dominance'].max()
            count = mask.sum()

            deciles_data.append({
                "decile": int(decile) + 1,
                "dominance_min": float(dom_min),
                "dominance_max": float(dom_max),
                "success_top1_pct": float(success_rate * 100),
                "count": int(count)
            })

        # Drop the temporary column
        df = df.drop(columns=['dominance_decile'])

        # Spearman correlation
        corr, p_val = spearmanr(df['dominance'], df['success_top1'])

        # Collect output data
        output_data = {
            "total_queries": len(df),
            "mean_dominance": mean_dom,
            "median_dominance": median_dom,
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
    analyze_single_ngram_dominance(datapath)
