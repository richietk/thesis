import ijson
import ast
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
    """Parse n-gram keys from string or list format, handling Decimal objects."""
    if not keys_str:
        return []
    try:
        # If it's already a list, use it directly
        if isinstance(keys_str, list):
            keys_list = keys_str
        else:
            # Try to parse string format
            keys_list = ast.literal_eval(keys_str)

        # Convert Decimal to float for all score values
        result = []
        for ngram, freq, score in keys_list:
            result.append((ngram, int(freq), float(score)))
        return result
    except:
        return []

def analyze_single_ngram_dominance(datapath="data/seal_output.json"):
    """Analyze score concentration in a single n-gram."""
    script_name = "single_ngram_dominance"
    print(f"running {script_name}")

    try:
        dataset_name = get_dataset_name(datapath)
        output_dir = f"generated_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Redirect stdout to log file
        log_file = os.path.join(output_dir, f"{script_name}_log.txt")
        original_stdout = sys.stdout
        sys.stdout = open(log_file, 'w', encoding='utf-8')

        print("\n" + "="*80)
        print("ANALYSIS 6: SINGLE N-GRAM DOMINANCE")
        print("="*80)

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
            print("\nNo data to analyze (empty dataframe)")
            sys.stdout.close()
            sys.stdout = original_stdout
            print(f"success running {script_name}")
            return

        # Summary statistics
        mean_dom = df['dominance'].mean()
        median_dom = df['dominance'].median()
        print(f"\nAnalyzed {len(df)} queries")
        print(f"Mean dominance: {mean_dom:.3f}")
        print(f"Median dominance: {median_dom:.3f}")

        # Decile analysis using pd.qcut for proper percentile-based binning
        print("\nSuccess Rate by Dominance Decile:")
        print("-" * 70)
        print(f"{'Decile':8s} | {'Dominance Range':18s} | {'Success':>10s} | {'Count':>8s}")
        print("-" * 70)

        # Create decile bins
        df['dominance_decile'] = pd.qcut(df['dominance'], q=10, labels=False, duplicates='drop')

        for decile in sorted(df['dominance_decile'].unique()):
            mask = df['dominance_decile'] == decile
            if mask.sum() == 0:
                continue

            success_rate = df.loc[mask, 'success_top1'].mean()
            dom_min = df.loc[mask, 'dominance'].min()
            dom_max = df.loc[mask, 'dominance'].max()
            count = mask.sum()

            print(f"D{int(decile)+1:1d}       | {dom_min:6.3f}–{dom_max:6.3f}      | {success_rate:9.2%} | {count:8d}")

        print("-" * 70)

        # Drop the temporary column
        df = df.drop(columns=['dominance_decile'])

        # Spearman correlation
        corr, p_val = spearmanr(df['dominance'], df['success_top1'])
        print(f"\nSpearman correlation (dominance vs top-1 success): ρ = {corr:.3f}, p = {p_val:.4e}")

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
    analyze_single_ngram_dominance(datapath)
