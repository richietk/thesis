import ijson
import ast
import pandas as pd
from collections import Counter
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

def analyze_repetitive_generation(datapath="data/seal_output.json"):
    """Analyze token diversity in generated n-grams."""
    script_name = "repetitive_tokens"
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
        print("ANALYSIS 5: REPETITIVE GENERATION")
        print("="*80)

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

                results.append({
                    'query': query,
                    'success': success,
                    'num_ngrams': len(ngrams),
                    'total_tokens': len(all_tokens),
                    'unique_tokens': len(unique_tokens),
                    'diversity_ratio': diversity_ratio,
                    'max_repetition': max_repetition
                })

        df = pd.DataFrame(results)

        # Check if dataframe is empty
        if len(df) == 0:
            print("\nNo data to analyze (empty dataframe)")
            sys.stdout.close()
            sys.stdout = original_stdout
            print(f"success running {script_name}")
            return

        # Statistics by diversity
        low_div = df[df['diversity_ratio'] < 0.70]
        mid_div = df[(df['diversity_ratio'] >= 0.70) & (df['diversity_ratio'] < 0.85)]
        high_div = df[df['diversity_ratio'] >= 0.85]

        print(f"\nAnalyzed {len(df)} queries")
        print(f"Mean diversity: {df['diversity_ratio'].mean():.3f}")
        print(f"\nSuccess Rate by Diversity:")
        print(f"  Low (<0.70): {low_div['success'].mean():.2%} ({len(low_div)} queries)")
        print(f"  Mid (0.70-0.85): {mid_div['success'].mean():.2%} ({len(mid_div)} queries)")
        print(f"  High (>0.85): {high_div['success'].mean():.2%} ({len(high_div)} queries)")

        corr, p_val = spearmanr(df['diversity_ratio'], df['success'])
        print(f"\nSpearman correlation: ρ = {corr:.3f}, p = {p_val:.4e}")

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
    analyze_repetitive_generation(datapath)
