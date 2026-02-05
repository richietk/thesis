import ijson
import pandas as pd
from collections import Counter
from scipy.stats import spearmanr
import sys
import os
from utils import get_dataset_name, strip_ngram_markers, parse_ngrams

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

        # Statistics by diversity deciles
        print(f"\nAnalyzed {len(df)} queries")
        print(f"Mean diversity: {df['diversity_ratio'].mean():.3f}")

        print(f"\nSuccess Rate by Diversity Decile:")
        print("-" * 70)
        print(f"{'Decile':8s} | {'Diversity Range':18s} | {'Success':>10s} | {'Count':>8s}")
        print("-" * 70)

        # Create decile bins
        df['diversity_decile'] = pd.qcut(df['diversity_ratio'], q=10, labels=False, duplicates='drop')

        for decile in sorted(df['diversity_decile'].unique()):
            mask = df['diversity_decile'] == decile
            if mask.sum() == 0:
                continue
            success_rate = df.loc[mask, 'success'].mean()
            div_min = df.loc[mask, 'diversity_ratio'].min()
            div_max = df.loc[mask, 'diversity_ratio'].max()
            count = mask.sum()

            print(f"D{int(decile)+1:1d}       | {div_min:6.3f}–{div_max:6.3f}      | {success_rate:9.2%} | {count:8d}")

        print("-" * 70)

        # Drop the temporary column
        df = df.drop(columns=['diversity_decile'])

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
