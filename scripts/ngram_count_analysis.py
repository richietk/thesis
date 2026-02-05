import ijson
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import ast
import os
from utils import strip_ngram_markers, get_dataset_name, parse_ngrams

# OUTPUT_DIR will be set dynamically based on dataset_name

def analyze_ngram_distribution(datapath="data/seal_output.json"):
    import sys

    script_name = "ngram_count_analysis"
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

        print(f"Loading {datapath}...")
        with open(datapath, 'r', encoding='utf-8') as f:
            for entry in ijson.items(f, 'item'):
                top_ctx = entry['ctxs'][0] if entry.get('ctxs') else None
                if not top_ctx: continue

                positive_ids = {ctx['passage_id'] for ctx in entry.get('positive_ctxs', [])}
                success_top1 = top_ctx['passage_id'] in positive_ids
                ngrams = parse_ngrams(top_ctx.get('keys', ''))

                count = len(ngrams)
                # Corpus Frequency is index 1 in the (ngram, freq, score) tuple
                avg_freq = np.mean([n[1] for n in ngrams]) if ngrams else 0

                results.append({
                    'ngram_count': count,
                    'avg_freq': avg_freq,
                    'success_top1': success_top1,
                })

        df = pd.DataFrame(results)

        # ================================================================================
        # GLOBAL STATISTICS BLOCK
        # ================================================================================
        print("\n" + "=" * 80)
        print("EMPIRICAL ANALYSIS: N-GRAM COUNT VS. RETRIEVAL EFFECTIVENESS")
        print("=" * 80)

        print(f"\nSummary Statistics (N={len(df)}):")
        print(f"  Mean Count:   {df['ngram_count'].mean():.2f}")
        print(f"  Median:       {df['ngram_count'].median():.2f}")
        print(f"  Std. Dev:     {df['ngram_count'].std():.2f}")
        print(f"  Range:        [{df['ngram_count'].min()}, {df['ngram_count'].max()}]")

        # Define Decile-based intervals
        print("\nRetrieval Performance by N-gram Count Decile:")
        print("-" * 75)
        header = f"{'Decile':8s} | {'Count Range':15s} | {'N':8s} | {'Success@1':>10s} | {'Mean Corpus Freq'}"
        print(header)
        print("-" * 75)

        bin_data = []

        # Create decile bins
        df['ngram_decile'] = pd.qcut(df['ngram_count'], q=10, labels=False, duplicates='drop')

        for decile in sorted(df['ngram_decile'].unique()):
            mask = df['ngram_decile'] == decile
            count_n = mask.sum()
            if count_n > 0:
                success = df.loc[mask, 'success_top1'].mean() * 100
                mean_f = df.loc[mask, 'avg_freq'].mean()

                count_min = int(df.loc[mask, 'ngram_count'].min())
                count_max = int(df.loc[mask, 'ngram_count'].max())
                range_label = f"{count_min}–{count_max}"

                print(f" D{int(decile)+1:1d}      | {range_label:15s} | {count_n:8d} | {success:9.2f}% | {mean_f:,.0f}")
                bin_data.append({'range': range_label, 'success': success, 'freq': mean_f, 'decile': int(decile)+1})

        # Drop the temporary column
        df = df.drop(columns=['ngram_decile'])

        # Spearman Correlation
        rho_success, p_success = spearmanr(df['ngram_count'], df['success_top1'])
        rho_freq, p_freq = spearmanr(df['ngram_count'], df['avg_freq'])

        print("-" * 75)
        print(f"Spearman ρ (Count vs. Success): {rho_success:.3f} (p={p_success:.2e})")
        print(f"Spearman ρ (Count vs. Frequency): {rho_freq:.3f} (p={p_freq:.2e})")

        # ================================================================================
        # VISUALIZATION BLOCK
        # ================================================================================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ranges = [b['range'] for b in bin_data]
        rates = [b['success'] for b in bin_data]
        freqs = [b['freq'] for b in bin_data]

        # Success Plot
        ax1.bar(ranges, rates, color='skyblue', edgecolor='navy')
        ax1.set_ylabel("Top-1 Success Rate (%)", fontweight='bold')
        ax1.set_xlabel("N-gram Count Interval", fontweight='bold')
        ax1.set_title("Retrieval Effectiveness by Identifier Count", fontweight='bold')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # Frequency Plot
        ax2.bar(ranges, freqs, color='salmon', edgecolor='darkred')
        ax2.set_ylabel(r"Mean Corpus Frequency ($\bar{F}_c$)", fontweight='bold')
        ax2.set_xlabel("N-gram Count Interval", fontweight='bold')
        ax2.set_title("Mean Specificity by Identifier Count", fontweight='bold')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        output_file = f"{output_dir}/ngram_distribution_analysis_{dataset_name}.png"
        plt.savefig(output_file, dpi=300)
        print(f"\nPlots saved to {output_file}")
        plt.close()  # Close instead of show to avoid blocking

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
    import sys
    datapath = sys.argv[1] if len(sys.argv) > 1 else 'data/seal_output.json'
    analyze_ngram_distribution(datapath)
