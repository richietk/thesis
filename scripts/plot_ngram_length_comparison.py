import json
import matplotlib.pyplot as plt
import os
import sys

def plot_ngram_length_comparison():
    """Create unified chart comparing SEAL and MINDER ngram length bias results."""
    script_name = "plot_ngram_length_comparison"
    print(f"running {script_name}")

    try:
        # Paths to the JSON files
        seal_path = "generated_data/seal/ngram_length_bias_results.json"
        minder_path = "generated_data/minder/ngram_length_bias_results.json"

        # Load JSON data
        with open(seal_path, 'r') as f:
            seal_data = json.load(f)

        with open(minder_path, 'r') as f:
            minder_data = json.load(f)

        # Extract decile data for SEAL
        seal_deciles = seal_data['deciles']
        seal_x = [(d['unigram_frac_min'] + d['unigram_frac_max']) / 2 for d in seal_deciles]
        seal_hits1 = [d['hits_at_1_pct'] for d in seal_deciles]
        seal_hits10 = [d['hits_at_10_pct'] for d in seal_deciles]

        # Extract decile data for MINDER
        minder_deciles = minder_data['deciles']
        minder_x = [(d['unigram_frac_min'] + d['unigram_frac_max']) / 2 for d in minder_deciles]
        minder_hits1 = [d['hits_at_1_pct'] for d in minder_deciles]
        minder_hits10 = [d['hits_at_10_pct'] for d in minder_deciles]

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(seal_x, seal_hits1, marker='o', label='SEAL Hits@1', linewidth=2, markersize=6)
        plt.plot(seal_x, seal_hits10, marker='s', label='SEAL Hits@10', linewidth=2, markersize=6)
        plt.plot(minder_x, minder_hits1, marker='^', label='MINDER Hits@1', linewidth=2, markersize=6)
        plt.plot(minder_x, minder_hits10, marker='d', label='MINDER Hits@10', linewidth=2, markersize=6)

        plt.xlabel('Mean Unigram Fraction (Decile Centers)', fontsize=16, fontweight='bold')
        plt.ylabel('Hits (%)', fontsize=16, fontweight='bold')
        plt.title('Hits@1 and Hits@10 vs Mean Unigram Fraction', fontsize=18, fontweight='bold')
        plt.legend(fontsize=13)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        output_dir = "generated_data/shared"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "ngram_length_bias_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plot saved to: {output_path}")
        print(f"success running {script_name}")

    except Exception as e:
        print(f"error: running {script_name} {e}")
        raise

if __name__ == "__main__":
    plot_ngram_length_comparison()
