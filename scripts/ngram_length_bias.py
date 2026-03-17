import ijson
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from utils.utils import get_dataset_name, strip_ngram_markers, parse_ngrams, calculate_retrieval_metrics

def analyze_ngram_length_bias(datapath="data/seal_output.json"):
    """Analyze over-generation of unigrams vs multi-grams with bins and deciles."""
    script_name = "ngram_length_bias"
    print(f"running {script_name}")

    try:
        dataset_name = get_dataset_name(datapath)
        output_dir = f"generated_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

        results = []

        def _mrr(retrieved, positive):
            for rank, pid in enumerate(retrieved, 1):
                if pid in positive:
                    return 1.0 / rank
            return 0.0

        def _recall_at_k(retrieved, positive, k):
            if not positive:
                return 0.0
            return sum(1 for pid in retrieved[:k] if pid in positive) / len(positive)

        def _ndcg_at_k(retrieved, positive, k):
            if not positive:
                return 0.0
            dcg = sum(1.0 / np.log2(r + 1) for r, pid in enumerate(retrieved[:k], 1) if pid in positive)
            idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(positive), k) + 1))
            return dcg / idcg if idcg > 0 else 0.0

        with open(datapath, 'r', encoding='utf-8') as f:
            for entry in ijson.items(f, 'item'):
                query = entry['question']

                positive_ids = {ctx['passage_id'] for ctx in entry.get('positive_ctxs', [])}

                ctxs = entry.get('ctxs', [])
                top_ctx = ctxs[0] if ctxs else None
                if not top_ctx:
                    continue

                hits_at_1 = top_ctx['passage_id'] in positive_ids
                hits_at_2 = any(ctx['passage_id'] in positive_ids for ctx in ctxs[:2])
                hits_at_10 = any(ctx['passage_id'] in positive_ids for ctx in ctxs[:10])
                hits_at_20 = any(ctx['passage_id'] in positive_ids for ctx in ctxs[:20])
                hits_at_100 = any(ctx['passage_id'] in positive_ids for ctx in ctxs[:100])

                ngrams = parse_ngrams(top_ctx.get('keys', ''))
                if not ngrams:
                    continue

                lengths = [len(strip_ngram_markers(ngram, datapath).strip().split()) for ngram, _, _ in ngrams]
                unigram_frac = sum(1 for l in lengths if l == 1) / len(lengths)
                avg_length = np.mean(lengths)

                retrieved_ids = [ctx['passage_id'] for ctx in ctxs]
                metrics = calculate_retrieval_metrics(retrieved_ids, positive_ids)
                mrr = _mrr(retrieved_ids, positive_ids)
                recall_10 = _recall_at_k(retrieved_ids, positive_ids, 10)
                recall_100 = _recall_at_k(retrieved_ids, positive_ids, 100)
                ndcg_10 = _ndcg_at_k(retrieved_ids, positive_ids, 10)
                ndcg_100 = _ndcg_at_k(retrieved_ids, positive_ids, 100)

                results.append({
                    'query': query,
                    'unigram_frac': unigram_frac,
                    'hits_at_1': hits_at_1,
                    'hits_at_2': hits_at_2,
                    'hits_at_10': hits_at_10,
                    'hits_at_20': hits_at_20,
                    'hits_at_100': hits_at_100,
                    'num_ngrams': len(ngrams),
                    'avg_length': avg_length,
                    'precision_at_1': metrics['precision_at_1'],
                    'r_precision': metrics['r_precision'],
                    'mrr': mrr,
                    'recall@10': recall_10,
                    'recall@100': recall_100,
                    'ndcg@10': ndcg_10,
                    'ndcg@100': ndcg_100,
                })

        df = pd.DataFrame(results)

        # Check if dataframe is empty
        if len(df) == 0:
            print(f"success running {script_name}")
            return

        # Calculate statistics
        mean_unigram_frac = float(df['unigram_frac'].mean())
        median_unigram_frac = float(df['unigram_frac'].median())
        std_unigram_frac = float(df['unigram_frac'].std())
        mean_avg_length = float(df['avg_length'].mean())
        median_avg_length = float(df['avg_length'].median())
        std_avg_length = float(df['avg_length'].std())

        # Decile analysis using equal-frequency bins
        df['unigram_decile'] = pd.qcut(df['unigram_frac'], q=10, labels=False, duplicates='drop')

        deciles_data = []
        for decile in sorted(df['unigram_decile'].dropna().unique()):
            mask = df['unigram_decile'] == decile
            bin_data = df[mask]
            deciles_data.append({
                "decile": int(decile) + 1,
                "unigram_frac_min": float(bin_data['unigram_frac'].min()),
                "unigram_frac_max": float(bin_data['unigram_frac'].max()),
                "unigram_frac_mean": float(bin_data['unigram_frac'].mean()),
                "unigram_frac_std": float(bin_data['unigram_frac'].std()),
                "hits_at_1_pct": float(bin_data['hits_at_1'].mean() * 100),
                "hits_at_1_std": float(bin_data['hits_at_1'].std() * 100),
                "hits_at_2_pct": float(bin_data['hits_at_2'].mean() * 100),
                "hits_at_2_std": float(bin_data['hits_at_2'].std() * 100),
                "hits_at_10_pct": float(bin_data['hits_at_10'].mean() * 100),
                "hits_at_10_std": float(bin_data['hits_at_10'].std() * 100),
                "hits_at_20_pct": float(bin_data['hits_at_20'].mean() * 100),
                "hits_at_20_std": float(bin_data['hits_at_20'].std() * 100),
                "hits_at_100_pct": float(bin_data['hits_at_100'].mean() * 100),
                "hits_at_100_std": float(bin_data['hits_at_100'].std() * 100),
                "mrr": float(bin_data['mrr'].mean()),
                "mrr_std": float(bin_data['mrr'].std()),
                "recall@10": float(bin_data['recall@10'].mean()),
                "recall@10_std": float(bin_data['recall@10'].std()),
                "recall@100": float(bin_data['recall@100'].mean()),
                "recall@100_std": float(bin_data['recall@100'].std()),
                "ndcg@10": float(bin_data['ndcg@10'].mean()),
                "ndcg@10_std": float(bin_data['ndcg@10'].std()),
                "ndcg@100": float(bin_data['ndcg@100'].mean()),
                "ndcg@100_std": float(bin_data['ndcg@100'].std()),
                "count": int(mask.sum())
            })

        df = df.drop(columns=['unigram_decile'])

        # Correlations
        corr_hits1, p_val_hits1 = spearmanr(df['unigram_frac'], df['hits_at_1'])
        corr_hits10, p_val_hits10 = spearmanr(df['unigram_frac'], df['hits_at_10'])
        corr_mrr, p_val_mrr = spearmanr(df['unigram_frac'], df['mrr'])
        corr_recall10, p_val_recall10 = spearmanr(df['unigram_frac'], df['recall@10'])
        corr_ndcg10, p_val_ndcg10 = spearmanr(df['unigram_frac'], df['ndcg@10'])

        # Collect output data
        output_data = {
            "total_queries": len(df),
            "mean_unigram_frac": mean_unigram_frac,
            "median_unigram_frac": median_unigram_frac,
            "std_unigram_frac": std_unigram_frac,
            "mean_avg_length": mean_avg_length,
            "median_avg_length": median_avg_length,
            "std_avg_length": std_avg_length,
            "hits_at_1": float(df['hits_at_1'].mean()),
            "std_hits_at_1": float(df['hits_at_1'].std()),
            "hits_at_2": float(df['hits_at_2'].mean()),
            "std_hits_at_2": float(df['hits_at_2'].std()),
            "hits_at_10": float(df['hits_at_10'].mean()),
            "std_hits_at_10": float(df['hits_at_10'].std()),
            "hits_at_20": float(df['hits_at_20'].mean()),
            "std_hits_at_20": float(df['hits_at_20'].std()),
            "hits_at_100": float(df['hits_at_100'].mean()),
            "std_hits_at_100": float(df['hits_at_100'].std()),
            "avg_mrr": float(df['mrr'].mean()),
            "avg_recall@10": float(df['recall@10'].mean()),
            "avg_recall@100": float(df['recall@100'].mean()),
            "avg_ndcg@10": float(df['ndcg@10'].mean()),
            "avg_ndcg@100": float(df['ndcg@100'].mean()),
            "precision_at_1": float(df['precision_at_1'].mean()),
            "r_precision": float(df['r_precision'].mean()),
            "deciles": deciles_data,
            "spearman_unigram_frac_vs_hits_at_1": {
                "correlation": float(corr_hits1),
                "p_value": float(p_val_hits1)
            },
            "spearman_unigram_frac_vs_hits_at_10": {
                "correlation": float(corr_hits10),
                "p_value": float(p_val_hits10)
            },
            "spearman_unigram_frac_vs_mrr": {
                "correlation": float(corr_mrr),
                "p_value": float(p_val_mrr)
            },
            "spearman_unigram_frac_vs_recall10": {
                "correlation": float(corr_recall10),
                "p_value": float(p_val_recall10)
            },
            "spearman_unigram_frac_vs_ndcg10": {
                "correlation": float(corr_ndcg10),
                "p_value": float(p_val_ndcg10)
            },
        }

        # Plot: Unigram fraction vs Hits@1 (deciles)
        bin_centers = [d['unigram_frac_mean'] for d in deciles_data]
        hits_at_1_rates = [d['hits_at_1_pct'] for d in deciles_data]

        plt.figure(figsize=(8, 5))
        plt.plot(bin_centers, hits_at_1_rates, marker='o', linestyle='-', linewidth=3, markersize=8)
        plt.xlabel("Mean Unigram Fraction (decile)", fontsize=16, fontweight='bold')
        plt.ylabel("Hits@1 (%)", fontsize=16, fontweight='bold')
        plt.title("Unigram Fraction vs Hits@1 (Deciles)", fontsize=18, fontweight='bold')
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{script_name}_unigram_frac_vs_hits_at_1.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

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
        analyze_ngram_length_bias(datapath)
