import ijson
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, rankdata
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.utils import get_dataset_name, strip_ngram_markers, parse_ngrams, calculate_retrieval_metrics

def analyze_ngram_frequency(datapath="data/seal_output.json"):
    script_name = "nonspecific_highfreq_ngrams"
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
                if not ctxs:
                    continue

                # Collect top n queries
                top_passages = ctxs[:10]  # only consider top-10
                passage_ids = [ctx['passage_id'] for ctx in top_passages]

                # Success flags for top-1/top-2/top-10/top-20/top-100
                success_top1 = int(passage_ids[0] in positive_ids) if len(passage_ids) >= 1 else 0
                success_top2 = int(any(pid in positive_ids for pid in passage_ids[:2]))
                success_top10 = int(any(pid in positive_ids for pid in passage_ids[:10]))
                success_top20 = int(any(ctx['passage_id'] in positive_ids for ctx in ctxs[:20]))
                success_top100 = int(any(ctx['passage_id'] in positive_ids for ctx in ctxs[:100]))

                # N-gram statistics from top-1 passage only
                ngrams = parse_ngrams(ctxs[0].get('keys', ''))
                if not ngrams:
                    continue

                sorted_ngrams = sorted(ngrams, key=lambda x: x[2], reverse=True)
                top_5 = sorted_ngrams[:5]
                top_10 = sorted_ngrams[:10]

                frequencies = [freq for _, freq, _ in ngrams]
                top_5_freq = [ng[1] for ng in top_5]
                top_10_freq = [ng[1] for ng in top_10]
                top_5_lengths = [len(ng[0].strip()) for ng in top_5]
                per_passage_ngram_counts = [len(ngrams)]

                # Calculate retrieval metrics
                retrieved_ids = [ctx['passage_id'] for ctx in ctxs]
                metrics = calculate_retrieval_metrics(retrieved_ids, positive_ids)
                mrr = _mrr(retrieved_ids, positive_ids)
                recall_10 = _recall_at_k(retrieved_ids, positive_ids, 10)
                recall_20 = _recall_at_k(retrieved_ids, positive_ids, 20)
                recall_100 = _recall_at_k(retrieved_ids, positive_ids, 100)
                ndcg_10 = _ndcg_at_k(retrieved_ids, positive_ids, 10)
                ndcg_100 = _ndcg_at_k(retrieved_ids, positive_ids, 100)

                top5_freq_within_std = float(np.std(top_5_freq, ddof=1)) if len(top_5_freq) > 1 else 0.0
                avg_top5_length = float(np.mean(top_5_lengths)) if top_5_lengths else 0.0

                results.append({
                    'query': query,
                    'hits@1': success_top1,
                    'hits@2': success_top2,
                    'hits@10': success_top10,
                    'hits@20': success_top20,
                    'hits@100': success_top100,
                    'num_ngrams': int(np.mean(per_passage_ngram_counts)) if per_passage_ngram_counts else 0,
                    'avg_frequency_all': np.mean(frequencies),
                    'median_frequency_all': np.median(frequencies),
                    'avg_top5_frequency': np.mean(top_5_freq) if top_5_freq else 0,
                    'avg_top10_frequency': np.mean(top_10_freq) if top_10_freq else 0,
                    'top5_freq_within_std': top5_freq_within_std,
                    'avg_top5_length': avg_top5_length,
                    'precision_at_1': metrics['precision_at_1'],
                    'r_precision': metrics['r_precision'],
                    'mrr': mrr,
                    'recall@10': recall_10,
                    'recall@20': recall_20,
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
        avg_freq_all = float(df['avg_frequency_all'].mean())
        std_freq_all = float(df['avg_frequency_all'].std())
        avg_freq_top5 = float(df['avg_top5_frequency'].mean())
        std_freq_top5 = float(df['avg_top5_frequency'].std())
        avg_freq_top10 = float(df['avg_top10_frequency'].mean())
        std_freq_top10 = float(df['avg_top10_frequency'].std())
        avg_top5_within_std = float(df['top5_freq_within_std'].mean())
        std_top5_within_std = float(df['top5_freq_within_std'].std())

        # Helper function for decile analysis
        def compute_deciles(df, freq_column, temp_col_name):
            freq_values = df[freq_column]
            df[temp_col_name] = pd.qcut(freq_values, q=10, labels=False, duplicates='drop')

            deciles_data = []
            for decile in sorted(df[temp_col_name].unique()):
                mask = df[temp_col_name] == decile
                if mask.sum() == 0:
                    continue
                bin_min = freq_values[mask].min()
                bin_max = freq_values[mask].max()
                top1_rate = df.loc[mask, 'hits@1'].mean()
                top1_std = df.loc[mask, 'hits@1'].std()
                top2_rate = df.loc[mask, 'hits@2'].mean()
                top2_std = df.loc[mask, 'hits@2'].std()
                top10_rate = df.loc[mask, 'hits@10'].mean()
                top10_std = df.loc[mask, 'hits@10'].std()
                top20_rate = df.loc[mask, 'hits@20'].mean()
                top20_std = df.loc[mask, 'hits@20'].std()
                top100_rate = df.loc[mask, 'hits@100'].mean()
                top100_std = df.loc[mask, 'hits@100'].std()
                avg_length = df.loc[mask, 'avg_top5_length'].mean()
                mrr_mean = df.loc[mask, 'mrr'].mean()
                mrr_std = df.loc[mask, 'mrr'].std()
                recall10_mean = df.loc[mask, 'recall@10'].mean()
                recall10_std = df.loc[mask, 'recall@10'].std()
                recall100_mean = df.loc[mask, 'recall@100'].mean()
                recall100_std = df.loc[mask, 'recall@100'].std()
                ndcg10_mean = df.loc[mask, 'ndcg@10'].mean()
                ndcg10_std = df.loc[mask, 'ndcg@10'].std()
                ndcg100_mean = df.loc[mask, 'ndcg@100'].mean()
                ndcg100_std = df.loc[mask, 'ndcg@100'].std()
                count = mask.sum()


                deciles_data.append({
                    "decile": int(decile) + 1,
                    "freq_min": float(bin_min),
                    "freq_max": float(bin_max),
                    "hits@1_pct": float(top1_rate * 100),
                    "hits@1_std": float(top1_std * 100),
                    "hits@2_pct": float(top2_rate * 100),
                    "hits@2_std": float(top2_std * 100),
                    "hits@10_pct": float(top10_rate * 100),
                    "hits@10_std": float(top10_std * 100),
                    "hits@20_pct": float(top20_rate * 100),
                    "hits@20_std": float(top20_std * 100),
                    "hits@100_pct": float(top100_rate * 100),
                    "hits@100_std": float(top100_std * 100),
                    "mrr": float(mrr_mean),
                    "mrr_std": float(mrr_std),
                    "recall@10": float(recall10_mean),
                    "recall@10_std": float(recall10_std),
                    "recall@100": float(recall100_mean),
                    "recall@100_std": float(recall100_std),
                    "ndcg@10": float(ndcg10_mean),
                    "ndcg@10_std": float(ndcg10_std),
                    "ndcg@100": float(ndcg100_mean),
                    "ndcg@100_std": float(ndcg100_std),
                    "avg_ngram_length": float(avg_length),
                    "count": int(count)
                })


            df = df.drop(columns=[temp_col_name])
            return deciles_data, df

        def partial_spearman(x, y, z):
            """Partial Spearman correlation of x and y controlling for z, via rank residualization."""
            rx = rankdata(x).astype(float)
            ry = rankdata(y).astype(float)
            rz = rankdata(z).astype(float)
            rz_c = rz - rz.mean()
            denom = np.dot(rz_c, rz_c)
            resid_x = rx - (np.dot(rz_c, rx) / denom) * rz_c
            resid_y = ry - (np.dot(rz_c, ry) / denom) * rz_c
            return pearsonr(resid_x, resid_y)

        def compute_length_stratified_freq_deciles(df):
            """Within each length quartile (computed via qcut), compute frequency deciles and hits@k."""
            df = df.copy()
            df['length_quartile'] = pd.qcut(df['avg_top5_length'], q=4, labels=False, duplicates='drop')

            stratified_data = []
            for quartile in sorted(df['length_quartile'].dropna().unique()):
                qmask = df['length_quartile'] == quartile
                qdf = df[qmask].copy()

                qdf['freq_decile'] = pd.qcut(qdf['avg_top5_frequency'], q=10, labels=False, duplicates='drop')

                freq_deciles = []
                for decile in sorted(qdf['freq_decile'].dropna().unique()):
                    dmask = qdf['freq_decile'] == decile
                    freq_deciles.append({
                        "decile": int(decile) + 1,
                        "freq_min": float(qdf.loc[dmask, 'avg_top5_frequency'].min()),
                        "freq_max": float(qdf.loc[dmask, 'avg_top5_frequency'].max()),
                        "hits@1_pct": float(qdf.loc[dmask, 'hits@1'].mean() * 100),
                        "hits@1_std": float(qdf.loc[dmask, 'hits@1'].std() * 100),
                        "hits@10_pct": float(qdf.loc[dmask, 'hits@10'].mean() * 100),
                        "hits@10_std": float(qdf.loc[dmask, 'hits@10'].std() * 100),
                        "hits@100_pct": float(qdf.loc[dmask, 'hits@100'].mean() * 100),
                        "hits@100_std": float(qdf.loc[dmask, 'hits@100'].std() * 100),
                        "mrr": float(qdf.loc[dmask, 'mrr'].mean()),
                        "mrr_std": float(qdf.loc[dmask, 'mrr'].std()),
                        "recall@10": float(qdf.loc[dmask, 'recall@10'].mean()),
                        "recall@10_std": float(qdf.loc[dmask, 'recall@10'].std()),
                        "recall@100": float(qdf.loc[dmask, 'recall@100'].mean()),
                        "recall@100_std": float(qdf.loc[dmask, 'recall@100'].std()),
                        "ndcg@10": float(qdf.loc[dmask, 'ndcg@10'].mean()),
                        "ndcg@10_std": float(qdf.loc[dmask, 'ndcg@10'].std()),
                        "ndcg@100": float(qdf.loc[dmask, 'ndcg@100'].mean()),
                        "ndcg@100_std": float(qdf.loc[dmask, 'ndcg@100'].std()),
                        "count": int(dmask.sum())
                    })

                stratified_data.append({
                    "length_quartile": int(quartile) + 1,
                    "length_min": float(df.loc[qmask, 'avg_top5_length'].min()),
                    "length_max": float(df.loc[qmask, 'avg_top5_length'].max()),
                    "count": int(qmask.sum()),
                    "freq_deciles": freq_deciles
                })

            return stratified_data

        # Decile-based frequency analysis for top-5 n-grams
        deciles_top5, df = compute_deciles(df, 'avg_top5_frequency', 'freq_decile_top5')

        # Decile-based frequency analysis for all n-grams
        deciles_all, df = compute_deciles(df, 'avg_frequency_all', 'freq_decile_all')

        # Spearman correlations
        corr_top5, p_val_top5 = spearmanr(df['avg_top5_frequency'], df['hits@1'])
        corr_all, p_val_all = spearmanr(df['avg_frequency_all'], df['hits@1'])
        corr_len_freq, p_val_len_freq = spearmanr(df['avg_top5_length'], df['avg_top5_frequency'])

        # Partial Spearman: frequency vs hits@1, controlling for length
        partial_corr, partial_pval = partial_spearman(
            df['avg_top5_frequency'], df['hits@1'], df['avg_top5_length']
        )

        # Spearman correlations for new metrics vs frequency
        corr_freq_mrr, p_val_freq_mrr = spearmanr(df['avg_top5_frequency'], df['mrr'])
        corr_freq_recall10, p_val_freq_recall10 = spearmanr(df['avg_top5_frequency'], df['recall@10'])
        corr_freq_ndcg10, p_val_freq_ndcg10 = spearmanr(df['avg_top5_frequency'], df['ndcg@10'])

        # Length-stratified frequency decile analysis
        length_stratified = compute_length_stratified_freq_deciles(df)

        # Collect output data
        output_data = {
            "total_queries": len(df),
            "avg_frequency_all": avg_freq_all,
            "std_frequency_all": std_freq_all,
            "avg_frequency_top5": avg_freq_top5,
            "std_frequency_top5": std_freq_top5,
            "avg_frequency_top10": avg_freq_top10,
            "std_frequency_top10": std_freq_top10,
            "avg_top5_freq_within_std": avg_top5_within_std,
            "std_top5_freq_within_std": std_top5_within_std,
            "precision_at_1": float(df['precision_at_1'].mean()),
            "r_precision": float(df['r_precision'].mean()),
            "deciles_top5": deciles_top5,
            "deciles_all": deciles_all,
            "spearman_top5_vs_hit1": {
                "correlation": float(corr_top5),
                "p_value": float(p_val_top5)
            },
            "spearman_all_vs_hit1": {
                "correlation": float(corr_all),
                "p_value": float(p_val_all)
            },
            "avg_top5_ngram_length": float(df['avg_top5_length'].mean()),
            "std_top5_ngram_length": float(df['avg_top5_length'].std()),
            "spearman_length_vs_frequency": {
                "correlation": float(corr_len_freq),
                "p_value": float(p_val_len_freq)
            },
            "partial_spearman_freq_vs_hit1_controlling_length": {
                "correlation": float(partial_corr),
                "p_value": float(partial_pval)
            },
            "avg_mrr": float(df['mrr'].mean()),
            "avg_recall@10": float(df['recall@10'].mean()),
            "avg_recall@100": float(df['recall@100'].mean()),
            "avg_ndcg@10": float(df['ndcg@10'].mean()),
            "avg_ndcg@100": float(df['ndcg@100'].mean()),
            "spearman_top5_vs_mrr": {
                "correlation": float(corr_freq_mrr),
                "p_value": float(p_val_freq_mrr)
            },
            "spearman_top5_vs_recall10": {
                "correlation": float(corr_freq_recall10),
                "p_value": float(p_val_freq_recall10)
            },
            "spearman_top5_vs_ndcg10": {
                "correlation": float(corr_freq_ndcg10),
                "p_value": float(p_val_freq_ndcg10)
            },
            "length_stratified_freq_deciles": length_stratified
        }

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
        analyze_ngram_frequency(datapath)