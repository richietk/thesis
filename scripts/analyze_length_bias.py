"""Analyze correlation between passage length and retrieval performance in SEAL outputs."""

import ijson
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import sys
from utils import strip_pseudoqueries, get_dataset_name

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def get_token_count(text, datapath=""):
    """Approximate BART token count using whitespace splitting."""
    text = strip_pseudoqueries(text, datapath)
    return len(text.split())


def extract_analysis_data(file_path, datapath=""):
    """Extract analysis data from SEAL output (streaming)."""
    print(f"Streaming data from {file_path}...")
    rows = []
    query_idx = 0

    with open(file_path, 'rb') as f:
        parser = ijson.items(f, 'item')

        for item in parser:
            query_idx += 1
            if query_idx % 1000 == 0:
                print(f"  Processed {query_idx} queries...")
            question = item['question']

            positive_ids = set()
            if 'positive_ctxs' in item and item['positive_ctxs']:
                positive_ids = {ctx['passage_id'] for ctx in item['positive_ctxs']}

            for rank, ctx in enumerate(item.get('ctxs', []), start=1):
                passage_id = ctx['passage_id']
                text = ctx['text']
                score = ctx.get('score', 0.0)

                passage_length = get_token_count(text, datapath)

                keys_data = ctx.get('keys', '')
                if keys_data:
                    if isinstance(keys_data, list):
                        num_keys = len(keys_data)
                        key_scores = [float(item[2]) for item in keys_data if len(item) >= 3]
                    else:
                        parts = keys_data.split(' | ')
                        num_keys = len(parts) // 3
                        key_scores = []
                        for i in range(2, len(parts), 3):
                            try:
                                key_scores.append(float(parts[i]))
                            except (ValueError, IndexError):
                                continue

                    total_key_score = sum(key_scores) if key_scores else 0.0
                    avg_key_score = np.mean(key_scores) if key_scores else 0.0
                    max_key_score = max(key_scores) if key_scores else 0.0
                else:
                    num_keys = 0
                    total_key_score = 0.0
                    avg_key_score = 0.0
                    max_key_score = 0.0

                rows.append({
                    'query_id': query_idx - 1,
                    'question': question,
                    'passage_id': passage_id,
                    'rank': rank,
                    'score': score,
                    'passage_length': passage_length,
                    'is_positive': passage_id in positive_ids,
                    'num_keys': num_keys,
                    'total_key_score': total_key_score,
                    'avg_key_score': avg_key_score,
                    'max_key_score': max_key_score
                })

    df = pd.DataFrame(rows)

    numeric_columns = ['rank', 'score', 'passage_length', 'num_keys', 'total_key_score', 'avg_key_score', 'max_key_score']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"\nExtracted {len(df)} passage retrievals across {df['query_id'].nunique()} queries")
    print(f"Passage length statistics:")
    print(f"  Mean: {df['passage_length'].mean():.1f} tokens")
    print(f"  Median: {df['passage_length'].median():.1f} tokens")
    print(f"  Std: {df['passage_length'].std():.1f} tokens")
    print(f"  Min: {df['passage_length'].min()}, Max: {df['passage_length'].max()}")

    return df


def analyze_length_rank_correlation(df, output_dir, dataset_name="seal"):
    """Analyze correlation between passage length and retrieval rank."""
    print("\n" + "="*80)
    print("ANALYSIS 1: Passage Length vs Retrieval Rank")
    print("="*80)

    correlations = []
    for query_id, group in df.groupby('query_id'):
        if len(group) > 10:
            corr, p_value = spearmanr(group['rank'], group['passage_length'])
            correlations.append({
                'query_id': query_id,
                'spearman_corr': corr,
                'p_value': p_value,
                'n_passages': len(group),
                'length_range': group['passage_length'].max() - group['passage_length'].min()
            })

    corr_df = pd.DataFrame(correlations)

    print(f"\nPer-Query Spearman Correlation (Rank vs Length):")
    print(f"  Mean: {corr_df['spearman_corr'].mean():.4f}")
    print(f"  Median: {corr_df['spearman_corr'].median():.4f}")
    print(f"  Std: {corr_df['spearman_corr'].std():.4f}")

    biased_queries = corr_df[corr_df['spearman_corr'] < 0]
    significant_bias = corr_df[(corr_df['spearman_corr'] < -0.2) & (corr_df['p_value'] < 0.05)]

    print(f"\n  Queries with length bias (ρ < 0): {len(biased_queries)} / {len(corr_df)} ({100*len(biased_queries)/len(corr_df):.1f}%)")
    print(f"  Queries with STRONG bias (ρ < -0.2, p < 0.05): {len(significant_bias)} ({100*len(significant_bias)/len(corr_df):.1f}%)")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(corr_df['spearman_corr'], bins=40, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='No correlation')
    axes[0, 0].axvline(corr_df['spearman_corr'].mean(), color='orange', linestyle='--',
                       linewidth=2, label=f'Mean = {corr_df["spearman_corr"].mean():.3f}')
    axes[0, 0].set_xlabel('Spearman Correlation (Rank vs Length)')
    axes[0, 0].set_ylabel('Number of Queries')
    axes[0, 0].set_title('Distribution of Rank-Length Correlations')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    sample = df.sample(min(5000, len(df)))
    axes[0, 1].scatter(sample['passage_length'], sample['rank'], alpha=0.3, s=10)
    axes[0, 1].set_xlabel('Passage Length (tokens)')
    axes[0, 1].set_ylabel('Retrieval Rank (1 = best)')
    axes[0, 1].set_title('Passage Length vs Rank (Overall)')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].invert_yaxis()

    rank_strata = [(1, 10), (11, 20), (21, 50), (51, 100)]
    length_by_strata = []
    labels = []
    for low, high in rank_strata:
        subset = df[(df['rank'] >= low) & (df['rank'] <= high)]
        length_by_strata.append(subset['passage_length'].values)
        labels.append(f'Ranks {low}-{high}')

    axes[1, 0].boxplot(length_by_strata, labels=labels)
    axes[1, 0].set_ylabel('Passage Length (tokens)')
    axes[1, 0].set_title('Passage Length Distribution by Rank Strata')
    axes[1, 0].grid(alpha=0.3, axis='y')
    axes[1, 0].tick_params(axis='x', rotation=15)

    if len(significant_bias) > 0:
        example_qid = significant_bias.iloc[0]['query_id']
        example = df[df['query_id'] == example_qid].sort_values('rank')
        axes[1, 1].scatter(example['rank'], example['passage_length'], s=50, alpha=0.7)
        axes[1, 1].set_xlabel('Retrieval Rank')
        axes[1, 1].set_ylabel('Passage Length (tokens)')
        query_text = example.iloc[0]['question'][:50]
        corr_val = significant_bias.iloc[0]['spearman_corr']
        axes[1, 1].set_title(f'Example Query (ρ = {corr_val:.3f})\n"{query_text}..."')
        axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    output_file = f'{output_dir}/01_length_rank_correlation_{dataset_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_file}")

    return corr_df


def analyze_length_score_correlation(df, output_dir, dataset_name="seal"):
    """Analyze correlation between passage length and SEAL's aggregate score."""
    print("\n" + "="*80)
    print("ANALYSIS 2: Passage Length vs Aggregate Score")
    print("="*80)

    corr_spearman, p_spearman = spearmanr(df['passage_length'], df['score'])
    corr_pearson, p_pearson = pearsonr(df['passage_length'], df['score'])

    print(f"\nOverall Correlations:")
    print(f"  Spearman: ρ = {corr_spearman:.4f}, p = {p_spearman:.4e}")
    print(f"  Pearson:  r = {corr_pearson:.4f}, p = {p_pearson:.4e}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    hb = axes[0].hexbin(df['passage_length'], df['score'], gridsize=50, cmap='YlOrRd', mincnt=1)
    axes[0].set_xlabel('Passage Length (tokens)')
    axes[0].set_ylabel('SEAL Aggregate Score')
    axes[0].set_title(f'Length vs Score (Spearman ρ = {corr_spearman:.3f})')
    plt.colorbar(hb, ax=axes[0], label='Count')

    df['length_bin'] = pd.cut(df['passage_length'],
                               bins=[0, 50, 75, 100, 125, 150, 500],
                               labels=['0-50', '51-75', '76-100', '101-125', '126-150', '150+'])
    df.boxplot(column='score', by='length_bin', ax=axes[1])
    axes[1].set_xlabel('Passage Length Bin (tokens)')
    axes[1].set_ylabel('SEAL Aggregate Score')
    axes[1].set_title('Score Distribution by Length')
    plt.suptitle('')

    plt.tight_layout()
    output_file = f'{output_dir}/02_length_score_correlation_{dataset_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_file}")


def analyze_positive_vs_negative_lengths(df, output_dir, dataset_name="seal"):
    """Compare passage lengths: ground truth (positive) vs retrieved (negative)."""
    print("\n" + "="*80)
    print("ANALYSIS 3: Ground Truth vs Retrieved Passage Lengths")
    print("="*80)

    positive = df[df['is_positive'] == True]
    negative = df[df['is_positive'] == False]

    top10 = df[df['rank'] <= 10]
    top10_positive = top10[top10['is_positive'] == True]
    top10_negative = top10[top10['is_positive'] == False]

    print(f"\nPassage Length Statistics:")
    print(f"  Ground truth (all positive passages): mean={positive['passage_length'].mean():.1f}, median={positive['passage_length'].median():.1f}")
    print(f"  Retrieved negative (all ranks): mean={negative['passage_length'].mean():.1f}, median={negative['passage_length'].median():.1f}")
    print(f"\n  Top-10 positive: mean={top10_positive['passage_length'].mean():.1f}, median={top10_positive['passage_length'].median():.1f}")
    print(f"  Top-10 negative: mean={top10_negative['passage_length'].mean():.1f}, median={top10_negative['passage_length'].median():.1f}")

    if len(positive) > 0 and len(top10_negative) > 0:
        stat, p = mannwhitneyu(positive['passage_length'], top10_negative['passage_length'], alternative='two-sided')
        print(f"\n  Mann-Whitney U test (Ground truth vs Top-10 negative): p = {p:.4e}")
        if p < 0.05:
            mean_diff = top10_negative['passage_length'].mean() - positive['passage_length'].mean()
            print(f"    Mean difference: {mean_diff:+.1f} tokens")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if len(positive) > 0:
        axes[0].hist(positive['passage_length'], bins=50, alpha=0.6, label='Ground truth (positive)',
                     color='green', edgecolor='black', density=True)
    axes[0].hist(top10_negative['passage_length'], bins=50, alpha=0.6, label='Top-10 negative',
                 color='red', edgecolor='black', density=True)
    axes[0].set_xlabel('Passage Length (tokens)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Length Distribution: Ground Truth vs Retrieved')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if len(positive) > 0:
        pos_sorted = np.sort(positive['passage_length'])
        axes[1].plot(pos_sorted, np.arange(len(pos_sorted)) / len(pos_sorted),
                     label='Ground truth (positive)', linewidth=2, color='green')

    neg_sorted = np.sort(top10_negative['passage_length'])
    axes[1].plot(neg_sorted, np.arange(len(neg_sorted)) / len(neg_sorted),
                 label='Top-10 negative', linewidth=2, color='red')

    axes[1].set_xlabel('Passage Length (tokens)')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('Cumulative Distribution Function')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    output_file = f'{output_dir}/03_positive_vs_negative_lengths_{dataset_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_file}")


def analyze_ngrams_vs_length(df, output_dir, dataset_name="seal"):
    """Analyze relationship between passage length and number of matched n-grams."""
    print("\n" + "="*80)
    print("ANALYSIS 4: Matched N-grams vs Passage Length (Accumulation Effect)")
    print("="*80)

    df_with_keys = df[df['num_keys'] > 0].copy()

    if len(df_with_keys) == 0:
        print("  No n-gram data available")
        return

    print(f"\nPassages with n-gram data: {len(df_with_keys)} / {len(df)}")

    corr_spearman, p_spearman = spearmanr(df_with_keys['passage_length'], df_with_keys['num_keys'])
    corr_pearson, p_pearson = pearsonr(df_with_keys['passage_length'], df_with_keys['num_keys'])

    print(f"\nCorrelation (Length vs Number of Matched N-grams):")
    print(f"  Spearman: ρ = {corr_spearman:.4f}, p = {p_spearman:.4e}")
    print(f"  Pearson:  r = {corr_pearson:.4f}, p = {p_pearson:.4e}")

    print(f"\nN-gram Matching Statistics:")
    print(f"  Mean n-grams per passage: {df_with_keys['num_keys'].mean():.2f}")
    print(f"  Median: {df_with_keys['num_keys'].median():.1f}")
    print(f"  Max: {df_with_keys['num_keys'].max()}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    hb = axes[0, 0].hexbin(df_with_keys['passage_length'], df_with_keys['num_keys'],
                           gridsize=40, cmap='YlOrRd', mincnt=1)
    axes[0, 0].set_xlabel('Passage Length (tokens)')
    axes[0, 0].set_ylabel('Number of Matched N-grams')
    axes[0, 0].set_title(f'Length vs N-gram Count (ρ = {corr_spearman:.3f})')
    plt.colorbar(hb, ax=axes[0, 0])

    df_with_keys.boxplot(column='num_keys', by='length_bin', ax=axes[0, 1])
    axes[0, 1].set_xlabel('Passage Length Bin (tokens)')
    axes[0, 1].set_ylabel('Number of Matched N-grams')
    axes[0, 1].set_title('N-gram Count by Length Bin')
    plt.suptitle('')

    corr_score_len, _ = spearmanr(df_with_keys['passage_length'], df_with_keys['total_key_score'])
    hb2 = axes[1, 0].hexbin(df_with_keys['passage_length'], df_with_keys['total_key_score'],
                            gridsize=40, cmap='YlGnBu', mincnt=1)
    axes[1, 0].set_xlabel('Passage Length (tokens)')
    axes[1, 0].set_ylabel('Total N-gram Score (sum)')
    axes[1, 0].set_title(f'Length vs Total Score (ρ = {corr_score_len:.3f})')
    plt.colorbar(hb2, ax=axes[1, 0])

    corr_avg_len, _ = spearmanr(df_with_keys['passage_length'], df_with_keys['avg_key_score'])
    hb3 = axes[1, 1].hexbin(df_with_keys['passage_length'], df_with_keys['avg_key_score'],
                            gridsize=40, cmap='viridis', mincnt=1)
    axes[1, 1].set_xlabel('Passage Length (tokens)')
    axes[1, 1].set_ylabel('Average N-gram Score')
    axes[1, 1].set_title(f'Length vs Avg Score (ρ = {corr_avg_len:.3f})')
    plt.colorbar(hb3, ax=axes[1, 1])

    plt.tight_layout()
    output_file = f'{output_dir}/04_ngrams_vs_length_{dataset_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_file}")


def analyze_retrieval_quality_by_length(df, output_dir, dataset_name="seal"):
    """Analyze whether length affects retrieval precision."""
    print("\n" + "="*80)
    print("ANALYSIS 5: Retrieval Precision by Passage Length")
    print("="*80)

    results = []
    for k in [5, 10, 20, 50, 100]:
        topk = df[df['rank'] <= k]
        for length_bin in df['length_bin'].cat.categories:
            subset = topk[topk['length_bin'] == length_bin]
            if len(subset) > 0:
                precision = subset['is_positive'].sum() / len(subset)
                results.append({
                    'k': k,
                    'length_bin': length_bin,
                    'precision': precision,
                    'n_passages': len(subset),
                    'n_positive': subset['is_positive'].sum()
                })

    results_df = pd.DataFrame(results)

    print(f"\nPrecision@K by Passage Length Bin:")
    print(f"{'Length Bin':<15} {'P@5':>8} {'P@10':>8} {'P@20':>8} {'P@50':>8} {'P@100':>9}")
    print("-" * 70)
    for length_bin in df['length_bin'].cat.categories:
        row = results_df[results_df['length_bin'] == length_bin]
        p5 = row[row['k'] == 5]['precision'].values
        p10 = row[row['k'] == 10]['precision'].values
        p20 = row[row['k'] == 20]['precision'].values
        p50 = row[row['k'] == 50]['precision'].values
        p100 = row[row['k'] == 100]['precision'].values

        p5_str = f"{p5[0]:.4f}" if len(p5) > 0 else "N/A"
        p10_str = f"{p10[0]:.4f}" if len(p10) > 0 else "N/A"
        p20_str = f"{p20[0]:.4f}" if len(p20) > 0 else "N/A"
        p50_str = f"{p50[0]:.4f}" if len(p50) > 0 else "N/A"
        p100_str = f"{p100[0]:.4f}" if len(p100) > 0 else "N/A"

        print(f"{length_bin:<15} {p5_str:>8} {p10_str:>8} {p20_str:>8} {p50_str:>8} {p100_str:>9}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for k in [5, 10, 20, 50, 100]:
        subset = results_df[results_df['k'] == k]
        ax.plot(range(len(subset)), subset['precision'], marker='o', linewidth=2, label=f'P@{k}')

    ax.set_xticks(range(len(df['length_bin'].cat.categories)))
    ax.set_xticklabels(df['length_bin'].cat.categories, rotation=45)
    ax.set_xlabel('Passage Length Bin (tokens)')
    ax.set_ylabel('Precision')
    ax.set_title('Retrieval Precision by Passage Length')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_file = f'{output_dir}/05_precision_by_length_{dataset_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {output_file}")



def main(datapath="data/seal_output.json"):
    """Run all analyses (streaming version)."""
    import sys
    from io import StringIO

    script_name = "analyze_length_bias"
    print(f"running {script_name}")

    try:
        dataset_name = get_dataset_name(datapath)
        output_dir = Path(f"generated_data/{dataset_name}/length_bias_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)

        log_file = output_dir / f"{script_name}_log.txt"
        original_stdout = sys.stdout
        sys.stdout = open(log_file, 'w', encoding='utf-8')

        print(f"\nOutput directory: {output_dir}\n")
        print(f"Dataset: {dataset_name}\n")

        df = extract_analysis_data(datapath, datapath)

        print("\nRunning analyses...")
        corr_df = analyze_length_rank_correlation(df, output_dir, dataset_name)
        analyze_length_score_correlation(df, output_dir, dataset_name)
        analyze_positive_vs_negative_lengths(df, output_dir, dataset_name)
        analyze_ngrams_vs_length(df, output_dir, dataset_name)
        analyze_retrieval_quality_by_length(df, output_dir, dataset_name)

        processed_file = f'{output_dir}/processed_data_{dataset_name}.csv'
        correlations_file = f'{output_dir}/correlations_{dataset_name}.csv'
        df.to_csv(processed_file, index=False, encoding='utf-8')
        if len(corr_df) > 0:
            corr_df.to_csv(correlations_file, index=False, encoding='utf-8')
        print(f"\n  Saved: {processed_file}")
        print(f"  Saved: {correlations_file}")

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
    main(datapath)
