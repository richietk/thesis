"""
Comprehensive Failure Mode Analysis for SEAL Outputs

This script analyzes observable failure modes in SEAL's generative retrieval:
1. Repetitive Generation - semantic redundancy in n-grams
2. Additive Scoring Bias - quantity over quality in scoring
3. Query-N-gram Overlap - connection between query and generated n-grams
4. Answer Coverage - whether answer appears in generated n-grams
5. Scoring Failure - correct passage retrieved but mis-ranked
6. N-gram Frequency Analysis - diagnostic metric

Author: Richard
Date: 2026-01-07
"""

import json
import ast
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from typing import List, Dict, Tuple, Set
import sys

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = "generated_data"

# ============================================================================
# Data Loading
# ============================================================================

def load_data(path: str) -> List[Dict]:
    """Load SEAL output JSON."""
    print(f"Loading data from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} queries\n")
    return data


def parse_ngrams(keys_str: str) -> List[Tuple[str, int, float]]:
    """
    Parse n-gram keys from string format.

    Returns:
        List of (ngram_text, corpus_frequency, score) tuples
    """
    if not keys_str:
        return []

    try:
        keys_list = ast.literal_eval(keys_str)
        return [(ngram, freq, score) for ngram, freq, score in keys_list]
    except:
        return []


# ============================================================================
# Failure Mode 1: Repetitive Generation
# ============================================================================

def analyze_repetitive_generation(data: List[Dict]) -> pd.DataFrame:
    """
    Analyze semantic redundancy in generated n-grams.

    Metrics:
    - Token diversity ratio: unique_tokens / total_tokens
    - Token repetition: most frequently repeated tokens
    - Success rate correlation with diversity
    """
    print("="*80)
    print("FAILURE MODE 1: REPETITIVE GENERATION")
    print("="*80)

    results = []

    for entry in data:
        query = entry['question']

        # Check if retrieval succeeded
        positive_ids = set()
        if 'positive_ctxs' in entry and entry['positive_ctxs']:
            positive_ids = {ctx['passage_id'] for ctx in entry['positive_ctxs']}

        top_ctx = entry['ctxs'][0] if entry.get('ctxs') else None
        if not top_ctx:
            continue

        success = top_ctx['passage_id'] in positive_ids

        # Parse n-grams
        ngrams = parse_ngrams(top_ctx.get('keys', ''))
        if not ngrams:
            continue

        # Extract all tokens from all n-grams
        all_ngram_text = ' '.join([ng[0] for ng in ngrams])
        all_tokens = all_ngram_text.split()

        if len(all_tokens) == 0:
            continue

        unique_tokens = set(all_tokens)
        diversity_ratio = len(unique_tokens) / len(all_tokens)

        # Count token repetitions
        token_counts = Counter(all_tokens)
        max_repetition = max(token_counts.values()) if token_counts else 0
        avg_repetition = np.mean(list(token_counts.values())) if token_counts else 0

        # Top repeated tokens
        most_repeated = token_counts.most_common(3)

        results.append({
            'query': query,
            'success': success,
            'num_ngrams': len(ngrams),
            'total_tokens': len(all_tokens),
            'unique_tokens': len(unique_tokens),
            'diversity_ratio': diversity_ratio,
            'max_repetition': max_repetition,
            'avg_repetition': avg_repetition,
            'most_repeated_token': most_repeated[0][0] if most_repeated else '',
            'most_repeated_count': most_repeated[0][1] if most_repeated else 0
        })

    df = pd.DataFrame(results)

    # Statistics
    success_high_diversity = df[df['diversity_ratio'] > 0.85]['success'].mean()
    success_low_diversity = df[df['diversity_ratio'] < 0.70]['success'].mean()

    print(f"\nAnalyzed {len(df)} queries")
    print(f"\nDiversity Statistics:")
    print(f"  Mean diversity ratio: {df['diversity_ratio'].mean():.3f}")
    print(f"  Median diversity ratio: {df['diversity_ratio'].median():.3f}")
    print(f"  Std: {df['diversity_ratio'].std():.3f}")

    print(f"\nSuccess Rate by Diversity:")
    print(f"  High diversity (>0.85): {success_high_diversity:.2%} ({sum(df['diversity_ratio'] > 0.85)} queries)")
    print(f"  Low diversity (<0.70): {success_low_diversity:.2%} ({sum(df['diversity_ratio'] < 0.70)} queries)")
    print(f"  Difference: {success_high_diversity - success_low_diversity:+.2%}")

    # Correlation
    corr, p_val = spearmanr(df['diversity_ratio'], df['success'])
    print(f"\nSpearman correlation (diversity vs success): ρ = {corr:.3f}, p = {p_val:.4e}")

    if corr > 0.1 and p_val < 0.05:
        print("  → SIGNIFICANT: Higher diversity correlates with better retrieval")

    return df


# ============================================================================
# Failure Mode 2: Additive Scoring Bias (N-gram Quantity)
# ============================================================================

def analyze_additive_scoring_bias(data: List[Dict]) -> pd.DataFrame:
    """
    Analyze whether passages with many weak n-grams outrank passages with few strong n-grams.

    Metrics:
    - Number of n-grams per passage
    - Correlation: num_ngrams vs rank
    - Correlation: num_ngrams vs score
    - Compare: successful vs failed queries
    """
    print("\n" + "="*80)
    print("FAILURE MODE 2: ADDITIVE SCORING BIAS (N-gram Quantity)")
    print("="*80)

    results = []

    for entry in data:
        query = entry['question']

        # Ground truth
        positive_ids = set()
        if 'positive_ctxs' in entry and entry['positive_ctxs']:
            positive_ids = {ctx['passage_id'] for ctx in entry['positive_ctxs']}

        # Analyze top-10 passages
        for rank, ctx in enumerate(entry.get('ctxs', [])[:10], start=1):
            ngrams = parse_ngrams(ctx.get('keys', ''))

            if not ngrams:
                continue

            scores = [score for _, _, score in ngrams]

            results.append({
                'query': query,
                'rank': rank,
                'passage_id': ctx['passage_id'],
                'is_positive': ctx['passage_id'] in positive_ids,
                'num_ngrams': len(ngrams),
                'total_score': sum(scores),
                'avg_ngram_score': np.mean(scores),
                'max_ngram_score': max(scores),
                'score_std': np.std(scores)
            })

    df = pd.DataFrame(results)

    # Analysis: Do passages with more n-grams rank higher?
    corr_num_rank, p_num_rank = spearmanr(df['num_ngrams'], df['rank'])
    corr_num_score, p_num_score = spearmanr(df['num_ngrams'], df['total_score'])

    print(f"\nAnalyzed {len(df)} passage retrievals")
    print(f"\nN-gram Count Statistics:")
    print(f"  Mean n-grams per passage: {df['num_ngrams'].mean():.1f}")
    print(f"  Median: {df['num_ngrams'].median():.1f}")
    print(f"  Range: {df['num_ngrams'].min()} - {df['num_ngrams'].max()}")

    print(f"\nCorrelations:")
    print(f"  Num n-grams vs Rank: ρ = {corr_num_rank:.3f}, p = {p_num_rank:.4e}")
    print(f"  Num n-grams vs Total Score: ρ = {corr_num_score:.3f}, p = {p_num_score:.4e}")

    if corr_num_score > 0.3 and p_num_score < 0.05:
        print("  → STRONG BIAS: More n-grams = higher scores")

    # Compare positive vs negative passages
    positive = df[df['is_positive'] == True]
    negative = df[df['is_positive'] == False]

    if len(positive) > 0 and len(negative) > 0:
        print(f"\nPositive vs Negative Passages:")
        print(f"  Positive: {positive['num_ngrams'].mean():.1f} n-grams on average")
        print(f"  Negative: {negative['num_ngrams'].mean():.1f} n-grams on average")

        stat, p = mannwhitneyu(positive['num_ngrams'], negative['num_ngrams'], alternative='two-sided')
        if p < 0.05:
            diff = negative['num_ngrams'].mean() - positive['num_ngrams'].mean()
            print(f"  → Significant difference (p={p:.4e}): {diff:+.1f} n-grams")

    return df


# ============================================================================
# Failure Mode 3: Query-N-gram Overlap
# ============================================================================

def analyze_query_ngram_overlap(data: List[Dict]) -> pd.DataFrame:
    """
    Analyze lexical overlap between query and generated n-grams.

    Metrics:
    - Jaccard similarity: intersection / union
    - Percentage of query tokens in n-grams
    - Success rate by overlap level
    """
    print("\n" + "="*80)
    print("FAILURE MODE 3: QUERY-N-GRAM OVERLAP")
    print("="*80)

    results = []

    for entry in data:
        query = entry['question']
        query_tokens = set(query.lower().split())

        # Ground truth
        positive_ids = set()
        if 'positive_ctxs' in entry and entry['positive_ctxs']:
            positive_ids = {ctx['passage_id'] for ctx in entry['positive_ctxs']}

        top_ctx = entry['ctxs'][0] if entry.get('ctxs') else None
        if not top_ctx:
            continue

        success = top_ctx['passage_id'] in positive_ids

        # Parse n-grams
        ngrams = parse_ngrams(top_ctx.get('keys', ''))
        if not ngrams:
            continue

        # Extract tokens from n-grams
        ngram_text = ' '.join([ng[0].lower() for ng in ngrams])
        ngram_tokens = set(ngram_text.split())

        # Calculate overlap
        intersection = query_tokens & ngram_tokens
        union = query_tokens | ngram_tokens

        jaccard = len(intersection) / len(union) if len(union) > 0 else 0
        query_coverage = len(intersection) / len(query_tokens) if len(query_tokens) > 0 else 0

        results.append({
            'query': query,
            'success': success,
            'num_query_tokens': len(query_tokens),
            'num_ngram_tokens': len(ngram_tokens),
            'overlap_tokens': len(intersection),
            'jaccard_similarity': jaccard,
            'query_coverage': query_coverage
        })

    df = pd.DataFrame(results)

    # Statistics
    success_high_overlap = df[df['query_coverage'] > 0.6]['success'].mean()
    success_low_overlap = df[df['query_coverage'] < 0.3]['success'].mean()

    print(f"\nAnalyzed {len(df)} queries")
    print(f"\nOverlap Statistics:")
    print(f"  Mean Jaccard similarity: {df['jaccard_similarity'].mean():.3f}")
    print(f"  Mean query coverage: {df['query_coverage'].mean():.3f}")

    print(f"\nSuccess Rate by Query Coverage:")
    print(f"  High coverage (>60%): {success_high_overlap:.2%} ({sum(df['query_coverage'] > 0.6)} queries)")
    print(f"  Low coverage (<30%): {success_low_overlap:.2%} ({sum(df['query_coverage'] < 0.3)} queries)")
    print(f"  Difference: {success_high_overlap - success_low_overlap:+.2%}")

    # Correlation
    corr, p_val = spearmanr(df['query_coverage'], df['success'])
    print(f"\nSpearman correlation (query coverage vs success): ρ = {corr:.3f}, p = {p_val:.4e}")

    return df


# ============================================================================
# Failure Mode 4: Answer Coverage
# ============================================================================

def analyze_answer_coverage(data: List[Dict]) -> pd.DataFrame:
    """
    Analyze whether answer string appears in generated n-grams.

    Metrics:
    - Percentage of queries where answer appears in n-grams
    - Success rate when answer is/isn't in n-grams
    """
    print("\n" + "="*80)
    print("FAILURE MODE 4: ANSWER COVERAGE")
    print("="*80)

    results = []

    for entry in data:
        query = entry['question']
        answers = entry.get('answers', [])

        if not answers:
            continue

        # Ground truth
        positive_ids = set()
        if 'positive_ctxs' in entry and entry['positive_ctxs']:
            positive_ids = {ctx['passage_id'] for ctx in entry['positive_ctxs']}

        top_ctx = entry['ctxs'][0] if entry.get('ctxs') else None
        if not top_ctx:
            continue

        success = top_ctx['passage_id'] in positive_ids

        # Parse n-grams
        ngrams = parse_ngrams(top_ctx.get('keys', ''))
        if not ngrams:
            continue

        # Check if any answer appears in any n-gram
        ngram_text = ' '.join([ng[0].lower() for ng in ngrams])

        answer_in_ngrams = any(ans.lower() in ngram_text for ans in answers)

        results.append({
            'query': query,
            'answer': answers[0] if answers else '',
            'success': success,
            'answer_in_ngrams': answer_in_ngrams,
            'num_ngrams': len(ngrams)
        })

    df = pd.DataFrame(results)

    # Statistics
    answer_present = df[df['answer_in_ngrams'] == True]
    answer_absent = df[df['answer_in_ngrams'] == False]

    print(f"\nAnalyzed {len(df)} queries")
    print(f"\nAnswer Coverage:")
    print(f"  Answer in n-grams: {len(answer_present)} ({100*len(answer_present)/len(df):.1f}%)")
    print(f"  Answer NOT in n-grams: {len(answer_absent)} ({100*len(answer_absent)/len(df):.1f}%)")

    if len(answer_present) > 0 and len(answer_absent) > 0:
        print(f"\nSuccess Rate:")
        print(f"  When answer in n-grams: {answer_present['success'].mean():.2%}")
        print(f"  When answer NOT in n-grams: {answer_absent['success'].mean():.2%}")
        print(f"  Difference: {answer_present['success'].mean() - answer_absent['success'].mean():+.2%}")

    return df


# ============================================================================
# Failure Mode 5: Scoring Failure
# ============================================================================

def analyze_scoring_failure(data: List[Dict]) -> pd.DataFrame:
    """
    Analyze cases where correct passage is retrieved but mis-ranked.

    Metrics:
    - Percentage of queries where ground truth is in top-10 but not rank-1
    - Average rank of ground truth when it's retrieved
    - Score differential between rank-1 and ground truth
    """
    print("\n" + "="*80)
    print("FAILURE MODE 5: SCORING FAILURE")
    print("="*80)

    results = []

    for entry in data:
        query = entry['question']

        # Ground truth
        positive_ids = set()
        if 'positive_ctxs' in entry and entry['positive_ctxs']:
            positive_ids = {ctx['passage_id'] for ctx in entry['positive_ctxs']}

        if not positive_ids:
            continue

        # Find ground truth in top-K
        gt_rank = None
        gt_score = None
        rank1_score = None

        for rank, ctx in enumerate(entry.get('ctxs', [])[:100], start=1):
            if rank == 1:
                rank1_score = ctx.get('score', 0)

            if ctx['passage_id'] in positive_ids:
                gt_rank = rank
                gt_score = ctx.get('score', 0)
                break

        if gt_rank is not None:
            scoring_failure = (gt_rank > 1)

            results.append({
                'query': query,
                'gt_rank': gt_rank,
                'gt_score': gt_score,
                'rank1_score': rank1_score,
                'score_diff': rank1_score - gt_score if (rank1_score and gt_score) else None,
                'scoring_failure': scoring_failure,
                'retrieved': True
            })
        else:
            results.append({
                'query': query,
                'gt_rank': None,
                'gt_score': None,
                'rank1_score': rank1_score,
                'score_diff': None,
                'scoring_failure': False,
                'retrieved': False
            })

    df = pd.DataFrame(results)

    retrieved = df[df['retrieved'] == True]
    scoring_failures = df[df['scoring_failure'] == True]

    print(f"\nAnalyzed {len(df)} queries")
    print(f"\nRetrieval Statistics:")
    print(f"  Ground truth retrieved (in top-100): {len(retrieved)} ({100*len(retrieved)/len(df):.1f}%)")
    print(f"  Ground truth NOT retrieved: {len(df) - len(retrieved)} ({100*(len(df)-len(retrieved))/len(df):.1f}%)")

    if len(retrieved) > 0:
        print(f"\nScoring Failure:")
        print(f"  Retrieved but mis-ranked (rank > 1): {len(scoring_failures)} ({100*len(scoring_failures)/len(retrieved):.1f}% of retrieved)")
        print(f"  Retrieved and correctly ranked (rank = 1): {len(retrieved) - len(scoring_failures)}")

        print(f"\nAverage Ground Truth Rank: {retrieved['gt_rank'].mean():.1f}")
        print(f"Median Ground Truth Rank: {retrieved['gt_rank'].median():.1f}")

        if len(scoring_failures) > 0 and scoring_failures['score_diff'].notna().any():
            print(f"\nScore Differential (Rank-1 score - Ground truth score):")
            print(f"  Mean: {scoring_failures['score_diff'].mean():.2f}")
            print(f"  Median: {scoring_failures['score_diff'].median():.2f}")

    return df


# ============================================================================
# Diagnostic: N-gram Frequency Analysis
# ============================================================================

def analyze_ngram_frequency(data: List[Dict]) -> pd.DataFrame:
    """
    Analyze corpus frequency of generated n-grams.

    This is a diagnostic metric to understand other failures.

    Metrics:
    - Average frequency of top-K n-grams
    - Percentage of high-frequency n-grams (freq > threshold)
    - Correlation with success
    """
    print("\n" + "="*80)
    print("DIAGNOSTIC: N-GRAM FREQUENCY ANALYSIS")
    print("="*80)

    results = []

    for entry in data:
        query = entry['question']

        # Ground truth
        positive_ids = set()
        if 'positive_ctxs' in entry and entry['positive_ctxs']:
            positive_ids = {ctx['passage_id'] for ctx in entry['positive_ctxs']}

        top_ctx = entry['ctxs'][0] if entry.get('ctxs') else None
        if not top_ctx:
            continue

        success = top_ctx['passage_id'] in positive_ids

        # Parse n-grams
        ngrams = parse_ngrams(top_ctx.get('keys', ''))
        if not ngrams:
            continue

        # Extract frequencies and scores
        frequencies = [freq for _, freq, _ in ngrams]
        scores = [score for _, _, score in ngrams]

        # Top-K n-grams by score
        sorted_ngrams = sorted(ngrams, key=lambda x: x[2], reverse=True)
        top_5_freq = [ng[1] for ng in sorted_ngrams[:5]]

        # High-frequency count (e.g., > 1000)
        high_freq_count = sum(1 for freq in frequencies if freq > 1000)
        high_freq_pct = high_freq_count / len(frequencies) if frequencies else 0

        results.append({
            'query': query,
            'success': success,
            'num_ngrams': len(ngrams),
            'avg_frequency': np.mean(frequencies),
            'median_frequency': np.median(frequencies),
            'max_frequency': max(frequencies),
            'min_frequency': min(frequencies),
            'avg_top5_frequency': np.mean(top_5_freq) if top_5_freq else 0,
            'high_freq_count': high_freq_count,
            'high_freq_pct': high_freq_pct
        })

    df = pd.DataFrame(results)

    # Statistics
    print(f"\nAnalyzed {len(df)} queries")
    print(f"\nFrequency Statistics:")
    print(f"  Mean avg frequency: {df['avg_frequency'].mean():.1f}")
    print(f"  Mean median frequency: {df['median_frequency'].mean():.1f}")
    print(f"  Mean of top-5 scored n-grams frequency: {df['avg_top5_frequency'].mean():.1f}")

    print(f"\nHigh-Frequency N-grams (freq > 1000):")
    print(f"  Average count per query: {df['high_freq_count'].mean():.1f}")
    print(f"  Average percentage: {df['high_freq_pct'].mean():.2%}")

    # Success rate by frequency
    success_low_freq = df[df['avg_top5_frequency'] < 500]['success'].mean()
    success_high_freq = df[df['avg_top5_frequency'] > 2000]['success'].mean()

    print(f"\nSuccess Rate by Top-5 Frequency:")
    print(f"  Low frequency (<500): {success_low_freq:.2%} ({sum(df['avg_top5_frequency'] < 500)} queries)")
    print(f"  High frequency (>2000): {success_high_freq:.2%} ({sum(df['avg_top5_frequency'] > 2000)} queries)")
    print(f"  Difference: {success_low_freq - success_high_freq:+.2%}")

    # Correlation
    corr, p_val = spearmanr(df['avg_top5_frequency'], df['success'])
    print(f"\nSpearman correlation (avg top-5 frequency vs success): ρ = {corr:.3f}, p = {p_val:.4e}")

    if corr < -0.15 and p_val < 0.05:
        print("  → SIGNIFICANT: Lower frequency correlates with better retrieval")

    return df


# ============================================================================
# Visualization
# ============================================================================

def create_visualizations(
    repetitive_df: pd.DataFrame,
    scoring_bias_df: pd.DataFrame,
    query_overlap_df: pd.DataFrame,
    answer_coverage_df: pd.DataFrame,
    scoring_failure_df: pd.DataFrame,
    frequency_df: pd.DataFrame,
    output_dir: Path
):
    """Create comprehensive visualizations for all failure modes."""

    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('SEAL Failure Mode Analysis', fontsize=16, fontweight='bold')

    # 1. Repetitive Generation
    ax = axes[0, 0]
    bins = np.linspace(0.5, 1.0, 20)
    success = repetitive_df[repetitive_df['success'] == True]['diversity_ratio']
    failure = repetitive_df[repetitive_df['success'] == False]['diversity_ratio']
    ax.hist([success, failure], bins=bins, label=['Success', 'Failure'], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Token Diversity Ratio')
    ax.set_ylabel('Count')
    ax.set_title('1. Repetitive Generation')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Additive Scoring Bias
    ax = axes[0, 1]
    sample = scoring_bias_df.sample(min(1000, len(scoring_bias_df)))
    positive = sample[sample['is_positive'] == True]
    negative = sample[sample['is_positive'] == False]
    ax.scatter(negative['num_ngrams'], negative['total_score'], alpha=0.3, s=10, label='Negative', color='red')
    ax.scatter(positive['num_ngrams'], positive['total_score'], alpha=0.6, s=20, label='Positive', color='green')
    ax.set_xlabel('Number of N-grams')
    ax.set_ylabel('Total Score')
    ax.set_title('2. Additive Scoring Bias')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Query-N-gram Overlap
    ax = axes[1, 0]
    bins = np.linspace(0, 1, 20)
    success = query_overlap_df[query_overlap_df['success'] == True]['query_coverage']
    failure = query_overlap_df[query_overlap_df['success'] == False]['query_coverage']
    ax.hist([success, failure], bins=bins, label=['Success', 'Failure'], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Query Coverage (% query tokens in n-grams)')
    ax.set_ylabel('Count')
    ax.set_title('3. Query-N-gram Overlap')
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Answer Coverage
    ax = axes[1, 1]
    answer_in = answer_coverage_df[answer_coverage_df['answer_in_ngrams'] == True]['success'].mean()
    answer_not_in = answer_coverage_df[answer_coverage_df['answer_in_ngrams'] == False]['success'].mean()
    ax.bar(['Answer in\nN-grams', 'Answer NOT\nin N-grams'], [answer_in, answer_not_in],
           color=['green', 'red'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Success Rate')
    ax.set_title('4. Answer Coverage')
    ax.set_ylim([0, 1])
    for i, v in enumerate([answer_in, answer_not_in]):
        ax.text(i, v + 0.03, f'{v:.2%}', ha='center', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # 5. Scoring Failure
    ax = axes[2, 0]
    retrieved = scoring_failure_df[scoring_failure_df['retrieved'] == True]
    if len(retrieved) > 0:
        rank_counts = retrieved['gt_rank'].value_counts().sort_index()
        ax.bar(rank_counts.index[:20], rank_counts.values[:20], alpha=0.7, edgecolor='black', color='steelblue')
        ax.set_xlabel('Rank of Ground Truth')
        ax.set_ylabel('Count')
        ax.set_title('5. Scoring Failure - Ground Truth Rank Distribution')
        ax.grid(alpha=0.3, axis='y')

    # 6. N-gram Frequency
    ax = axes[2, 1]
    bins = np.logspace(0, 5, 30)
    success = frequency_df[frequency_df['success'] == True]['avg_top5_frequency']
    failure = frequency_df[frequency_df['success'] == False]['avg_top5_frequency']
    ax.hist([success, failure], bins=bins, label=['Success', 'Failure'], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Average Top-5 N-gram Frequency (log scale)')
    ax.set_ylabel('Count')
    ax.set_title('6. N-gram Frequency (Diagnostic)')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'failure_mode_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'failure_mode_analysis.png'}")


# ============================================================================
# Summary Report
# ============================================================================

def generate_summary_report(
    repetitive_df: pd.DataFrame,
    scoring_bias_df: pd.DataFrame,
    query_overlap_df: pd.DataFrame,
    answer_coverage_df: pd.DataFrame,
    scoring_failure_df: pd.DataFrame,
    frequency_df: pd.DataFrame,
    output_dir: Path,
    datapath: str
):
    """Generate comprehensive summary report."""

    report_path = output_dir / 'FAILURE_MODE_SUMMARY.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SEAL FAILURE MODE ANALYSIS - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Dataset: {datapath}\n")
        f.write(f"Total queries analyzed: {len(repetitive_df)}\n\n")

        f.write("-"*80 + "\n")
        f.write("FAILURE MODE FINDINGS\n")
        f.write("-"*80 + "\n\n")

        # 1. Repetitive Generation
        f.write("1. REPETITIVE GENERATION\n")
        f.write(f"   Mean diversity ratio: {repetitive_df['diversity_ratio'].mean():.3f}\n")
        success_high = repetitive_df[repetitive_df['diversity_ratio'] > 0.85]['success'].mean()
        success_low = repetitive_df[repetitive_df['diversity_ratio'] < 0.70]['success'].mean()
        f.write(f"   Success rate (high diversity >0.85): {success_high:.2%}\n")
        f.write(f"   Success rate (low diversity <0.70): {success_low:.2%}\n")
        if success_high - success_low > 0.1:
            f.write(f"   → IMPACT: {success_high - success_low:+.2%} difference\n")
        f.write("\n")

        # 2. Additive Scoring Bias
        f.write("2. ADDITIVE SCORING BIAS\n")
        positive = scoring_bias_df[scoring_bias_df['is_positive'] == True]
        negative = scoring_bias_df[scoring_bias_df['is_positive'] == False]
        f.write(f"   Avg n-grams in positive passages: {positive['num_ngrams'].mean():.1f}\n")
        f.write(f"   Avg n-grams in negative passages: {negative['num_ngrams'].mean():.1f}\n")
        corr, _ = spearmanr(scoring_bias_df['num_ngrams'], scoring_bias_df['total_score'])
        f.write(f"   Correlation (num n-grams vs score): ρ = {corr:.3f}\n")
        if corr > 0.3:
            f.write(f"   → STRONG BIAS: More n-grams = higher scores\n")
        f.write("\n")

        # 3. Query-N-gram Overlap
        f.write("3. QUERY-N-GRAM OVERLAP\n")
        f.write(f"   Mean query coverage: {query_overlap_df['query_coverage'].mean():.3f}\n")
        corr, _ = spearmanr(query_overlap_df['query_coverage'], query_overlap_df['success'])
        f.write(f"   Correlation (coverage vs success): ρ = {corr:.3f}\n")
        f.write("\n")

        # 4. Answer Coverage
        f.write("4. ANSWER COVERAGE\n")
        answer_in = answer_coverage_df[answer_coverage_df['answer_in_ngrams'] == True]
        answer_not = answer_coverage_df[answer_coverage_df['answer_in_ngrams'] == False]
        f.write(f"   Answer in n-grams: {len(answer_in)} ({100*len(answer_in)/len(answer_coverage_df):.1f}%)\n")
        f.write(f"   Success when answer present: {answer_in['success'].mean():.2%}\n")
        f.write(f"   Success when answer absent: {answer_not['success'].mean():.2%}\n")
        f.write("\n")

        # 5. Scoring Failure
        f.write("5. SCORING FAILURE\n")
        retrieved = scoring_failure_df[scoring_failure_df['retrieved'] == True]
        mis_ranked = scoring_failure_df[scoring_failure_df['scoring_failure'] == True]
        f.write(f"   Ground truth retrieved: {len(retrieved)} ({100*len(retrieved)/len(scoring_failure_df):.1f}%)\n")
        f.write(f"   Retrieved but mis-ranked: {len(mis_ranked)} ({100*len(mis_ranked)/len(retrieved):.1f}% of retrieved)\n")
        f.write(f"   Avg rank when retrieved: {retrieved['gt_rank'].mean():.1f}\n")
        f.write("\n")

        # 6. N-gram Frequency
        f.write("6. N-GRAM FREQUENCY (DIAGNOSTIC)\n")
        f.write(f"   Mean avg frequency: {frequency_df['avg_frequency'].mean():.1f}\n")
        f.write(f"   Mean top-5 frequency: {frequency_df['avg_top5_frequency'].mean():.1f}\n")
        corr, _ = spearmanr(frequency_df['avg_top5_frequency'], frequency_df['success'])
        f.write(f"   Correlation (frequency vs success): ρ = {corr:.3f}\n")
        if corr < -0.15:
            f.write(f"   → Lower frequency n-grams correlate with success\n")
        f.write("\n")

        f.write("-"*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-"*80 + "\n\n")

        f.write("Based on this analysis:\n\n")

        if success_high - success_low > 0.1:
            f.write("• Repetitive Generation is a significant issue\n")
            f.write("  → Consider penalizing redundant n-grams during generation\n\n")

        if corr > 0.3:
            f.write("• Additive Scoring Bias exists\n")
            f.write("  → Consider normalizing scores by number of n-grams\n")
            f.write("  → Or use average score instead of sum\n\n")

        retrieved_pct = 100*len(retrieved)/len(scoring_failure_df)
        mis_ranked_pct = 100*len(mis_ranked)/len(retrieved) if len(retrieved) > 0 else 0
        if mis_ranked_pct > 30:
            f.write(f"• {mis_ranked_pct:.1f}% of retrieved passages are mis-ranked\n")
            f.write("  → Scoring mechanism needs refinement\n\n")

    print(f"\n  Saved summary: {report_path}")


# ============================================================================
# Main Execution
# ============================================================================

def main(datapath="data/seal_output.json"):
    """Run all failure mode analyses."""

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Load data
    data = load_data(datapath)

    # Run analyses
    print("\nRunning failure mode analyses...\n")

    repetitive_df = analyze_repetitive_generation(data)
    scoring_bias_df = analyze_additive_scoring_bias(data)
    query_overlap_df = analyze_query_ngram_overlap(data)
    answer_coverage_df = analyze_answer_coverage(data)
    scoring_failure_df = analyze_scoring_failure(data)
    frequency_df = analyze_ngram_frequency(data)

    # Create visualizations
    create_visualizations(
        repetitive_df,
        scoring_bias_df,
        query_overlap_df,
        answer_coverage_df,
        scoring_failure_df,
        frequency_df,
        output_dir
    )

    # Generate summary
    generate_summary_report(
        repetitive_df,
        scoring_bias_df,
        query_overlap_df,
        answer_coverage_df,
        scoring_failure_df,
        frequency_df,
        output_dir,
        datapath
    )

    # Save DataFrames
    repetitive_df.to_csv(output_dir / '01_repetitive_generation.csv', index=False)
    scoring_bias_df.to_csv(output_dir / '02_additive_scoring_bias.csv', index=False)
    query_overlap_df.to_csv(output_dir / '03_query_ngram_overlap.csv', index=False)
    answer_coverage_df.to_csv(output_dir / '04_answer_coverage.csv', index=False)
    scoring_failure_df.to_csv(output_dir / '05_scoring_failure.csv', index=False)
    frequency_df.to_csv(output_dir / '06_ngram_frequency.csv', index=False)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - 6 CSV files with detailed data")
    print(f"  - 1 visualization (PNG)")
    print(f"  - 1 summary report (TXT)")


if __name__ == "__main__":
    main()
