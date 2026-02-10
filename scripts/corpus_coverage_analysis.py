#!/usr/bin/env python
"""
Corpus Coverage Analysis for SEAL/MINDER

This script analyzes whether query term availability in the corpus
correlates with retrieval success.

Research Question: Do queries with more terms in the corpus perform better?

Usage:
    python scripts/corpus_coverage_analysis.py data/seal_output.json seal_nq/SEAL-checkpoint+index.NQ/NQ.fm_index
"""

import json
import sys
from collections import defaultdict
from seal.index import FMIndex
from utils.utils import stream_data, get_tokenizer, get_ground_truth_ids
import numpy as np


def analyze_corpus_coverage(fm_index, output_data_path, dataset_name, sample_size=None):
    """
    Analyze correlation between corpus coverage and retrieval success.

    Args:
        fm_index: Loaded SEAL FM-Index
        output_data_path: Path to SEAL/MINDER output JSON
        dataset_name: Name of dataset (for output path)
        sample_size: Number of queries to analyze (None = all queries)

    Returns:
        Dict with analysis results
    """
    print("\n" + "="*70)
    print("CORPUS COVERAGE ANALYSIS")
    print("="*70)
    print(f"Dataset: {dataset_name}")
    if sample_size:
        print(f"Sample size: {sample_size}")
    else:
        print(f"Analyzing ALL queries")
    print()

    tokenizer = get_tokenizer()
    all_data = []

    count = 0
    for query_data in stream_data(output_data_path):
        if sample_size and count >= sample_size:
            break
        count += 1

        # Get query and ground truth
        gold_ids = get_ground_truth_ids(query_data)
        question = query_data.get('question', '')

        # Calculate coverage: % of query words that exist in corpus
        # Tokenize, then decode to get words (handles subword tokenization)
        query_tokens = tokenizer.encode(question.lower(), add_special_tokens=False)
        query_words = tokenizer.decode(query_tokens).split()

        if not query_words:
            continue

        # Check which words exist in corpus (word-level, not token-level)
        words_in_corpus = 0
        for word in query_words:
            # Encode the word to token IDs and check if it exists
            word_tokens = tokenizer.encode(word, add_special_tokens=False)
            if word_tokens:
                word_count = fm_index.get_count(word_tokens)
                if word_count > 0:
                    words_in_corpus += 1

        coverage_score = words_in_corpus / len(query_words)

        # Check retrieval success
        retrieved_ids = [ctx.get('passage_id', '') for ctx in query_data.get('ctxs', [])[:10]]
        hits_1 = retrieved_ids[0] in gold_ids if retrieved_ids else False
        hits_10 = any(pid in gold_ids for pid in retrieved_ids)

        all_data.append({
            "coverage": coverage_score,
            "hits_1": hits_1,
            "hits_10": hits_10,
            "question": question
        })

    print(f"Analyzed {len(all_data)} queries\n")

    # Calculate coverage statistics
    coverages = [d["coverage"] for d in all_data]
    mean_coverage = np.mean(coverages)
    median_coverage = np.median(coverages)
    std_coverage = np.std(coverages)
    pct_high_coverage = sum(1 for c in coverages if c >= 0.7) / len(coverages) * 100

    print("COVERAGE STATISTICS:")
    print(f"  Mean coverage:   {mean_coverage:.1%}")
    print(f"  Median coverage: {median_coverage:.1%}")
    print(f"  Queries with ≥70% coverage: {pct_high_coverage:.1f}%")
    print()

    # Calculate correlations with p-values
    from scipy import stats

    hits_1_vals = [1 if d["hits_1"] else 0 for d in all_data]
    hits_10_vals = [1 if d["hits_10"] else 0 for d in all_data]

    corr_hits_1, p_hits_1 = stats.pearsonr(coverages, hits_1_vals)
    corr_hits_10, p_hits_10 = stats.pearsonr(coverages, hits_10_vals)

    print("CORRELATION RESULTS (Pearson's r):")
    print(f"  Coverage vs Hits@1:  r = {corr_hits_1:.4f}, p = {p_hits_1:.4f}")
    print(f"  Coverage vs Hits@10: r = {corr_hits_10:.4f}, p = {p_hits_10:.4f}")

    # Interpret statistical significance
    if p_hits_1 < 0.001:
        sig_1 = "***"
    elif p_hits_1 < 0.01:
        sig_1 = "**"
    elif p_hits_1 < 0.05:
        sig_1 = "*"
    else:
        sig_1 = "n.s."

    if p_hits_10 < 0.001:
        sig_10 = "***"
    elif p_hits_10 < 0.01:
        sig_10 = "**"
    elif p_hits_10 < 0.05:
        sig_10 = "*"
    else:
        sig_10 = "n.s."

    print(f"\n  Significance: * p<0.05, ** p<0.01, *** p<0.001, n.s. = not significant")
    print(f"  Hits@1:  {sig_1}")
    print(f"  Hits@10: {sig_10}")
    print()

    # Create deciles for detailed analysis
    all_data.sort(key=lambda x: x["coverage"])
    decile_size = len(all_data) // 10

    deciles = {}
    for i in range(10):
        start_idx = i * decile_size
        end_idx = (i + 1) * decile_size if i < 9 else len(all_data)
        decile_data = all_data[start_idx:end_idx]

        if decile_data:
            hits_1_rate = sum(d["hits_1"] for d in decile_data) / len(decile_data) * 100
            hits_1_std = np.std([d["hits_1"] for d in decile_data]) * 100
            hits_10_rate = sum(d["hits_10"] for d in decile_data) / len(decile_data) * 100
            hits_10_std = np.std([d["hits_10"] for d in decile_data]) * 100
            min_cov = min(d["coverage"] for d in decile_data)
            max_cov = max(d["coverage"] for d in decile_data)
            coverage_std = np.std([d["coverage"] for d in decile_data])

        deciles[f"D{i+1}"] = {
            "coverage_range": [min_cov, max_cov],
            "coverage_std": coverage_std,
            "hits_1_rate": hits_1_rate,
            "hits_1_std": hits_1_std,
            "hits_10_rate": hits_10_rate,
            "hits_10_std": hits_10_std,
            "count": len(decile_data)
        }


    print("DECILE BREAKDOWN:")
    print(f"  {'Decile':<8} {'Coverage':<20} {'Hits@1':<10} {'Hits@10':<10}")
    print(f"  {'-'*60}")
    for decile_name, data in sorted(deciles.items()):
        min_c, max_c = data["coverage_range"]
        h1 = data["hits_1_rate"]
        h10 = data["hits_10_rate"]
        print(f"  {decile_name:<8} {min_c:.2f} - {max_c:.2f}         {h1:>6.1f}%    {h10:>6.1f}%")

    return {
        "total_analyzed": len(all_data),
        "mean_coverage": float(mean_coverage),
        "median_coverage": float(median_coverage),
        "std_coverage": float(std_coverage),
        "pct_high_coverage": float(pct_high_coverage),
        "correlation_hits_1": float(corr_hits_1),
        "p_value_hits_1": float(p_hits_1),
        "correlation_hits_10": float(corr_hits_10),
        "p_value_hits_10": float(p_hits_10),
        "deciles": deciles
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python corpus_coverage_analysis.py <output_json> <fm_index_base_path>")
        print("\nExample:")
        print("  python scripts/corpus_coverage_analysis.py \\")
        print("         data/seal_output.json \\")
        print("         seal_nq/SEAL-checkpoint+index.NQ/NQ.fm_index")
        sys.exit(1)

    output_json = sys.argv[1]
    fm_index_path = sys.argv[2]

    # Determine dataset name
    if "seal" in output_json.lower():
        dataset_name = "seal"
    elif "minder" in output_json.lower():
        dataset_name = "minder"
    else:
        dataset_name = "unknown"

    print("\n" + "="*70)
    print("CORPUS COVERAGE ANALYSIS")
    print("="*70)
    print(f"Output JSON: {output_json}")
    print(f"FM-Index: {fm_index_path}")
    print()

    # Load FM-Index
    print("Loading FM-Index (this may take 30-60 seconds)...")
    fm_index = FMIndex.load(fm_index_path)
    print(f"✓ Loaded: {fm_index.n_docs:,} documents\n")

    # Run analysis
    results = analyze_corpus_coverage(fm_index, output_json, dataset_name)

    # Save results
    output_dir = f"generated_data/{dataset_name}"
    import os
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "corpus_coverage_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✓ Results saved to: {output_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
