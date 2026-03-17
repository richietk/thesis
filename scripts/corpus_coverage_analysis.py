#!/usr/bin/env python
"""
Corpus Coverage Analysis for SEAL/MINDER

This script analyzes whether query term availability in the corpus
correlates with retrieval success.

The corpus vocabulary is loaded directly from the .oth pickle file
(no C++ FM-Index required). The .oth file stores the set of all
token IDs present in the corpus.

Research Question: Do queries with more terms in the corpus perform better?

Usage:
    # SEAL NQ
    python scripts/corpus_coverage_analysis.py data/seal_nq_output.json \
        wip_dirs/SEAL-checkpoint+index.NQ/NQ.fm_index

    # MINDER NQ
    python scripts/corpus_coverage_analysis.py data/minder_nq_output.json \
        wip_dirs/MINDER-checkpoint+index.NQ/NQ.fm_index

    # MINDER MSMARCO
    python scripts/corpus_coverage_analysis.py data/minder_msmarco_output.json \
        wip_dirs/MINDER-checkpoint+index.MSMARCO/MSMARCO.fm_index
"""

import json
import os
import pickle
import sys

_thesis_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_thesis_root, 'scripts'))

from utils.utils import stream_data, get_tokenizer, get_ground_truth_ids
import numpy as np


def load_corpus_vocab(fm_index_base_path):
    """
    Load the set of token IDs present in the corpus from the .oth pickle file.
    No C++ extension required.

    Returns:
        set of int token IDs
    """
    oth_path = fm_index_base_path + '.oth'
    print(f"Loading corpus vocabulary from {oth_path} ...")
    with open(oth_path, 'rb') as f:
        beginnings, occurring, labels = pickle.load(f)
    vocab = set(occurring)
    n_docs = len(beginnings) - 1
    print(f"✓ Loaded: {n_docs:,} documents, {len(vocab):,} unique tokens\n")
    return vocab


def analyze_corpus_coverage(corpus_vocab, output_data_path, dataset_name, sample_size=None):
    """
    Analyze correlation between corpus coverage and retrieval success.

    Coverage is defined as the fraction of query tokens whose token ID
    appears anywhere in the corpus vocabulary.

    Args:
        corpus_vocab: set of token IDs present in corpus (from load_corpus_vocab)
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

        gold_ids = get_ground_truth_ids(query_data)
        question = query_data.get('question', '')

        # Tokenize query and measure what fraction of tokens exist in corpus
        query_tokens = tokenizer.encode(question.lower(), add_special_tokens=False)
        if not query_tokens:
            continue

        tokens_in_corpus = sum(1 for t in query_tokens if t in corpus_vocab)
        coverage_score = tokens_in_corpus / len(query_tokens)

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

    # Coverage statistics
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

    # Correlations with p-values
    from scipy import stats

    hits_1_vals = [1 if d["hits_1"] else 0 for d in all_data]
    hits_10_vals = [1 if d["hits_10"] else 0 for d in all_data]

    corr_hits_1, p_hits_1 = stats.pearsonr(coverages, hits_1_vals)
    corr_hits_10, p_hits_10 = stats.pearsonr(coverages, hits_10_vals)

    print("CORRELATION RESULTS (Pearson's r):")
    print(f"  Coverage vs Hits@1:  r = {corr_hits_1:.4f}, p = {p_hits_1:.4f}")
    print(f"  Coverage vs Hits@10: r = {corr_hits_10:.4f}, p = {p_hits_10:.4f}")

    def sig_label(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "n.s."

    print(f"\n  Significance: * p<0.05, ** p<0.01, *** p<0.001, n.s. = not significant")
    print(f"  Hits@1:  {sig_label(p_hits_1)}")
    print(f"  Hits@10: {sig_label(p_hits_10)}")
    print()

    # Decile breakdown
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


DATASETS = [
    ('data/seal_nq_output.json',      'data/seal_fm_index/NQ.fm_index'),
    ('data/minder_nq_output.json',    'data/minder_fm_index/psgs_w100.fm_index'),
    ('data/minder_msmarco_output.json', 'data/minder_fm_index/msmarco-passage-corpus.fm_index'),
]

def main():
    for output_json, fm_index_path in DATASETS:
        basename = os.path.splitext(os.path.basename(output_json))[0]
        dataset_name = '_'.join(basename.split('_')[:2])

        print("\n" + "="*70)
        print("CORPUS COVERAGE ANALYSIS")
        print("="*70)
        print(f"Output JSON: {output_json}")
        print(f"FM-Index:    {fm_index_path}")
        print()

        corpus_vocab = load_corpus_vocab(fm_index_path)

        results = analyze_corpus_coverage(corpus_vocab, output_json, dataset_name)

        output_dir = f"generated_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "corpus_coverage_analysis.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*70}")
        print(f"✓ Results saved to: {output_path}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
