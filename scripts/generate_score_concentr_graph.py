
import json
import ijson
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from typing import List, Dict
from utils.utils import parse_ngrams, stream_data, get_dataset_name

def get_top_k_ngram_scores(datapath: str, k: int = 10) -> List[np.ndarray]:
    """
    For each query, get the scores of the top-k n-grams in the top-1 retrieved context.
    Returns a list of arrays, each containing up to k scores.
    """
    all_query_scores = []
    count = 0
    print(f"Processing {datapath}...")
    
    for entry in stream_data(datapath):
        ctxs = entry.get('ctxs', [])
        if not ctxs:
            continue
            
        # Get top-1 retrieved context
        top_ctx = ctxs[0]
        # Parse n-grams: [(text, freq, score), ...]
        ngrams = parse_ngrams(top_ctx.get('keys', ''))
        
        if not ngrams:
            continue
            
        # Sort n-grams by score descending
        all_scores = [float(ng[2]) for ng in ngrams]
        total_score = sum(all_scores)
        
        if total_score == 0:
            continue
            
        # Sort and take top k, then convert to percentage
        scores_sorted = sorted(all_scores, reverse=True)
        top_k_percentages = [(s / total_score) * 100 for s in scores_sorted[:k]]
        
        all_query_scores.append(top_k_percentages)
        
        count += 1
        if count % 1000 == 0:
            print(f"  Processed {count} queries...")
            
    return all_query_scores

def calculate_averages(scores_list: List[List[float]], k: int = 10) -> np.ndarray:
    """Calculate the mean score for each rank (1 to k)."""
    averages = []
    for i in range(k):
        # Collect all scores at rank i (if they exist for that query)
        rank_i_scores = [s[i] for s in scores_list if len(s) > i]
        if rank_i_scores:
            averages.append(np.mean(rank_i_scores))
        else:
            averages.append(0.0)
    return np.array(averages)

DATASETS = [
    'data/seal_nq_output.json',
    'data/minder_nq_output.json',
    'data/minder_msmarco_output.json',
]

def main():
    k = 10

    all_scores = {}
    all_avgs = {}
    for datapath in DATASETS:
        scores = get_top_k_ngram_scores(datapath, k)
        dataset_name = get_dataset_name(datapath)
        all_scores[dataset_name] = scores
        all_avgs[dataset_name] = calculate_averages(scores, k)

    # Comparison plot: seal_nq vs minder_nq
    seal_avg = all_avgs.get('seal_nq', np.zeros(k))
    minder_avg = all_avgs.get('minder_nq', np.zeros(k))

    x = np.arange(1, k + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, seal_avg, marker='o', label='SEAL (NQ)', color='skyblue', linewidth=2)
    ax.plot(x, minder_avg, marker='s', label='MINDER (NQ)', color='salmon', linewidth=2)
    if 'minder_msmarco' in all_avgs:
        ax.plot(x, all_avgs['minder_msmarco'], marker='^', label='MINDER (MSMARCO)', color='green', linewidth=2)

    ax.set_ylabel('Average N-gram Score Contribution (%)')
    ax.set_xlabel('N-gram Rank (by Score)')
    ax.set_title('Mean N-gram Contribution to Total Passage Score')
    ax.set_xticks(x)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    plt.tight_layout()
    output_dir = 'generated_data/shared'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'ngram_score_comparison.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")

    # Print numerical results
    print("\nNumerical Averages (Percentage of total score):")
    print(f"{'Rank':<5} | {'seal_nq (%)':<12} | {'minder_nq (%)':<14} | {'minder_msmarco (%)'}")
    print("-" * 60)
    msmarco_avg = all_avgs.get('minder_msmarco', np.zeros(k))
    for i in range(k):
        print(f"{i+1:<5} | {seal_avg[i]:<12.2f} | {minder_avg[i]:<14.2f} | {msmarco_avg[i]:<.2f}")

if __name__ == "__main__":
    main()
