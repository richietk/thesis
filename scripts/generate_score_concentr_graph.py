
import json
import ijson
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from utils.utils import parse_ngrams, stream_data

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

def main():
    k = 10
    seal_path = '../data/seal_output.json'
    minder_path = '../data/minder_output.json'
    
    seal_scores = get_top_k_ngram_scores(seal_path, k)
    minder_scores = get_top_k_ngram_scores(minder_path, k)
    
    seal_avg = calculate_averages(seal_scores, k)
    minder_avg = calculate_averages(minder_scores, k)
    
    # Plotting
    x = np.arange(1, k + 1)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, seal_avg, marker='o', label='SEAL', color='skyblue', linewidth=2)
    ax.plot(x, minder_avg, marker='s', label='MINDER', color='salmon', linewidth=2)
    
    ax.set_ylabel('Average N-gram Score Contribution (%)')
    ax.set_xlabel('N-gram Rank (by Score)')
    ax.set_title('Mean N-gram Contribution to Total Passage Score on NQ')
    ax.set_xticks(x)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('ngram_score_comparison.png')
    print("Plot saved to ngram_score_comparison.png")
    
    # Print numerical results for quick check
    print("\nNumerical Averages (Percentage of total score):")
    print(f"{'Rank':<5} | {'SEAL (%)':<10} | {'MINDER (%)':<10}")
    print("-" * 35)
    for i in range(k):
        print(f"{i+1:<5} | {seal_avg[i]:<10.2f} | {minder_avg[i]:<10.2f}")

if __name__ == "__main__":
    main()
