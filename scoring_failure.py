import ijson
import ast
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

DATA_PATH = "data/seal_output.json"

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def parse_ngrams(keys_str):
    if not keys_str:
        return []
    try:
        keys_list = ast.literal_eval(keys_str)
        return [(ngram, freq, score) for ngram, freq, score in keys_list]
    except:
        return []

def analyze_scoring_failure(output_dir="generated_data"):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*80)
    print("ANALYSIS 2: SCORING FAILURE")
    print("="*80)

    results = []

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for entry in ijson.items(f, 'item'):
            query = entry['question']
            positive_ids = {ctx['passage_id'] for ctx in entry.get('positive_ctxs', [])}

            if not positive_ids:
                continue

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
                score_diff = rank1_score - gt_score if (rank1_score is not None and gt_score is not None) else None
                score_diff_rel = (score_diff / rank1_score) if rank1_score and score_diff is not None else None

                results.append({
                    'query': query,
                    'gt_rank': gt_rank,
                    'gt_score': gt_score,
                    'rank1_score': rank1_score,
                    'score_diff': score_diff,
                    'score_diff_rel': score_diff_rel,
                    'scoring_failure': gt_rank > 1,
                    'retrieved': True
                })
            else:
                results.append({
                    'query': query,
                    'gt_rank': None,
                    'gt_score': None,
                    'rank1_score': rank1_score,
                    'score_diff': None,
                    'score_diff_rel': None,
                    'scoring_failure': False,
                    'retrieved': False
                })

    df = pd.DataFrame(results)
    retrieved = df[df['retrieved'] == True]
    scoring_failures = df[df['scoring_failure'] == True]

    # Summary statistics
    print(f"\nAnalyzed {len(df)} queries")
    print(f"  Ground truth retrieved (top-100): {len(retrieved)} ({100*len(retrieved)/len(df):.1f}%)")
    print(f"  Retrieved but mis-ranked: {len(scoring_failures)} ({100*len(scoring_failures)/len(retrieved):.1f}% of retrieved)")
    print(f"  Average GT rank: {retrieved['gt_rank'].mean():.1f}")
    print(f"  Median GT rank: {retrieved['gt_rank'].median():.1f}")

    if len(scoring_failures) > 0:
        print(f"\nScore Differential (Rank-1 - GT):")
        print(f"  Mean: {scoring_failures['score_diff'].mean():.2f}")
        print(f"  Median: {scoring_failures['score_diff'].median():.2f}")
        print(f"\nRelative Score Differential (diff / rank1_score):")
        print(f"  Mean: {scoring_failures['score_diff_rel'].mean():.3f}")
        print(f"  Median: {scoring_failures['score_diff_rel'].median():.3f}")

        # Boxplot of relative score differentials
        rel_diff_values = scoring_failures['score_diff_rel'].dropna().astype(float)
        plt.figure(figsize=(8,5))
        plt.boxplot(rel_diff_values, vert=True)
        plt.title("Relative Score Differential (Rank-1 vs GT) for Mis-ranked Queries")
        plt.ylabel("Relative Score Diff (Rank1 - GT) / Rank1")
        plt.xticks([1], ["Mis-ranked Queries"])
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.savefig(output_dir / "score_diff_rel_boxplot.png", bbox_inches='tight')
        plt.close()


    # Rank distribution
    rank_dist = retrieved['gt_rank'].value_counts().sort_index()
    print(f"\nRank Distribution (top-10):")
    for rank in range(1, 11):
        count = rank_dist.get(rank, 0)
        pct = 100 * count / len(retrieved) if len(retrieved) > 0 else 0
        print(f"  Rank {rank}: {count} ({pct:.1f}%)")

if __name__ == "__main__":
    analyze_scoring_failure()
