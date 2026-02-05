import ijson
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json
from utils import get_dataset_name, parse_ngrams

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def analyze_scoring_failure(datapath="data/seal_output.json"):
    import sys

    script_name = "scoring_failure"
    print(f"running {script_name}")

    try:
        dataset_name = get_dataset_name(datapath)
        output_dir = Path(f"generated_data/{dataset_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        with open(datapath, 'r', encoding='utf-8') as f:
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

        # Build JSON output
        output_data = {
            "total_queries": len(df),
            "gt_retrieved_top100": len(retrieved),
            "gt_retrieved_pct": float(len(retrieved) / len(df) * 100) if len(df) > 0 else 0.0,
            "misranked_count": len(scoring_failures),
            "misranked_pct_of_retrieved": float(len(scoring_failures) / len(retrieved) * 100) if len(retrieved) > 0 else 0.0,
            "avg_gt_rank": float(retrieved['gt_rank'].mean()) if len(retrieved) > 0 else None,
            "median_gt_rank": float(retrieved['gt_rank'].median()) if len(retrieved) > 0 else None,
        }

        if len(scoring_failures) > 0:
            output_data.update({
                "mean_score_diff": float(scoring_failures['score_diff'].mean()),
                "median_score_diff": float(scoring_failures['score_diff'].median()),
                "mean_score_diff_rel": float(scoring_failures['score_diff_rel'].mean()),
                "median_score_diff_rel": float(scoring_failures['score_diff_rel'].median()),
            })

            # Boxplot of relative score differentials
            rel_diff_values = scoring_failures['score_diff_rel'].dropna().astype(float)
            plt.figure(figsize=(8,5))
            plt.boxplot(rel_diff_values, vert=True)
            plt.title("Relative Score Differential (Rank-1 vs GT) for Mis-ranked Queries")
            plt.ylabel("Relative Score Diff (Rank1 - GT) / Rank1")
            plt.xticks([1], ["Mis-ranked Queries"])
            plt.grid(True, axis='y', linestyle='--', alpha=0.6)
            plt.savefig(output_dir / f"score_diff_rel_boxplot_{dataset_name}.png", bbox_inches='tight')
            plt.close()

        # Rank distribution
        rank_dist = retrieved['gt_rank'].value_counts().sort_index()
        rank_distribution = {}
        for rank in range(1, 11):
            count = int(rank_dist.get(rank, 0))
            pct = float(count / len(retrieved) * 100) if len(retrieved) > 0 else 0.0
            rank_distribution[f"rank_{rank}"] = count
            rank_distribution[f"rank_{rank}_pct"] = pct

        output_data["rank_distribution_top10"] = rank_distribution

        # Save JSON output
        output_json = output_dir / f"{script_name}_results.json"
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"success running {script_name}")

    except Exception as e:
        print(f"error: running {script_name} {e}")
        raise

if __name__ == "__main__":
    import sys
    datapath = sys.argv[1] if len(sys.argv) > 1 else 'data/seal_output.json'
    analyze_scoring_failure(datapath)
