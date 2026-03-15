import ijson
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from pathlib import Path
import sys
import os
from utils.utils import get_dataset_name


def recall_at_k(retrieved_ids, gold_ids, k):
    """Recall@k: fraction of relevant docs found in top-k."""
    if not gold_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    found = sum(1 for pid in top_k if pid in gold_ids)
    return found / len(gold_ids)


def analyze_title_repetition(datapath="data/seal_output.json"):
    """
    Analyze the relationship between title repetition in top-10 retrieved contexts
    and recall@1/10/100.

    For each query, determines the maximum number of top-10 contexts that share
    the same title, then calculates recall rates for each repetition level.
    """
    script_name = "title_rep_analysis2"
    print(f"running {script_name}")

    try:
        dataset_name = get_dataset_name(datapath)
        output_dir = f"generated_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

        results = []

        with open(datapath, 'r', encoding='utf-8') as f:
            for entry in ijson.items(f, 'item'):
                query = entry['question']
                positive_ids = {ctx['passage_id'] for ctx in entry.get('positive_ctxs', [])}

                ctxs = entry.get('ctxs', [])
                if not ctxs:
                    continue

                # Get top-10 retrieved contexts
                top_10_ctxs = ctxs[:10]

                # Extract titles from top-10 contexts
                titles = []
                for ctx in top_10_ctxs:
                    title = ctx.get('title', '')
                    if title:
                        titles.append(title)

                # Skip if no titles available
                if not titles:
                    continue

                # Count title occurrences
                title_counts = Counter(titles)

                # Find how many contexts share the most common title
                same_title_count = max(title_counts.values()) if title_counts else 1
                most_common_title = title_counts.most_common(1)[0][0] if title_counts else ""

                # Calculate recall@1, @10, @100
                retrieved_ids = [ctx['passage_id'] for ctx in ctxs]
                r1 = recall_at_k(retrieved_ids, positive_ids, 1)
                r10 = recall_at_k(retrieved_ids, positive_ids, 10)
                r100 = recall_at_k(retrieved_ids, positive_ids, 100)

                results.append({
                    'query': query,
                    'same_title_count': same_title_count,
                    'most_common_title': most_common_title,
                    'total_titles_in_top10': len(titles),
                    'unique_titles_in_top10': len(title_counts),
                    'recall@1': r1,
                    'recall@10': r10,
                    'recall@100': r100,
                })

        df = pd.DataFrame(results)

        # Check if dataframe is empty
        if len(df) == 0:
            print(f"success running {script_name}")
            return

        # Group by same_title_count and calculate statistics
        grouped_stats = []

        for count in sorted(df['same_title_count'].unique()):
            group = df[df['same_title_count'] == count]

            grouped_stats.append({
                'same_title_count': int(count),
                'num_queries': len(group),
                'pct_of_total': float(100 * len(group) / len(df)),
                'recall@1_rate': float(group['recall@1'].mean() * 100),
                'recall@1_std': float(group['recall@1'].std() * 100),
                'recall@10_rate': float(group['recall@10'].mean() * 100),
                'recall@10_std': float(group['recall@10'].std() * 100),
                'recall@100_rate': float(group['recall@100'].mean() * 100),
                'recall@100_std': float(group['recall@100'].std() * 100),
                'avg_unique_titles': float(group['unique_titles_in_top10'].mean()),
                'unique_titles_std': float(group['unique_titles_in_top10'].std())
            })

        # Overall statistics
        overall_stats = {
            'total_queries': len(df),
            'overall_recall@1': float(df['recall@1'].mean() * 100),
            'overall_recall@1_std': float(df['recall@1'].std() * 100),
            'overall_recall@10': float(df['recall@10'].mean() * 100),
            'overall_recall@10_std': float(df['recall@10'].std() * 100),
            'overall_recall@100': float(df['recall@100'].mean() * 100),
            'overall_recall@100_std': float(df['recall@100'].std() * 100),
            'avg_same_title_count': float(df['same_title_count'].mean()),
            'std_same_title_count': float(df['same_title_count'].std()),
            'median_same_title_count': float(df['same_title_count'].median())
        }

        # Collect output data
        output_data = {
            "overall_statistics": overall_stats,
            "by_title_repetition_count": grouped_stats
        }

        # Write JSON output
        json_path = os.path.join(output_dir, f"{script_name}_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        # Print summary table
        print("\n" + "="*90)
        print("Title Repetition Analysis - Recall Summary")
        print("="*90)
        print(f"Total queries analyzed: {len(df)}")
        print(f"Overall recall@1:   {overall_stats['overall_recall@1']:.2f}%")
        print(f"Overall recall@10:  {overall_stats['overall_recall@10']:.2f}%")
        print(f"Overall recall@100: {overall_stats['overall_recall@100']:.2f}%")
        print("\n" + "-"*90)
        print(f"{'Same Title Count':<20} {'# Queries':<12} {'% of Total':<12} {'Recall@1':<12} {'Recall@10':<12} {'Recall@100':<12}")
        print("-"*90)

        for stat in grouped_stats:
            print(f"{stat['same_title_count']:<20} "
                  f"{stat['num_queries']:<12} "
                  f"{stat['pct_of_total']:<11.2f}% "
                  f"{stat['recall@1_rate']:<11.2f}% "
                  f"{stat['recall@10_rate']:<11.2f}% "
                  f"{stat['recall@100_rate']:<11.2f}%")

        print("="*90)

        print(f"success running {script_name}")

    except Exception as e:
        print(f"error: running {script_name} {e}")
        raise


def create_comparison_chart():
    """Create comparison chart for recall@1, @10, @100 across SEAL and Minder."""
    print("Creating comparison chart...")

    seal_path = "generated_data/seal/title_rep_analysis2_results.json"
    minder_path = "generated_data/minder/title_rep_analysis2_results.json"
    output_dir = "generated_data/shared"

    try:
        with open(seal_path, "r") as f:
            seal_data = json.load(f)
        with open(minder_path, "r") as f:
            minder_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find results file: {e}")
        return

    def extract_stats(data, metric):
        stats_dict = {}
        for s in data.get("by_title_repetition_count", []):
            count = s.get("same_title_count")
            if count is not None:
                stats_dict[count] = s.get(metric, 0)
        return stats_dict

    all_counts = sorted(
        set(s["same_title_count"] for s in seal_data.get("by_title_repetition_count", []))
        | set(s["same_title_count"] for s in minder_data.get("by_title_repetition_count", []))
    )
    x = np.array(all_counts)

    metrics = [
        ("recall@1_rate", "Recall@1"),
        ("recall@10_rate", "Recall@10"),
        ("recall@100_rate", "Recall@100"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=False)

    seal_colors = {"recall@1_rate": "#1f77b4", "recall@10_rate": "#1f77b4", "recall@100_rate": "#1f77b4"}
    minder_colors = {"recall@1_rate": "#ff7f0e", "recall@10_rate": "#ff7f0e", "recall@100_rate": "#ff7f0e"}

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        seal_dict = extract_stats(seal_data, metric_key)
        minder_dict = extract_stats(minder_data, metric_key)

        seal_rates = [seal_dict.get(c, np.nan) for c in all_counts]
        minder_rates = [minder_dict.get(c, np.nan) for c in all_counts]

        ax.plot(x, seal_rates, marker="o", linewidth=3.5, label="SEAL", color="#1f77b4")
        ax.plot(x, minder_rates, marker="s", linewidth=3.5, label="Minder", color="#ff7f0e")

        ax.set_xlabel("Max Same Title Count (out of top-10)", fontsize=15)
        ax.set_ylabel(f"{metric_label} (%)", fontsize=15)
        ax.set_title(f"{metric_label} by Title Repetition", fontsize=17, fontweight="bold")
        ax.grid(True, linestyle='--', alpha=1)
        ax.legend(loc="upper right", prop={'size': 14}, markerscale=3,
                  borderaxespad=1, labelspacing=1.2, handlelength=2.5)

        for xi, yi in zip(x, seal_rates):
            if not np.isnan(yi):
                ax.text(xi, yi - 1.8, f"{yi:.1f}%", ha="center", va="top",
                       fontsize=13, color="#1f77b4", fontweight='bold')

        for xi, yi in zip(x, minder_rates):
            if not np.isnan(yi):
                ax.text(xi, yi + 1.8, f"{yi:.1f}%", ha="center", va="bottom",
                       fontsize=13, color="#ff7f0e", fontweight='bold')

        ax.set_xticks(all_counts)
        ax.tick_params(axis='both', which='major', labelsize=14)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "title_rep_analysis2_comparison.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Chart saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        datapath = sys.argv[1]
        analyze_title_repetition(datapath)
    else:
        seal_json_path = "generated_data/seal/title_rep_analysis2_results.json"
        minder_json_path = "generated_data/minder/title_rep_analysis2_results.json"

        seal_exists = os.path.exists(seal_json_path)
        minder_exists = os.path.exists(minder_json_path)

        if seal_exists and minder_exists:
            print("JSON results already exist. Skipping analysis, regenerating image only...\n")
        else:
            print("No arguments provided. Running on both SEAL and Minder datasets...\n")

            if not seal_exists:
                analyze_title_repetition('data/seal_output.json')
                print()
            else:
                print("SEAL results already exist, skipping...\n")

            if not minder_exists:
                analyze_title_repetition('data/minder_output.json')
                print()
            else:
                print("Minder results already exist, skipping...\n")

        create_comparison_chart()
