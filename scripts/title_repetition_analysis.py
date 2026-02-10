import ijson
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from pathlib import Path
import sys
import os
from utils.utils import get_dataset_name, calculate_retrieval_metrics


def analyze_title_repetition(datapath="data/seal_output.json"):
    """
    Analyze the relationship between title repetition in top-10 retrieved contexts
    and hits@10 success rate.

    For each query, determines the maximum number of top-10 contexts that share
    the same title, then calculates hits@10 rate for each repetition level.
    """
    script_name = "title_repetition_analysis"
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

                # Calculate retrieval metrics
                retrieved_ids = [ctx['passage_id'] for ctx in ctxs]
                metrics = calculate_retrieval_metrics(retrieved_ids, positive_ids)

                results.append({
                    'query': query,
                    'same_title_count': same_title_count,
                    'most_common_title': most_common_title,
                    'total_titles_in_top10': len(titles),
                    'unique_titles_in_top10': len(title_counts),
                    'hits@10': metrics['hits_at_10'],
                    'hits@1': metrics['precision_at_1'],
                    'r_precision': metrics['r_precision']
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
            'hits@10_rate': float(group['hits@10'].mean() * 100),
            'hits@10_std': float(group['hits@10'].std() * 100),
            'hits@1_rate': float(group['hits@1'].mean() * 100),
            'hits@1_std': float(group['hits@1'].std() * 100),
            'r_precision_avg': float(group['r_precision'].mean()),
            'r_precision_std': float(group['r_precision'].std()),
            'avg_unique_titles': float(group['unique_titles_in_top10'].mean()),
            'unique_titles_std': float(group['unique_titles_in_top10'].std())
        })


        # Overall statistics
        overall_stats = {
            'total_queries': len(df),
            'overall_hits@10_rate': float(df['hits@10'].mean() * 100),
            'overall_hits@10_std': float(df['hits@10'].std() * 100),
            'overall_hits@1_rate': float(df['hits@1'].mean() * 100),
            'overall_hits@1_std': float(df['hits@1'].std() * 100),
            'overall_r_precision': float(df['r_precision'].mean()),
            'overall_r_precision_std': float(df['r_precision'].std()),
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
        print("\n" + "="*80)
        print("Title Repetition Analysis - Summary")
        print("="*80)
        print(f"Total queries analyzed: {len(df)}")
        print(f"Overall hits@10 rate: {overall_stats['overall_hits@10_rate']:.2f}%")
        print("\n" + "-"*80)
        print(f"{'Same Title Count':<20} {'# Queries':<15} {'% of Total':<15} {'Hits@10 Rate':<15}")
        print("-"*80)

        for stat in grouped_stats:
            print(f"{stat['max_same_title_count']:<20} "
                  f"{stat['num_queries']:<15} "
                  f"{stat['pct_of_total']:<14.2f}% "
                  f"{stat['hits@10_rate']:<14.2f}%")

        print("="*80)

        print(f"success running {script_name}")

    except Exception as e:
        print(f"error: running {script_name} {e}")
        raise

def create_comparison_chart():
    """Create comparison chart with large legend and labels."""
    print("Creating comparison chart...")
    
    # Load data
    seal_path = "generated_data/seal/title_repetition_analysis_results.json"
    minder_path = "generated_data/minder/title_repetition_analysis_results.json"
    output_dir = "generated_data/shared"
    
    try:
        with open(seal_path, "r") as f:
            seal_data = json.load(f)
        with open(minder_path, "r") as f:
            minder_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find results file: {e}")
        return
    
    # Extract stats
    def extract_stats(data):
        stats_dict = {}
        for s in data.get("by_title_repetition_count", []):
            count = s.get("same_title_count", s.get("max_same_title_count"))
            if count is not None:
                stats_dict[count] = s.get("hits@10_rate", 0)
        return stats_dict
    
    seal_dict = extract_stats(seal_data)
    minder_dict = extract_stats(minder_data)
    all_counts = sorted(set(seal_dict.keys()) | set(minder_dict.keys()))
    
    seal_rates = [seal_dict.get(c, np.nan) for c in all_counts]
    minder_rates = [minder_dict.get(c, np.nan) for c in all_counts]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 9))
    x = np.array(all_counts)
    
    ax.plot(x, seal_rates, marker="o", linewidth=3.5, label="SEAL", color="#1f77b4")
    ax.plot(x, minder_rates, marker="s", linewidth=3.5, label="Minder", color="#ff7f0e")
    
    ax.set_xlabel("Max Same Title Count (out of top-10)", fontsize=18)
    ax.set_ylabel("Hits@10 Rate (%)", fontsize=18)
    ax.set_title("Hits@10 Rate by Title Repetition", fontsize=21, fontweight="bold")
    ax.grid(True, linestyle='--', alpha=1)
    
    ax.legend(loc="upper right", prop={'size': 18}, markerscale=4, 
              borderaxespad=1, labelspacing=1.5, handlelength=3)
    
    # Data labels
    for xi, yi in zip(x, seal_rates):
        if not np.isnan(yi):
            ax.text(xi, yi - 1.8, f"{yi:.1f}%", ha="center", va="top", 
                   fontsize=18, color="#1f77b4", fontweight='bold')
    
    for xi, yi in zip(x, minder_rates):
        if not np.isnan(yi):
            ax.text(xi, yi + 1.8, f"{yi:.1f}%", ha="center", va="bottom", 
                   fontsize=18, color="#ff7f0e", fontweight='bold')
    
    ax.set_xticks(all_counts)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_ylim(45, 105)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "title_repetition_comparison.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Chart saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If argument provided, run on that specific file
        datapath = sys.argv[1]
        analyze_title_repetition(datapath)
    else:
        # No arguments: run on both datasets and create comparison chart
        seal_json_path = "generated_data/seal/title_repetition_analysis_results.json"
        minder_json_path = "generated_data/minder/title_repetition_analysis_results.json"

        seal_exists = os.path.exists(seal_json_path)
        minder_exists = os.path.exists(minder_json_path)

        if seal_exists and minder_exists:
            print("JSON results already exist. Skipping analysis, regenerating image only...\n")
        else:
            print("No arguments provided. Running on both SEAL and Minder datasets...\n")

            # Run on SEAL if needed
            if not seal_exists:
                analyze_title_repetition('data/seal_output.json')
                print()
            else:
                print("SEAL results already exist, skipping...\n")

            # Run on Minder if needed
            if not minder_exists:
                analyze_title_repetition('data/minder_output.json')
                print()
            else:
                print("Minder results already exist, skipping...\n")

        # Create comparison chart
        create_comparison_chart()
