import ijson
from collections import Counter
import pandas as pd
import sys
import os
import json
from utils import get_dataset_name

def analyze_article_diversity(datapath='data/seal_output.json'):
    """Exploratory analysis: how many unique article titles appear in top-10 passages."""
    script_name = "article_concentration"
    print(f"running {script_name}")

    try:
        dataset_name = get_dataset_name(datapath)
        output_dir = f"generated_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

        results = []

        with open(datapath, 'r', encoding='utf-8') as f:
            for entry in ijson.items(f, 'item'):
                query = entry['question']

                positive_ids = set()
                if 'positive_ctxs' in entry and entry['positive_ctxs']:
                    positive_ids = {ctx['passage_id'] for ctx in entry['positive_ctxs']}

                top10 = entry.get('ctxs', [])[:10]
                if not top10:
                    continue

                titles = [ctx['title'] for ctx in top10]
                passage_ids = [ctx['passage_id'] for ctx in top10]

                unique_titles = len(set(titles))
                unique_passages = len(set(passage_ids))

                success_top1 = top10[0]['passage_id'] in positive_ids
                success_top2 = any(ctx['passage_id'] in positive_ids for ctx in top10[:2])
                success_top10 = any(ctx['passage_id'] in positive_ids for ctx in top10)

                results.append({
                    'query': query,
                    'unique_titles': unique_titles,
                    'unique_passages': unique_passages,
                    'success_top1': success_top1,
                    'success_top2': success_top2,
                    'success_top10': success_top10
                })

        df = pd.DataFrame(results)

        grouped = df.groupby('unique_titles').agg({
            'success_top1': 'mean',
            'success_top2': 'mean',
            'success_top10': 'mean',
            'query': 'count'
        }).rename(columns={'query': 'count'})

        # Build JSON output
        output_data = {
            "total_queries": len(df),
            "by_unique_titles": {}
        }

        for idx in sorted(grouped.index):
            row = grouped.loc[idx]
            output_data["by_unique_titles"][str(idx)] = {
                "count": int(row['count']),
                "success_top1_pct": float(row['success_top1'] * 100),
                "success_top2_pct": float(row['success_top2'] * 100),
                "success_top10_pct": float(row['success_top10'] * 100)
            }

        # Low vs high diversity examples
        low_div = df[df['unique_titles'] == 1]
        high_div = df[df['unique_titles'] >= 8]

        output_data["low_diversity_1_title"] = {
            "count": len(low_div),
            "success_top1_pct": float(low_div['success_top1'].mean() * 100) if len(low_div) > 0 else None
        }
        output_data["high_diversity_8plus_titles"] = {
            "count": len(high_div),
            "success_top1_pct": float(high_div['success_top1'].mean() * 100) if len(high_div) > 0 else None
        }

        # Save JSON output
        output_json = os.path.join(output_dir, f"{script_name}_results.json")
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"success running {script_name}")

    except Exception as e:
        print(f"error: running {script_name} {e}")
        raise

if __name__ == "__main__":
    datapath = sys.argv[1] if len(sys.argv) > 1 else 'data/seal_output.json'
    analyze_article_diversity(datapath)
