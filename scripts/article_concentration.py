import ijson
from collections import Counter
import pandas as pd
import sys
import os
from utils import get_dataset_name

def analyze_article_diversity(datapath='data/seal_output.json'):
    """Exploratory analysis: how many unique article titles appear in top-10 passages."""
    script_name = "article_concentration"
    print(f"running {script_name}")

    try:
        dataset_name = get_dataset_name(datapath)
        output_dir = f"generated_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Redirect stdout to log file
        log_file = os.path.join(output_dir, f"{script_name}_log.txt")
        original_stdout = sys.stdout
        sys.stdout = open(log_file, 'w', encoding='utf-8')

        print("\n" + "="*80)
        print("ANALYSIS 6: ARTICLE-LEVEL DIVERSITY (EXPLORATORY)")
        print("="*80)

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

        print(f"\nAnalyzed {len(df)} queries")
        print("\nSuccess Rate by Article Diversity (Unique Titles in Top-10):")
        print("Unique Titles | Count | Top-1 % | Top-2 % | Top-10 %")
        print("-----------------------------------------------")
        for idx in sorted(grouped.index):
            row = grouped.loc[idx]
            print(f"{idx:13d} | {int(row['count']):5d} | {100*row['success_top1']:7.2f}% | "
                  f"{100*row['success_top2']:7.2f}% | {100*row['success_top10']:8.2f}%")

        # Low vs high diversity examples
        low_div = df[df['unique_titles'] == 1]
        high_div = df[df['unique_titles'] >= 8]

        print(f"\nLow diversity (1 unique title): {len(low_div)} queries, {100*low_div['success_top1'].mean():.1f}% success")
        print(f"High diversity (8+ unique titles): {len(high_div)} queries, {100*high_div['success_top1'].mean():.1f}% success")

        # Restore stdout
        sys.stdout.close()
        sys.stdout = original_stdout

        print(f"success running {script_name}")

    except Exception as e:
        # Restore stdout in case of error
        if sys.stdout != original_stdout:
            sys.stdout.close()
            sys.stdout = original_stdout
        print(f"error: running {script_name} {e}")
        raise

if __name__ == "__main__":
    datapath = sys.argv[1] if len(sys.argv) > 1 else 'data/seal_output.json'
    analyze_article_diversity(datapath)
