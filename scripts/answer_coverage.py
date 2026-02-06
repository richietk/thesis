import ijson
import json
import pandas as pd
from pathlib import Path
import sys
import os
from utils.utils import strip_ngram_markers, get_dataset_name, parse_ngrams, calculate_retrieval_metrics

def analyze_answer_coverage(datapath="data/seal_output.json"):
    """Analyze whether answer string appears in generated n-grams."""
    script_name = "answer_coverage"
    print(f"running {script_name}")

    try:
        dataset_name = get_dataset_name(datapath)
        output_dir = f"generated_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

        results = []
        passage_results = []

        with open(datapath, 'r', encoding='utf-8') as f:
            for entry in ijson.items(f, 'item'):
                query = entry['question']
                answers = entry.get('answers', [])

                if not answers:
                    continue

                positive_ids = {ctx['passage_id'] for ctx in entry.get('positive_ctxs', [])}
                retrieved_ids = [ctx['passage_id'] for ctx in entry.get('ctxs', [])]

                top_ctx = entry.get('ctxs', [None])[0]
                if not top_ctx:
                    continue

                success = top_ctx['passage_id'] in positive_ids

                # Calculate retrieval metrics
                metrics = calculate_retrieval_metrics(retrieved_ids, positive_ids)

                ngrams = parse_ngrams(top_ctx.get('keys', ''))
                if not ngrams:
                    continue

                ngram_text = ' '.join([strip_ngram_markers(ng[0], datapath).lower() for ng in ngrams])
                answer_in_ngrams = any(ans.lower() in ngram_text for ans in answers)

                results.append({
                    'query': query,
                    'answer': answers[0],
                    'success': success,
                    'answer_in_ngrams': answer_in_ngrams,
                    'num_ngrams': len(ngrams),
                    'hits@1': metrics['precision_at_1'],
                    'hits@10': metrics['hits_at_10'],
                    'r_precision': metrics['r_precision']
                })

                # Passage-level analysis: check top-10 retrieved passages
                for ctx in entry.get('ctxs', [])[:10]:
                    passage_id = ctx['passage_id']
                    is_positive = passage_id in positive_ids

                    ngrams_p = parse_ngrams(ctx.get('keys', ''))
                    if not ngrams_p:
                        continue

                    ngram_text_p = ' '.join([strip_ngram_markers(ng[0], datapath).lower() for ng in ngrams_p])
                    answer_in_passage = any(ans.lower() in ngram_text_p for ans in answers)

                    passage_results.append({
                        'passage_id': passage_id,
                        'answer_in_ngrams': answer_in_passage,
                        'is_positive': is_positive
                    })

        df = pd.DataFrame(results)

        # Check if dataframe is empty
        if len(df) == 0:
            print(f"success running {script_name}")
            return

        answer_present = df[df['answer_in_ngrams']]
        answer_absent = df[~df['answer_in_ngrams']]

        # Passage-level statistics
        passage_df = pd.DataFrame(passage_results)
        passages_with_answer = passage_df[passage_df['answer_in_ngrams']]
        passages_without_answer = passage_df[~passage_df['answer_in_ngrams']]

        # 2x2 Contingency Table
        # A: has answer AND is ground truth
        # B: has answer AND NOT ground truth
        # C: no answer AND is ground truth
        # D: no answer AND NOT ground truth
        A = int(passages_with_answer['is_positive'].sum())
        B = len(passages_with_answer) - A
        C = int(passages_without_answer['is_positive'].sum())
        D = len(passages_without_answer) - C

        # Standard correlation metrics
        precision = A / (A + B) if (A + B) > 0 else 0.0
        recall = A / (A + C) if (A + C) > 0 else 0.0
        odds_ratio = (A * D) / (B * C) if (B > 0 and C > 0) else float('inf')

        # Collect output data
        output_data = {
            # Query-level analysis (top-1 passage)
            "query_level": {
                "total_queries": len(df),
                "answer_in_top1_count": len(answer_present),
                "answer_in_top1_pct": float(100 * len(answer_present) / len(df)),
                "answer_not_in_top1_count": len(answer_absent),
                "answer_not_in_top1_pct": float(100 * len(answer_absent) / len(df)),
                "hits@1": float(df['hits@1'].mean()),
                "hits@1_answer_present": float(answer_present['hits@1'].mean()) if len(answer_present) > 0 else 0.0,
                "hits@1_answer_absent": float(answer_absent['hits@1'].mean()) if len(answer_absent) > 0 else 0.0,
                "hits@10": float(df['hits@10'].mean()),
                "hits@10_answer_present": float(answer_present['hits@10'].mean()) if len(answer_present) > 0 else 0.0,
                "hits@10_answer_absent": float(answer_absent['hits@10'].mean()) if len(answer_absent) > 0 else 0.0
            },
            # Passage-level analysis (top-20 passages)
            "passage_level": {
                "contingency_table": {
                    "has_answer_and_ground_truth": A,
                    "has_answer_not_ground_truth": B,
                    "no_answer_and_ground_truth": C,
                    "no_answer_not_ground_truth": D
                },
                "precision": float(precision),
                "recall": float(recall),
                "odds_ratio": float(odds_ratio) if odds_ratio != float('inf') else "inf"
            }
        }

        # Write JSON output
        json_path = os.path.join(output_dir, f"{script_name}_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        print(f"success running {script_name}")

    except Exception as e:
        print(f"error: running {script_name} {e}")
        raise

if __name__ == "__main__":
    datapath = sys.argv[1] if len(sys.argv) > 1 else 'data/seal_output.json'
    analyze_answer_coverage(datapath)
