"""
Answer Location Analysis for SEAL Outputs (Memory-Efficient Version)
=====================================================================
Identifies whether SEAL found the correct answer, and if so, whether it was
in the expected (ground-truth) passage or a different passage.

This version streams the JSON file and writes directly to CSV without 
loading everything into memory.

Categories:
- True success: Ground-truth passage found in top-k
- Answer in different passage: Answer found in top-k, but not in the ground-truth passage
- True failure: Answer not found in any top-k passage
"""

import ijson
import os
from collections import defaultdict
import csv
from utils import strip_pseudoqueries, get_dataset_name, normalize_text, answer_in_text, get_ground_truth_ids


def analyze_retrieval_outcome(query_data, top_k=10, datapath=""):
    """
    Analyze a single query's retrieval outcome.

    Returns dict with:
    - question: the query
    - answers: list of acceptable answers
    - outcome: 'true_success', 'answer_in_different_passage', or 'true_failure'
    - gt_in_topk: whether ground-truth passage is in top-k
    - answer_in_topk: whether any answer string appears in top-k passages
    - answer_found_at_rank: first rank where answer appears (or -1)
    - gt_found_at_rank: first rank where ground-truth appears (or -1)
    """

    question = query_data.get('question', '')
    answers = query_data.get('answers', [])
    gold_ids = get_ground_truth_ids(query_data)
    retrieved = query_data.get('ctxs', [])[:top_k]

    # Check for ground-truth passage in top-k
    gt_found_at_rank = -1
    for rank, ctx in enumerate(retrieved, 1):
        pid = ctx.get('passage_id', '').split('...')[0]
        if pid in gold_ids:
            gt_found_at_rank = rank
            break

    gt_in_topk = gt_found_at_rank > 0

    # Check for answer string in top-k passages
    answer_found_at_rank = -1
    for rank, ctx in enumerate(retrieved, 1):
        passage_text = ctx.get('text', '') + ' ' + ctx.get('title', '')
        passage_text = strip_pseudoqueries(passage_text, datapath)
        for ans in answers:
            if answer_in_text(ans, passage_text):
                answer_found_at_rank = rank
                break
        if answer_found_at_rank > 0:
            break

    answer_in_topk = answer_found_at_rank > 0
    
    # Classify outcome
    if gt_in_topk:
        outcome = 'true_success'
    elif answer_in_topk:
        outcome = 'answer_in_different_passage'
    else:
        outcome = 'true_failure'
    
    return {
        'question': question,
        'answers': answers,
        'outcome': outcome,
        'gt_in_topk': gt_in_topk,
        'answer_in_topk': answer_in_topk,
        'gt_found_at_rank': gt_found_at_rank,
        'answer_found_at_rank': answer_found_at_rank,
    }


def main(datapath="data/seal_output.json"):
    import sys
    import json

    script_name = "answer_location_analysis"
    print(f"running {script_name}")

    try:
        file_path = datapath
        dataset_name = get_dataset_name(datapath)
        output_dir = f"generated_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

        output_csv = f"{output_dir}/answer_location_analysis_{dataset_name}.csv"
        diff_passage_csv = f"{output_dir}/answer_in_different_passage_{dataset_name}.csv"
        top_k = 2

        if not os.path.exists(file_path):
            print(f"error: running {script_name} File not found at {file_path}")
            return

        fieldnames = ['question', 'answers', 'outcome', 'gt_in_topk',
                      'answer_in_topk', 'gt_found_at_rank', 'answer_found_at_rank']

        # Counters
        outcome_counts = defaultdict(int)
        total = 0

        with open(file_path, 'rb') as json_file, \
             open(output_csv, 'w', newline='', encoding='utf-8') as main_csv_file, \
             open(diff_passage_csv, 'w', newline='', encoding='utf-8') as diff_csv_file:

            main_writer = csv.DictWriter(main_csv_file, fieldnames=fieldnames)
            main_writer.writeheader()

            diff_writer = csv.DictWriter(diff_csv_file, fieldnames=fieldnames)
            diff_writer.writeheader()

            parser = ijson.items(json_file, 'item')

            for entry in parser:
                total += 1

                result = analyze_retrieval_outcome(entry, top_k=top_k, datapath=file_path)
                outcome_counts[result['outcome']] += 1

                csv_row = result.copy()
                csv_row['answers'] = str(csv_row['answers'])

                # Write all to main CSV
                main_writer.writerow(csv_row)

                # Write the different passage cases to separate CSV
                if result['outcome'] == 'answer_in_different_passage':
                    diff_writer.writerow(csv_row)

        # Build JSON summary
        output_data = {
            "total_queries": total,
            "top_k": top_k,
            "true_success_count": outcome_counts.get('true_success', 0),
            "true_success_pct": float(outcome_counts.get('true_success', 0) / total * 100) if total > 0 else 0.0,
            "answer_in_different_passage_count": outcome_counts.get('answer_in_different_passage', 0),
            "answer_in_different_passage_pct": float(outcome_counts.get('answer_in_different_passage', 0) / total * 100) if total > 0 else 0.0,
            "true_failure_count": outcome_counts.get('true_failure', 0),
            "true_failure_pct": float(outcome_counts.get('true_failure', 0) / total * 100) if total > 0 else 0.0
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
    import sys
    datapath = sys.argv[1] if len(sys.argv) > 1 else 'data/seal_output.json'
    main(datapath)
