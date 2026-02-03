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
import re
from collections import defaultdict
import csv


def strip_pseudoqueries(text, datapath):
    """Strip pseudoquery markers from text if using Minder data."""
    if "minder_output.json" in datapath:
        # Remove || ... @@ patterns
        text = re.sub(r'\|\|[^@]*@@', '', text)
    return text


def normalize_text(text):
    """Normalize text for matching: lowercase, strip whitespace."""
    return text.lower().strip()


def answer_in_text(answer, text):
    """Check if answer appears in text (case-insensitive)."""
    return normalize_text(answer) in normalize_text(text)


def get_ground_truth_ids(query_data):
    """Extract ground-truth passage IDs."""
    gold_ids = set()
    if 'positive_ctxs' in query_data:
        for ctx in query_data['positive_ctxs']:
            gold_ids.add(ctx.get('passage_id', '').split('...')[0])
    return gold_ids


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
    file_path = datapath
    output_csv = "generated_data/answer_location_analysis.csv"
    diff_passage_csv = "generated_data/answer_in_different_passage.csv"
    top_k = 2
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
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
            if total % 1000 == 0:
                print(f"  Processed {total} queries...")

            result = analyze_retrieval_outcome(entry, top_k=top_k, datapath=file_path)
            outcome_counts[result['outcome']] += 1
            
            csv_row = result.copy()
            csv_row['answers'] = str(csv_row['answers'])
            
            # Write all to main CSV
            main_writer.writerow(csv_row)
            
            # Write the 426 cases to separate CSV
            if result['outcome'] == 'answer_in_different_passage':
                diff_writer.writerow(csv_row)
    
    # Summary (unchanged)
    print(f"\nProcessed {total} queries.")
    print(f"All results saved to: {output_csv}")
    print(f"'Answer in different passage' cases saved to: {diff_passage_csv}")



if __name__ == "__main__":
    main()
