"""
Answer Location Analysis for SEAL Outputs
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
from utils.utils import strip_pseudoqueries, get_dataset_name, normalize_text, answer_in_text, answer_in_text_substring, get_ground_truth_ids, calculate_retrieval_metrics


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
        top_k = 10

        if not os.path.exists(file_path):
            print(f"error: running {script_name} File not found at {file_path}")
            return

        fieldnames = ['question', 'answers', 'outcome', 'gt_in_topk',
                      'answer_in_topk', 'gt_found_at_rank', 'answer_found_at_rank']

        # Counters
        outcome_counts = defaultdict(int)
        total = 0
        sum_precision_at_1 = 0.0
        sum_r_precision = 0.0

        # Track disagreements for comparison
        disagreement_examples = []
        max_examples = 3

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

                # Calculate retrieval metrics
                gold_ids = get_ground_truth_ids(entry)
                retrieved_ids = [ctx['passage_id'] for ctx in entry.get('ctxs', [])]
                metrics = calculate_retrieval_metrics(retrieved_ids, gold_ids)
                sum_precision_at_1 += metrics['precision_at_1']
                sum_r_precision += metrics['r_precision']

                csv_row = result.copy()
                csv_row['answers'] = str(csv_row['answers'])

                # Write all to main CSV
                main_writer.writerow(csv_row)

                # Write the different passage cases to separate CSV
                if result['outcome'] == 'answer_in_different_passage':
                    diff_writer.writerow(csv_row)

                # Compare substring vs tokenized matching for disagreement examples
                if len(disagreement_examples) < max_examples and not result['gt_in_topk']:
                    # Only check when GT is not in top-k (to find false positives)
                    answers = entry.get('answers', [])
                    retrieved = entry.get('ctxs', [])[:top_k]

                    for ctx in retrieved:
                        passage_text = ctx.get('text', '') + ' ' + ctx.get('title', '')
                        passage_text = strip_pseudoqueries(passage_text, file_path)

                        for ans in answers:
                            substring_match = answer_in_text_substring(ans, passage_text)
                            tokenized_match = answer_in_text(ans, passage_text)

                            # Found a disagreement: substring matched but tokenized didn't
                            if substring_match and not tokenized_match:
                                # Get tokenization details for debugging
                                from transformers import GPT2TokenizerFast
                                tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

                                answer_norm = normalize_text(ans)
                                text_norm = normalize_text(passage_text)

                                answer_tokens = tokenizer.encode(answer_norm, add_special_tokens=False)
                                text_tokens = tokenizer.encode(text_norm, add_special_tokens=False)

                                disagreement_examples.append({
                                    'question': entry.get('question', ''),
                                    'answer': ans,
                                    'answer_normalized': answer_norm,
                                    'answer_tokens': answer_tokens,
                                    'answer_tokens_decoded': [tokenizer.decode([t]) for t in answer_tokens],
                                    'passage_snippet': passage_text[:300] + '...' if len(passage_text) > 300 else passage_text,
                                    'passage_normalized': text_norm[:300] + '...' if len(text_norm) > 300 else text_norm,
                                    'passage_tokens': text_tokens[:50],  # First 50 tokens
                                    'passage_tokens_decoded': [tokenizer.decode([t]) for t in text_tokens[:50]]
                                })
                                break
                        if len(disagreement_examples) >= max_examples:
                            break

        # Build JSON summary
        output_data = {
            "total_queries": total,
            "top_k": top_k,
            "precision_at_1": float(sum_precision_at_1 / total) if total > 0 else 0.0,
            "r_precision": float(sum_r_precision / total) if total > 0 else 0.0,
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

        # Print disagreement examples
        if disagreement_examples:
            print("\n" + "="*80)
            print("EXAMPLES: Substring matched but Tokenized didn't (False Positives?)")
            print("="*80)
            for i, example in enumerate(disagreement_examples, 1):
                print(f"\nExample {i}:")
                print(f"Question: {example['question']}")
                print(f"Answer: '{example['answer']}'")
                print(f"Answer normalized: '{example['answer_normalized']}'")
                print(f"Answer tokens: {example['answer_tokens']}")
                print(f"Answer decoded: {example['answer_tokens_decoded']}")
                print(f"\nPassage snippet (original): {example['passage_snippet']}")
                print(f"\nPassage normalized: {example['passage_normalized']}")
                print(f"Passage tokens (first 50): {example['passage_tokens']}")
                print(f"Passage decoded: {example['passage_tokens_decoded']}")
                print("-" * 80)

        print(f"success running {script_name}")

    except Exception as e:
        print(f"error: running {script_name} {e}")
        raise



if __name__ == "__main__":
    import sys
    datapath = sys.argv[1] if len(sys.argv) > 1 else 'data/seal_output.json'
    main(datapath)
