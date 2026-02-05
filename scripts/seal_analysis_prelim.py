import ijson
import json
import ast
import numpy as np
from collections import defaultdict
import os
import csv
from typing import Dict, List, Set, Any, Tuple
from scripts.utils.utils import strip_ngram_markers, get_dataset_name, stream_data


def parse_keys_field(keys_field: Any) -> List:
    """Parse keys field and convert Decimal to float."""
    if isinstance(keys_field, str):
        try:
            keys_list = json.loads(keys_field)
        except json.JSONDecodeError:
            try:
                keys_list = ast.literal_eval(keys_field)
            except:
                return []
    else:
        keys_list = keys_field if keys_field else []

    result = []
    for item in keys_list:
        if len(item) >= 3:
            ngram = item[0]
            freq = int(item[1]) if item[1] is not None else 0
            score = float(item[2]) if item[2] is not None else 0.0
            result.append([ngram, freq, score])
        else:
            result.append(item)
    return result


def get_recall_category(retrieved_ctxs: List[Dict], gold_ids: Set[str]) -> str:
    """Classify query based on top 2 retrieved docs (PP, PN, NP, NN)."""
    if len(retrieved_ctxs) < 2:
        return 'X'
        
    r1_id = retrieved_ctxs[0].get('passage_id')
    r1_is_pos = r1_id in gold_ids
    
    r2_id = retrieved_ctxs[1].get('passage_id')
    r2_is_pos = r2_id in gold_ids
    
    if r1_is_pos and r2_is_pos:
        return 'PP'
    if r1_is_pos and not r2_is_pos:
        return 'PN'
    if not r1_is_pos and r2_is_pos:
        return 'NP'
    if not r1_is_pos and not r2_is_pos:
        return 'NN'
    return 'X'


def get_scores_list(id_set: Set[int], source_map: Dict[int, List[float]]) -> List[float]:
    """Extract all scores for a set of n-gram IDs."""
    scores = []
    for i in id_set:
        scores.extend(source_map[i])
    return scores


def extract_ngram_examples(query_data: Dict, top_k: int = 10, datapath: str = "") -> Dict:
    """Extract detailed n-gram examples for a query."""
    gold_ids = set()
    if 'positive_ctxs' in query_data:
        for ctx in query_data['positive_ctxs']:
            gold_ids.add(ctx['passage_id'])

    retrieved_ctxs = query_data.get('ctxs', [])[:top_k]

    p_keys_details = {}
    n_keys_details = {}

    for ctx in retrieved_ctxs:
        pid = ctx.get('passage_id')
        is_positive = pid in gold_ids

        raw_keys = parse_keys_field(ctx.get('keys', []))

        for item in raw_keys:
            if len(item) < 3:
                continue

            ngram_string = strip_ngram_markers(item[0], datapath)
            ngram_id = item[1]
            score = item[2]

            if is_positive:
                if ngram_id not in p_keys_details or score > p_keys_details[ngram_id][2]:
                    p_keys_details[ngram_id] = (ngram_string, ngram_id, score)
            else:
                if ngram_id not in n_keys_details or score > n_keys_details[ngram_id][2]:
                    n_keys_details[ngram_id] = (ngram_string, ngram_id, score)

    set_P = set(p_keys_details.keys())
    set_N = set(n_keys_details.keys())

    unique_P_ids = set_P - set_N
    unique_N_ids = set_N - set_P
    shared_ids = set_P & set_N

    unique_P_examples = sorted(
        [p_keys_details[i] for i in unique_P_ids],
        key=lambda x: x[2], reverse=True
    )
    unique_N_examples = sorted(
        [n_keys_details[i] for i in unique_N_ids],
        key=lambda x: x[2], reverse=True
    )
    shared_examples = sorted(
        [(p_keys_details[i][0], p_keys_details[i][1], p_keys_details[i][2], n_keys_details[i][2])
         for i in shared_ids],
        key=lambda x: x[2], reverse=True
    )

    return {
        "question": query_data.get("question", ""),
        "unique_positive": unique_P_examples,
        "unique_negative": unique_N_examples,
        "shared": shared_examples,
    }


def print_ngram_examples(examples: Dict, max_examples: int = 10) -> None:
    """Print detailed n-gram examples for a query."""
    print("\n" + "=" * 100)
    print(f"N-GRAM EXAMPLES FOR: {examples['question']}")
    print("=" * 100)

    print(f"\nUNIQUE TO POSITIVE PASSAGES ({len(examples['unique_positive'])} total)")
    print(f"{'N-gram':<50} {'Corpus Freq':>12} {'Score':>10}")
    for ngram, freq, score in examples['unique_positive'][:max_examples]:
        display_ngram = repr(ngram) if len(ngram) < 45 else repr(ngram[:42]) + "..."
        print(f"{display_ngram:<50} {freq:>12} {score:>10.2f}")
    if len(examples['unique_positive']) > max_examples:
        print(f"  ... and {len(examples['unique_positive']) - max_examples} more")

    if examples['unique_positive']:
        scores = [x[2] for x in examples['unique_positive']]
        print(f"\n  Total: {len(scores)} keys, Sum={sum(scores):.1f}, Avg={np.mean(scores):.2f}")

    print(f"\nUNIQUE TO NEGATIVE PASSAGES ({len(examples['unique_negative'])} total)")
    print(f"{'N-gram':<50} {'Corpus Freq':>12} {'Score':>10}")
    for ngram, freq, score in examples['unique_negative'][:max_examples]:
        display_ngram = repr(ngram) if len(ngram) < 45 else repr(ngram[:42]) + "..."
        print(f"{display_ngram:<50} {freq:>12} {score:>10.2f}")
    if len(examples['unique_negative']) > max_examples:
        print(f"  ... and {len(examples['unique_negative']) - max_examples} more")

    if examples['unique_negative']:
        scores = [x[2] for x in examples['unique_negative']]
        print(f"\n  Total: {len(scores)} keys, Sum={sum(scores):.1f}, Avg={np.mean(scores):.2f}")

    print(f"\nSHARED BETWEEN POSITIVE AND NEGATIVE ({len(examples['shared'])} total)")
    print(f"{'N-gram':<40} {'Freq':>8} {'Score(P)':>10} {'Score(N)':>10}")
    for item in examples['shared'][:max_examples]:
        ngram, freq, score_p, score_n = item
        display_ngram = repr(ngram) if len(ngram) < 35 else repr(ngram[:32]) + "..."
        print(f"{display_ngram:<40} {freq:>8} {score_p:>10.2f} {score_n:>10.2f}")
    if len(examples['shared']) > max_examples:
        print(f"  ... and {len(examples['shared']) - max_examples} more")

    print("=" * 100)


def find_query_by_substring(data: List[Dict], substring: str) -> Dict:
    """Find a query containing the given substring."""
    substring_lower = substring.lower()
    for entry in data:
        if substring_lower in entry.get('question', '').lower():
            return entry
    return None


def analyze_query_sets(query_data: Dict, top_k: int = 10, datapath: str = "") -> Dict:
    """Analyze n-gram keys for a single query."""
    gold_ids = set()
    if 'positive_ctxs' in query_data:
        for ctx in query_data['positive_ctxs']:
            gold_ids.add(ctx['passage_id'])

    retrieved_ctxs = query_data.get('ctxs', [])[:top_k]
    category = get_recall_category(retrieved_ctxs, gold_ids)

    p_keys_map = defaultdict(list)
    n_keys_map = defaultdict(list)
    
    for ctx in retrieved_ctxs:
        pid = ctx.get('passage_id')
        is_positive = pid in gold_ids
        
        raw_keys = parse_keys_field(ctx.get('keys', []))
        
        for item in raw_keys:
            if len(item) < 3:
                continue
                
            ngram_id = item[1]
            score = item[2]
            
            if is_positive:
                p_keys_map[ngram_id].append(score)
            else:
                n_keys_map[ngram_id].append(score)

    set_P = set(p_keys_map.keys())
    set_N = set(n_keys_map.keys())
    
    unique_P_ids = set_P - set_N
    unique_N_ids = set_N - set_P
    intersection_ids = set_P & set_N

    scores_unique_P = get_scores_list(unique_P_ids, p_keys_map)
    scores_unique_N = get_scores_list(unique_N_ids, n_keys_map)
    scores_shared_in_P = get_scores_list(intersection_ids, p_keys_map)
    scores_shared_in_N = get_scores_list(intersection_ids, n_keys_map)
    
    stats = {
        "question": query_data.get("question", ""),
        "category": category,
        "count_P_total": len(set_P),
        "count_N_total": len(set_N),
        "count_unique_P": len(unique_P_ids),
        "count_unique_N": len(unique_N_ids),
        "count_shared": len(intersection_ids),
        "avg_score_unique_P": np.mean(scores_unique_P) if scores_unique_P else 0.0,
        "avg_score_unique_N": np.mean(scores_unique_N) if scores_unique_N else 0.0,
        "sum_score_unique_P": np.sum(scores_unique_P) if scores_unique_P else 0.0,
        "sum_score_unique_N": np.sum(scores_unique_N) if scores_unique_N else 0.0,
        "avg_score_shared_in_P": np.mean(scores_shared_in_P) if scores_shared_in_P else 0.0,
        "avg_score_shared_in_N": np.mean(scores_shared_in_N) if scores_shared_in_N else 0.0,
        "sum_score_shared_in_P": np.sum(scores_shared_in_P) if scores_shared_in_P else 0.0,
        "sum_score_shared_in_N": np.sum(scores_shared_in_N) if scores_shared_in_N else 0.0,
    }
    
    return stats


def print_category_summary(all_results: List[Dict]) -> None:
    """Print summary table."""
    groups = defaultdict(list)
    for r in all_results:
        groups[r['category']].append(r)
    
    print("\n" + "=" * 120)
    print("CATEGORY-WISE SUMMARY")
    print("=" * 120)
    print(f"{'CAT':<5} | {'COUNT':>6} | {'Unique P (Cnt/Avg/Sum)':<25} | "
          f"{'Unique N (Cnt/Avg/Sum)':<25} | {'Shared (Cnt/AvgP/AvgN)':<25}")
    
    for cat in ['PP', 'PN', 'NP', 'NN', 'X']:
        entries = groups.get(cat, [])
        if not entries:
            continue
        
        n = len(entries)
        
        # unique P stats
        cnt_uP = np.mean([x['count_unique_P'] for x in entries])
        avg_uP = np.mean([x['avg_score_unique_P'] for x in entries])
        sum_uP = np.mean([x['sum_score_unique_P'] for x in entries])
        
        # unique N stats
        cnt_uN = np.mean([x['count_unique_N'] for x in entries])
        avg_uN = np.mean([x['avg_score_unique_N'] for x in entries])
        sum_uN = np.mean([x['sum_score_unique_N'] for x in entries])
        
        # shared stats
        cnt_sh = np.mean([x['count_shared'] for x in entries])
        avg_sh_P = np.mean([x['avg_score_shared_in_P'] for x in entries])
        avg_sh_N = np.mean([x['avg_score_shared_in_N'] for x in entries])
        
        print(f"{cat:<5} | {n:>6} | {cnt_uP:>5.1f} / {avg_uP:>6.2f} / {sum_uP:>8.0f} | "
              f"{cnt_uN:>5.1f} / {avg_uN:>6.2f} / {sum_uN:>8.0f} | "
              f"{cnt_sh:>5.1f} / {avg_sh_P:>6.2f} / {avg_sh_N:>6.2f}")
    
    print("=" * 120)
    print("Legend: Cnt=Count, Avg=Average Score, Sum=Sum Score, AvgP=Avg in Pos, AvgN=Avg in Neg")


def print_global_summary(all_results: List[Dict]) -> None:
    """Print global stats."""
    print("\n" + "=" * 80)
    print(f"GLOBAL SUMMARY (N={len(all_results)} queries)")
    print("=" * 80)

    cat_counts = defaultdict(int)
    for r in all_results:
        cat_counts[r['category']] += 1
    
    print("\nCategory Distribution:")
    for cat in ['PP', 'PN', 'NP', 'NN', 'X']:
        count = cat_counts.get(cat, 0)
        pct = (count / len(all_results)) * 100 if all_results else 0
        print(f"  {cat}: {count:>6} ({pct:>5.1f}%)")
    
    print("\nOverall Metrics (Averaged across all queries):")
    print(f"  Avg Unique Keys (Positive): {np.mean([r['count_unique_P'] for r in all_results]):.2f}")
    print(f"  Avg Unique Keys (Negative): {np.mean([r['count_unique_N'] for r in all_results]):.2f}")
    print(f"  Avg Shared Keys:            {np.mean([r['count_shared'] for r in all_results]):.2f}")
    
    print(f"\n  Avg Score - Unique P: {np.mean([r['avg_score_unique_P'] for r in all_results]):.2f}")
    print(f"  Avg Score - Unique N: {np.mean([r['avg_score_unique_N'] for r in all_results]):.2f}")
    print(f"  Avg Score - Shared in P: {np.mean([r['avg_score_shared_in_P'] for r in all_results]):.2f}")
    print(f"  Avg Score - Shared in N: {np.mean([r['avg_score_shared_in_N'] for r in all_results]):.2f}")
    
    print("=" * 80)


def save_results_to_csv(all_results: List[Dict], output_path: str) -> None:
    """Save results to CSV."""
    if not all_results:
        print("No results to save.")
        return
        
    fieldnames = [
        "question", "category",
        "count_P_total", "count_N_total",
        "count_unique_P", "count_unique_N", "count_shared",
        "avg_score_unique_P", "avg_score_unique_N",
        "sum_score_unique_P", "sum_score_unique_N",
        "avg_score_shared_in_P", "avg_score_shared_in_N",
        "sum_score_shared_in_P", "sum_score_shared_in_N"
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\nResults saved to: {output_path}")


def find_query_by_substring_streaming(file_path: str, query_substring: str) -> Dict:
    """Find a query by substring in the question field (streaming version)."""
    query_lower = query_substring.lower()
    for entry in stream_data(file_path):
        if query_lower in entry.get('question', '').lower():
            return entry
    return None


def main(datapath="data/seal_output.json"):
    import sys

    script_name = "seal_analysis_prelim"
    print(f"running {script_name}")

    try:
        INPUT_FILE = datapath
        dataset_name = get_dataset_name(datapath)
        output_dir = f"generated_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        OUTPUT_CSV = f"{output_dir}/seal_unified_analysis_{dataset_name}.csv"
        TOP_K = 10

        all_results = []

        if not os.path.exists(INPUT_FILE):
            print(f"error: running {script_name} File not found: {INPUT_FILE}")
            raise Exception(f"File not found: {INPUT_FILE}")

        for entry in stream_data(INPUT_FILE):
            result = analyze_query_sets(entry, top_k=TOP_K, datapath=INPUT_FILE)
            all_results.append(result)

        if all_results:
            save_results_to_csv(all_results, OUTPUT_CSV)

            # Build JSON summary
            groups = defaultdict(list)
            for r in all_results:
                groups[r['category']].append(r)

            output_data = {
                "total_queries": len(all_results),
                "category_distribution": {},
                "overall_metrics": {
                    "avg_unique_keys_positive": float(np.mean([r['count_unique_P'] for r in all_results])),
                    "avg_unique_keys_negative": float(np.mean([r['count_unique_N'] for r in all_results])),
                    "avg_shared_keys": float(np.mean([r['count_shared'] for r in all_results])),
                    "avg_score_unique_P": float(np.mean([r['avg_score_unique_P'] for r in all_results])),
                    "avg_score_unique_N": float(np.mean([r['avg_score_unique_N'] for r in all_results])),
                    "avg_score_shared_in_P": float(np.mean([r['avg_score_shared_in_P'] for r in all_results])),
                    "avg_score_shared_in_N": float(np.mean([r['avg_score_shared_in_N'] for r in all_results]))
                }
            }

            # Category-wise summary
            for cat in ['PP', 'PN', 'NP', 'NN', 'X']:
                entries = groups.get(cat, [])
                if not entries:
                    continue

                output_data["category_distribution"][cat] = {
                    "count": len(entries),
                    "pct": float(len(entries) / len(all_results) * 100),
                    "avg_count_unique_P": float(np.mean([x['count_unique_P'] for x in entries])),
                    "avg_score_unique_P": float(np.mean([x['avg_score_unique_P'] for x in entries])),
                    "avg_sum_unique_P": float(np.mean([x['sum_score_unique_P'] for x in entries])),
                    "avg_count_unique_N": float(np.mean([x['count_unique_N'] for x in entries])),
                    "avg_score_unique_N": float(np.mean([x['avg_score_unique_N'] for x in entries])),
                    "avg_sum_unique_N": float(np.mean([x['sum_score_unique_N'] for x in entries])),
                    "avg_count_shared": float(np.mean([x['count_shared'] for x in entries])),
                    "avg_score_shared_in_P": float(np.mean([x['avg_score_shared_in_P'] for x in entries])),
                    "avg_score_shared_in_N": float(np.mean([x['avg_score_shared_in_N'] for x in entries]))
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
    datapath = sys.argv[1] if len(sys.argv) > 1 else "data/seal_output.json"
    main(datapath)
