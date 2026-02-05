import json
import numpy as np
import os
from typing import Dict, List, Any
from utils.utils import get_dataset_name, stream_data


def parse_keys_field(keys_field: Any) -> List:
    """Parse keys field and convert Decimal to float."""
    if isinstance(keys_field, str):
        try:
            keys_list = json.loads(keys_field)
        except json.JSONDecodeError:
            try:
                import ast
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


def get_recall_category(retrieved_ctxs: List[Dict], gold_ids: set) -> str:
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


def analyze_query_sets(query_data: Dict, top_k: int = 10, datapath: str = "") -> Dict:
    """Analyze n-gram keys for a single query."""
    gold_ids = set()
    if 'positive_ctxs' in query_data:
        for ctx in query_data['positive_ctxs']:
            gold_ids.add(ctx['passage_id'])

    retrieved_ctxs = query_data.get('ctxs', [])[:top_k]
    category = get_recall_category(retrieved_ctxs, gold_ids)

    # Store just the score for each ngram_id (not a list)
    # Since each ngram has a fixed score, we only need to store it once
    p_keys_scores = {}
    n_keys_scores = {}

    for ctx in retrieved_ctxs:
        pid = ctx.get('passage_id')
        is_positive = pid in gold_ids

        raw_keys = parse_keys_field(ctx.get('keys', []))

        for item in raw_keys:
            if len(item) < 3:
                continue

            ngram_id = item[1]
            score = item[2]

            # Store just one score per ngram_id (they should all be the same)
            if is_positive:
                if ngram_id not in p_keys_scores:
                    p_keys_scores[ngram_id] = score
            else:
                if ngram_id not in n_keys_scores:
                    n_keys_scores[ngram_id] = score

    set_P = set(p_keys_scores.keys())
    set_N = set(n_keys_scores.keys())

    unique_P_ids = set_P - set_N
    unique_N_ids = set_N - set_P
    intersection_ids = set_P & set_N

    # Get scores for each category
    scores_unique_P = [p_keys_scores[i] for i in unique_P_ids]
    scores_unique_N = [n_keys_scores[i] for i in unique_N_ids]
    scores_shared = [p_keys_scores[i] for i in intersection_ids]

    # Calculate total keys
    total_keys = len(set_P | set_N)

    # Calculate percentages
    pct_unique_P = (len(unique_P_ids) / total_keys * 100) if total_keys > 0 else 0.0
    pct_unique_N = (len(unique_N_ids) / total_keys * 100) if total_keys > 0 else 0.0
    pct_shared = (len(intersection_ids) / total_keys * 100) if total_keys > 0 else 0.0

    stats = {
        "question": query_data.get("question", ""),
        "category": category,
        "total_keys": total_keys,
        "pct_unique_P": pct_unique_P,
        "pct_unique_N": pct_unique_N,
        "pct_shared": pct_shared,
        "count_unique_P": len(unique_P_ids),
        "count_unique_N": len(unique_N_ids),
        "count_shared": len(intersection_ids),
        "avg_score_unique_P": float(np.mean(scores_unique_P)) if scores_unique_P else 0.0,
        "avg_score_unique_N": float(np.mean(scores_unique_N)) if scores_unique_N else 0.0,
        "avg_score_shared": float(np.mean(scores_shared)) if scores_shared else 0.0,
        "sum_score_unique_P": float(np.sum(scores_unique_P)) if scores_unique_P else 0.0,
        "sum_score_unique_N": float(np.sum(scores_unique_N)) if scores_unique_N else 0.0,
        "sum_score_shared": float(np.sum(scores_shared)) if scores_shared else 0.0,
    }

    return stats


def main(datapath="data/seal_output.json"):
    script_name = "prelim_analysis"
    print(f"running {script_name}")

    try:
        INPUT_FILE = datapath
        dataset_name = get_dataset_name(datapath)
        output_dir = f"generated_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        TOP_K = 10

        all_results = []

        if not os.path.exists(INPUT_FILE):
            print(f"error: File not found: {INPUT_FILE}")
            raise Exception(f"File not found: {INPUT_FILE}")

        for entry in stream_data(INPUT_FILE):
            result = analyze_query_sets(entry, top_k=TOP_K, datapath=INPUT_FILE)
            all_results.append(result)

        if all_results:
            # Calculate category distribution
            category_counts = {}
            for r in all_results:
                cat = r['category']
                category_counts[cat] = category_counts.get(cat, 0) + 1

            category_distribution = {}
            total_queries = len(all_results)
            for cat in ['PP', 'PN', 'NP', 'NN', 'X']:
                count = category_counts.get(cat, 0)
                category_distribution[cat] = {
                    "count": count,
                    "pct": float(count / total_queries * 100) if total_queries > 0 else 0.0
                }

            # Calculate overall metrics
            output_data = {
                "total_queries": len(all_results),
                "overall_metrics": {
                    "category_distribution": category_distribution,
                    "avg_total_keys": float(np.mean([r['total_keys'] for r in all_results])),
                    "avg_pct_unique_P": float(np.mean([r['pct_unique_P'] for r in all_results])),
                    "avg_pct_unique_N": float(np.mean([r['pct_unique_N'] for r in all_results])),
                    "avg_pct_shared": float(np.mean([r['pct_shared'] for r in all_results])),
                    "avg_count_unique_P": float(np.mean([r['count_unique_P'] for r in all_results])),
                    "avg_count_unique_N": float(np.mean([r['count_unique_N'] for r in all_results])),
                    "avg_count_shared": float(np.mean([r['count_shared'] for r in all_results])),
                    "avg_score_unique_P": float(np.mean([r['avg_score_unique_P'] for r in all_results])),
                    "avg_score_unique_N": float(np.mean([r['avg_score_unique_N'] for r in all_results])),
                    "avg_score_shared": float(np.mean([r['avg_score_shared'] for r in all_results]))
                },
                "per_query_results": all_results
            }

            # Save JSON output
            output_json = os.path.join(output_dir, f"{script_name}_results.json")
            with open(output_json, 'w') as f:
                json.dump(output_data, f, indent=2)

        print(f"success running {script_name}")

    except Exception as e:
        print(f"error: {e}")
        raise


if __name__ == "__main__":
    import sys
    datapath = sys.argv[1] if len(sys.argv) > 1 else "data/seal_output.json"
    main(datapath)
