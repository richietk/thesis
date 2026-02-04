import ijson
import json
import ast
import numpy as np
from collections import defaultdict
import os
import csv
from typing import Dict, List, Set, Any, Tuple


def strip_ngram_markers(ngram: str, datapath: str) -> str:
    """Strip pseudoquery markers from ngrams if using Minder data."""
    if "minder_output.json" in datapath:
        # Remove " ||" prefix from ngrams
        ngram = ngram.replace(" ||", "").strip()
    return ngram


def get_dataset_name(datapath: str) -> str:
    """Extract dataset name (seal or minder) from datapath."""
    if "minder" in datapath.lower():
        return "minder"
    elif "seal" in datapath.lower():
        return "seal"
    else:
        return os.path.splitext(os.path.basename(datapath))[0]


def parse_keys_field(keys_field: Any) -> List:
    """
    Parse the keys field
    Returns a list of parsed key entries
    """
    if isinstance(keys_field, str):
        try:
            return json.loads(keys_field)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(keys_field)
            except:
                return []
    return keys_field if keys_field else []


def get_recall_category(retrieved_ctxs: List[Dict], gold_ids: Set[str]) -> str:
    """
    Classifies the query based on the top 2 retrieved documents.
    PP, PN, NP, NN
    Retrieved ctxs: list of retrieved context dicts
    Gold_ids: ground truth passage ids
    Returns the category string
    """
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
    """
    Extracts all scores for a set of n-gram id-s.
    id_set: set of ngram integer id-s
    source_map: dict mapping id-s to sources
    returns list of all scores
    """
    scores = []
    for i in id_set:
        scores.extend(source_map[i])
    return scores


def extract_ngram_examples(query_data: Dict, top_k: int = 10, datapath: str = "") -> Dict:
    """
    Extracts detailed n-gram examples for a query, including the actual n-gram strings.
    Returns dict with lists of (ngram_string, corpus_freq, score) tuples for each category.
    """
    # 1. identify the positives
    gold_ids = set()
    if 'positive_ctxs' in query_data:
        for ctx in query_data['positive_ctxs']:
            gold_ids.add(ctx['passage_id'])

    # 2. get retrieved docs
    retrieved_ctxs = query_data.get('ctxs', [])[:top_k]

    # 3. build maps: ngram_id -> (ngram_string, corpus_freq, max_score)
    # We store the actual ngram strings along with their scores
    p_keys_details = {}  # {ngram_id: (ngram_string, corpus_freq, score)}
    n_keys_details = {}

    for ctx in retrieved_ctxs:
        pid = ctx.get('passage_id')
        is_positive = pid in gold_ids

        raw_keys = parse_keys_field(ctx.get('keys', []))

        for item in raw_keys:
            if len(item) < 3:
                continue

            ngram_string = strip_ngram_markers(item[0], datapath)  # The actual n-gram text
            ngram_id = item[1]      # Corpus frequency (used as ID)
            score = item[2]         # Score

            if is_positive:
                # Keep the highest score for each ngram
                if ngram_id not in p_keys_details or score > p_keys_details[ngram_id][2]:
                    p_keys_details[ngram_id] = (ngram_string, ngram_id, score)
            else:
                if ngram_id not in n_keys_details or score > n_keys_details[ngram_id][2]:
                    n_keys_details[ngram_id] = (ngram_string, ngram_id, score)

    # 4. identify unique and shared keys
    set_P = set(p_keys_details.keys())
    set_N = set(n_keys_details.keys())

    unique_P_ids = set_P - set_N
    unique_N_ids = set_N - set_P
    shared_ids = set_P & set_N

    # 5. extract details for each category, sorted by score descending
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
        "unique_positive": unique_P_examples,  # List of (ngram, freq, score)
        "unique_negative": unique_N_examples,
        "shared": shared_examples,  # List of (ngram, freq, score_in_P, score_in_N)
    }


def print_ngram_examples(examples: Dict, max_examples: int = 10) -> None:
    """
    Prints detailed n-gram examples for a query.
    """
    print("\n" + "=" * 100)
    print(f"N-GRAM EXAMPLES FOR: {examples['question']}")
    print("=" * 100)

    # Unique Positive Keys
    print(f"\n--- UNIQUE TO POSITIVE PASSAGES ({len(examples['unique_positive'])} total) ---")
    print(f"{'N-gram':<50} {'Corpus Freq':>12} {'Score':>10}")
    print("-" * 75)
    for ngram, freq, score in examples['unique_positive'][:max_examples]:
        # Escape/clean the ngram for display
        display_ngram = repr(ngram) if len(ngram) < 45 else repr(ngram[:42]) + "..."
        print(f"{display_ngram:<50} {freq:>12} {score:>10.2f}")
    if len(examples['unique_positive']) > max_examples:
        print(f"  ... and {len(examples['unique_positive']) - max_examples} more")

    # Summary stats for unique positive
    if examples['unique_positive']:
        scores = [x[2] for x in examples['unique_positive']]
        print(f"\n  Total: {len(scores)} keys, Sum={sum(scores):.1f}, Avg={np.mean(scores):.2f}")

    # Unique Negative Keys
    print(f"\n--- UNIQUE TO NEGATIVE PASSAGES ({len(examples['unique_negative'])} total) ---")
    print(f"{'N-gram':<50} {'Corpus Freq':>12} {'Score':>10}")
    print("-" * 75)
    for ngram, freq, score in examples['unique_negative'][:max_examples]:
        display_ngram = repr(ngram) if len(ngram) < 45 else repr(ngram[:42]) + "..."
        print(f"{display_ngram:<50} {freq:>12} {score:>10.2f}")
    if len(examples['unique_negative']) > max_examples:
        print(f"  ... and {len(examples['unique_negative']) - max_examples} more")

    # Summary stats for unique negative
    if examples['unique_negative']:
        scores = [x[2] for x in examples['unique_negative']]
        print(f"\n  Total: {len(scores)} keys, Sum={sum(scores):.1f}, Avg={np.mean(scores):.2f}")

    # Shared Keys
    print(f"\n--- SHARED BETWEEN POSITIVE AND NEGATIVE ({len(examples['shared'])} total) ---")
    print(f"{'N-gram':<40} {'Freq':>8} {'Score(P)':>10} {'Score(N)':>10}")
    print("-" * 75)
    for item in examples['shared'][:max_examples]:
        ngram, freq, score_p, score_n = item
        display_ngram = repr(ngram) if len(ngram) < 35 else repr(ngram[:32]) + "..."
        print(f"{display_ngram:<40} {freq:>8} {score_p:>10.2f} {score_n:>10.2f}")
    if len(examples['shared']) > max_examples:
        print(f"  ... and {len(examples['shared']) - max_examples} more")

    print("=" * 100)


def find_query_by_substring(data: List[Dict], substring: str) -> Dict:
    """
    Finds a query containing the given substring.
    """
    substring_lower = substring.lower()
    for entry in data:
        if substring_lower in entry.get('question', '').lower():
            return entry
    return None


def analyze_query_sets(query_data: Dict, top_k: int = 10, datapath: str = "") -> Dict:
    """
    Analyzes the n-gram keys for a single query.
    query_data: a single entry
    top_k: Number of retrieved documents to consider
    returns a dictionary containing the statistics
    """

    # 1. identify the positives
    gold_ids = set()
    if 'positive_ctxs' in query_data:
        for ctx in query_data['positive_ctxs']:
            gold_ids.add(ctx['passage_id'])
            
    # 2. get retrieved docs and determine category
    retrieved_ctxs = query_data.get('ctxs', [])[:top_k]
    category = get_recall_category(retrieved_ctxs, gold_ids)
    
    # 3. build key maps for positive and negative scores
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

    # 4. create sets
    set_P = set(p_keys_map.keys())
    set_N = set(n_keys_map.keys())
    
    unique_P_ids = set_P - set_N
    unique_N_ids = set_N - set_P
    intersection_ids = set_P & set_N
    
    # 5. get score lists for each set
    scores_unique_P = get_scores_list(unique_P_ids, p_keys_map)
    scores_unique_N = get_scores_list(unique_N_ids, n_keys_map)
    scores_shared_in_P = get_scores_list(intersection_ids, p_keys_map)
    scores_shared_in_N = get_scores_list(intersection_ids, n_keys_map)
    
    stats = {
        "question": query_data.get("question", ""),
        "category": category,
        
        # total counts
        "count_P_total": len(set_P),
        "count_N_total": len(set_N),
        
        # unique counts
        "count_unique_P": len(unique_P_ids),
        "count_unique_N": len(unique_N_ids),
        "count_shared": len(intersection_ids),
        
        # avg scores for unique keys
        "avg_score_unique_P": np.mean(scores_unique_P) if scores_unique_P else 0.0,
        "avg_score_unique_N": np.mean(scores_unique_N) if scores_unique_N else 0.0,
        
        # sum scores for unique keys
        "sum_score_unique_P": np.sum(scores_unique_P) if scores_unique_P else 0.0,
        "sum_score_unique_N": np.sum(scores_unique_N) if scores_unique_N else 0.0,
        
        # shared key scores
        "avg_score_shared_in_P": np.mean(scores_shared_in_P) if scores_shared_in_P else 0.0,
        "avg_score_shared_in_N": np.mean(scores_shared_in_N) if scores_shared_in_N else 0.0,
        "sum_score_shared_in_P": np.sum(scores_shared_in_P) if scores_shared_in_P else 0.0,
        "sum_score_shared_in_N": np.sum(scores_shared_in_N) if scores_shared_in_N else 0.0,
    }
    
    return stats


def print_category_summary(all_results: List[Dict]) -> None:
    """
    Prints summary table
    """
    groups = defaultdict(list)
    for r in all_results:
        groups[r['category']].append(r)
    
    print("\n" + "=" * 120)
    print("CATEGORY-WISE SUMMARY")
    print("=" * 120)
    print(f"{'CAT':<5} | {'COUNT':>6} | {'Unique P (Cnt/Avg/Sum)':<25} | "
          f"{'Unique N (Cnt/Avg/Sum)':<25} | {'Shared (Cnt/AvgP/AvgN)':<25}")
    print("-" * 120)
    
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
    """
    Prints stats
    """
    print("\n" + "=" * 80)
    print(f"GLOBAL SUMMARY (N={len(all_results)} queries)")
    print("=" * 80)
    
    # category distr
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
    """
    output to csv
    """
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


def stream_data(file_path: str):
    """Stream data from unified json using ijson."""
    with open(file_path, 'rb') as f:
        parser = ijson.items(f, 'item')
        for entry in parser:
            yield entry


def find_query_by_substring_streaming(file_path: str, query_substring: str) -> Dict:
    """Find a query by substring in the question field (streaming version)."""
    query_lower = query_substring.lower()
    for entry in stream_data(file_path):
        if query_lower in entry.get('question', '').lower():
            return entry
    return None


def main(datapath="data/seal_output.json"):
    INPUT_FILE = datapath
    dataset_name = get_dataset_name(datapath)
    OUTPUT_CSV = f"generated_data/seal_unified_analysis_{dataset_name}.csv"
    TOP_K = 10

    all_results = []

    if not os.path.exists(INPUT_FILE):
        raise Exception(f"File not found: {INPUT_FILE}")

    print(f"\nStreaming and analyzing data (top_k={TOP_K})...")
    i = 0
    for entry in stream_data(INPUT_FILE):
        i += 1
        if i > 0 and i % 1000 == 0:
            print(f"  Processing query {i}...")

        result = analyze_query_sets(entry, top_k=TOP_K, datapath=INPUT_FILE)
        all_results.append(result)

    print(f"Analysis complete. Processed {len(all_results)} queries.")

    if all_results:
        save_results_to_csv(all_results, OUTPUT_CSV)

        print_global_summary(all_results)
        print_category_summary(all_results)

        # sample output of first 3 queries
        print("\n" + "=" * 80)
        print("SAMPLE OUTPUT (First 3 queries)")
        print("=" * 80)
        for i, r in enumerate(all_results[:3]):
            print(f"\n[{i+1}] {r['question'][:60]}...")
            print(f"    Category: {r['category']}")
            print(f"    Unique P: {r['count_unique_P']} keys (avg={r['avg_score_unique_P']:.2f}, sum={r['sum_score_unique_P']:.0f})")
            print(f"    Unique N: {r['count_unique_N']} keys (avg={r['avg_score_unique_N']:.2f}, sum={r['sum_score_unique_N']:.0f})")
            print(f"    Shared:   {r['count_shared']} keys (avgP={r['avg_score_shared_in_P']:.2f}, avgN={r['avg_score_shared_in_N']:.2f})")

        # ============================================================
        # DETAILED N-GRAM EXAMPLES FOR SPECIFIC QUERIES
        # ============================================================
        print("\n\n" + "#" * 100)
        print("# DETAILED N-GRAM EXAMPLES FOR SPECIFIC QUERIES")
        print("#" * 100)

        # List of queries to examine in detail
        example_queries = [
            "who sings does he love me with reba",  # NN case from thesis
            "when does the new my hero academia movie come out",  # Another example
        ]

        for query_substring in example_queries:
            query_entry = find_query_by_substring_streaming(INPUT_FILE, query_substring)
            if query_entry:
                examples = extract_ngram_examples(query_entry, top_k=TOP_K, datapath=INPUT_FILE)
                print_ngram_examples(examples, max_examples=15)
            else:
                print(f"\nQuery not found: '{query_substring}'")


if __name__ == "__main__":
    import sys
    datapath = sys.argv[1] if len(sys.argv) > 1 else "data/seal_output.json"
    main(datapath)
