import sys
import ijson
import numpy as np
import os
import re
import json
import argparse
from collections import defaultdict
from utils.utils import parse_ngrams

def analyze_dataset(datapath):
    """Analyze dataset and return results as a dictionary."""
    dataset_name = "MINDER" if "minder" in datapath.lower() else "SEAL"

    # Per-query statistics for % increases
    per_query_title_pct_increases = []
    per_query_overall_pct_increases = []
    per_query_other_pct_increases = []

    # Per-query absolute values
    per_query_pos_title = []
    per_query_neg_title = []
    per_query_pos_overall = []
    per_query_neg_overall = []
    per_query_pos_other = []
    per_query_neg_other = []

    # Word coverage statistics
    per_query_pos_word_coverage = []
    per_query_neg_word_coverage = []

    # Track queries for top/bottom analysis
    queries_data = []  # List of {query_id, query_text, title_pct_increase, ...}

    queries_processed = 0
    queries_with_both = 0  # Queries that have both positive and negative passages

    try:
        with open(datapath, "rb") as f:
            for item in ijson.items(f, "item"):
                if not item.get("ctxs"):
                    continue

                queries_processed += 1

                # Get positive passage IDs
                positive_ids = set()
                if item.get("positive_ctxs"):
                    for pctx in item["positive_ctxs"]:
                        if "passage_id" in pctx:
                            positive_ids.add(pctx["passage_id"])

                if not positive_ids:
                    continue

                # Track scores for positive and negative passages
                positive_title_scores = []
                positive_overall_scores = []
                positive_other_scores = []
                positive_word_coverages = []
                positive_title_to_other_ratios = []  # For passages with title present

                negative_title_scores = []
                negative_overall_scores = []
                negative_other_scores = []
                negative_word_coverages = []
                negative_title_to_other_ratios = []  # For passages with title present

                # Process each passage in ctxs
                for ctx in item["ctxs"]:
                    passage_id = ctx.get("passage_id")
                    ngrams = parse_ngrams(ctx.get("keys", ""))

                    if not ngrams:
                        continue

                    # Calculate scores for this passage
                    title_score = 0.0
                    pseudo_score = 0.0
                    other_score = 0.0
                    total_score = 0.0

                    # Collect ngram texts for word coverage analysis
                    ngram_texts = []

                    for text, freq, score in ngrams:
                        total_score += score
                        stripped_text = text.strip()
                        is_title_ngram = stripped_text.startswith("</s>")
                        is_pseudo_ngram = stripped_text.startswith("||")

                        is_other_ngram = (
                            bool(text)
                            and not is_title_ngram
                            and not is_pseudo_ngram
                            and (
                                text[0].isalnum()
                                or text[0] == " "
                                or text[0] == "("
                            )
                        )

                        if is_title_ngram:
                            title_score += score
                        elif is_pseudo_ngram:
                            pseudo_score += score
                        elif is_other_ngram:
                            other_score += score

                        # Clean ngram text for word coverage
                        cleaned_ngram = text.replace("</s>", "").replace("||", "").replace("@@", "").strip()
                        if cleaned_ngram:
                            ngram_texts.append(cleaned_ngram)

                    # Calculate word coverage
                    passage_text = ctx.get("text", "")
                    word_coverage = 0.0
                    if passage_text and ngram_texts:
                        # Split passage into words (alphanumeric sequences)
                        passage_words = re.findall(r'\w+', passage_text.lower())
                        passage_word_set = set(passage_words)

                        # Combine all ngram texts into one string for searching
                        combined_ngrams = " ".join(ngram_texts).lower()

                        # Calculate coverage: what % of passage words appear in ngrams
                        if passage_word_set:
                            covered_words = set()
                            for word in passage_word_set:
                                if word in combined_ngrams:
                                    covered_words.add(word)
                            word_coverage = (len(covered_words) / len(passage_word_set)) * 100

                    # Calculate title-to-other ratio (only for passages with title present)
                    title_to_other_ratio = None
                    if title_score > 0:
                        # Use ratio: title_score / (other_score + small_epsilon)
                        # Small epsilon to avoid division by zero
                        epsilon = 1e-6
                        title_to_other_ratio = title_score / (other_score + epsilon)

                    # Determine if this passage is positive or negative
                    is_positive = passage_id in positive_ids

                    # Add to appropriate lists
                    if is_positive:
                        positive_title_scores.append(title_score)
                        positive_overall_scores.append(total_score)
                        positive_other_scores.append(other_score)
                        positive_word_coverages.append(word_coverage)
                        if title_to_other_ratio is not None:
                            positive_title_to_other_ratios.append(title_to_other_ratio)
                    else:
                        negative_title_scores.append(title_score)
                        negative_overall_scores.append(total_score)
                        negative_other_scores.append(other_score)
                        negative_word_coverages.append(word_coverage)
                        if title_to_other_ratio is not None:
                            negative_title_to_other_ratios.append(title_to_other_ratio)

                # Calculate averages for this query
                if positive_title_scores and negative_title_scores:
                    queries_with_both += 1

                    avg_pos_title = np.mean(positive_title_scores)
                    avg_neg_title = np.mean(negative_title_scores)

                    avg_pos_overall = np.mean(positive_overall_scores)
                    avg_neg_overall = np.mean(negative_overall_scores)

                    avg_pos_other = np.mean(positive_other_scores)
                    avg_neg_other = np.mean(negative_other_scores)

                    avg_pos_word_cov = np.mean(positive_word_coverages) if positive_word_coverages else 0
                    avg_neg_word_cov = np.mean(negative_word_coverages) if negative_word_coverages else 0

                    # Calculate average title-to-other ratios (only for passages with title present)
                    avg_pos_title_to_other = np.mean(positive_title_to_other_ratios) if positive_title_to_other_ratios else None
                    avg_neg_title_to_other = np.mean(negative_title_to_other_ratios) if negative_title_to_other_ratios else None

                    # Calculate difference in title-to-other ratio (positive - negative)
                    title_to_other_diff = None
                    if avg_pos_title_to_other is not None and avg_neg_title_to_other is not None:
                        title_to_other_diff = avg_pos_title_to_other - avg_neg_title_to_other

                    # Store absolute values
                    per_query_pos_title.append(avg_pos_title)
                    per_query_neg_title.append(avg_neg_title)
                    per_query_pos_overall.append(avg_pos_overall)
                    per_query_neg_overall.append(avg_neg_overall)
                    per_query_pos_other.append(avg_pos_other)
                    per_query_neg_other.append(avg_neg_other)

                    per_query_pos_word_coverage.append(avg_pos_word_cov)
                    per_query_neg_word_coverage.append(avg_neg_word_cov)

                    # Calculate percentage increase: ((positive - negative) / negative) * 100
                    title_pct_increase = None
                    overall_pct_increase = None
                    other_pct_increase = None

                    if avg_neg_title > 0:
                        title_pct_increase = ((avg_pos_title - avg_neg_title) / avg_neg_title) * 100
                        per_query_title_pct_increases.append(title_pct_increase)

                    if avg_neg_overall > 0:
                        overall_pct_increase = ((avg_pos_overall - avg_neg_overall) / avg_neg_overall) * 100
                        per_query_overall_pct_increases.append(overall_pct_increase)

                    if avg_neg_other > 0:
                        other_pct_increase = ((avg_pos_other - avg_neg_other) / avg_neg_other) * 100
                        per_query_other_pct_increases.append(other_pct_increase)

                    # Store query data for top/bottom analysis
                    query_text = item.get("question", "")
                    query_id = item.get("id", queries_with_both - 1)
                    queries_data.append({
                        "query_id": query_id,
                        "query_text": query_text,
                        "title_pct_increase": title_pct_increase,
                        "overall_pct_increase": overall_pct_increase,
                        "other_pct_increase": other_pct_increase,
                        "pos_title": avg_pos_title,
                        "neg_title": avg_neg_title,
                        "pos_overall": avg_pos_overall,
                        "neg_overall": avg_neg_overall,
                        "pos_other": avg_pos_other,
                        "neg_other": avg_neg_other,
                        "pos_word_cov": avg_pos_word_cov,
                        "neg_word_cov": avg_neg_word_cov,
                        "pos_title_to_other_ratio": avg_pos_title_to_other,
                        "neg_title_to_other_ratio": avg_neg_title_to_other,
                        "title_to_other_diff": title_to_other_diff,
                    })

        # Calculate summary statistics
        results = {
            "dataset_name": dataset_name,
            "queries_processed": queries_processed,
            "queries_with_both": queries_with_both,
            "summary": {},
            "top_bottom_queries": {}
        }

        # Title Score
        if per_query_title_pct_increases and per_query_pos_title and per_query_neg_title:
            results["summary"]["title_score"] = {
                "positive_absolute": {
                    "mean": float(np.mean(per_query_pos_title)),
                    "median": float(np.median(per_query_pos_title)),
                    "std": float(np.std(per_query_pos_title))
                },
                "negative_absolute": {
                    "mean": float(np.mean(per_query_neg_title)),
                    "median": float(np.median(per_query_neg_title)),
                    "std": float(np.std(per_query_neg_title))
                },
                "pct_increase": {
                    "mean": float(np.mean(per_query_title_pct_increases)),
                    "median": float(np.median(per_query_title_pct_increases)),
                    "std": float(np.std(per_query_title_pct_increases))
                }
            }

        # Overall Score
        if per_query_overall_pct_increases and per_query_pos_overall and per_query_neg_overall:
            results["summary"]["overall_score"] = {
                "positive_absolute": {
                    "mean": float(np.mean(per_query_pos_overall)),
                    "median": float(np.median(per_query_pos_overall)),
                    "std": float(np.std(per_query_pos_overall))
                },
                "negative_absolute": {
                    "mean": float(np.mean(per_query_neg_overall)),
                    "median": float(np.median(per_query_neg_overall)),
                    "std": float(np.std(per_query_neg_overall))
                },
                "pct_increase": {
                    "mean": float(np.mean(per_query_overall_pct_increases)),
                    "median": float(np.median(per_query_overall_pct_increases)),
                    "std": float(np.std(per_query_overall_pct_increases))
                }
            }

        # Other Score
        if per_query_other_pct_increases and per_query_pos_other and per_query_neg_other:
            results["summary"]["other_score"] = {
                "positive_absolute": {
                    "mean": float(np.mean(per_query_pos_other)),
                    "median": float(np.median(per_query_pos_other)),
                    "std": float(np.std(per_query_pos_other))
                },
                "negative_absolute": {
                    "mean": float(np.mean(per_query_neg_other)),
                    "median": float(np.median(per_query_neg_other)),
                    "std": float(np.std(per_query_neg_other))
                },
                "pct_increase": {
                    "mean": float(np.mean(per_query_other_pct_increases)),
                    "median": float(np.median(per_query_other_pct_increases)),
                    "std": float(np.std(per_query_other_pct_increases))
                }
            }

        # Word Coverage
        if per_query_pos_word_coverage and per_query_neg_word_coverage:
            results["summary"]["word_coverage"] = {
                "positive": {
                    "mean": float(np.mean(per_query_pos_word_coverage)),
                    "median": float(np.median(per_query_pos_word_coverage)),
                    "std": float(np.std(per_query_pos_word_coverage))
                },
                "negative": {
                    "mean": float(np.mean(per_query_neg_word_coverage)),
                    "median": float(np.median(per_query_neg_word_coverage)),
                    "std": float(np.std(per_query_neg_word_coverage))
                }
            }

        # Top and bottom 5 queries by title % increase
        valid_queries = [q for q in queries_data if q["title_pct_increase"] is not None]
        if valid_queries:
            # Sort by title_pct_increase
            sorted_by_title = sorted(valid_queries, key=lambda x: x["title_pct_increase"])

            # Bottom 5 (smallest/most negative)
            bottom_5 = sorted_by_title[:5]
            # Top 5 (largest/most positive)
            top_5 = sorted_by_title[-5:][::-1]  # Reverse to get descending order

            results["top_bottom_queries"]["title_pct_increase"] = {
                "top_5_largest": top_5,
                "top_5_smallest": bottom_5
            }

        # Top and bottom 5 queries by title-to-other ratio difference (positive - negative)
        valid_queries_ratio = [q for q in queries_data if q["title_to_other_diff"] is not None]
        if valid_queries_ratio:
            # Sort by title_to_other_diff
            sorted_by_ratio = sorted(valid_queries_ratio, key=lambda x: x["title_to_other_diff"])

            # Bottom 5 (smallest difference - title overwhelms LESS in positive vs negative)
            bottom_5_ratio = sorted_by_ratio[:5]
            # Top 5 (largest difference - title overwhelms MORE in positive vs negative)
            top_5_ratio = sorted_by_ratio[-5:][::-1]

            results["top_bottom_queries"]["title_to_other_diff"] = {
                "top_5_largest": top_5_ratio,
                "top_5_smallest": bottom_5_ratio
            }

        return results

    except Exception as e:
        print(f"Error processing {datapath}: {e}")
        import traceback
        traceback.print_exc()
        return None

def display_results(results):
    """Display results in a formatted way."""
    if not results:
        return

    print(f"\n{'='*60}")
    print(f"Analyzing {results['dataset_name']}")
    print(f"{'='*60}")
    print(f"\nQueries processed: {results['queries_processed']}")
    print(f"Queries with both positive and negative passages: {results['queries_with_both']}")

    print(f"\n{'='*60}")
    print("COMPARISON: POSITIVE vs NEGATIVE PASSAGES")
    print(f"{'='*60}\n")

    # Title Score
    if "title_score" in results["summary"]:
        ts = results["summary"]["title_score"]
        print("="*60)
        print("TITLE SCORE")
        print("="*60)

        print("\nPositive passages (absolute):")
        print(f"  Mean:   {ts['positive_absolute']['mean']:>10.4f}")
        print(f"  Median: {ts['positive_absolute']['median']:>10.4f}")
        print(f"  Std:    {ts['positive_absolute']['std']:>10.4f}")

        print("\nNegative passages (absolute):")
        print(f"  Mean:   {ts['negative_absolute']['mean']:>10.4f}")
        print(f"  Median: {ts['negative_absolute']['median']:>10.4f}")
        print(f"  Std:    {ts['negative_absolute']['std']:>10.4f}")

        print("\n% Increase (Positive vs Negative):")
        print(f"  Mean:   {ts['pct_increase']['mean']:>10.2f}%")
        print(f"  Median: {ts['pct_increase']['median']:>10.2f}%")
        print(f"  Std:    {ts['pct_increase']['std']:>10.2f}%")

    # Overall Score
    if "overall_score" in results["summary"]:
        os = results["summary"]["overall_score"]
        print("\n" + "="*60)
        print("OVERALL NGRAM SCORE (all ngrams)")
        print("="*60)

        print("\nPositive passages (absolute):")
        print(f"  Mean:   {os['positive_absolute']['mean']:>10.4f}")
        print(f"  Median: {os['positive_absolute']['median']:>10.4f}")
        print(f"  Std:    {os['positive_absolute']['std']:>10.4f}")

        print("\nNegative passages (absolute):")
        print(f"  Mean:   {os['negative_absolute']['mean']:>10.4f}")
        print(f"  Median: {os['negative_absolute']['median']:>10.4f}")
        print(f"  Std:    {os['negative_absolute']['std']:>10.4f}")

        print("\n% Increase (Positive vs Negative):")
        print(f"  Mean:   {os['pct_increase']['mean']:>10.2f}%")
        print(f"  Median: {os['pct_increase']['median']:>10.2f}%")
        print(f"  Std:    {os['pct_increase']['std']:>10.2f}%")

    # Other Score
    if "other_score" in results["summary"]:
        ots = results["summary"]["other_score"]
        print("\n" + "="*60)
        print("'OTHER' NGRAM SCORE (excluding title and pseudoquery)")
        print("="*60)

        print("\nPositive passages (absolute):")
        print(f"  Mean:   {ots['positive_absolute']['mean']:>10.4f}")
        print(f"  Median: {ots['positive_absolute']['median']:>10.4f}")
        print(f"  Std:    {ots['positive_absolute']['std']:>10.4f}")

        print("\nNegative passages (absolute):")
        print(f"  Mean:   {ots['negative_absolute']['mean']:>10.4f}")
        print(f"  Median: {ots['negative_absolute']['median']:>10.4f}")
        print(f"  Std:    {ots['negative_absolute']['std']:>10.4f}")

        print("\n% Increase (Positive vs Negative):")
        print(f"  Mean:   {ots['pct_increase']['mean']:>10.2f}%")
        print(f"  Median: {ots['pct_increase']['median']:>10.2f}%")
        print(f"  Std:    {ots['pct_increase']['std']:>10.2f}%")

    # Word Coverage
    if "word_coverage" in results["summary"]:
        wc = results["summary"]["word_coverage"]
        print("\n" + "="*60)
        print("WORD COVERAGE (% of passage words in ngrams)")
        print("="*60)

        print("\nPositive passages:")
        print(f"  Mean:   {wc['positive']['mean']:>10.2f}%")
        print(f"  Median: {wc['positive']['median']:>10.2f}%")
        print(f"  Std:    {wc['positive']['std']:>10.2f}%")

        print("\nNegative passages:")
        print(f"  Mean:   {wc['negative']['mean']:>10.2f}%")
        print(f"  Median: {wc['negative']['median']:>10.2f}%")
        print(f"  Std:    {wc['negative']['std']:>10.2f}%")

    # Top/Bottom Queries
    if "title_pct_increase" in results["top_bottom_queries"]:
        tb = results["top_bottom_queries"]["title_pct_increase"]

        print("\n" + "="*60)
        print("TOP 5 QUERIES - LARGEST TITLE % INCREASE")
        print("="*60)
        for i, q in enumerate(tb["top_5_largest"], 1):
            print(f"\n{i}. Query ID: {q['query_id']}")
            print(f"   Title % Increase: {q['title_pct_increase']:.2f}%")
            print(f"   Query: {q['query_text'][:100]}{'...' if len(q['query_text']) > 100 else ''}")

        print("\n" + "="*60)
        print("TOP 5 QUERIES - SMALLEST TITLE % INCREASE")
        print("="*60)
        for i, q in enumerate(tb["top_5_smallest"], 1):
            print(f"\n{i}. Query ID: {q['query_id']}")
            print(f"   Title % Increase: {q['title_pct_increase']:.2f}%")
            print(f"   Query: {q['query_text'][:100]}{'...' if len(q['query_text']) > 100 else ''}")

    # Top/Bottom Queries by Title-to-Other Ratio Difference
    if "title_to_other_diff" in results["top_bottom_queries"]:
        tb_ratio = results["top_bottom_queries"]["title_to_other_diff"]

        print("\n" + "="*60)
        print("TOP 5 QUERIES - LARGEST TITLE-TO-OTHER RATIO DIFFERENCE")
        print("(Title overwhelms other ngrams MORE in positive vs negative)")
        print("="*60)
        for i, q in enumerate(tb_ratio["top_5_largest"], 1):
            print(f"\n{i}. Query ID: {q['query_id']}")
            print(f"   Ratio Diff: {q['title_to_other_diff']:.4f}")
            print(f"   Positive ratio: {q['pos_title_to_other_ratio']:.4f}")
            print(f"   Negative ratio: {q['neg_title_to_other_ratio']:.4f}")
            print(f"   Query: {q['query_text'][:100]}{'...' if len(q['query_text']) > 100 else ''}")

        print("\n" + "="*60)
        print("TOP 5 QUERIES - SMALLEST TITLE-TO-OTHER RATIO DIFFERENCE")
        print("(Title overwhelms other ngrams LESS in positive vs negative)")
        print("="*60)
        for i, q in enumerate(tb_ratio["top_5_smallest"], 1):
            print(f"\n{i}. Query ID: {q['query_id']}")
            print(f"   Ratio Diff: {q['title_to_other_diff']:.4f}")
            print(f"   Positive ratio: {q['pos_title_to_other_ratio']:.4f}")
            print(f"   Negative ratio: {q['neg_title_to_other_ratio']:.4f}")
            print(f"   Query: {q['query_text'][:100]}{'...' if len(q['query_text']) > 100 else ''}")

    print(f"\n{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description='Analyze positive vs negative passage retrieval')
    parser.add_argument('-re', '--rerun', action='store_true',
                        help='Force rerun of analysis even if cached results exist')
    args = parser.parse_args()

    # Create output directories
    os.makedirs("generated_data/seal", exist_ok=True)
    os.makedirs("generated_data/minder", exist_ok=True)

    # Analyze SEAL data
    seal_path = "data/seal_output.json"
    seal_results_path = "generated_data/seal/positive_vs_negative_results.json"

    if os.path.exists(seal_path):
        if args.rerun or not os.path.exists(seal_results_path):
            print(f"Analyzing SEAL dataset...")
            results = analyze_dataset(seal_path)
            if results:
                with open(seal_results_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {seal_results_path}")
        else:
            print(f"Loading cached SEAL results from {seal_results_path}")
            with open(seal_results_path, 'r') as f:
                results = json.load(f)

        if results:
            display_results(results)
    else:
        print(f"SEAL data not found: {seal_path}")

    # Analyze Minder data
    minder_path = "data/minder_output.json"
    minder_results_path = "generated_data/minder/positive_vs_negative_results.json"

    if os.path.exists(minder_path):
        if args.rerun or not os.path.exists(minder_results_path):
            print(f"Analyzing MINDER dataset...")
            results = analyze_dataset(minder_path)
            if results:
                with open(minder_results_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {minder_results_path}")
        else:
            print(f"Loading cached MINDER results from {minder_results_path}")
            with open(minder_results_path, 'r') as f:
                results = json.load(f)

        if results:
            display_results(results)
    else:
        print(f"\nSkipping Minder analysis: {minder_path} not found.")

if __name__ == "__main__":
    main()
