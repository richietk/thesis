import sys
import ijson
import numpy as np
import os
from collections import defaultdict
from transformers import GPT2TokenizerFast
from utils.utils import parse_ngrams

def analyze_dataset(datapath, tokenizer):
    dataset_name = "MINDER" if "minder" in datapath.lower() else "SEAL"
    print(f"\n{'='*20}\nAnalyzing {dataset_name}: {datapath}\n{'='*20}")

    # Track unique ngrams seen: {text: {'length': L, 'freq': F, 'total_score': S, 'count': C}}
    unique_ngrams = {}

    # Per-passage analysis: track passages with title vs non-title ngrams
    passages_with_dual_ngrams = 0
    passages_with_dual_ngrams_gt = 0
    title_higher_counts = []  # Per passage: count of title ngrams ranked higher
    title_lower_counts = []   # Per passage: count of title ngrams ranked lower
    title_higher_counts_gt = []
    title_lower_counts_gt = []
    title_score_pct_diffs = []  # Per passage: average % difference in scores
    title_score_pct_diffs_gt = []  # Same for ground truth passages

    # Track scores for successfully retrieved (ground truth) passages only
    gt_title_score = 0.0
    gt_pseudo_score = 0.0
    gt_other_score = 0.0
    gt_uncategorized_score = 0.0

    # Track scores for top-10 retrieved passages
    top10_title_score = 0.0
    top10_pseudo_score = 0.0
    top10_other_score = 0.0
    top10_uncategorized_score = 0.0

    try:
        with open(datapath, "rb") as f:
            for item in ijson.items(f, "item"):
                if not item.get("ctxs"): continue

                # Get positive passage IDs for ground truth filtering
                positive_ids = set()
                if item.get("positive_ctxs"):
                    for pctx in item["positive_ctxs"]:
                        if "passage_id" in pctx:
                            positive_ids.add(pctx["passage_id"])

                # Process each passage in ctxs for per-passage analysis
                for ctx in item["ctxs"]:
                    passage_id = ctx.get("passage_id")
                    is_ground_truth = passage_id in positive_ids
                    ngrams = parse_ngrams(ctx.get("keys", ""))

                    if not ngrams:
                        continue

                    # Build a map of ngrams by their cleaned text (without title markers)
                    # Map: cleaned_text -> [(original_text, score, is_title), ...]
                    ngram_map = defaultdict(list)
                    for i, (text, freq, score) in enumerate(ngrams):
                        stripped = text.strip()
                        is_title = stripped.startswith("</s>")
                        # Clean the text to get the actual words
                        cleaned = text.replace("</s>", "").replace("@@", "").strip()
                        ngram_map[cleaned].append((text, score, is_title, i))

                    # Find cases where same word appears as both title and non-title
                    has_dual = False
                    title_higher_this_passage = 0
                    title_lower_this_passage = 0
                    score_pct_diffs_this_passage = []

                    for _, variants in ngram_map.items():
                        if len(variants) < 2:
                            continue

                        # Check if we have both title and non-title versions
                        title_variants = [v for v in variants if v[2]]  # is_title = True
                        non_title_variants = [v for v in variants if not v[2]]  # is_title = False

                        if title_variants and non_title_variants:
                            has_dual = True
                            # Compare scores: higher score means ranked higher (better)
                            for _, t_score, _, _ in title_variants:
                                for _, nt_score, _, _ in non_title_variants:
                                    if t_score > nt_score:
                                        title_higher_this_passage += 1
                                    elif t_score < nt_score:
                                        title_lower_this_passage += 1

                                    # Calculate percentage difference: (title - normal) / normal * 100
                                    if nt_score != 0:
                                        pct_diff = ((t_score - nt_score) / nt_score) * 100
                                        score_pct_diffs_this_passage.append(pct_diff)

                    if has_dual:
                        passages_with_dual_ngrams += 1
                        total_comparisons = title_higher_this_passage + title_lower_this_passage
                        if total_comparisons > 0:
                            title_higher_counts.append(title_higher_this_passage)
                            title_lower_counts.append(title_lower_this_passage)

                        # Calculate average percentage difference for this passage
                        if score_pct_diffs_this_passage:
                            avg_pct_diff = np.mean(score_pct_diffs_this_passage)
                            title_score_pct_diffs.append(avg_pct_diff)

                        if is_ground_truth:
                            passages_with_dual_ngrams_gt += 1
                            if total_comparisons > 0:
                                title_higher_counts_gt.append(title_higher_this_passage)
                                title_lower_counts_gt.append(title_lower_this_passage)

                            if score_pct_diffs_this_passage:
                                avg_pct_diff_gt = np.mean(score_pct_diffs_this_passage)
                                title_score_pct_diffs_gt.append(avg_pct_diff_gt)

                    # If this is a ground truth passage, accumulate scores by category
                    if is_ground_truth:
                        for text, freq, score in ngrams:
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
                                gt_title_score += score
                            elif is_pseudo_ngram:
                                gt_pseudo_score += score
                            elif is_other_ngram:
                                gt_other_score += score
                            else:
                                gt_uncategorized_score += score

                # Process n-grams from the top context for unique ngram tracking
                ngrams = parse_ngrams(item["ctxs"][0].get("keys", ""))

                for text, freq, score in ngrams:
                    if text not in unique_ngrams:
                        length = len(tokenizer.encode(text, add_special_tokens=False))
                        unique_ngrams[text] = {
                            'length': length,
                            'freq': freq,
                            'total_score': 0.0,
                            'count': 0
                        }

                    # Accumulate score for this unique ngram from all occurrences
                    unique_ngrams[text]['total_score'] += score
                    unique_ngrams[text]['count'] += 1

                # Process top-10 passages for score categorization
                for ctx in item["ctxs"][:10]:
                    top10_ngrams = parse_ngrams(ctx.get("keys", ""))

                    for text, freq, score in top10_ngrams:
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
                            top10_title_score += score
                        elif is_pseudo_ngram:
                            top10_pseudo_score += score
                        elif is_other_ngram:
                            top10_other_score += score
                        else:
                            top10_uncategorized_score += score
                    
        if not unique_ngrams:
            print("No ngrams found.")
            return

        # --- Statistics Containers ---
        title_lengths = []
        title_freqs = []
        title_scores = []
        pseudo_scores = []

        # Aggregators by length
        total_unique_counts_by_len = defaultdict(int)
        
        title_unique_counts_by_len = defaultdict(int)
        title_score_by_len = defaultdict(float)
        title_examples_by_len = defaultdict(list)
        
        pseudo_unique_counts_by_len = defaultdict(int)
        pseudo_score_by_len = defaultdict(float)
        pseudo_examples_by_len = defaultdict(list)
        
        global_total_score = sum(d['total_score'] for d in unique_ngrams.values())
        
        # Explicit accumulator for 'Other' category
        total_other_score = 0.0
        
        # Track uncategorized examples
        uncategorized_examples = []
        uncategorized_count = 0
        total_uncategorized_score = 0.0

        # --- Iterate through unique ngrams to classify and aggregate ---
        for text, data in unique_ngrams.items():
            length = data['length']
            freq = data['freq']
            score_accum = data['total_score']
            
            total_unique_counts_by_len[length] += 1
            
            stripped_text = text.strip()
            
            # Identify Categories
            is_title = stripped_text.startswith("</s>")
            is_pseudo = stripped_text.startswith("||")
            
            is_other = False
            is_other = (
                    bool(text)
                    and not is_title
                    and not is_pseudo
                    and (
                        text[0].isalnum()
                        or text[0] == " "
                        or text[0] == "("
                    )
                )
            
            # --- Logic for Titles ---
            if is_title:
                title_lengths.append(length)
                title_freqs.append(freq)
                title_scores.append(score_accum)
                title_unique_counts_by_len[length] += 1
                title_score_by_len[length] += score_accum

                if len(title_examples_by_len[length]) < 2:
                    clean_text = text.replace("</s>", "").replace("@@", "").strip()
                    title_examples_by_len[length].append(clean_text)

            # --- Logic for Pseudoqueries ---
            elif is_pseudo:
                pseudo_scores.append(score_accum)
                pseudo_unique_counts_by_len[length] += 1
                pseudo_score_by_len[length] += score_accum

                if len(pseudo_examples_by_len[length]) < 2:
                    clean_text = text.replace("||", "").replace("@@", "").strip()
                    pseudo_examples_by_len[length].append(clean_text)

            # --- Logic for Other ---
            elif is_other:
                total_other_score += score_accum
            
            # --- Uncategorized ---
            else:
                uncategorized_count += 1
                total_uncategorized_score += score_accum
                if len(uncategorized_examples) < 3:
                    uncategorized_examples.append(text)

        # --- Calculate Global Percentages ---
        total_title_score = sum(title_score_by_len.values())
        total_pseudo_score = sum(pseudo_score_by_len.values())
        
        pct_title_global = (total_title_score / global_total_score) * 100 if global_total_score > 0 else 0
        pct_pseudo_global = (total_pseudo_score / global_total_score) * 100 if global_total_score > 0 else 0
        pct_other_global = (total_other_score / global_total_score) * 100 if global_total_score > 0 else 0
        pct_uncategorized_global = (total_uncategorized_score / global_total_score) * 100 if global_total_score > 0 else 0
        
        print(f"Found {len(unique_ngrams)} unique ngrams.")
        print(f"Global Total Score:       {global_total_score:.4f}")
        print(f"Total Title Score:        {total_title_score:.4f} ({pct_title_global:.2f}% of global)")
        print(f"Total Pseudoquery Score:  {total_pseudo_score:.4f} ({pct_pseudo_global:.2f}% of global)")
        print(f"Total 'Other' Score:      {total_other_score:.4f} ({pct_other_global:.2f}% of global)")

        sum_pct = pct_title_global + pct_pseudo_global + pct_other_global

        # --- Ground Truth (Successfully Retrieved) Passages Score Analysis ---
        gt_total_score = gt_title_score + gt_pseudo_score + gt_other_score + gt_uncategorized_score
        if gt_total_score > 0:
            pct_gt_title = (gt_title_score / gt_total_score) * 100
            pct_gt_pseudo = (gt_pseudo_score / gt_total_score) * 100
            pct_gt_other = (gt_other_score / gt_total_score) * 100
            pct_gt_uncategorized = (gt_uncategorized_score / gt_total_score) * 100

            print(f"\n--- Successfully Retrieved (Ground Truth) Passages Only ---")
            print(f"GT Total Score:           {gt_total_score:.4f}")
            print(f"GT Title Score:           {gt_title_score:.4f} ({pct_gt_title:.2f}% of GT total)")
            print(f"GT Pseudoquery Score:     {gt_pseudo_score:.4f} ({pct_gt_pseudo:.2f}% of GT total)")
            print(f"GT 'Other' Score:         {gt_other_score:.4f} ({pct_gt_other:.2f}% of GT total)")

            if pct_gt_uncategorized >= 0.01:
                print(f"GT Uncategorized Score:   {gt_uncategorized_score:.4f} ({pct_gt_uncategorized:.2f}% of GT total)")

        # --- Top-10 Passages Score Analysis ---
        top10_total_score = top10_title_score + top10_pseudo_score + top10_other_score + top10_uncategorized_score
        if top10_total_score > 0:
            pct_top10_title = (top10_title_score / top10_total_score) * 100
            pct_top10_pseudo = (top10_pseudo_score / top10_total_score) * 100
            pct_top10_other = (top10_other_score / top10_total_score) * 100
            pct_top10_uncategorized = (top10_uncategorized_score / top10_total_score) * 100

            print(f"\n--- Top-10 Retrieved Passages ---")
            print(f"Top-10 Total Score:       {top10_total_score:.4f}")
            print(f"Top-10 Title Score:       {top10_title_score:.4f} ({pct_top10_title:.2f}% of Top-10 total)")
            print(f"Top-10 Pseudoquery Score: {top10_pseudo_score:.4f} ({pct_top10_pseudo:.2f}% of Top-10 total)")
            print(f"Top-10 'Other' Score:     {top10_other_score:.4f} ({pct_top10_other:.2f}% of Top-10 total)")

            if pct_top10_uncategorized >= 0.01:
                print(f"Top-10 Uncategorized Score: {top10_uncategorized_score:.4f} ({pct_top10_uncategorized:.2f}% of Top-10 total)")

        if pct_uncategorized_global >= 0.01:
            print(f"\nRemaining {100-sum_pct:.2f}% ({pct_uncategorized_global:.2f}% score share) belongs to Uncategorized n-grams.")
            if uncategorized_examples:
                print(f"Examples of Uncategorized n-grams (neither Title, Pseudo, nor Alphanumeric start):")
                for i, ex in enumerate(uncategorized_examples, 1):
                    print(f"  {i}. '{ex}'")

        if title_lengths:
            print(f"\nFound {len(title_lengths)} unique title ngrams.")

            print("\nTitle N-gram Token Length Stats (Unique):")
            print(f"  Mean:   {np.mean(title_lengths):.4f}")
            print(f"  Median: {np.median(title_lengths):.4f}")
            print(f"  SD:     {np.std(title_lengths):.4f}")

            print("\nTitle N-gram Frequency Stats (Unique):")
            print(f"  Mean:   {np.mean(title_freqs):.4f}")
            print(f"  Median: {np.median(title_freqs):.4f}")
            print(f"  SD:     {np.std(title_freqs):.4f}")

            print("\nTitle N-gram Score Stats (Unique):")
            print(f"  Mean:   {np.mean(title_scores):.4f}")
            print(f"  Median: {np.median(title_scores):.4f}")
            print(f"  SD:     {np.std(title_scores):.4f}")

        if pseudo_scores:
            print(f"\nFound {len(pseudo_scores)} unique pseudoquery ngrams.")

            print("\nPseudoquery N-gram Score Stats (Unique):")
            print(f"  Mean:   {np.mean(pseudo_scores):.4f}")
            print(f"  Median: {np.median(pseudo_scores):.4f}")
            print(f"  SD:     {np.std(pseudo_scores):.4f}")

        # --- Per-Passage Analysis Report ---
        print("\n" + "="*60)
        print("PER-PASSAGE ANALYSIS: Title vs Non-Title N-gram Ranking")
        print("="*60)

        print(f"\nTotal passages with dual ngrams (same word as title and non-title): {passages_with_dual_ngrams}")

        if title_higher_counts and title_lower_counts:
            # Calculate per-passage percentages
            passage_percentages = []
            for higher, lower in zip(title_higher_counts, title_lower_counts):
                total = higher + lower
                if total > 0:
                    pct_higher = (higher / total) * 100
                    passage_percentages.append(pct_higher)

            avg_pct_title_higher = np.mean(passage_percentages) if passage_percentages else 0
            avg_pct_title_lower = 100 - avg_pct_title_higher

            print(f"\nAll Passages:")
            print(f"  Average % of time title ngram ranked HIGHER: {avg_pct_title_higher:.2f}%")
            print(f"  Average % of time title ngram ranked LOWER:  {avg_pct_title_lower:.2f}%")

            # Report average score percentage difference
            if title_score_pct_diffs:
                avg_score_pct_diff = np.mean(title_score_pct_diffs)
                print(f"  Average score % difference (title vs normal): {avg_score_pct_diff:.2f}%")
        else:
            print("\nAll Passages: No dual ngrams found for comparison.")

        print(f"\nGround truth passages with dual ngrams: {passages_with_dual_ngrams_gt}")

        if title_higher_counts_gt and title_lower_counts_gt:
            # Calculate per-passage percentages for ground truth
            passage_percentages_gt = []
            for higher, lower in zip(title_higher_counts_gt, title_lower_counts_gt):
                total = higher + lower
                if total > 0:
                    pct_higher = (higher / total) * 100
                    passage_percentages_gt.append(pct_higher)

            avg_pct_title_higher_gt = np.mean(passage_percentages_gt) if passage_percentages_gt else 0
            avg_pct_title_lower_gt = 100 - avg_pct_title_higher_gt

            print(f"\nGround Truth Passages:")
            print(f"  Average % of time title ngram ranked HIGHER: {avg_pct_title_higher_gt:.2f}%")
            print(f"  Average % of time title ngram ranked LOWER:  {avg_pct_title_lower_gt:.2f}%")

            # Report average score percentage difference for ground truth
            if title_score_pct_diffs_gt:
                avg_score_pct_diff_gt = np.mean(title_score_pct_diffs_gt)
                print(f"  Average score % difference (title vs normal): {avg_score_pct_diff_gt:.2f}%")
        else:
            print("\nGround Truth Passages: No dual ngrams found for comparison.")
        print("="*60)

        # --- Table Printout ---
        print("\nUnique N-gram Analysis by Token Length:")
        
        if dataset_name == "MINDER":
             # Header for Minder
            header = (
                f"{'Len':<4} | {'Total':<8} | "
                f"{'Titles':<6} | {'%Title':<6} | {'%Sc(T)':<6} | "
                f"{'Pseudos':<7} | {'%Pseudo':<7} | {'%Sc(P)':<6} | "
                f"{'Examples (Title / Pseudo)'}"
            )
            print(header)
            print("-" * len(header))
        else:
            # Header for SEAL
            header = (
                f"{'Len':<4} | {'Total':<8} | "
                f"{'Titles':<6} | {'%Title':<6} | {'%Sc(T)':<6} | "
                f"{'Examples (First 2)'}"
            )
            print(header)
            print("-" * len(header))
        
        for length in sorted(total_unique_counts_by_len.keys()):
            total_unique = total_unique_counts_by_len[length]
            
            # Title Stats
            titles_unique = title_unique_counts_by_len[length]
            pct_title_unique = (titles_unique / total_unique) * 100 if total_unique > 0 else 0
            title_score_this_len = title_score_by_len[length]
            pct_score_title = (title_score_this_len / global_total_score) * 100 if global_total_score > 0 else 0
            
            # Title Examples
            t_ex = title_examples_by_len.get(length, [])
            t_ex_str = ", ".join([f"'{e[:12]}'" for e in t_ex])
            
            if dataset_name == "MINDER":
                # Pseudoquery Stats
                pseudo_unique = pseudo_unique_counts_by_len[length]
                pct_pseudo_unique = (pseudo_unique / total_unique) * 100 if total_unique > 0 else 0
                pseudo_score_this_len = pseudo_score_by_len[length]
                pct_score_pseudo = (pseudo_score_this_len / global_total_score) * 100 if global_total_score > 0 else 0
                
                # Pseudo Examples
                p_ex = pseudo_examples_by_len.get(length, [])
                p_ex_str = ", ".join([f"'{e[:12]}'" for e in p_ex])
                
                # Combined Example String
                examples_display = f"T:[{t_ex_str}] P:[{p_ex_str}]"
                
                print(
                    f"{length:<4} | {total_unique:<8} | "
                    f"{titles_unique:<6} | {pct_title_unique:6.2f}% | {pct_score_title:6.2f}% | "
                    f"{pseudo_unique:<7} | {pct_pseudo_unique:7.2f}% | {pct_score_pseudo:6.2f}% | "
                    f"{examples_display}"
                )
            else:
                # SEAL Print
                print(
                    f"{length:<4} | {total_unique:<8} | "
                    f"{titles_unique:<6} | {pct_title_unique:6.2f}% | {pct_score_title:6.2f}% | "
                    f"{t_ex_str}"
                )

    except Exception as e:
        print(f"Error processing {datapath}: {e}")

def main():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Analyze SEAL data
    analyze_dataset("data/seal_output.json", tokenizer)
    
    # Analyze Minder data if it exists
    minder_path = "data/minder_output.json"
    if os.path.exists(minder_path):
        analyze_dataset(minder_path, tokenizer)
    else:
        print(f"\nSkipping Minder analysis: {minder_path} not found.")

if __name__ == "__main__":
    main()
