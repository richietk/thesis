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

    try:
        with open(datapath, "rb") as f:
            for item in ijson.items(f, "item"):
                if not item.get("ctxs"): continue
                
                # Process n-grams from the top context
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
                    
        if not unique_ngrams:
            print("No ngrams found.")
            return

        # --- Statistics Containers ---
        title_lengths = []
        title_freqs = []
        
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
                title_unique_counts_by_len[length] += 1
                title_score_by_len[length] += score_accum
                
                if len(title_examples_by_len[length]) < 2:
                    clean_text = text.replace("</s>", "").replace("@@", "").strip()
                    title_examples_by_len[length].append(clean_text)
            
            # --- Logic for Pseudoqueries ---
            elif is_pseudo:
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
        print(f"(Sum of tracked categories: {sum_pct:.2f}%) - Note: 'Other' is strictly alphanumeric start.")
        
        if uncategorized_examples:
            print(f"\nRemaining {100-sum_pct:.2f}% ({pct_uncategorized_global:.2f}% score share) belongs to Uncategorized n-grams.")
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
            t_ex_str = ", ".join([f"'{e[:20]}...'" if len(e) > 20 else f"'{e}'" for e in t_ex])
            
            if dataset_name == "MINDER":
                # Pseudoquery Stats
                pseudo_unique = pseudo_unique_counts_by_len[length]
                pct_pseudo_unique = (pseudo_unique / total_unique) * 100 if total_unique > 0 else 0
                pseudo_score_this_len = pseudo_score_by_len[length]
                pct_score_pseudo = (pseudo_score_this_len / global_total_score) * 100 if global_total_score > 0 else 0
                
                # Pseudo Examples
                p_ex = pseudo_examples_by_len.get(length, [])
                p_ex_str = ", ".join([f"'{e[:20]}...'" if len(e) > 20 else f"'{e}'" for e in p_ex])
                
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
