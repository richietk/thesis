import sys
import ijson
import numpy as np
from collections import defaultdict
from transformers import GPT2TokenizerFast
from utils.utils import parse_ngrams

def main(datapath="data/seal_output.json"):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Track unique ngrams seen: {text: {'length': L, 'freq': F, 'total_score': S, 'count': C}}
    unique_ngrams = {}

    print(f"Analyzing unique ngrams and scores in {datapath}...")

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

        # Separate stats
        title_lengths = []
        title_freqs = []
        
        # Aggregators by length
        total_unique_counts_by_len = defaultdict(int)
        title_unique_counts_by_len = defaultdict(int)
        total_score_by_len = defaultdict(float)
        title_score_by_len = defaultdict(float)
        title_examples_by_len = defaultdict(list)
        
        global_total_score = sum(d['total_score'] for d in unique_ngrams.values())

        for text, data in unique_ngrams.items():
            length = data['length']
            freq = data['freq']
            score_accum = data['total_score']
            
            total_unique_counts_by_len[length] += 1
            total_score_by_len[length] += score_accum
            
            if text.strip().startswith("</s>"):
                title_lengths.append(length)
                title_freqs.append(freq)
                title_unique_counts_by_len[length] += 1
                title_score_by_len[length] += score_accum
                
                if len(title_examples_by_len[length]) < 2:
                    clean_text = text.replace("</s>", "").replace("@@", "").strip()
                    title_examples_by_len[length].append(clean_text)

        total_title_score = sum(title_score_by_len.values())
        total_title_score_pct = (total_title_score / global_total_score) * 100 if global_total_score > 0 else 0

        print(f"Found {len(unique_ngrams)} unique ngrams.")
        print(f"Global Total Score: {global_total_score:.4f}")
        print(f"Total Title Score:  {total_title_score:.4f} ({total_title_score_pct:.2f}% of global score)")

        if title_lengths:
            print(f"Found {len(title_lengths)} unique title ngrams.")
            
            print("\nTitle N-gram Token Length Stats (Unique):")
            print(f"  Mean:   {np.mean(title_lengths):.4f}")
            print(f"  Median: {np.median(title_lengths):.4f}")
            print(f"  SD:     {np.std(title_lengths):.4f}")

            print("\nTitle N-gram Frequency Stats (Unique):")
            print(f"  Mean:   {np.mean(title_freqs):.4f}")
            print(f"  Median: {np.median(title_freqs):.4f}")
            print(f"  SD:     {np.std(title_freqs):.4f}")

        print("\nUnique Title N-gram Analysis by Token Length:")
        print(f"{'Len':<4} | {'Total':<8} | {'Titles':<8} | {'% Title':<8} | {'% Score':<8} | {'Examples (First 2)'}")
        print("-" * 110)
        
        for length in sorted(total_unique_counts_by_len.keys()):
            total_unique = total_unique_counts_by_len[length]
            titles_unique = title_unique_counts_by_len[length]
            pct_title_unique = (titles_unique / total_unique) * 100 if total_unique > 0 else 0
            
            title_score_this_len = title_score_by_len[length]
            pct_score = (title_score_this_len / global_total_score) * 100 if global_total_score > 0 else 0
            
            examples = title_examples_by_len.get(length, [])
            ex_str = ", ".join([f"'{e[:30]}...'" if len(e) > 30 else f"'{e}'" for e in examples])
            
            print(f"{length:<4} | {total_unique:<8} | {titles_unique:<8} | {pct_title_unique:6.2f}% | {pct_score:6.2f}% | {ex_str}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/seal_output.json"
    main(path)
