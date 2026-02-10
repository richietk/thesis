import sys
import ijson
import numpy as np
from collections import defaultdict
from transformers import GPT2TokenizerFast
from utils.utils import parse_ngrams

def main(datapath="data/seal_output.json"):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Track unique ngrams seen: {text: (length, freq)}
    unique_ngrams = {}

    print(f"Analyzing unique ngrams in {datapath}...")

    try:
        with open(datapath, "rb") as f:
            for item in ijson.items(f, "item"):
                if not item.get("ctxs"): continue
                
                # Process n-grams from the top context
                ngrams = parse_ngrams(item["ctxs"][0].get("keys", ""))
                
                for text, freq, _ in ngrams:
                    if text not in unique_ngrams:
                        length = len(tokenizer.encode(text, add_special_tokens=False))
                        unique_ngrams[text] = (length, freq)
                    
        if not unique_ngrams:
            print("No ngrams found.")
            return

        # Separate stats
        title_lengths = []
        title_freqs = []
        total_counts_by_len = defaultdict(int)
        title_counts_by_len = defaultdict(int)
        
        # Store examples: {length: [title1, title2]}
        title_examples_by_len = defaultdict(list)

        for text, (length, freq) in unique_ngrams.items():
            total_counts_by_len[length] += 1
            
            if text.strip().startswith("</s>"):
                title_lengths.append(length)
                title_freqs.append(freq)
                title_counts_by_len[length] += 1
                
                # Capture up to 2 examples per length
                if len(title_examples_by_len[length]) < 2:
                    # Clean up the display string slightly for readability
                    clean_text = text.replace("</s>", "").replace("@@", "").strip()
                    title_examples_by_len[length].append(clean_text)

        print(f"Found {len(unique_ngrams)} unique ngrams.")
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

        print("\nPercentage of Unique Title N-grams by Token Length:")
        # Layout: Length | Total | Titles | % Title | Examples
        print(f"{'Len':<4} | {'Total':<8} | {'Titles':<8} | {'% Title':<8} | {'Examples (First 2)'}")
        print("-" * 100)
        
        for length in sorted(total_counts_by_len.keys()):
            total = total_counts_by_len[length]
            titles = title_counts_by_len[length]
            pct = (titles / total) * 100
            
            # Format examples
            examples = title_examples_by_len.get(length, [])
            # Truncate long examples for display
            ex_str = ", ".join([f"'{e[:500]}...'" if len(e) > 500 else f"'{e}'" for e in examples])
            
            print(f"{length:<4} | {total:<8} | {titles:<8} | {pct:6.2f}% | {ex_str}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/seal_output.json"
    main(path)
