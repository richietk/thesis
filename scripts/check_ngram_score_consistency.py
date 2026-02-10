import sys
import ijson
import math
from collections import defaultdict
from utils.utils import parse_ngrams

def main(datapath="data/seal_output.json"):
    print(f"Checking n-gram score consistency in {datapath}...")
    
    # Store the first seen score for each n-gram text
    # text -> (score, frequency, example_location)
    seen_ngrams = {}
    
    mismatch_count = 0
    total_ngrams_checked = 0
    unique_ngrams_count = 0
    
    # To avoid flooding output, only print first few examples
    max_examples = 10
    examples_printed = 0
    
    try:
        with open(datapath, "rb") as f:
            for i, item in enumerate(ijson.items(f, "item")):
                ctxs = item.get("ctxs", [])
                if not ctxs:
                    continue
                
                # Iterate through all retrieved passages
                for ctx_idx, ctx in enumerate(ctxs):
                    keys_str = ctx.get("keys", "")
                    ngrams = parse_ngrams(keys_str)
                    
                    for text, freq, score in ngrams:
                        total_ngrams_checked += 1
                        
                        if text not in seen_ngrams:
                            seen_ngrams[text] = (score, freq, f"Entry {i}, Ctx {ctx_idx}")
                            unique_ngrams_count += 1
                        else:
                            prev_score, prev_freq, prev_loc = seen_ngrams[text]
                            
                            # Check for score mismatch (using tolerance for floats)
                            if not math.isclose(score, prev_score, rel_tol=1e-9, abs_tol=1e-9):
                                mismatch_count += 1
                                if examples_printed < max_examples:
                                    print(f"\nMismatch found for n-gram: '{text}'")
                                    print(f"  Location 1 ({prev_loc}): Score={prev_score}, Freq={prev_freq}")
                                    print(f"  Location 2 (Entry {i}, Ctx {ctx_idx}): Score={score}, Freq={freq}")
                                    print(f"  Diff: {abs(score - prev_score)}")
                                    examples_printed += 1

        print(f"\nAnalysis Complete.")
        print(f"Total n-grams checked: {total_ngrams_checked}")
        print(f"Unique n-grams found: {unique_ngrams_count}")
        
        if mismatch_count == 0:
            print("\nSUCCESS: No score inconsistencies found. All n-grams with the same text have consistent scores.")
        else:
            print(f"\nWARNING: Found {mismatch_count} instances where the same n-gram had different scores.")

    except FileNotFoundError:
        print(f"File not found: {datapath}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/seal_output.json"
    main(path)
