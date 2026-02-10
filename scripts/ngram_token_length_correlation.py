import sys
import ijson
from scipy.stats import spearmanr
from transformers import GPT2TokenizerFast
from utils.utils import parse_ngrams

def main(datapath="data/seal_output.json"):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Use a dictionary to dedupe by text
    # We store (length, freq) for each unique n-gram text
    unique_ngrams = {}

    print(f"Reading and deduping n-grams from {datapath}...")

    try:
        with open(datapath, "rb") as f:
            for item in ijson.items(f, "item"):
                if not item.get("ctxs"): continue
                
                # Process n-grams from the top context
                # Assuming we analyze the first context as per previous conventions
                ngrams = parse_ngrams(item["ctxs"][0].get("keys", ""))
                
                for text, freq, _ in ngrams:
                    if text not in unique_ngrams:
                        # Calculate length only once
                        length = len(tokenizer.encode(text, add_special_tokens=False))
                        unique_ngrams[text] = (length, freq)
                    
        if not unique_ngrams:
            print("No n-grams found.")
            return

        # Extract vectors for correlation
        lengths = [val[0] for val in unique_ngrams.values()]
        freqs = [val[1] for val in unique_ngrams.values()]

        correlation, p_value = spearmanr(lengths, freqs)
        
        print(f"Analyzed {len(lengths)} unique n-grams.")
        print(f"Spearman Correlation (Length vs Frequency): {correlation:.4f}")
        print(f"P-value: {p_value:.4g}")

    except FileNotFoundError:
        print(f"File not found: {datapath}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/seal_output.json"
    main(path)
