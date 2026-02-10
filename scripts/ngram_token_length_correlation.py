import sys
import ijson
from scipy.stats import spearmanr
from transformers import GPT2TokenizerFast
from utils.utils import parse_ngrams

def main(datapath="data/seal_output.json"):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    token_len_cache = {}
    lengths = []
    freqs = []

    try:
        with open(datapath, "rb") as f:
            for item in ijson.items(f, "item"):
                if not item.get("ctxs"): continue
                
                # Process n-grams from the top context
                ngrams = parse_ngrams(item["ctxs"][0].get("keys", ""))
                
                for text, freq, _ in ngrams:
                    if text not in token_len_cache:
                        token_len_cache[text] = len(tokenizer.encode(text, add_special_tokens=False))
                    
                    lengths.append(token_len_cache[text])
                    freqs.append(freq)
                    
        if not lengths:
            print("No n-grams found.")
            return

        correlation, p_value = spearmanr(lengths, freqs)
        print(f"Analyzed {len(lengths)} n-grams.")
        print(f"Spearman Correlation (Length vs Frequency): {correlation:.4f}")
        print(f"P-value: {p_value:.4g}")

    except FileNotFoundError:
        print(f"File not found: {datapath}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/seal_output.json"
    main(path)
