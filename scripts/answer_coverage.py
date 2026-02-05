import ijson
import json
import pandas as pd
from pathlib import Path
import sys
import os
from scripts.utils import strip_ngram_markers, get_dataset_name, parse_ngrams

def analyze_answer_coverage(datapath="data/seal_output.json"):
    """Analyze whether answer string appears in generated n-grams."""
    script_name = "answer_coverage"
    print(f"running {script_name}")

    try:
        dataset_name = get_dataset_name(datapath)
        output_dir = f"generated_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

        results = []

        with open(datapath, 'r', encoding='utf-8') as f:
            for entry in ijson.items(f, 'item'):
                query = entry['question']
                answers = entry.get('answers', [])

                if not answers:
                    continue

                positive_ids = {ctx['passage_id'] for ctx in entry.get('positive_ctxs', [])}

                top_ctx = entry.get('ctxs', [None])[0]
                if not top_ctx:
                    continue

                success = top_ctx['passage_id'] in positive_ids

                ngrams = parse_ngrams(top_ctx.get('keys', ''))
                if not ngrams:
                    continue

                ngram_text = ' '.join([strip_ngram_markers(ng[0], datapath).lower() for ng in ngrams])
                answer_in_ngrams = any(ans.lower() in ngram_text for ans in answers)

                results.append({
                    'query': query,
                    'answer': answers[0],
                    'success': success,
                    'answer_in_ngrams': answer_in_ngrams,
                    'num_ngrams': len(ngrams)
                })

        df = pd.DataFrame(results)

        # Check if dataframe is empty
        if len(df) == 0:
            print(f"success running {script_name}")
            return

        answer_present = df[df['answer_in_ngrams']]
        answer_absent = df[~df['answer_in_ngrams']]

        # Collect output data
        output_data = {
            "total_queries": len(df),
            "answer_in_ngrams_count": len(answer_present),
            "answer_in_ngrams_pct": float(100 * len(answer_present) / len(df)),
            "answer_not_in_ngrams_count": len(answer_absent),
            "answer_not_in_ngrams_pct": float(100 * len(answer_absent) / len(df)),
            "success_rate_answer_present": float(answer_present['success'].mean()),
            "success_rate_answer_absent": float(answer_absent['success'].mean())
        }

        # Write JSON output
        json_path = os.path.join(output_dir, f"{script_name}_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        print(f"success running {script_name}")

    except Exception as e:
        print(f"error: running {script_name} {e}")
        raise

if __name__ == "__main__":
    datapath = sys.argv[1] if len(sys.argv) > 1 else 'data/seal_output.json'
    analyze_answer_coverage(datapath)
