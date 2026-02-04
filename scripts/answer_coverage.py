import ijson
import ast
import pandas as pd
from pathlib import Path
import sys
import os

def strip_ngram_markers(ngram: str, datapath: str) -> str:
    """Strip pseudoquery markers from ngrams if using Minder data."""
    if "minder_output.json" in datapath:
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

def parse_ngrams(keys_str):
    if not keys_str:
        return []
    try:
        keys_list = ast.literal_eval(keys_str)
        return [(ngram, freq, score) for ngram, freq, score in keys_list]
    except:
        return []

def analyze_answer_coverage(datapath="data/seal_output.json"):
    """Analyze whether answer string appears in generated n-grams."""
    script_name = "answer_coverage"
    print(f"running {script_name}")

    try:
        dataset_name = get_dataset_name(datapath)
        output_dir = f"generated_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Redirect stdout to log file
        log_file = os.path.join(output_dir, f"{script_name}_log.txt")
        original_stdout = sys.stdout
        sys.stdout = open(log_file, 'w', encoding='utf-8')

        print("\n" + "="*80)
        print("ANALYSIS 4: ANSWER COVERAGE")
        print("="*80)

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

        answer_present = df[df['answer_in_ngrams']]
        answer_absent = df[~df['answer_in_ngrams']]

        print(f"\nAnalyzed {len(df)} queries")
        print(f"Answer in n-grams: {len(answer_present)} ({100*len(answer_present)/len(df):.1f}%)")
        print(f"Answer NOT in n-grams: {len(answer_absent)} ({100*len(answer_absent)/len(df):.1f}%)")
        print(f"\nSuccess Rate:")
        print(f"  When answer present: {answer_present['success'].mean():.2%}")
        print(f"  When answer absent: {answer_absent['success'].mean():.2%}")

        # Restore stdout
        sys.stdout.close()
        sys.stdout = original_stdout

        print(f"success running {script_name}")

    except Exception as e:
        # Restore stdout in case of error
        if sys.stdout != original_stdout:
            sys.stdout.close()
            sys.stdout = original_stdout
        print(f"error: running {script_name} {e}")
        raise

if __name__ == "__main__":
    datapath = sys.argv[1] if len(sys.argv) > 1 else 'data/seal_output.json'
    analyze_answer_coverage(datapath)
