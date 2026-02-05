import ijson
import ast
import sys
import os
from utils import get_dataset_name, strip_ngram_markers

def main(datapath='data/seal_output.json'):
    script_name = "ngram_length_test"
    print(f"running {script_name}")

    original_stdout = sys.stdout
    try:
        dataset_name = get_dataset_name(datapath)
        output_dir = f"generated_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Redirect stdout to log file
        log_file = os.path.join(output_dir, f"{script_name}_log.txt")
        sys.stdout = open(log_file, 'w', encoding='utf-8')

        records = []

        with open(datapath, 'r', encoding='utf-8') as f:
            # Incrementally parse each element of the top-level JSON array
            for entry in ijson.items(f, 'item'):
                # Ground truth
                positive_ids = set()
                if 'positive_ctxs' in entry and entry['positive_ctxs']:
                    positive_ids = {ctx['passage_id'] for ctx in entry['positive_ctxs']}

                # Top-1 success
                top1_ctx = entry['ctxs'][0]
                success_top1 = top1_ctx['passage_id'] in positive_ids

                # Top-2 success
                top2_ctxs = entry['ctxs'][:2]
                success_top2 = any(ctx['passage_id'] in positive_ids for ctx in top2_ctxs)

                # Top-10 success
                top10_ctxs = entry['ctxs'][:10]
                success_top10 = any(ctx['passage_id'] in positive_ids for ctx in top10_ctxs)

                # Compute unigram fraction only for top-1 passage
                keys_str = top1_ctx.get('keys', '')
                if not keys_str:
                    continue

                # Handle both string and list formats
                try:
                    if isinstance(keys_str, list):
                        keys_list = keys_str
                    else:
                        keys_list = ast.literal_eval(keys_str)
                except:
                    continue

                if len(keys_list) == 0:
                    continue

                lengths = [len(strip_ngram_markers(ngram, datapath).strip().split()) for ngram, _, _ in keys_list]
                unigram_frac = sum(1 for l in lengths if l == 1) / len(lengths)

                records.append({
                    'unigram_frac': unigram_frac,
                    'success_top1': success_top1,
                    'success_top2': success_top2,
                    'success_top10': success_top10
                })

        # Sort by unigram fraction
        records.sort(key=lambda x: x['unigram_frac'])

        # Split into deciles
        num_bins = 10
        bin_size = len(records) // num_bins

        print("Unigram fraction | Top-1 | Top-2 | Top-10 | Count")
        print("-----------------------------------------------")

        for i in range(num_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < num_bins - 1 else len(records)
            bin_records = records[start:end]

            if not bin_records:
                continue

            top1_rate = 100 * sum(r['success_top1'] for r in bin_records) / len(bin_records)
            top2_rate = 100 * sum(r['success_top2'] for r in bin_records) / len(bin_records)
            top10_rate = 100 * sum(r['success_top10'] for r in bin_records) / len(bin_records)

            u_min = bin_records[0]['unigram_frac']
            u_max = bin_records[-1]['unigram_frac']

            print(f"{u_min:.2f}–{u_max:.2f}       | {top1_rate:6.2f}% | {top2_rate:6.2f}% | {top10_rate:6.2f}% | {len(bin_records)}")

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
    main(datapath)
