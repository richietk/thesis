import ijson
import ast
import sys
import os
from utils import get_dataset_name, strip_ngram_markers

def main(datapath='data/seal_output.json'):
    script_name = "single_ngram_test"
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
            # ijson.items parses each element of the top-level array incrementally
            for entry in ijson.items(f, 'item'):
                # Ground truth
                positive_ids = set()
                if 'positive_ctxs' in entry and entry['positive_ctxs']:
                    positive_ids = {ctx['passage_id'] for ctx in entry['positive_ctxs']}

                # === top 1 success ===
                top_ctx = entry['ctxs'][0]
                success_top1 = top_ctx['passage_id'] in positive_ids

                # === top 2 success ===
                top2_ctxs = entry['ctxs'][:2]
                success_top2 = any(ctx['passage_id'] in positive_ids for ctx in top2_ctxs)

                # === top 10 success ===
                top10_ctxs = entry['ctxs'][:10]
                success_top10 = any(ctx['passage_id'] in positive_ids for ctx in top10_ctxs)

                # Only compute dominance for top 1 passage
                keys_str = top_ctx.get('keys', '')
                if not keys_str:
                    continue

                # Handle both string and list formats, convert Decimal to float
                try:
                    if isinstance(keys_str, list):
                        keys_list = keys_str
                    else:
                        keys_list = ast.literal_eval(keys_str)
                except:
                    continue

                if len(keys_list) <= 1:
                    continue

                scores = [float(score) for _, _, score in keys_list]
                total = sum(scores)
                if total <= 0:
                    continue

                dominance = max(scores) / total

                records.append({
                    'dominance': dominance,
                    'success_top1': success_top1,
                    'success_top2': success_top2,
                    'success_top10': success_top10
                })

        # Sort by dominance
        records.sort(key=lambda x: x['dominance'])

        # Split into deciles
        num_bins = 10
        bin_size = len(records) // num_bins

        print("Dominance range | Top-1 | Top-2 | Top-10 | Count")
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

            d_min = bin_records[0]['dominance']
            d_max = bin_records[-1]['dominance']

            print(f"{d_min:.2f}–{d_max:.2f}      | {top1_rate:6.2f}% | {top2_rate:6.2f}% | {top10_rate:6.2f}% | {len(bin_records)}")

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
