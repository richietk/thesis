import ijson
import ast
import sys
import os
import json
from scripts.utils.utils import get_dataset_name, strip_ngram_markers, calculate_retrieval_metrics

def main(datapath='data/seal_output.json'):
    script_name = "single_ngram_test"
    print(f"running {script_name}")

    try:
        dataset_name = get_dataset_name(datapath)
        output_dir = f"generated_data/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)

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

                # Calculate retrieval metrics
                retrieved_ids = [ctx['passage_id'] for ctx in entry['ctxs']]
                metrics = calculate_retrieval_metrics(retrieved_ids, positive_ids)

                records.append({
                    'dominance': dominance,
                    'success_top1': success_top1,
                    'success_top2': success_top2,
                    'success_top10': success_top10,
                    'precision_at_1': metrics['precision_at_1'],
                    'r_precision': metrics['r_precision']
                })

        # Sort by dominance
        records.sort(key=lambda x: x['dominance'])

        # Split into deciles
        num_bins = 10
        bin_size = len(records) // num_bins

        output_data = {
            "total_queries": len(records),
            "precision_at_1": sum(r['precision_at_1'] for r in records) / len(records) if records else 0.0,
            "r_precision": sum(r['r_precision'] for r in records) / len(records) if records else 0.0,
            "deciles": []
        }

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

            output_data["deciles"].append({
                "decile": i + 1,
                "dominance_min": float(d_min),
                "dominance_max": float(d_max),
                "success_top1_pct": float(top1_rate),
                "success_top2_pct": float(top2_rate),
                "success_top10_pct": float(top10_rate),
                "count": len(bin_records)
            })

        # Save JSON output
        output_json = os.path.join(output_dir, f"{script_name}_results.json")
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"success running {script_name}")

    except Exception as e:
        print(f"error: running {script_name} {e}")
        raise


if __name__ == "__main__":
    datapath = sys.argv[1] if len(sys.argv) > 1 else 'data/seal_output.json'
    main(datapath)
