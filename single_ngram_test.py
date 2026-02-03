import ijson
import ast

records = []

with open('data/seal_output.json', 'r', encoding='utf-8') as f:
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

        keys_list = ast.literal_eval(keys_str)
        if len(keys_list) <= 1:
            continue

        scores = [score for _, _, score in keys_list]
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
