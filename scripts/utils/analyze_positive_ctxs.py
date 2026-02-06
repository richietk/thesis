#!/usr/bin/env python3
import ijson
import json
import os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

def analyze(json_file):
    counts = []
    examples = defaultdict(list)

    with open(json_file, 'rb') as f:
        items = ijson.items(f, 'item')
        for item in items:
            pc = item.get('positive_ctxs', [])
            count = len(pc)
            counts.append(count)
            examples[count].append(item.get('question', ''))

    total = len(counts)
    avg = sum(counts) / total
    min_count = min(counts)
    max_count = max(counts)
    distribution = Counter(counts)

    mid_count = None
    for target in range(10, 0, -1):
        if target in examples:
            mid_count = target
            break

    return {
        'total': total,
        'avg': avg,
        'min': min_count,
        'max': max_count,
        'distribution': dict(distribution),
        'mid_count': mid_count
    }

# Hardcoded input file
input_file = 'data/seal_output.json'

# Analyze the file
results = analyze(input_file)

# Create output directory if it doesn't exist
output_dir = 'generated_data/shared'
os.makedirs(output_dir, exist_ok=True)

# Save results to JSON
json_output_path = os.path.join(output_dir, 'positive_ctxs_analysis.json')
with open(json_output_path, 'w') as f:
    json.dump(results, f, indent=2)

# Generate histogram with decile bins
# Flatten the distribution data to get all individual counts
all_counts = []
for count, freq in results['distribution'].items():
    all_counts.extend([count] * freq)

plt.figure(figsize=(10, 6))
plt.hist(all_counts, bins=10, edgecolor='black')
plt.xlabel('Number of Positive Contexts')
plt.ylabel('Frequency')
plt.title('Distribution of Positive Contexts Count (Deciles)')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

histogram_path = os.path.join(output_dir, 'positive_ctxs_histogram.png')
plt.savefig(histogram_path, dpi=300)
plt.close()
