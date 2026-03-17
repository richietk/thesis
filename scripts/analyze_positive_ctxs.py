#!/usr/bin/env python3
import ijson
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np

DATASETS = [
    'data/seal_nq_output.json',
    'data/minder_nq_output.json',
    'data/minder_msmarco_output.json',
]

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
    std_count = float(np.std(counts))
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
        'std': std_count,
        'min': min_count,
        'max': max_count,
        'distribution': dict(distribution),
        'mid_count': mid_count
    }


for input_file in DATASETS:
    basename = os.path.splitext(os.path.basename(input_file))[0]
    dataset_name = '_'.join(basename.split('_')[:2])

    results = analyze(input_file)

    output_dir = f'generated_data/{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)

    json_output_path = os.path.join(output_dir, 'positive_ctxs_analysis.json')
    with open(json_output_path, 'w') as f:
        json.dump(results, f, indent=2)

    all_counts = []
    for count, freq in results['distribution'].items():
        all_counts.extend([count] * freq)

    plt.figure(figsize=(10, 6))
    plt.hist(all_counts, bins=10, edgecolor='black')
    plt.xlabel('Number of Positive Contexts')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Positive Contexts Count ({dataset_name})')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    histogram_path = os.path.join(output_dir, 'positive_ctxs_histogram.png')
    plt.savefig(histogram_path, dpi=300)
    plt.close()
