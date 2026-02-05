#!/usr/bin/env python3
import ijson
import sys
from collections import Counter, defaultdict

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
        'distribution': distribution,
        'examples': examples,
        'mid_count': mid_count
    }

file1 = sys.argv[1]
file2 = sys.argv[2]

r1 = analyze(file1)
r2 = analyze(file2)

print(f"Total: {r1['total']} {r2['total']}")
print(f"Average: {r1['avg']:.2f} {r2['avg']:.2f}")
print(f"Minimum: {r1['min']} {r2['min']}")
print(f"Maximum: {r1['max']} {r2['max']}")

print()
all_counts = sorted(set(r1['distribution'].keys()) | set(r2['distribution'].keys()))
for count in all_counts:
    print(f"{count} {r1['distribution'].get(count, 0)} {r2['distribution'].get(count, 0)}")

print()
print(f"Min example: {r1['examples'][r1['min']][0]}")
print(f"Min example: {r2['examples'][r2['min']][0]}")

if r1['mid_count']:
    print(f"Mid example ({r1['mid_count']}): {r1['examples'][r1['mid_count']][0]}")
if r2['mid_count']:
    print(f"Mid example ({r2['mid_count']}): {r2['examples'][r2['mid_count']][0]}")

print(f"Max example: {r1['examples'][r1['max']][0]}")
print(f"Max example: {r2['examples'][r2['max']][0]}")
