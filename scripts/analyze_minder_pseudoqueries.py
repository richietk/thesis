#!/usr/bin/env python3
import ijson
import json
from pathlib import Path
from collections import defaultdict

def is_pseudoquery(key_text):
    return '@@' in key_text or key_text.strip().startswith('||')

def analyze_minder_output(filepath):
    stats = {
        'total_questions': 0,
        'total_ngrams': 0,
        'pseudoquery_ngrams': 0,
        'pseudoquery_score': 0.0,
        'regular_score': 0.0,
        'questions_with_keys': 0,
    }

    with open(filepath, 'rb') as f:
        for item in ijson.items(f, 'item'):
            stats['total_questions'] += 1

            has_keys = False
            for ctx_list_name in ['ctxs', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs']:
                if ctx_list_name in item:
                    for ctx in item[ctx_list_name]:
                        if 'keys' in ctx and ctx['keys']:
                            has_keys = True
                            for key_data in ctx['keys']:
                                if len(key_data) >= 3:
                                    key_text, key_count, key_score = key_data[0], key_data[1], float(key_data[2])
                                    stats['total_ngrams'] += 1

                                    if is_pseudoquery(key_text):
                                        stats['pseudoquery_ngrams'] += 1
                                        stats['pseudoquery_score'] += key_score
                                    else:
                                        stats['regular_score'] += key_score

            if has_keys:
                stats['questions_with_keys'] += 1

    total_score = stats['pseudoquery_score'] + stats['regular_score']
    stats['pseudoquery_ngram_pct'] = (stats['pseudoquery_ngrams'] / stats['total_ngrams'] * 100) if stats['total_ngrams'] > 0 else 0
    stats['pseudoquery_score_pct'] = (stats['pseudoquery_score'] / total_score * 100) if total_score > 0 else 0
    stats['regular_ngrams'] = stats['total_ngrams'] - stats['pseudoquery_ngrams']
    stats['total_score'] = total_score

    return stats

def main():
    print("running script")
    try:
        input_file = Path('data/minder_output.json')
        output_dir = Path('generated_data/minder')
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = analyze_minder_output(input_file)

        output_file = output_dir / 'pseudoquery_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print("script success")
    except Exception as e:
        print(f"script error {e}")
        raise

if __name__ == '__main__':
    main()
