#!/bin/bash
set -e

#python scripts/answer_coverage.py
#python scripts/repetitive_tokens.py
#python scripts/single_ngram_dominance.py
#python scripts/ngram_length_bias.py
#python scripts/query_n_gram_overlap.py
#python scripts/nonspecific_highfreq_ngrams.py
python scripts/ngram_token_length_correlation.py
python scripts/title_repetition_analysis.py
python scripts/analyze_minder_pseudoqueries.py
python scripts/positive_vs_negative_analysis.py
python scripts/title_ngram_analysis.py
python scripts/generate_score_concentr_graph.py
python scripts/analyze_positive_ctxs.py
python scripts/plot_ngram_length_comparison.py
python scripts/corpus_coverage_analysis.py
