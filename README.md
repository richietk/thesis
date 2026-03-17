Bachelor's thesis: Taxonomy and Analysis of Failure Modes in N-gram Based Generative Retrieval

Data: https://drive.google.com/drive/folders/1efondSjYtrF2LzWv2Zi0_KVe5CxRUik-?usp=sharing

## Scripts (`scripts/`)

- `ngram_token_length_correlation.py` — Section 2 (ρ = −0.835 number, page 6). Computes Spearman correlation between n-gram token length and corpus frequency. Outputs to terminal.
- `analyze_positive_ctxs.py` — Section 3 "Methodology". Computes statistics and histogram of ground-truth positive context counts per query. Outputs results to generated_data/[model_dataset]/positive_ctxs_histogram.png
- `nonspecific_highfreq_ngrams.py` — Tables 2, 3. Correlates top-5 n-gram corpus frequency with retrieval metrics across deciles. Outputs results to generated_data/[model_dataset]/nonspecific_highfreq_ngrams_results.json
- `ngram_length_bias.py` — Tables 4, 5. Correlates unigram proportion with retrieval metrics across deciles. Outputs results to generated_data/[model_dataset]/ngram_length_bias_results.json
- `title_ngram_analysis.py` — Section 5 "Metadata Dependency". Compares title vs. non-title n-gram score contributions. Outputs results to terminal.
- `positive_vs_negative_analysis.py` — Section 5 "Metadata Dependency". Computes title score differential between positive and negative passages. Outputs results to terminal.
- `answer_coverage.py` — Table 6. Checks whether the answer string appears in generated n-grams and correlates with hit rate. Reported only on SEAL NQ and MINDER NQ. Outptus to generated_data/[model_dataset]/answer_coverage_results.json
- `analyze_minder_pseudoqueries.py` — Section 5 "Metadata Dependency". Measures pseudoquery n-gram prevalence and score share (page 26/27 in thesis). Outputs to generated_data/[model_dataset]/pseudoquery_analysis.json
- `query_n_gram_overlap.py` — Table 7. Measures token overlap between query terms and generated n-grams. Outputs to generated_data/[model_dataset]/query_n_gram_overlap_results.json
- `single_ngram_dominance.py` — Table 8. Computes ratio of top n-gram score to total passage score and correlates with retrieval success. Reported only for NQ. Outputs to generated_data/[model_dataset]/single_ngram_dominance_results.json
- `generate_score_concentr_graph.py` — Generates score concentration comparison chart for SEAL and MINDER for the presentation. Outputs to generated_data/shared/ngram_score_comparison.png
- `repetitive_tokens.py` — Table 9. Computes token diversity ratio across generated n-grams and correlates with retrieval success.
- `title_repetition_analysis.py` — Table 10 + chart. Counts passages sharing the same title in top-10 results and correlates with Hits@10.
- `utils/utils.py` — Shared utility functions (tokenization, metrics, n-gram parsing) used across all scripts.