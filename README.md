## Bachelor's thesis: Taxonomy and Analysis of Failure Modes in N-gram Based Generative Retrieval

Data: https://drive.google.com/drive/folders/1efondSjYtrF2LzWv2Zi0_KVe5CxRUik-?usp=sharing

Download data from the google drive link above, place it in the folder called data/

```
├── data
│   ├── minder_msmarco_output.json
│   ├── minder_nq_output.json
│   └── seal_nq_output.json
```
Running the scripts:
- run from the base directory e.g. `python scripts/[scriptname].py`
- OR run ./run_all_scripts.sh

Each script will run itself on each 3 model output data.

### Tying data in thesis to scripts

#### Section 2 — Information Retrieval

- "N-gram length is strongly correlated with frequency (ρ = −0.835, p < 0.001)."
  - `scripts/ngram_token_length_correlation.py`, output to terminal.

- Table 1
  - this table was constructed manually from inspecting the raw output data (`data/seal_nq_output.json`).

#### Section 3 — Methodology

- Positive Context Stats / Histogram
  - `scripts/analyze_positive_ctxs.py`
  - Output to `generated_data/[model_dataset]/positive_ctxs_histogram.png`, stats outputted to terminal

#### Section 5 — RQ2

- Non-discriminative DocIDs
  - N-gram corpus frequency stats
    - `scripts/nonspecific_highfreq_ngrams.py`, output in `generated_data/[dataset]/nonspecific_highfreq_ngrams_results.json` -> fields `avg_frequency_top5` and `avg_frequency_all`.
  - Table 2
    - `scripts/nonspecific_highfreq_ngrams.py`, output in ``generated_data/[dataset]/nonspecific_highfreq_ngrams_results.json`-> field `deciles_top5`, subfields `hits@1_pct`, `hits@10_pct`, `hits@100_pct`.
  - Table 3
    - `scripts/nonspecific_highfreq_ngrams.py`, output in `generated_data/[dataset]/nonspecific_highfreq_ngrams_results.json` -> field `deciles_top5`, subfields `mrr`, `recall@10`, `recall@100`, `ndcg@10`, `ndcg@100`.
  - Spearman correlations in text
    - `scripts/nonspecific_highfreq_ngrams.py`, output in`generated_data/[dataset]/nonspecific_highfreq_ngrams_results.json` -> fields `spearman_top5_vs_mrr`, `spearman_top5_vs_ndcg10`.

- Length Bias
  - Unigram proportion, ngrma length stats in text
    - `scripts/ngram_length_bias.py`, output in `generated_data/[dataset]/ngram_length_bias_results.json` -> fields `mean_unigram_frac`, `std_unigram_frac`, `mean_avg_length`.
  - Table 4
    - `scripts/ngram_length_bias.py`, output in `generated_data/[dataset]/ngram_length_bias_results.json` -> field `deciles`, subfields `hits_at_1_pct`, `hits_at_10_pct`, `hits_at_100_pct`.
  - Table 5
    - `scripts/ngram_length_bias.py`, output in `generated_data/[dataset]/ngram_length_bias_results.json` -> field `deciles`, subfields `mrr`, `recall@10`, `recall@100`, `ndcg@10`, `ndcg@100`.
  - Spearman correlations in text
    - `scripts/ngram_length_bias.py`, output in `generated_data/[dataset]/ngram_length_bias_results.json` -> fields `spearman_unigram_frac_vs_mrr`, `spearman_unigram_frac_vs_ndcg10`.
  - Chart: Unigram fraction vs Hits@1/Hits@10 on NQ
    - `scripts/plot_ngram_length_comparison.py`, output in`generated_data/shared/ngram_length_bias_comparison.png`

- Metadata Dependency
  - Stats in text
    -`scripts/title_ngram_analysis.py`, output in terminal.
  - Table 6
    - `scripts/answer_coverage.py`, output in `generated_data/[dataset]/answer_coverage_results.json`
  - Pseudoquery stats in text
    - `scripts/analyze_minder_pseudoqueries.py` output in `generated_data/[dataset]/pseudoquery_analysis.json`

- Query-to-DocID Overlap
  - Spearman correlations in text (query overlap section)
    - `scripts/query_n_gram_overlap.py`, output in `generated_data/[dataset]/query_n_gram_overlap_results.json` -> Spearman correlation fields.
  - Table 7
    - `scripts/query_n_gram_overlap.py`, output in `generated_data/[dataset]/query_n_gram_overlap_results.json`

- Overreliance on a Single Identifier
  - Stats in text and Table 8
    - `scripts/single_ngram_dominance.py`, output in `generated_data/[dataset]/single_ngram_dominance_results.json`

- Overlapping N-gram Tokens
  - Table 9 and Spearman correlations in text
    - `scripts/repetitive_tokens.py`, output in `generated_data/[dataset]/repetitive_tokens_results.json`

- Unique Articles among Passages
  - Table 10 and line chart
    - `scripts/title_repetition_analysis.py`, output in `generated_data/[dataset]/title_repetition_results.json`. Also produces the line chart.


#### Script Explanations (`scripts/`)

- `ngram_token_length_correlation.py` — Section 2 (ρ = −0.835 number, page 6). Computes Spearman correlation between n-gram token length and corpus frequency. Outputs to terminal.
- `analyze_positive_ctxs.py` — Section 3 "Methodology". Computes statistics and histogram of ground-truth positive context counts per query. Outputs results to generated_data/[model_dataset]/positive_ctxs_histogram.png
- `nonspecific_highfreq_ngrams.py` — Tables 2, 3. Correlates top-5 n-gram corpus frequency with retrieval metrics across deciles. Outputs results to generated_data/[model_dataset]/nonspecific_highfreq_ngrams_results.json
- `ngram_length_bias.py` — Tables 4, 5. Correlates unigram proportion with retrieval metrics across deciles. Outputs results to generated_data/[model_dataset]/ngram_length_bias_results.json
- `plot_ngram_length_comparison.py` — Plots unigram fraction vs. Hits@1 and Hits@10 for both models on NQ. Outputs to generated_data/shared/ngram_length_bias_comparison.png
- `title_ngram_analysis.py` — Section 5 "Metadata Dependency". Compares title vs. non-title n-gram score contributions. Outputs results to terminal.
- `positive_vs_negative_analysis.py` — Section 5 "Metadata Dependency". Computes title score differential between positive and negative passages. Outputs results to terminal.
- `answer_coverage.py` — Table 6. Checks whether the answer string appears in generated n-grams and correlates with hit rate. Reported only on SEAL NQ and MINDER NQ. Outptus to generated_data/[model_dataset]/answer_coverage_results.json
- `analyze_minder_pseudoqueries.py` — Section 5 "Metadata Dependency". Measures pseudoquery n-gram prevalence and score share (page 26/27 in thesis). Outputs to generated_data/[model_dataset]/pseudoquery_analysis.json
- `query_n_gram_overlap.py` — Table 7. Measures token overlap between query terms and generated n-grams. Outputs to generated_data/[model_dataset]/query_n_gram_overlap_results.json
- `single_ngram_dominance.py` — Table 8. Computes ratio of top n-gram score to total passage score and correlates with retrieval success. Reported only for NQ. Outputs to generated_data/[model_dataset]/single_ngram_dominance_results.json
- `generate_score_concentr_graph.py` — Generates score concentration comparison chart for SEAL and MINDER for the presentation. Outputs to generated_data/shared/ngram_score_comparison.png
- `repetitive_tokens.py` — Table 9. Computes token diversity ratio across generated n-grams and correlates with retrieval success.
- `title_repetition_analysis.py` — Table 10 + chart. Counts passages sharing the same title in top-10 results and correlates with Hits@10.
- `utils/utils.py` — Shared utility functions (e.g. tokenization, metrics, n-gram parsing) used across all scripts.