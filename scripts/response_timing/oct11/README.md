# Response timing and frozen helpfulness (October 11)

Run from the repository root: `python scripts/response_timing/oct11/reproduce.py`.
Python standard library only; no GPU, network, model or private paths required.
Inputs: `tables/response_timing/oct11/joined_data.json.gz` (IDs, timestamps,
memberships and frozen scores; no post bodies). Outputs are verified against
`analysis.json`, including group and year-specific summaries.

This reproduces aggregation from the frozen join, not extraction from the full
archive. The recorded source hashes and original verification identify that
separate extraction. First answer is selected by timestamp then answer ID.
2,000 scored questions; 1,996 usable baseline scores; Beyer support is the
earlier397 complete cases, NOT the ongoing expanded classification. Topic timing
uses19,998 questions,17,749 observed answers; only44 overlap the scored sample.
Topic groups are not help-seeking intents. Rates by delay are conditional on
observed answers; archive censoring is unknown. Report denominators and medians/
quartiles, not only means. Year stratification is descriptive, not causal.

The exploratory time-adjusted index is mean(score * weight), NOT a sampling-
weighted mean. Weights use pooled2,000-question average ranks for ties and floors
0.25/0.5/0.75; never rank independently inside categories. The0/1/2 score is
ordinal and has known validity limitations; the product is not validated quality.
Existing collaborator topic assignments are reused, not re-created by Zi Hu.
