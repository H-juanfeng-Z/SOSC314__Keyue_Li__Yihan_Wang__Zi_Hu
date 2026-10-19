# Authoritative scores and historical exports

For new analyses use [reusable_evaluation/unified_v1](reusable_evaluation/unified_v1/README.txt): 5,000 unique first-answer coarse records, the nested 1,000-question fine sample and its matched coarse subset. Run `python tables/reusable_evaluation/unified_v1/verify.py` from the root. Invalid/null/skipped observations remain explicit; they are not zero scores.

The archived `fig_intent_helpfulness_pooled_行级数据.xlsx` has 4,977 question rows. Six coarse scores in that historical alternative-policy export differ from the frozen unified baseline. They are retained as historical output, not silently relabeled. The submitted report uses the unified baseline, as documented in `final_report/README.md`.

| question_id | Historical Excel score | Unified baseline score |
| ---: | ---: | ---: |
| 3115374 | 1 | 2 |
| 19433044 | 0 | 1 |
| 20882174 | 0 | 1 |
| 36530649 | 1 | 2 |
| 28511386 | 1 | 2 |
| 29474916 | 2 | 1 |

The companion fine-score workbook `fig_intent_candidate010_oct13_行级数据.xlsx` has 991 unique question rows; its scores/answer IDs agree with the unified release, but it is not the full 1,000-question fine sample. Classification coverage and valid-score support need separate denominators. Do not append this workbook to the JSON release or count it as additional questions.

Different dated folders represent different experiments or sampling batches. Join question groups using `question_id`, and compare ratings using `question_id` plus `answer_id`. Preserve prompt versions, invalids, skipped inputs and missing classification. Consult each release manifest and README instead of inferring valid counts from filenames.
