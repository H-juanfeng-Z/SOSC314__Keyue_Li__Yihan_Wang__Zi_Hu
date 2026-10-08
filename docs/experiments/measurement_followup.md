# Measurement sensitivity and answer trajectories

This follow-up examines whether our measurements support the project's main question: how help-seeking intent is associated with first-answer helpfulness. Multi-answer trajectories are a secondary sensitivity analysis, not a replacement primary outcome. All numerical findings below use reused development data. The original 80 reserved pairs remain unused.

## Completed work

| Workflow directory | Completed scope | Interpretation |
| --- | --- | --- |
| `measurement_expansion_20260930` | 2,693 task records: request-summary, prompt-example and separated-dimension comparisons | Summary extraction and prompt design change measurements; dimension separation is not independent validation. |
| `semantic_diagnostics_20261005` | 768 records: 48 pairs, four answer conditions, four prompt conditions | Examples and their order influence ratings. Only `results/newserver_20261007` is the completed reference run. |
| `rubric_check_20261007` | 768 records: 96 reused pairs, four answer conditions, two prompts | The concise rubric classified 5/96 empty answers as substantial. Prompt changes bundled several wording changes. |
| `evidence_dependence_20261007` | 1,728 records: title/full-question intent and answer-line ablations | Title-only judgments omit information. Random deletion can overlap cited lines and break code; this is not causal evidence. |
| `association_sensitivity_20261007` | Seven intents across primary, extension and pooled samples; four measurement pipelines | Within each cell, comparisons use identical complete-case support. Some association directions change; rare groups are suppressed. |
| `answer_trajectory_20261007` | 715 saved answers to 390 questions; 704 valid, four invalid, seven skipped | First five answers only, ordered by timestamp and ID; independent answer judgments, not full conversations. |
| `integrated_20261008` | Five stages, 8,393 records: 14B/7B presentation features, 7B intent/answers and 14B answer collections | Completion includes invalid/skipped outputs. Same-family agreement is not accuracy. The 7B answer stage has only 426/715 valid outputs. |
| `explicit_evidence_20261008` | 1,416 records, including 18 designed diagnostic texts | Format reminders increase valid outputs from 60/96 to 82/96, not proven semantic accuracy. Designed examples are not human gold. |
| `final_trajectory_20261008` | Fixed-prefix analyses, thresholds, author groups and count verification | Under 14B labels, 183 complete two-answer cases increase from 159 first-answer positives to 173 with either answer. This is a model-defined count, subsequently challenged by case review. |
| `gpt_case_review_20261008` | 31 questions, 62 answers with reasons and evidence IDs | Targeted GPT review is neither human gold nor strictly independent/blinded validation. |
| `post_review_20261008` | 36 sensitivity cells; input audit of 390 questions/715 answers; candidate preprocessing tests | Metadata loss and uncertain scope matter. Candidate inputs have NOT been rerun through the models. |

Counts are tasks/records, not independent sample sizes. Controls and questions are reused across workflows. Invalid, uncertain and skipped judgments must not be recoded as unhelpful.

## Reproduce from this repository

Programs are in [scripts/helpfulness/followup](../../scripts/helpfulness/followup/README.md). Frozen data and recorded outputs are in [tables/annotation_diagnostics/followup](../../tables/annotation_diagnostics/followup/README.md). No private server, API key or model is needed to recompute the CPU summaries.

From the repository root, using Python 3.10 or newer:

```sh
python scripts/helpfulness/followup/reproduce.py
python scripts/helpfulness/followup/reproduce.py --work-dir data/interim/followup_replay --run
python scripts/visualization/make_followup_figures.py
```

Use a NEW work directory on each run. The first command checks release hashes; the second expands the files and executes 13 CPU analysis/check steps without changing committed results. Substantive JSON outputs are compared with archived results where available; runtime/local-path fields are excluded. The explicit-evidence summary is newly generated from its frozen calls rather than compared with a previously saved top-level summary. Candidate-input tests do not validate model predictions.

Files larger than 150 KB are gzip-compressed. This is ordinary gzip, not Git LFS. The release manifest specifies stored hashes, decompressed hashes and restored paths. Unpacking restores the original task bytes, including exact prompts, code/text inputs and line IDs. The inputs/outputs deliberately preserve failed judgments and reasons.

The original `source_prepare.py` files document selection and transformations but are provenance snapshots, NOT portable entry points. Frozen inputs are the supported inference starting point. Rebuilding every input from raw Kaggle data additionally requires the earlier processed datasets and upstream records; it is not claimed to be covered by this replay command.

For the full raw-HTML audit, install DuckDB and place the original `questions_clean.parquet` and `answers_clean.parquet` under `<work-dir>/sosc314-stackoverflow/data/processed/`, then run `<work-dir>/experiments/post_review_20261008/analyze.py`. The release includes all 93 reviewed HTML snapshots and the full 1,105-document audit table, but does not duplicate the large Parquet files. No external images or linked page contents were fetched.

## Optional inference replay

After unpacking, each inference workflow has `run.py`, `request_protocol.py`, `run_helpfulness_grounding.py` and frozen `tasks.json` files. For example:

```sh
python data/interim/followup_replay/experiments/rubric_check_20261007/run.py --data data/interim/followup_replay/experiments/rubric_check_20261007/tasks.json --model /path/to/Qwen2.5-Coder-14B-Instruct --out /path/to/NEW-output-directory
```

This is a separate, optional GPU operation, not part of CPU reproduction. Install PyTorch, transformers and safetensors, provide local weights, and check free allocated GPU memory first. Preserve the original memory guard and validation rules. Consult each stage's runtime manifest for model name, library versions, seed, input/output limits and input hash; 7B and 14B stages are not interchangeable. Hardware/software changes can affect deterministic decoding results. The release excludes account-specific launchers, SSH access and model weights.

## How to interpret the new figures

- [Intent association sensitivity](../../figures/annotation_diagnostics/followup/intent_association_sensitivity.svg) compares four measurement pipelines. Intervals are conditional on fixed labels and the selected sample, not annotation uncertainty or causal effects. Small groups are explicitly suppressed.
- [GPT case review matrix](../../figures/annotation_diagnostics/followup/gpt_case_review_matrix.svg) shows both answers and invalid/uncertain judgments. Of 14 model-identified first-insufficient/second-substantial cases, GPT supports four, rates both answers substantial in two, neither substantial in seven, and reverses the order in one. These counts are not an accuracy estimate.

The review sampled all 14 added cases, 14 cross-model disagreements (five overlap), and eight deterministic non-target controls. Ratings and evidence were frozen before revealing chronology and prior labels. The reviewer already knew aggregate results and some examples. The file named `mapping_private.json` is now the published unmasking key: public post IDs, answer ranks and selection flags, not credentials. GPT uncertainty is retained; reviewed labels do not replace the full cohort's labels.

On the same 18 questions with determinate, valid judgments from all three raters, GPT and 14B each identify seven substantial transitions but share only three cases. Aggregate agreement can therefore conceal disagreement about which questions improved. Lowering the threshold to any assistance changes the outcome definition, not model accuracy.

## Limitations and unfinished work

The HTML audit found absent link targets in the readable text of 52 questions and 222 answers, and absent image addresses in 11 questions and 21 answers. Not every omission is substantively important. Some missing screenshots and formatting issues already exist in the archived HTML, so they cannot be attributed to this conversion step. The candidate converter only appends available metadata; it does not recover image contents or previous revisions.

The proposed original-versus-context-preserving model comparison has NOT been run. No new validated intent-helpfulness effect, population accuracy, full-dialogue resolution, causal improvement or human gold standard is claimed. Statistical figures are computed from saved results. See the [AI-use record](../AI_USE.md).
