# Reproduction guide

All commands below start from the repository root unless a `cd` is shown. Use a fresh clone when rerunning stages that write historical exploratory outputs. Never overwrite the frozen score release to make a new run agree.

## 1. What can be reproduced?

| Level | Inputs | What the command establishes |
| --- | --- | --- |
| Frozen-release verification | Committed scores, IDs, manifests and diagnostic inputs | File integrity, joins, valid denominators and replayed saved summaries; not correctness of model annotations |
| Submitted-report replay | Committed `final_report/` plus pinned CPU packages | Fresh execution of notebook, eight generated figures, three preserved supplied figures, three tables and HTML |
| Raw-data workflow | Original three Kaggle CSVs and base packages | Screening manifest, normalized copies, question-level features and original exploratory figures |
| Topic/annotation reconstruction | Appropriate raw/processed snapshot, selections, local model weights, additional libraries and configuration | A rerun under recorded settings; exact historical labels/text/weights cannot be guaranteed when original snapshots or revisions are unavailable |

The first two levels need neither raw data access nor a GPU. Replaying a summary is not re-estimating all bootstrap statistics from raw posts. The notebook uses saved audited estimates and interval endpoints, and the separate frozen-input check scripts verify the available supporting analyses.

## 2. Environment and dependencies

Recommended interpreter: **Python 3.12**. Create and activate an isolated environment as shown in the [root README](../README.md). Use `python -m pip`, ensuring it refers to that environment.

| Workflow | Installation |
| --- | --- |
| Report replay only | `python -m pip install -r requirements-report.txt` |
| Raw screening/features/exploration | `python -m pip install -r requirements.txt` |
| Prediction/annotation | Base requirements plus `requirements-models.txt`; PyTorch/CUDA must match your hardware |
| NMF/Doc2Vec/LDA | Base requirements plus `requirements-topics.txt`; explicitly obtain spaCy `en_core_web_sm` for LDA |
| Partner Parquet/features/workbooks | Base requirements plus `requirements-features.txt` |

The submitted report pins exact top-level versions in `final_report/requirements.txt`. Optional profiles are dependency ranges, not the exact historical model server environment. Prefer separate environments for report replay and optional models if their resolvers conflict. The repository check never downloads weights or starts model inference. Hardware/provider access and credentials are not included.

## 3. Offline checks and report replay

```sh
python scripts/check_reproducibility.py
python -m pip install -r requirements-report.txt
python scripts/check_reproducibility.py --replay-report
```

Checks include unified 5k/1k IDs and matched first answers, Oct12/Oct13 valid denominators, historical response-timing tables, archived follow-up file hashes, report release hashes and a fresh notebook kernel. The replay writes only `final_report/reproduced/`. Inspect `final_report/reproduced/index.html`. It does not publish anything.

The image comparison is strict only when `--strict-images` is supplied; different font/FreeType/platform versions can change PNG bytes without changing the data. The verifier always requires eleven displayed image outputs, unchanged notebook sources and a successful fresh execution. The release hash checks still protect the committed assets. Supplied collaborator images are attributed in `final_report/assets/provenance.json`, not claimed to be newly regenerated from raw data.

To check the code-element statistics independently:

```sh
python final_report/recompute_code_element.py --verify final_report/code_element_summary.json
```

This uses the committed support data. See [final report instructions](../final_report/README.md) for the metadata-join version requiring archived Parquet. Figure 9 uses arm-specific valid observations (995 Format, 996 Scope); its error bars are mean ±1 SE, not 95% confidence intervals. Other comparisons may deliberately use common support; do not silently mix denominators.

## 4. Reconstruct from the original CSV files

Obtain the source as described in [data/README.md](../data/README.md), then:

```sh
python -m pip install -r requirements.txt
python scripts/inspect_raw_data.py --raw-dir data/raw_csv --out data/interim/raw_inventory.json
cd data/raw_csv
python ../../scripts/simple_exclusion.ipynb
cd ../..
python scripts/02_build_question_features.py --raw-dir data/raw_csv --exclusions data/raw_csv/excluded_records.csv
python scripts/03_make_figures.py
```

The `.ipynb` screening filename is historical; its contents are plain Python. The genuine report notebook is `final_report/report.ipynb`. Screening reports duplicate/missing/date checks; the downstream code consumes the actual exported time-inverted-answer and orphan-tag exclusions. Do not assume every printed check is a separate exported exclusion. The manifest is comment-delimited with two sections, not a single rectangular CSV.

Outputs: normalized UTF-8 CSVs in `data/interim/normalized_utf8/`, features in `data/processed/question_response_features.parquet`, summary tables in `tables/`, and original exploratory figures in `figures/`. Stage 2 visibly fails on remaining orphan answers/time inversions rather than inventing new exclusions. Preserve source files and the generated inventory hashes.

For optional body-dependent workflows:

```sh
python scripts/export_annotation_inputs.py --out-dir data/processed/annotation_inputs
```

This creates `questions_clean.parquet`, `answers_clean.parquet` and `reconstruction_manifest.json` in a NEW directory. Required question fields include `question_id`, `creation_year`, `title`, `body_html`, `question_created_at`; answer fields include `question_id`, `answer_id`, `body_html`, `answer_created_at`, `answer_score`. This export preserves the existing exclusions, does not score/classify posts, and records input/output hashes. It sorts IDs chronologically for portability; it is not asserted to match the archived team Parquet bytes or row order. Order-dependent random samples can therefore differ from earlier samples. Use archived inputs/selections for exact sample replay.

## 5. Optional models: explicit, separate workflows

### Response prediction (not helpfulness prediction)

```sh
python -m pip install -r requirements.txt -r requirements-models.txt
python scripts/04_prepare_model_sample.py --features data/processed/question_response_features.parquet --output data/interim/model_sample_60000.jsonl
python scripts/05_compare_text_models.py --data data/interim/model_sample_60000.jsonl --output data/interim/text_model_results.json
```

The comparison without `--minilm` is the TF-IDF baseline. To add MiniLM, supply `--minilm /path/to/local/all-MiniLM-L6-v2`. The frozen encoder comparison including CodeBERT is documented in [response_prediction/README.md](../scripts/response_prediction/README.md). These commands fit models and are NOT part of routine repository checks. Training/validation/test years are 2008–2014/2015/2016; the sample excludes incomplete 24-hour observation windows. Local model names/paths must not be confused with an immutable downloaded revision.

### Topic models

```sh
python -m pip install -r requirements.txt -r requirements-topics.txt
cd data/processed/annotation_inputs
python "../../../scripts/topic_modeling/question classfication NMF.py"
cd ../../..
```

The NMF filename retains the author's original spelling. Outputs are created under that working directory. The natural-language NMF, Doc2Vec/K-Means variants and saved diagnostics are documented in [topic_modeling/README.md](../scripts/topic_modeling/README.md). These programs fit models; rerunning them is not required to read the report.

For LDA, explicitly install the `en_core_web_sm` spaCy model, set `QUESTIONS_CSV` and `OUT_DIR` at the top of `scripts/topic_modeling/LDA classification.py` to your original archive path and a fresh output directory, then run that script. Other parameters/selections must remain as recorded for comparison. This older script is retained rather than silently rewritten into a new sampling/classification protocol. Recovered current-API text and author-reported five no-verb-phrase cases have documented provenance limitations; new extraction does not recover past edits/deleted content.

### Partner features

```sh
python -m pip install -r requirements.txt -r requirements-features.txt
python "scripts/Feature Extract/03_extract_features_full.py" --parquet data/processed/annotation_inputs/questions_clean.parquet --out data/interim/question_features.xlsx
```

Compatibility imports connect the preserved numbered source files; feature definitions are unchanged. The report's separate 20k topic-keyword check additionally needs the original topic assignments: `python final_report/partner_feature_review.py --clean /path/to/questions_clean.parquet --types /path/to/questions_with_types.csv`. Do not substitute the scored sample for the full 20k category denominators.

### Qwen annotation

The report uses **Qwen2.5-Coder-14B-Instruct** as a local annotator of questions and existing human answers, not a hosted paid API or a trained helpfulness predictor. Full prompt text, decoding settings/limits, validation rules and an actual example are in [qwen_prompts.json](../final_report/qwen_prompts.json), [qwen_example.json](../final_report/qwen_example.json) and the report. Earlier 1.5B/3B/7B development scripts are not the final 14B run merely because they are convenient entry points. Follow the dated guides under [docs/experiments](experiments/README.md) and use the correct archived selections, source task IDs, prompt version, local weights and environment.

The exact upstream Qwen weight revision was not pinned. Some historical task/log/processed snapshots are not bundled. Thus full re-inference is not promised to reproduce every historical label bit-for-bit even with source access. This limitation is explicit; no new labels are invented, invalids filled with zero, or model agreement called accuracy. For current result reuse prefer the manifested releases rather than rerunning annotation.

## 6. Tests, website and publication

For the small synthetic raw-data smoke test (no real posts/models):

```sh
python -m pip install -r requirements.txt -r requirements-features.txt
python -m unittest discover -s tests -p "test_*.py"
```

It exercises screening → feature construction → exploratory plotting → body export and the collaborator import path in a temporary directory. Synthetic rows test software interfaces, not substantive accuracy or original-dataset equivalence. The separate CPU workflow also replays the report from a fresh environment. Pages deploys only `final_report/`; changing README alone does not change the published report. Do not upload raw files, model weights or credentials to make CI run inference.

## 7. Status and interpretation

The [recorded clean-environment checks](reproducibility_checks.json) cover report
replay and ten small CPU tests. Optional inference/topic environments and the full
original archive are not claimed to have been rerun in this repository update.

The 1,000 fine-score records are nested within 5,000 coarse questions. Preserve `status`, `usable`, batch and prompt version; compare the same `question_id`/`answer_id` on the intended valid support. [Version notes](../tables/REUSE_NOTES.md) flag historical alternative-policy exports. No code addition here changes previous labels, definitions, supplied figures or the user-approved report prose.

Package and code verification is not technical validation of an answer, a formal annotation-accuracy study, or evidence of causality. The report states the original acquisition/processing, manual inspection and GPT-assisted diagnostic roles, and limits of generalization.
