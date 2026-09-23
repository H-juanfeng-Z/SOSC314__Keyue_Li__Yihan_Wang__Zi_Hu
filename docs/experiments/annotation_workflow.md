# Annotation and encoder experiment workflow

Run commands from the repository root. Programs are organized by function under `scripts/`, not by reporting week. This document retains the input contracts for the experiments originally submitted in Week 4. Taxonomy discussion and interpretation belong in the reports.

See [helpfulness](../../scripts/helpfulness/README.md), [intent annotation](../../scripts/intent_annotation/README.md), [encoder comparison](../../scripts/response_prediction/README.md), and [plotting](../../scripts/visualization/README.md).

## Rebuild the two figures

```sh
python -m pip install numpy matplotlib
python scripts/visualization/make_intent_figures.py
```

This reads `tables/intent_annotation/figure_data.json` and regenerates the PNG/SVG files in `figures/intent_annotation/` without a GPU or inference. The JSON contains aggregate counts and original log hashes. Output validity and synthetic constraint compliance are **not** accuracy against independently established real-question labels. Invalid outputs are not negative labels.

## Experiment inputs and dependencies

Inference additionally requires PyTorch (CUDA), transformers, safetensors, pandas, and scikit-learn. Sample preparation requires DuckDB. These are dependency names, not a frozen environment; model runs record runtime versions where implemented. Model weights and full input data are not bundled here, so this folder alone does not reproduce inference from scratch.

Supply existing `questions_clean.parquet` and `answers_clean.parquet`. Required question columns: `question_id`, `creation_year`, `title`, `body_html`, `question_created_at`. Required answer columns: `question_id`, `answer_id`, `body_html`, `answer_created_at`, `answer_score`.

Also supply the existing `model_sample_60000.jsonl`, containing `question_id`, `creation_year`, `title`, `natural_language_text`, `code_text`, `tags`, and `answered_within_24h`. Use the same input snapshots for exact sample reconstruction; changing them changes the experiment. Keep source attribution with the external Stack Overflow data.

```sh
python scripts/helpfulness/prepare_helpfulness.py --processed-dir /path/to/processed --out data/interim/helpfulness
python scripts/intent_annotation/prepare_intent_round3.py --pairs data/interim/helpfulness/blinded_pairs.json --model-sample /path/to/model_sample_60000.jsonl --out data/interim/intent_round3
python scripts/helpfulness/run_helpfulness_v3.py --data data/interim/helpfulness/blinded_pairs.json --model /path/to/Qwen2.5-Coder-7B-Instruct --out data/interim/results/helpfulness_7b
python scripts/intent_annotation/run_intent_round3.py --data data/interim/intent_round3/cases.json --model /path/to/Qwen2.5-Coder-7B-Instruct --out data/interim/intent_round3_downloaded/results/7b-examples --arm examples
python scripts/intent_annotation/run_intent_round4.py --data data/interim/intent_round3/cases.json --model /path/to/Qwen2.5-Coder-7B-Instruct --out data/interim/intent_round4_downloaded/results/7b-shared_json --arm examples --variant shared_json
```

Use local safetensors model directories. Repeat helpfulness with 1.5B and 3B models; repeat intent with 3B and 7B models and round-four variants `shared_json`, `isolated`, `isolated_json`, each in a fresh output directory. `run_intent_pilot.py` and `run_intent_binary.py` are retained because later scripts import their taxonomy and validator; they need not be rerun first. The prompts and synthetic controls are implemented in code. Generated review templates are blank, not completed human annotations.

## Encoder ablation

```sh
python scripts/response_prediction/run_encoder_ablation.py --data /path/to/model_sample_60000.jsonl --minilm /path/to/all-MiniLM-L6-v2 --codebert /path/to/codebert-base --out data/interim/results/encoder_60000 --n 60000 --batch-size 16
```

This is a separate 24-hour-response prediction experiment, not an answer-helpfulness model.

## Evaluate outputs

Invoke `summarize_round2.py` from `scripts/helpfulness/` and the two intent evaluation scripts from `scripts/intent_annotation/`, using their full script paths from the repository root. `PATH` below is an external/local run root (for the commands above, `data/interim`), not a code directory. To plot a newly aggregated file, pass `--data PATH/report_outputs/figure_data.json --out OUTPUT_DIR` to `scripts/visualization/make_intent_figures.py`.

- `summarize_round2.py --root PATH`: reads `PATH/results/helpfulness_{15b,3b,7b}` and optional `PATH/results/encoder_60000`, summarizes validity, prompt agreement, mismatch controls, and encoder comparisons.
- `audit_intent_cpu.py --root PATH`: expects `PATH/intent_round3/data/{cases,diagnostic_constraints}.json` and `PATH/intent_round{3,4}/results/*/calls.jsonl`; audits completeness, validation failures, and disagreements.
- `aggregate_intent_results.py --root PATH`: expects `PATH/intent_round3/diagnostic_constraints.json`, round-three logs under `PATH/intent_round3_downloaded/results/{3b,7b}-examples`, and round-four logs under `PATH/intent_round4_downloaded/results/{3b,7b}-{shared_json,isolated,isolated_json}`. Each run needs `calls.jsonl` and `summary.json`. Writes `PATH/report_outputs/figure_data.json`, the input to plotting.

The two evaluation tools retain the original server/download folder layouts; arrange results as specified. Raw inference logs and private server configuration are not included in this upload. No new GPU training or inference is triggered by plotting.
