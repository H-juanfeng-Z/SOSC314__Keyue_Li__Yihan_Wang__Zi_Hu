# Intent and first-answer helpfulness measurement

These completed development experiments support Week 5. They measure prompt sensitivity and annotation behavior, not independent accuracy or causal effects. Technical NMF topics remain distinct from help-seeking intent.

## Completed comparisons

- Development: 58 selected question-category tasks and 120 new stratified question–first-answer pairs; 356 generations. Intent review is blinded to earlier labels. Answer scores determine sampling strata but never enter prompts.
- Scope calibration: the same inputs, unchanged controls and scope revisions; 356 generations. Reused development data, not a fresh test set.
- Request-first: 200 prespecified primary pairs plus 190 extensions, from a frozen 1,200-pair pool. One extraction, seven direct and seven extraction-assisted intent decisions, and two helpfulness judgments per complete pair. 6,630 logical records, 6,585 generations, 6,574 schema-valid outputs. The eight-hour run stopped at a complete-pair boundary. Failed extraction blocks dependent calls; context skips are retained.

The primary helpfulness comparison agrees on 178/196 valid pairs; extensions agree on 167/185. Actionability equals sufficiency in 762/767 valid ratings. API-usage positives increase from 153 to 186 (198 paired primary decisions), and from 144 to 182 (189 extension decisions). These figures do not identify the more accurate procedure. The two arms share a model and differ in computational cost. The reserved 80 pairs were not annotated.

## Files and CPU reproduction

Run from the repository root with Python 3.9+; plotting requires Pillow 10.1+ (the checked rendering used 12.3). No GPU is needed to recompute results from exported annotations.

```sh
python scripts/helpfulness/test_request_protocol.py
python scripts/helpfulness/summarize_bridge.py --root tables/annotation_diagnostics/measurement/request_first
python scripts/helpfulness/summarize_scope.py --root tables/annotation_diagnostics/measurement/scope
python scripts/visualization/make_measurement_report_figure.py
```

The development analyzer refuses to overwrite an existing analysis.json. To rerun it, copy development/calls.jsonl to a NEW output directory and run:

```sh
python scripts/helpfulness/analyze_measurement.py --run NEW_OUTPUT_DIRECTORY --metadata tables/annotation_diagnostics/measurement/inputs
```

The report figure is `figures/annotation_diagnostics/first_answer_helpfulness_transitions.png`, based on the 120-pair development comparison, not the 390-pair request-first experiment.

## GPU reruns and input provenance

Inputs are frozen under `tables/annotation_diagnostics/measurement/inputs`. Text is derived from the existing cleaned Stack Overflow Python dataset used by this project. Question and answer IDs link to `https://stackoverflow.com/questions/ID` and `https://stackoverflow.com/a/ID`. Preserve the original dataset and Stack Overflow attribution/license conditions when redistributing. Text transformations retain marked code via `prepare_helpfulness.readable`; the original source data remain authoritative. Audit case IDs can be traced through earlier intent experiment records. Reference labels are not human annotations.

Use an environment with torch 2.2.0+cu121, transformers 4.53.0 and existing Qwen2.5-Coder-14B-Instruct weights. Greedy float16, seed 314, 7,500 input-token cap; extraction output cap 512 and other calls 256. The runner checks for more than 36 GiB free GPU memory. GPU reruns are not guaranteed byte-identical across environments.

```sh
python scripts/helpfulness/run_measurement.py --data tables/annotation_diagnostics/measurement/inputs --model MODEL_DIRECTORY --out NEW_RUN_DIRECTORY
python scripts/helpfulness/run_scope_calibration.py --data tables/annotation_diagnostics/measurement/inputs --model MODEL_DIRECTORY --out NEW_SCOPE_DIRECTORY
python scripts/helpfulness/run_bridge.py --data tables/annotation_diagnostics/measurement/inputs/request_pool.json --model MODEL_DIRECTORY --out NEW_BRIDGE_DIRECTORY --target-hours 8
```

Time-bounded reruns may finish a different number of extension cases; the original 390-case exported decisions are the basis of reported statistics. Scores and earlier selection labels in metadata are not read into inference prompts.

Preparation programs preserve original sampling logic, with `--workspace` specifying the data workspace. They require DuckDB 1.4.4 and the original layout: `sosc314-stackoverflow/data/processed/{questions_clean,answers_clean}.parquet`, `sosc314_week4_experiments/data/blinded_pairs.json`, `sosc314_week4_experiments/measurement_20260926/source_cases.json`, and the earlier intent exports in this repository. These historical source files are NOT all included here. Thus the frozen-input GPU rerun and CPU statistical reproduction are supported; independent reconstruction of the entire historical sampling pipeline additionally requires those sources. Preparation uses exclusive output creation; never overwrite original experiment inputs.

## Export boundaries

Decision exports retain IDs, labels, validity, token counts and timing. Full prompts, completions and free-text rationales remain in the original logs, whose SHA-256 hashes are recorded in provenance.json. Summary request text is omitted from decision exports; it is regenerated by the GPU workflow. No server credentials, login helpers, model weights or reserved-case text are included. The scripts contain the rubric and prompt definitions. Schema checks do not verify semantic evidence support.
