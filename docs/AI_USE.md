# Generative AI Use Record

## Purpose and Scope

This document records how generative AI tools are used in the project. It
provides a concise disclosure of the tasks for which AI assistance was used and,
where possible, connects that assistance to identifiable project files and
GitHub commits.

Generative AI may assist with tasks such as drafting or debugging code. The
group remains responsible for reviewing the output, deciding whether to use it,
verifying the analysis, and interpreting the results. Final research and design
decisions are made by the group.

The tables and figures in this repository are generated from the project data by
the documented Python programs. No generative-image model is used to create the
statistical figures.

## Updating This Record

This document will be updated weekly. Each entry will identify the project
stage, the person whose work is being documented, the AI tool used, the task for
which it was used, and the related file or commit when available. New entries
will be added only after the relevant use has been confirmed. Records and drafts
associated with AI-assisted work will be retained in accordance with the course
policy.

## Week 2

- Feature construction (Zi Hu): OpenAI Codex was used to assist with generating
  and debugging `scripts/02_build_question_features.py`. Related commit:
  `6698aac`.
- Figure generation (Zi Hu): OpenAI Codex was used to assist with generating and
  debugging `scripts/03_make_figures.py`. Related commit: `cb123ef`.
- Data cleaning (Keyue Li): Kimi was used to assist with generating and
  debugging `simple_exclusion.ipynb`. Related commit: `1a3eef0`.

## Week 3

- Text-model feasibility workflow (Zi Hu): OpenAI Codex was used to assist with
  generating and debugging the model-sample preparation and text-model
  comparison code. Related commits: `77b84d0`, `a2a5406`.
- Model-comparison visualization (Zi Hu): OpenAI Codex was used to assist with
  generating and debugging the figure-generation code and its versioned output.
  Related commit: `32547a1`.
  
## Week 4
- Reproduing A&S LDA model (Keyue Li): Kimi K3 was used to assist with generating and debugging. Related commits: '5981726'

## Annotation follow-up (September 22–25, 2026)

- Experimental programs (Zi Hu): OpenAI Codex assisted with generating and
  debugging the helpfulness diagnostics and question-intent stability programs.
  Related commits: `aa97012`, `a425aae`.
- Automated annotation (Zi Hu): locally deployed Qwen2.5-Coder-7B-Instruct and
  Qwen2.5-Coder-14B-Instruct produced experimental judgments, not human labels.
  Prompt definitions are preserved in the corresponding programs; exported
  decisions and summary results are recorded in `0f236e8`. Original prompt/input
  and completion logs are retained privately. Synthetic diagnostic cases are
  AI-generated checks, not independently established ground truth.
- Results and presentation (Zi Hu): OpenAI Codex assisted with diagnostic prompt
  revisions, exploratory interpretation, summary-code preparation, plotting and
  experiment documentation. Related commits: `0f236e8`, `420fbba`, `eca112d`,
  `6ccbd15`. Figures are computed from recorded results, not generated images.
  Automated schema tests and repeatability checks do not establish semantic
  annotation accuracy. No independent human validation is claimed for this stage.
## October 11 staged diagnostic publication (Zi Hu)

- OpenAI Codex assisted with the response-time aggregation, diagnostic prompt repair, inference/verification programs, case review, and publication packaging. See `scripts/response_timing/oct11/`, `scripts/helpfulness/scale_repair_oct11/`, and `scripts/intent_annotation/expansion_oct11/`. This entry does not attribute collaborators' topic-model work or AI use to Zi Hu.
- Qwen2.5-Coder-14B-Instruct generated the experimental scores and intent judgments. GPT conversational review is explicitly AI review, not human gold. The exact backend model identifier for that review is not available. Synthetic examples are AI-constructed diagnostic conditions, not independently validated gold labels.
- The scoring candidate remains unadopted: request-instruction copying improved but short-code penalties and invented requirements remain. Twenty-six exact controls measure deterministic reproduction, not accuracy. The ongoing intent expansion contributes code only in this release; incomplete results are not substituted for the frozen reusable score release.
- Public decision exports omit natural post bodies, raw completions, machine paths and connection credentials. They permit schema/summary replay, not full raw-log or new-model reproduction. Original records remain local. The timing bundle contains IDs, timestamps, memberships and existing scores. No paid API was used by these replay programs.

# Measurement follow-up September 26–27 2026

- Zi Hu: OpenAI Codex assisted with preparation and inference code, prompt revisions, testing, result export, plotting, interpretation and report editing. Related files: `scripts/helpfulness/{prepare_measurement,prepare_request_bridge,run_measurement,run_scope_calibration,run_bridge,request_protocol,measurement_taxonomy,analyze_measurement,summarize_bridge,summarize_scope,export_measurement,test_request_protocol}.py`, `scripts/visualization/make_measurement_report_figure.py`, and `docs/experiments/request_first_measurement.md`.
- Qwen2.5-Coder-14B-Instruct generated experimental intent and helpfulness labels, including question-only request extraction. Results are under `tables/annotation_diagnostics/measurement/`. They are model judgments, not human gold labels. Prompt definitions are in the programs; original logs are retained privately, and hashes accompany the decision exports.
- Figures are computed from recorded data, not generated by an image model. Schema tests and cross-prompt agreement do not establish semantic accuracy. This entry concerns Zi Hu's work, not unconfirmed assistance in teammates' work.

## Measurement follow-up September 30–October 8 2026

- Zi Hu: OpenAI Codex assisted with experiment design and prompt revisions, program generation/debugging, result interpretation, sensitivity analyses, input-fidelity checks, statistical plotting and reproduction documentation. Related files: `scripts/helpfulness/followup/`, `scripts/visualization/make_followup_figures.py`, and `docs/experiments/measurement_followup.md`. This disclosure does not attribute unconfirmed work or AI use to other group members.
- Locally deployed Qwen2.5-Coder-14B-Instruct and Qwen2.5-Coder-7B-Instruct generated experimental intent, presentation-feature and helpfulness judgments. Frozen prompts, inputs, outputs and validation failures are retained under `tables/annotation_diagnostics/followup/`; designed diagnostic texts are AI-assisted checks, not human gold examples.
- GPT in the Codex conversation reviewed 31 questions and 62 answers, recording labels, confidence, evidence lines and reasons. Exact backend model ID was not exposed. This was AI review, not human annotation or strictly independent validation: previous labels and answer order were hidden during rating, but the conversation contained aggregate context. Ratings were frozen before unmasking; the protocol, review packet, ratings and freeze receipt are included in `gpt_case_review_20261008`. The review scripts do not call an API or automatically regenerate these judgments.
- Plots are computed from recorded data, not generated images. Input-preservation tests do not establish model accuracy. The context-preserving model rerun remains unfinished. Records are linked by file here; commit references can be added after publication without relabeling AI-generated judgments as human work.


## Final repository reproducibility update (2026-10-19)

At Zi Hu's request, Codex assisted with README line-ending repair, dependency
profiles, data-access/run-order documentation, a read-only raw-input inventory,
a schema-compatible Parquet export, compatibility imports and synthetic CPU
workflow tests. The new export is not claimed to recreate historical Parquet
bytes/order; original model/annotation definitions and frozen scores are unchanged.
Tests check software interfaces and saved outputs, not annotation accuracy. The
user-approved report prose remains unchanged in this repository-only update.
