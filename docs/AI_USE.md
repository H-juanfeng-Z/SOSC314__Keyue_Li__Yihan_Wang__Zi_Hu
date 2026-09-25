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
