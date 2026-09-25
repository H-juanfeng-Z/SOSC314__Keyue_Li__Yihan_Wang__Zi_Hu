# Experiment documentation

- [Annotation and encoder workflow](annotation_workflow.md)
- [Annotation measurement diagnostics](annotation_diagnostics.md): intent stability, answer-content tests, prompt factorial, and reproduction limits.
- [Technical-topic methods](technical_topics.md)
- [NMF example questions](nmf_example_questions.md)
- [Code navigation](../../scripts/README.md)
- [NMF_example_questions_advanced](topic_model_advanced)

## Location migration

Programs are organized by function rather than reporting week. Experiment logic and published numerical results are retained. Report commit links remain immutable snapshots of the submitted layout.

| Former location | Current location |
| --- | --- |
| `experiments/week4/run_encoder_ablation.py` | `scripts/response_prediction/` |
| `experiments/week4/prepare_helpfulness.py`, `run_helpfulness_v3.py`, `summarize_round2.py` | `scripts/helpfulness/` |
| `experiments/week4/prepare_intent_round3.py`, `run_intent_*.py`, intent audit and aggregation | `scripts/intent_annotation/` |
| `experiments/week4/make_intent_figures.py` | `scripts/visualization/` |
| `experiments/week4/report_outputs/figure_data.json` | `tables/intent_annotation/` |
| `experiments/week4/report_outputs/*.png`, `*.svg` | `figures/intent_annotation/` |
| `experiments/week4/Question Type Classification/*.py` and its figure-generation code | `scripts/topic_modeling/` |
| `scripts/LDA classification.py` | `scripts/topic_modeling/` |
| Technical-topic result folders | `tables/topic_modeling/` |
| Technical-topic figures | `figures/topic_modeling/` |
| Week 4 and technical-topic READMEs and example questions | `docs/experiments/` |


Historical meeting and AI-use records are not rewritten. Early numbered scripts retain their paths. Local raw data, model weights, server experiments and untracked report materials were not migrated or uploaded.
