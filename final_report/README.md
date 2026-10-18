# SOSC314 report

**Authors:** Zi Hu, Keyue Li and Yihan Wang.

[Report website](https://h-juanfeng-z.github.io/SOSC314__Keyue_Li__Yihan_Wang__Zi_Hu/) · [Notebook source](report.ipynb) · [Project repository](https://github.com/H-juanfeng-Z/SOSC314__Keyue_Li__Yihan_Wang__Zi_Hu)

The report studies help-seeking intention, first-answer helpfulness and response speed on historical Python questions. It includes seven figures and two tables. The authors conducted many informal manual inspections of questions, answers and model ratings. Those checks did not use a uniform adjudication standard, so we do not report an accuracy estimate from them. The documented GPT-assisted 48-case review is a separate diagnostic activity.

## Read the report

Open `index.html` without running code. Figures are embedded. The full annotation instructions are in [qwen_prompts.json](qwen_prompts.json); [qwen_example.json](qwen_example.json) contains an actual recorded output with an input excerpt. The report evaluates existing Stack Overflow answers; Qwen acts as an annotator.

## Reproduce figures and tables

Use Python 3.12 in a separate environment:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

This starts a fresh kernel with the selected Python executable, executes `report.ipynb`, regenerates all seven figures and two tables from `report_data.json`, and exports `reproduced/index.html`. It writes to `reproduced/` and does not register a user-wide kernel. No GPU, model download, paid API or raw-data access is required for this report replay.

Replay from saved summaries is distinct from full regeneration of the underlying sample, annotations and bootstrap statistics. `report_data.json` retains the audited estimates, denominators and interval endpoints. `provenance.json` and `release_manifest.json` record content hashes and provenance.

## Original data and analyses

Obtain the historical [Python Questions from Stack Overflow](https://www.kaggle.com/datasets/stackoverflow/pythonquestions) archive, with `Questions.csv`, `Answers.csv` and `Tags.csv`, following the source terms. Raw data and model weights are not bundled. Existing repository instructions and earlier work remain available:

- `scripts/05_compare_text_models.py`: temporal response-prediction comparison after the initial data preparation workflow.
- `scripts/topic_modeling/README.md`: NMF and Doc2Vec grouping programs and saved diagnostics under `tables/topic_modeling/`.
- `tables/reusable_evaluation/unified_v1/`: 5,000 coarse-score question records, with the 1,000 fine-score questions nested within them.
- `scripts/helpfulness/` and `scripts/intent_annotation/`: historical annotation protocols and programs.

Full inference regeneration additionally needs the archived models, recorded selections, prompts and environments. The exact upstream Qwen weight revision is not pinned. Recovered LDA classifications use current API text rather than the historical snapshot; detailed retrieval/edit metadata and five statuses in the author's aggregate accounting still need clarification. The unsupported hand-entered keyword-percentage table is omitted. None of these gaps is silently filled with invented data.

## Deployment

The repository's `report-pages.yml` workflow uploads only `final_report/` as the static Pages artifact. Earlier weekly work is preserved in the repository. A successful commit alone is not deployment evidence; check the workflow result and public URL. The generated report is approximately 2,000 words, excluding its code and expandable example.

## Attribution

Qwen generated research annotations; the authors conducted informal manual checks. Codex assisted with analysis scripts, diagnostic review and writing. The report includes references to the source archive, question-category literature, Doc2Vec, MiniLM, Qwen and scikit-learn.
