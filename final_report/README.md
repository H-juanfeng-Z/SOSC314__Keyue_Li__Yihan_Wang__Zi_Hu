# SOSC314 report

**Authors:** Zi Hu, Keyue Li and Yihan Wang.

[Report website](https://h-juanfeng-z.github.io/SOSC314__Keyue_Li__Yihan_Wang__Zi_Hu/) · [Notebook source](report.ipynb) · [Project repository](https://github.com/H-juanfeng-Z/SOSC314__Keyue_Li__Yihan_Wang__Zi_Hu)

The report studies help-seeking intention, first-answer helpfulness and response speed on historical Python questions. It includes eleven figures and three tables. The authors conducted many informal manual inspections of questions, answers and model ratings. Those checks did not use a uniform adjudication standard, so we do not report an accuracy estimate from them. The documented GPT-assisted 48-case review is a separate diagnostic activity.

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

This starts a fresh kernel with the selected Python executable, executes `report.ipynb`, regenerates eight analytical figures and three tables from `report_data.json`, displays three attributed supplied figures from `assets/`, and exports `reproduced/index.html`. It writes to `reproduced/` and does not register a user-wide kernel. No GPU, model download, paid API or raw-data access is required for this report replay.

Replay from saved summaries is distinct from full regeneration of the underlying sample, annotations and bootstrap statistics. `report_data.json` retains the audited estimates, denominators and interval endpoints. `provenance.json` and `release_manifest.json` record content hashes and provenance.

## Original data and analyses

Obtain the historical [Python Questions from Stack Overflow](https://www.kaggle.com/datasets/stackoverflow/pythonquestions) archive, with `Questions.csv`, `Answers.csv` and `Tags.csv`, following the source terms. Raw data and model weights are not bundled. Existing repository instructions and earlier work remain available:

- `scripts/05_compare_text_models.py`: temporal response-prediction comparison after the initial data preparation workflow.
- `scripts/topic_modeling/README.md`: NMF and Doc2Vec grouping programs and saved diagnostics under `tables/topic_modeling/`.
- `tables/reusable_evaluation/unified_v1/`: 5,000 coarse-score question records, with the 1,000 fine-score questions nested within them.
- `scripts/helpfulness/` and `scripts/intent_annotation/`: historical annotation protocols and programs.

Full inference regeneration additionally needs the archived models, recorded selections, prompts and environments. The exact upstream Qwen weight revision is not pinned. Recovered LDA classifications use current API text rather than the historical snapshot; detailed per-ID retrieval/edit metadata remains unavailable. The author clarified that five posts lacked extractable verb phrases, reconciling 3,423 strict + 1,568 recovered + 4 deleted + 5 unclassified = 5,000. This reason is author-reported, not a new per-ID log audit. Yihan's keyword computation has now been rerun: all 28 proportions match on the 20,000-question technical corpus. Full-category denominators and 15-case review samples are kept separate.

## Deployment

The repository's `report-pages.yml` workflow uploads only `final_report/` as the static Pages artifact. Earlier weekly work is preserved in the repository. A successful commit alone is not deployment evidence; check the workflow result and public URL. The narrative is approximately 2,000 words excluding code, figure captions, references and the expandable example.

## Attribution

Qwen generated research annotations; the authors conducted informal manual checks. Codex assisted with analysis scripts, diagnostic review and writing. The report includes references to the source archive, question-category literature, Doc2Vec, MiniLM, Qwen and scikit-learn.


## Collaborator feature check

`partner_feature_audit.json` records input hashes, category denominators, counts, regex definitions and fixed-seed sample IDs. `feature_id_support.json` contains the category/ID/hit support without post bodies. The 20,000-question corpus is not interchangeable with the scored sample. These are exploratory textual indicators, not model-generation accuracy or causal estimates.

To recompute the original feature comparison given the archived inputs:

```sh
python -m pip install pandas pyarrow
python partner_feature_review.py --clean /path/to/questions_clean.parquet --types /path/to/questions_with_types.csv
```

The portable script retains Yihan's calculation logic, makes file paths explicit and fixes the console's per-case feature display. It does not create a workbook or change labels. The author's reported manual inspections are distinct from the program's automatic keyword matching.


## Revised report structure and figures

The report now starts with sample denominators, explains NMF/LDA/Qwen separately, then presents main coarse and fine group results, technical composition, response timing, measurement diagnostics and a separate reply-prediction benchmark. Figures 2 and 3 retain Yihan's and Keyue's supplied method visuals; Figure 9 retains Yihan's supplied code-element chart with explicit attribution. Their source hashes are in `assets/provenance.json`; they are not claimed as regenerated from summaries. The other eight displayed figures are generated by the notebook. Historical labels and scores are unchanged.

The new seven-intention figure uses 989 common-support questions and 4,000 whole-question bootstrap draws, seed 3141018, preserving overlapping group memberships. Group-mean intervals are pointwise; the separate 42-contrast simultaneous finding uses the previously saved analysis. The main LDA coarse distribution uses the unified baseline release, not the six discrepant alternative-policy export scores. Reply timing uses 1,987 complete-intention questions, not the 989 fine-score support.

## Small display fix in collaborator code

The original per-case console loop sampled rows without computed feature columns, so it could print `None detected` even when features were present. Percentage calculations were unaffected. The portable script fixes the display path and explicit input paths only; 28 proportions and the 75-case sample selection were rechecked. This implementation detail is kept here rather than in the public-facing narrative.


## Code-element analysis (Figure 9)

The supplied image is preserved in `assets/code_element_scores.png`; `code_element_original.py` is the unchanged partner source. Code elements count `<code>` elements, including inline and block code, not lines or characters. Its bins are 0, 1-2, 3-5, 6-10 and 11+. Valid ratings are selected separately by prompt: Format n=995 and Scope n=996 from the fixed 1,000 questions. The 993 common-support restriction used elsewhere is not applied to this descriptive plot. Black bars are mean ±1 SE, with sample SD (ddof=1) divided by sqrt(n), NOT 95% confidence intervals.

The included `code_element_support.json` contains only question/answer IDs, code-element counts and the existing valid scores (null otherwise), without post bodies. Its ten bin summaries were independently verified against the supplied plot. To check all means, counts, SDs and SEs without raw data or additional dependencies:

```sh
python recompute_code_element.py --verify code_element_summary.json
```

With access to the archived inputs, independently repeat the metadata join (requires `pyarrow`):

```sh
python -m pip install pyarrow
python recompute_code_element.py --fine /path/to/unified_v1/scores_0_10_1000.json --questions /path/to/questions_clean.parquet --verify code_element_summary.json
```

The original partner script expects a flat `fine_scores.json` (one row per question/prompt) and `questions_clean.parquet` in its working directory; the portable checker accepts the repository's nested unified release. Report replay preserves the supplied PNG and recalculates its statistical support, but does not claim to regenerate that PNG byte-for-byte. No new labels or scoring standards are introduced.
