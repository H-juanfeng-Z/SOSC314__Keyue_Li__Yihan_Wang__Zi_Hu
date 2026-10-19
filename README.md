# Question Characteristics, Response Time, and Answer Quality on Stack Overflow

**Authors:** Zi Hu, Keyue Li and Yihan Wang · SOSC314 course project

**[Read the public GitHub Pages report](https://h-juanfeng-z.github.io/SOSC314__Keyue_Li__Yihan_Wang__Zi_Hu/)** · [Source Jupyter notebook](final_report/report.ipynb) · [Rendered HTML](final_report/index.html)

We study how Python help-seeking intentions, technical topics and question presentation are associated with the helpfulness and speed of the first observed Stack Overflow answer. Qwen annotates existing human answers; a separate supervised benchmark predicts a reply within 24 hours. The main analyses are observational, not causal. The report contains eleven figures and three tables, with data provenance, methods, diagnostics, limitations, references and AI-use disclosure.

## Reproduce the submitted report (CPU; no original data or model required)

Clone this repository and open a terminal in its root:

```sh
git clone https://github.com/H-juanfeng-Z/SOSC314__Keyue_Li__Yihan_Wang__Zi_Hu.git
cd SOSC314__Keyue_Li__Yihan_Wang__Zi_Hu
```

Use **Python 3.12** in an isolated environment. On Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
```

Then, on either platform:

```sh
python -m pip install -r requirements-report.txt
python scripts/check_reproducibility.py --replay-report
```

The command checks frozen release hashes, IDs and denominators; re-executes the source notebook in a fresh kernel; and verifies its eleven displayed image outputs (use `--strict-images` to require identical PNG bytes). Open **`final_report/reproduced/index.html`** to read the rebuilt page. Eight figures are regenerated from saved analysis summaries; three attributed collaborator figures are preserved. This is report replay, **not** new annotation or complete raw-source regeneration. No paid API, model download or GPU is used.

For hash/data checks alone, use `python scripts/check_reproducibility.py` (standard library only). Detailed environment, run order, optional workflows and known limits are in [the reproduction guide](docs/REPRODUCIBILITY.md).

## Repository structure

| Path | Contents |
| --- | --- |
| [final_report/](final_report/README.md) | Submitted notebook, self-contained HTML, report data, full Qwen prompts, examples, assets, pinned report dependencies and release hashes |
| [data/](data/README.md) | Instructions to obtain the original archive; raw/processed data are created locally and ignored by Git |
| [scripts/](scripts/README.md) | Screening, feature construction, response prediction, topic models and annotation code |
| [tables/](tables/) | Frozen scores, classification exports, summary tables and diagnostic inputs |
| [tables/reusable_evaluation/unified_v1/](tables/reusable_evaluation/unified_v1/README.txt) | Authoritative 5,000-question 0–2 release, nested 1,000-question 0–10 release and exactly matched 0–2 subset |
| [figures/](figures/) | Earlier analytical figures; the submitted report's assets are also under `final_report/` |
| [docs/](docs/REPRODUCIBILITY.md) | Reproduction, experiment methods, AI use and historical project records |
| [.github/workflows/](.github/workflows/) | Separate report verification/replay and GitHub Pages deployment |

The notebook is intentionally under `final_report/`, separate from previous weekly work. Folder names such as `src/` or `notebooks/` are not prerequisites: `scripts/` contains the source code and `final_report/report.ipynb` is the report source.

## Obtain data and run the raw-data workflow

Download the historical [Python Questions from Stack Overflow archive on Kaggle](https://www.kaggle.com/datasets/stackoverflow/pythonquestions), following the source access and attribution terms. Place **`Questions.csv`, `Answers.csv`, `Tags.csv`** in `data/raw_csv/`. Large raw HTML, model weights, credentials and full private inference logs are not bundled. See [data access, schemas and provenance](data/README.md).

From the repository root, install the base dependencies and record the input inventory:

```sh
python -m pip install -r requirements.txt
python scripts/inspect_raw_data.py --raw-dir data/raw_csv --out data/interim/raw_inventory.json
```

Run these stages in order:

```sh
cd data/raw_csv
python ../../scripts/simple_exclusion.ipynb
cd ../..
python scripts/02_build_question_features.py --raw-dir data/raw_csv --exclusions data/raw_csv/excluded_records.csv
python scripts/03_make_figures.py
```

`simple_exclusion.ipynb` is a historical **plain Python script with an .ipynb filename**, not a Jupyter notebook; invoke it with Python exactly as above. It writes an exclusion manifest. Stage 2 applies the manifest and creates `data/processed/question_response_features.parquet`; Stage 3 creates exploratory tables and figures. These outputs can replace local copies of the earlier exploratory tables, so use a fresh clone/worktree when regenerating. None of these commands changes the frozen evaluation release.

For topic/annotation-compatible Parquet inputs, see `scripts/export_annotation_inputs.py` in the reproduction guide. Optional model inference requires local model weights, the correct historical input snapshots and the archived selections/prompts. It is not launched by the report replay or repository checks. Exact historical model/text reproduction has documented limits; we do not claim that installing packages reconstructs unavailable historical edits or unpinned weight revisions.

## Dependencies and data reuse

- [requirements.txt](requirements.txt): initial CSV/feature/exploratory workflow.
- [requirements-report.txt](requirements-report.txt): exact submitted-report replay dependencies (delegates to `final_report/requirements.txt`).
- [requirements-models.txt](requirements-models.txt): optional prediction/annotation libraries; install a suitable PyTorch/CUDA build separately for your hardware.
- [requirements-topics.txt](requirements-topics.txt): optional topic-model libraries; the spaCy language model is an additional explicit download.
- [requirements-features.txt](requirements-features.txt): optional partner feature/Parquet/workbook tools.

Join scores/classifications by `question_id`; compare the same answer using `question_id` **and** `answer_id`. Invalid, skipped and null scores are not zero. The 1,000 fine-score questions are a subset of the 5,000, not an extra sample. Earlier exports are preserved for provenance: [reuse/version notes](tables/REUSE_NOTES.md) distinguish them from the authoritative unified scores.

## Course deliverable and acknowledgment

The public report can be read without running code or logging in. Its notebook, HTML and support files are versioned separately from earlier work. [Final-report details](final_report/README.md), [references and Qwen disclosure](final_report/index.html), and [project AI-use record](docs/AI_USE.md) explain the sources and assistance used. Authors conducted informal manual inspections; no formal annotation-accuracy estimate is claimed.

## Historical project records and examples

The sections below retain earlier work and dated development notes. Statements such as “pending” in the October 11 notes describe that historical release, not the status of the final report. Use the instructions above for the current submission.

## Project Direction

This project investigates how the help-seeking intent and presentation of Python
questions on Stack Overflow are associated with first-answer helpfulness.
Technical topics provide context, while response occurrence and timing remain
complementary outcomes and prediction benchmarks. We use question-level text,
code, tags, and posting characteristics as potential explanatory variables.

The project is observational: it examines associations in a historical archive
and does not make causal claims. Intent and helpfulness annotations are under
development and require validation before substantive relationships can be
established. The workflow does not treat voting scores, output-format validity,
or agreement between prompts as verified technical correctness.

## October 11 staged additions

- [Response timing and frozen helpfulness](scripts/response_timing/oct11/README.md): standard-library replay of group, year and timing summaries; ordinal time-adjusted indices are exploratory, not validated quality.
- [Candidate 0–10 scoring diagnostics](scripts/helpfulness/scale_repair_oct11/README.md): completed242-task test, preserved failures and counterexamples; the candidate is **not adopted**.
- [Intent expansion program](scripts/intent_annotation/expansion_oct11/README.md): fixed seven-label protocol and runner; full expansion results are **pending**, not part of this release.

## Data

The analysis uses the historical Kaggle dataset [Python Questions from Stack
Overflow](https://www.kaggle.com/datasets/stackoverflow/pythonquestions). The
project copy contains records from 2008--2016 in three CSV files:

- `Questions.csv` contains question IDs, creation times, titles, bodies, and scores.
- `Answers.csv` contains answer IDs, parent question IDs, creation times, bodies, and scores.
- `Tags.csv` maps technical tags to question IDs.

Large raw files, model weights and full inference logs are not committed.
Selected generated tables, assignment snapshots and figures are versioned for
inspection. To run the initial workflow, place the three CSV files in `data/raw_csv/`.
The expected `data/` subdirectories are created locally during analysis and are
excluded from version control.

## Data Examples

The project uses three raw CSV files from the Kaggle *Python Questions from Stack Overflow* dataset. `Questions.csv` contains one row per question, `Answers.csv` contains one row per answer, and `Tags.csv` contains one question–tag association per row. The preliminary-screening script identifies logically invalid records and writes an exclusion manifest without modifying the raw files. The feature-construction script then applies that manifest, aggregates answers and tags, and creates one analytical row per question. The examples below are taken from the original data and the resulting question-level dataset. Long HTML body fields are shortened only for display in this README.

### Raw Questions

Each row in `Questions.csv` represents one Stack Overflow question.

| Id | OwnerUserId | CreationDate | Score | Title | Body excerpt |
|---:|---:|---|---:|---|---|
| 469 | 147 | 2008-08-02T15:11:16Z | 21 | How can I find the full path to a font from its display name on a Mac? | `<p>I am using the Photoshop's javascript API to find the fonts in a given PSD.</p>...` |
| 502 | 147 | 2008-08-02T17:01:58Z | 27 | Get a preview JPEG of a PDF on Windows? | `<p>I have a cross-platform (Python) application which needs to generate a JPEG preview...</p>` |
| 535 | 154 | 2008-08-02T18:43:54Z | 40 | Continuous Integration System for a Python Codebase | `<p>I'm starting work on a hobby project with a python codebase and would like to set up some form of continuous integration...</p>` |

### Raw Answers

Each row in `Answers.csv` represents one answer. `ParentId` identifies the question to which the answer belongs.

| Id | OwnerUserId | CreationDate | ParentId | Score | Body excerpt |
|---:|---:|---|---:|---:|---|
| 497 | 50 | 2008-08-02T16:56:53Z | 469 | 4 | `<p>open up a terminal... and type this in:</p><pre><code>locate InsertFontHere</code></pre>...` |
| 518 | 153 | 2008-08-02T17:42:28Z | 469 | 2 | `<p>I think you'll have to iterate through the various font folders on the system...</p>` |
| 536 | 161 | 2008-08-02T18:49:07Z | 502 | 9 | `<p>You can use ImageMagick's convert utility for this...</p><pre><code>Convert taxes.pdf taxes.jpg</code></pre>...` |

### Raw Tags

Each row in `Tags.csv` represents one question–tag association. A question can therefore appear in multiple rows when it has multiple tags.

| Id | Tag |
|---:|---|
| 469 | python |
| 469 | osx |
| 469 | fonts |

### Question-Level Data After Screening and Construction

The preliminary-screening script identified invalid records and stored them in `excluded_records.csv`. The feature-construction script applied those exclusions and aggregated the remaining answers and tags into one row per question. The processed dataset is stored as `data/processed/question_response_features.parquet`.

Because the processed table contains many columns, the same three example rows are displayed below in three parts: question information, response outcomes, and constructed features.

#### Question Information

| question_id | owner_user_id | question_created_at | question_score | title | body_html excerpt |
|---:|---:|---|---:|---|---|
| 469 | 147 | 2008-08-02T15:11:16Z | 21 | How can I find the full path to a font from its display name on a Mac? | `<p>I am using the Photoshop's javascript API...</p>` |
| 502 | 147 | 2008-08-02T17:01:58Z | 27 | Get a preview JPEG of a PDF on Windows? | `<p>I have a cross-platform (Python) application...</p>` |
| 535 | 154 | 2008-08-02T18:43:54Z | 40 | Continuous Integration System for a Python Codebase | `<p>I'm starting work on a hobby project with a python codebase...</p>` |

#### Response Outcomes

| question_id | answer_count | first_answer_id | first_answer_at | first_answer_score | maximum_answer_score | received_answer | first_response_minutes | answered_within_24h |
|---:|---:|---:|---|---:|---:|---|---:|---|
| 469 | 4 | 497 | 2008-08-02T16:56:53Z | 4 | 12 | True | 105.62 | True |
| 502 | 3 | 536 | 2008-08-02T18:49:07Z | 9 | 25 | True | 107.15 | True |
| 535 | 7 | 538 | 2008-08-02T18:56:56Z | 23 | 23 | True | 13.03 | True |

#### Constructed Question Features

| question_id | tag_count | tags | title_chars | body_html_chars | code_block_count | link_count | question_mark_count | creation_year | creation_month | creation_weekday_utc | creation_hour_utc |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 469 | 4 | fonts\|osx\|photoshop\|python | 70 | 445 | 0 | 0 | 1 | 2008 | 8 | 6 | 15 |
| 502 | 4 | image\|pdf\|python\|windows | 39 | 314 | 0 | 1 | 2 | 2008 | 8 | 6 | 17 |
| 535 | 3 | continuous-integration\|extreme-programming\|python | 51 | 624 | 0 | 0 | 1 | 2008 | 8 | 6 | 18 |

The response variables are constructed as follows:

- `received_answer` indicates whether the question has at least one valid observed answer.
- `first_response_minutes` is the time between question creation and the earliest valid answer.
- `answered_within_24h` indicates whether the first valid answer arrived within 24 hours.
- `first_answer_score` is the community score of the earliest answer.
- `maximum_answer_score` is the highest community score among all answers to the question.

The question-characteristic variables are descriptive features rather than complete classifications of question type. In particular, `body_html_chars` measures the length of the complete HTML body, including natural language, code, links, and HTML markup.

## Repository Guide

| Location | Purpose |
| --- | --- |
| [scripts/simple_exclusion.ipynb](scripts/simple_exclusion.ipynb) | Preliminary screening script and exclusion-manifest creation. |
| [scripts/](scripts/README.md) | Initial data workflow and functional code navigation. |
| [scripts/response_prediction/](scripts/response_prediction/README.md) | MiniLM/CodeBERT comparison for response prediction. |
| [scripts/helpfulness/](scripts/helpfulness/README.md) | First-answer sample preparation and helpfulness annotation. |
| [scripts/intent_annotation/](scripts/intent_annotation/README.md) | Question-intent prompts, sampling and validation. |
| [scripts/topic_modeling/](scripts/topic_modeling/README.md) | NMF, Doc2Vec/K-Means and verb-phrase LDA. |
| [scripts/visualization/](scripts/visualization/README.md) | Intent diagnostic figure generation. |
| [tables/](tables/) | Descriptive outputs, topic assignments and intent figure source counts. |
| [figures/](figures/) | Versioned descriptive, model and annotation figures. |
| [docs/experiments/](docs/experiments/README.md) | Experiment methods, input contracts, run instructions and migration guide. |
| `docs/` | Project documentation, including the [AI-use record](docs/AI_USE.md). |
| [docs/meetings/](docs/meetings/README.md) | Dated meeting records, including topic selection, oral preparation, and progress discussions. |
| `Topic choice reasoning` | Rationale for rejecting earlier project topics. |
| `Schedule.md` | Early topic exploration and project-planning notes. |
| `requirements.txt` | Python package requirements for the reproducible workflow. |


## Project Records

The repository retains early topic selection and meeting records so that the
development of the research question and data choice is traceable. Week-specific
reports and individual responsibilities are maintained with the relevant course
deliverables rather than duplicated in this long-lived project overview.

## Generative AI Disclosure

Generative-AI use connected to project code is documented in
[docs/AI_USE.md](docs/AI_USE.md). The record is designed to be updated as the
project develops.
# Measurement follow-up

Completed intent and first-answer helpfulness diagnostics, reproducible result summaries and the report figure are documented in [the measurement guide](docs/experiments/request_first_measurement.md). Technical-topic labels and intent labels represent different concepts; their integration remains a subsequent analysis step.

Subsequent [measurement sensitivity and answer-trajectory analyses](docs/experiments/measurement_followup.md) include frozen inputs, CPU reproduction commands, cross-model diagnostics, GPT case review and an input-fidelity audit. These analyses test measurement limitations; automated judgments are not verified answer correctness or human gold labels.
