# Question Characteristics and Response Time on Stack Overflow

## Project Direction

This project investigates how observable characteristics of Python questions on
Stack Overflow are associated with response outcomes. The core questions are
whether a question receives an answer and how quickly it receives its first
answer. We use question-level text, code, tags, and posting characteristics as
potential explanatory variables.

The project is observational: it examines associations in a historical archive
and does not make causal claims. A later project stage may investigate answer
adequacy or quality, but only after the group defines and evaluates a defensible
proxy or labeling procedure. The current workflow does not treat voting scores
as verified technical correctness.

## Data

The analysis uses the historical Kaggle dataset [Python Questions from Stack
Overflow](https://www.kaggle.com/datasets/stackoverflow/pythonquestions). The
project copy contains records from 2008--2016 in three CSV files:

- `Questions.csv` contains question IDs, creation times, titles, bodies, and scores.
- `Answers.csv` contains answer IDs, parent question IDs, creation times, bodies, and scores.
- `Tags.csv` maps technical tags to question IDs.

The raw and generated data files are intentionally not committed because of
their size. To run the workflow, place the three CSV files in `data/raw_csv/`.
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
| `simple_exclusion.ipynb` | Preliminary screening script and exclusion-manifest creation. |
| `scripts/` | Feature construction, descriptive analysis, and workflow instructions. |
| `tables/` | Versioned CSV outputs used for initial descriptive analysis. |
| `figures/` | Versioned exploratory figures generated from the question-level data. |
| `docs/` | Project documentation, including the [AI-use record](docs/AI_USE.md). |
| `group meeting 1`, `group meeting 2`, `group meeting 3` | Records of the group’s early project decisions. |
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
