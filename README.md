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
