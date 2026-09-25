# Reproducible data workflow

## Functional script directories

- [Response prediction](response_prediction/README.md): frozen encoder comparison; earlier numbered baseline scripts remain below.
- [First-answer helpfulness](helpfulness/README.md): sample preparation, annotation and diagnostics.
- [Question-intent annotation](intent_annotation/README.md): taxonomy, prompts and output audits.
- [Topic modeling](topic_modeling/README.md): NMF, Doc2Vec/K-Means and verb-phrase LDA.
- [Visualization](visualization/README.md): intent result figures.
- [Experiment workflow](../docs/experiments/annotation_workflow.md): data contracts and run commands.
- [Annotation diagnostics](../docs/experiments/annotation_diagnostics.md): follow-up runners, decision exports, paired analysis and figures.

These directories replace the former `experiments/week4/` code layout. Existing numbered preprocessing and response-model scripts retain their paths.

The project separates screening decisions from Zi Hu's construction and
visualization work.

## Inputs

Download `Questions.csv`, `Answers.csv`, and `Tags.csv` from Kaggle and place
them in `data/raw_csv/`.

## Stage 1: preliminary screening

Run the group repository's preliminary screening script from the directory that
contains the three raw CSV files:

```bash
cd data/raw_csv
python ../../scripts/simple_exclusion.ipynb
cd ../..
```

It writes `data/raw_csv/excluded_records.csv`. This stage is owned by Keyue Li
and defines which invalid answer and tag records are excluded.

## Stage 2: construct question-level variables

From the repository root, run:

```bash
python scripts/02_build_question_features.py \
  --raw-dir data/raw_csv \
  --exclusions data/raw_csv/excluded_records.csv
```

This stage applies, but does not redefine, the Stage 1 exclusions. It creates a
UTF-8 working copy without removing rows, links answers and tags to questions,
and writes:

- `data/processed/question_response_features.parquet`
- `tables/data_summary.csv`

## Stage 3: create exploratory tables and figures

```bash
python scripts/03_make_figures.py
```

This stage reads the Stage 2 Parquet file and writes three CSV summary tables
and three PNG figures. It does not perform further record exclusion.

## Dependencies

```bash
pip install -r requirements.txt
```

Both Stage 2 and Stage 3 are owned and should be explainable by Zi Hu. The code
was developed with generative-AI assistance; that assistance and the human
verification steps must be disclosed in the project AI-use record.

## Optional Week 3 text-model feasibility workflow

This optional workflow tests whether text representations add predictive
information for the preliminary outcome of receiving an answer within 24 hours.
It is not a final measure of answer quality or successful help-seeking.

Install the additional modeling dependencies:

```bash
pip install -r requirements.txt -r requirements-models.txt
```

Then create a deterministic sample from the Stage 2 Parquet file:

```bash
python scripts/04_prepare_model_sample.py \
  --features data/processed/question_response_features.parquet \
  --output data/interim/model_sample_60000.jsonl
```

The script separates natural-language text and code, excludes questions whose
full 24-hour response window is not observable, and writes a JSON metadata file
beside the sample. It does not alter the source Parquet file.

Run the text-model comparison. TF-IDF can run without a MiniLM path; supplying
the local path to a downloaded model adds the frozen-MiniLM comparison:

```bash
python scripts/05_compare_text_models.py \
  --data data/interim/model_sample_60000.jsonl \
  --output tables/text_model_results.json \
  --minilm path/to/local/MiniLM
```

The comparison uses questions from 2008--2014 for training, 2015 for
validation, and 2016 for testing. It uses title, natural-language body, and
tags as text input; raw code text is intentionally excluded in this initial
comparison.

The versioned model-comparison figure can be regenerated from its metric table:

```bash
python scripts/06_make_model_comparison_figure.py
```
