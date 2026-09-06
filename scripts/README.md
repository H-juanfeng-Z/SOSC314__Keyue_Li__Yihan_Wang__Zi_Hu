# Reproducible data workflow

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
python ../../simple_exclusion.ipynb
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
