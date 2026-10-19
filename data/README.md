# Data access and provenance

## Original archive

Source: [Python Questions from Stack Overflow (Kaggle)](https://www.kaggle.com/datasets/stackoverflow/pythonquestions). The project uses the historical 2008–2016 question/answer snapshot, not a live crawl. Download and extract the archive using your own authorized access if Kaggle requests sign-in. Keep its attribution and follow the source terms; the repository does not supply credentials or claim ownership of Stack Overflow posts.

Create `data/raw_csv/` in the repository root and place these files there, without modifying them:

| File | Required header columns | Unit |
| --- | --- | --- |
| `Questions.csv` | `Id,OwnerUserId,CreationDate,Score,Title,Body` | One question |
| `Answers.csv` | `Id,OwnerUserId,CreationDate,ParentId,Score,Body` | One answer; `ParentId` links to a question |
| `Tags.csv` | `Id,Tag` | One question–tag association |

CSV filenames are case-sensitive on Linux. Do not substitute current API posts or a differently filtered archive and assume unchanged samples. HTML may contain non-UTF-8 bytes: screening and the normalized working copy use replacement decoding, with the source kept unchanged.

```sh
python scripts/inspect_raw_data.py --raw-dir data/raw_csv --out data/interim/raw_inventory.json
```

This records SHA256 hashes, row counts, headers, file sizes and replacement-character counts. It validates the input contract but cannot certify a new download as identical to a historical file when no historical raw hash is available.

## Processing order

1. Run `scripts/simple_exclusion.ipynb` using Python from `data/raw_csv/`. Despite its extension it is a plain Python script. It reports data checks and writes the answer/tag exclusion manifest; the raw files are not overwritten.
2. Run `scripts/02_build_question_features.py` from the root. It applies that manifest, normalizes a separate UTF-8 working copy, joins answers/tags and writes one row per question to `data/processed/question_response_features.parquet`. Invalid joins or remaining time inversions stop the process rather than silently dropping new records.
3. Run `scripts/03_make_figures.py` for the original exploratory tables/figures.
4. For schema-compatible question/answer body inputs, run `scripts/export_annotation_inputs.py` after Stage 2. This is an explicitly documented reconstruction/export, not a claim that the archived team Parquet files have identical bytes, order or historical filtering. Preserve a fresh output directory and its generated provenance.

The historical topic/annotation workflows may additionally require the archived `questions_clean.parquet`, `answers_clean.parquet`, sample selections, model directories and classification assignments. Their contracts and paths are in [the reproduction guide](../docs/REPRODUCIBILITY.md). Current API recovery of edited/deleted posts is not an exact replacement for the historical text. Never invent missing text, labels or zero scores.

## Public derived inputs

- [Unified scores](../tables/reusable_evaluation/unified_v1/README.txt): 5,000 unique first-answer coarse records, a nested 1,000-question fine sample and its matched coarse subset, including invalid/skip/null status.
- [Oct13 inputs](../tables/reusable_evaluation/oct13/): question IDs, scores and seven overlapping intent decisions with manifests and verification code.
- [Report data](../final_report/report_data.json): audited estimates, denominators and interval endpoints used by the notebook.
- [Prompts](../final_report/qwen_prompts.json), [actual output example](../final_report/qwen_example.json), figure support and provenance under `final_report/`.

These are sufficient for the documented frozen-data checks and report replay without raw data access. They are not the full raw archive or every private inference log. Full re-inference additionally requires the historical selection/input snapshots and appropriate local weights; their absence and unpinned weight/text history constrain exact reproduction. The report describes these limits rather than claiming unrestricted end-to-end bitwise reproduction.

Raw posts, intermediate Parquet, weights and generated data remain ignored by Git. Do not upload credentials, personal access tokens, or a large raw archive as a workaround for missing access.
