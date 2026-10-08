# Measurement follow-up programs

These programs examine measurement sensitivity, intent-helpfulness associations, answer trajectories and GPT case review. Directory dates identify frozen experiment versions, not separate weekly projects.

Start with the [method and reproduction guide](../../../docs/experiments/measurement_followup.md).

```sh
python scripts/helpfulness/followup/reproduce.py
python scripts/helpfulness/followup/reproduce.py --work-dir data/interim/followup_replay --run
```

The runner verifies compressed and decompressed hashes, creates a fresh disposable workspace, and runs CPU analyses. It never launches inference or contacts a server. It refuses to overwrite an existing workspace. Materialized scripts preserve original relative layouts; do not run archived scripts directly inside this source directory because their frozen data live separately under `tables`.

`run.py` and validation modules support optional inference from frozen tasks; `analyze.py`, `experiment.py`, `batch.py` and `verify.py` perform analyses/checks. `source_prepare.py` files document historical preparation and are not promised standalone executables. `gpt_case_review_20261008/prepare.py` selects review cases; `freeze.py` fixes ratings before unmasking; `analyze_review.py` compares already recorded GPT judgments. None of these scripts calls a GPT API or regenerates GPT judgments.

The full post-review HTML audit needs DuckDB and the original processed Parquet inputs. The smaller metadata-preservation check uses included reviewed snapshots. See the guide for exact layout and boundaries.
