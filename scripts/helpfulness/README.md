# First-answer helpfulness

Relocated from `experiments/week4/`; annotation rules and sampling logic are unchanged. Commands below run from the repository root.

```sh
python scripts/helpfulness/prepare_helpfulness.py --processed-dir data/processed --out data/interim/helpfulness
python scripts/helpfulness/run_helpfulness_v3.py --data data/interim/helpfulness/blinded_pairs.json --model /path/to/Qwen2.5-Coder-7B-Instruct --out data/interim/helpfulness/results/helpfulness_7b
python scripts/helpfulness/summarize_round2.py --root data/interim/helpfulness
```

Preparation requires the existing `questions_clean.parquet` and `answers_clean.parquet`, not just the question-response feature table. See the project experiment instructions for their schema. Supply local model weights. Preparation uses DuckDB; inference uses PyTorch, transformers and safetensors. Summarization also uses NumPy and scikit-learn.

The summary script retains historical result names `results/helpfulness_{15b,3b,7b}` and optional `results/encoder_60000` below the supplied root. The two-prompt pilot evaluates first-answer helpfulness; output validation is not correctness. No raw data or model weights are bundled.
