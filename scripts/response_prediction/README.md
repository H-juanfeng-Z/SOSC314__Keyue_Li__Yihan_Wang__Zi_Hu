# Response prediction

`run_encoder_ablation.py` was relocated from `experiments/week4/` without changing its experiment logic. Run from the repository root:

```sh
python scripts/response_prediction/run_encoder_ablation.py --data data/interim/model_sample_60000.jsonl --minilm /path/to/all-MiniLM-L6-v2 --codebert /path/to/codebert-base --out data/interim/encoder_comparison --n 60000 --batch-size 16
```

This compares frozen encoders for the 24-hour-response outcome, not answer helpfulness. Supply local model weights and the existing model sample. Dependencies include PyTorch, transformers, safetensors, pandas, NumPy and scikit-learn.

Earlier preparation and baseline programs remain in [scripts](../README.md).
