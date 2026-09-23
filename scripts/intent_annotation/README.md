# Question-intent annotation

The intent scripts were relocated together from `experiments/week4/` to preserve their sibling imports. Taxonomy, prompts, validation and inference logic are unchanged. Run from the repository root:

```sh
python scripts/intent_annotation/prepare_intent_round3.py --pairs data/interim/helpfulness/blinded_pairs.json --model-sample data/interim/model_sample_60000.jsonl --out data/interim/intent_round3
python scripts/intent_annotation/run_intent_round3.py --data data/interim/intent_round3/cases.json --model /path/to/Qwen2.5-Coder-7B-Instruct --out data/interim/intent_round3_downloaded/results/7b-examples --arm examples
python scripts/intent_annotation/run_intent_round4.py --data data/interim/intent_round3/cases.json --model /path/to/Qwen2.5-Coder-7B-Instruct --out data/interim/intent_round4_downloaded/results/7b-shared_json --arm examples --variant shared_json
python scripts/intent_annotation/aggregate_intent_results.py --root data/interim
```

Supply local weights and PyTorch, transformers and safetensors. For the published comparison, repeat with 3B and 7B and all three round-four variants (`shared_json`, `isolated`, `isolated_json`), each in a fresh output directory. `run_intent_pilot.py` and `run_intent_binary.py` also supply imported taxonomy and validator definitions; no preliminary rerun is required.

Aggregation keeps the historical relative log layout beneath `--root` and writes `report_outputs/figure_data.json` there. `audit_intent_cpu.py` instead expects `intent_round3/data/{cases,diagnostic_constraints}.json` and `intent_round{3,4}/results/*/calls.jsonl` beneath its root. These layouts are explicit input contracts, not repository locations.

Outputs are provisional multi-label annotations. Formatting validity and synthetic constraints are not real-world accuracy. Raw logs and model weights are not included.
