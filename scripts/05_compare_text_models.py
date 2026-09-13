#!/usr/bin/env python3
"""Compare TF-IDF and frozen MiniLM text representations on a temporal split."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline


def evaluate(model, x_train, y_train, splits: list[tuple[str, object, np.ndarray]]) -> dict:
    started = time.perf_counter()
    model.fit(x_train, y_train)
    result = {"fit_seconds": time.perf_counter() - started, "splits": {}}
    for name, x, y in splits:
        probability = model.predict_proba(x)[:, 1]
        result["splits"][name] = {
            "n": int(len(y)),
            "positive_rate": float(np.mean(y)),
            "roc_auc": float(roc_auc_score(y, probability)),
            "average_precision": float(average_precision_score(y, probability)),
        }
    return result


def encode_minilm(texts, model_path: Path, batch_size: int, max_length: int):
    import torch
    from transformers import AutoModel, AutoTokenizer

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False, use_safetensors=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    vectors, truncated = [], 0
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            token_lengths = tokenizer(batch, truncation=False)["input_ids"]
            truncated += sum(len(tokens) > max_length for tokens in token_lengths)
            encoded = tokenizer(
                batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            ).to(device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            vectors.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy())
    return np.concatenate(vectors), {"device": str(device), "max_length": max_length, "truncated": int(truncated)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minilm", type=Path, help="Local path to a downloaded MiniLM model.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_json(args.data, lines=True)
    frame["model_text"] = (
        frame["title"].fillna("") + "\n" + frame["natural_language_text"].fillna("")
        + "\nTAGS " + frame["tags"].fillna("")
    )
    train = frame.creation_year <= 2014
    validation = frame.creation_year == 2015
    test = frame.creation_year == 2016
    y = frame.answered_within_24h.to_numpy()
    splits = [("validation_2015", validation), ("test_2016", test)]

    tfidf = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=3, max_features=60000, ngram_range=(1, 2), sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")),
    ])
    results = {
        "outcome": "At least one answer within 24 hours of posting",
        "temporal_split": "Train: 2008--2014; validation: 2015; test: 2016",
        "input": "Question title + natural-language body + tags; raw code text is excluded in this initial comparison.",
        "models": [{
            "name": "tfidf_logistic",
            "method": "TF-IDF word and word-pair features followed by logistic regression",
            **evaluate(
                tfidf, frame.loc[train, "model_text"], y[train],
                [(name, frame.loc[mask, "model_text"], y[mask]) for name, mask in splits],
            ),
        }],
    }

    if args.minilm:
        started = time.perf_counter()
        vectors, encoding = encode_minilm(
            frame.model_text.tolist(), args.minilm, args.batch_size, args.max_length
        )
        encoding["seconds"] = time.perf_counter() - started
        minilm = LogisticRegression(max_iter=1000, class_weight="balanced")
        results["models"].append({
            "name": "minilm_frozen_logistic",
            "method": "Frozen MiniLM embeddings followed by logistic regression",
            **evaluate(minilm, vectors[train], y[train], [(name, vectors[mask], y[mask]) for name, mask in splits]),
            "encoding": encoding,
        })

    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
