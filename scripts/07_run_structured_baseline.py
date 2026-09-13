#!/usr/bin/env python3
"""Run the interpretable structured logistic-regression baseline only.

This is a self-contained handoff version of the non-text baseline used in
the preliminary Week 3 feasibility comparison. It uses a temporal split:
2008--2014 training, 2015 validation, and 2016 testing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "tag_count", "title_chars", "body_html_chars", "code_block_count",
    "code_element_count", "link_count", "question_mark_count",
    "natural_language_chars", "code_chars",
]
TIME_FEATURES = ["creation_month", "creation_weekday_utc", "creation_hour_utc"]


def score(model, x, y) -> dict[str, float]:
    probability = model.predict_proba(x)[:, 1]
    return {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_json(args.data, lines=True)
    train = frame.creation_year <= 2014
    validation = frame.creation_year == 2015
    test = frame.creation_year == 2016
    y = frame["answered_within_24h"].to_numpy()
    predictors = NUMERIC_FEATURES + TIME_FEATURES

    feature_processing = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("time", OneHotEncoder(handle_unknown="ignore"), TIME_FEATURES),
    ])
    model = Pipeline([
        ("features", feature_processing),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(frame.loc[train, predictors], y[train])

    results = {
        "outcome": "At least one answer within 24 hours of posting",
        "temporal_split": "Train: 2008--2014; validation: 2015; test: 2016",
        "leakage_policy": (
            "Question score and all answer-derived fields are excluded because they "
            "are not available at posting time."
        ),
        "structured_features": {
            "numeric_features": NUMERIC_FEATURES,
            "time_features": TIME_FEATURES,
        },
        "validation_2015": score(model, frame.loc[validation, predictors], y[validation]),
        "test_2016": score(model, frame.loc[test, predictors], y[test]),
    }
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
