# -*- coding: utf-8 -*-
"""
Extract observable characteristics from the full cleaned Stack Overflow
question dataset and produce descriptive statistics.

The script:
1. Reads the full Parquet dataset directly with pandas and PyArrow.
2. Extracts question-level characteristics for every question.
3. Computes descriptive statistics for the extracted characteristics.
4. Saves both the question-level data and descriptive statistics into
   a single Excel workbook with two sheets.

Required packages:
    pip install pandas numpy pyarrow beautifulsoup4 openpyxl

Example:
    python extract_features_full.py \
        --parquet /path/to/questions_clean.parquet \
        --out stackoverflow_question_features.xlsx
"""

import argparse
import time

import numpy as np
import pandas as pd

from feature_extract import extract_features


def describe(name, values, is_binary=False):
    """Return descriptive statistics for one feature."""
    arr = np.asarray(values, dtype=float)

    if len(arr) == 0:
        return {
            "feature": name,
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "max": np.nan,
            "proportion": np.nan,
        }

    if is_binary:
        return {
            "feature": name,
            "n": len(arr),
            "mean": arr.mean(),
            "median": np.nan,
            "std": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "max": arr.max(),
            "proportion": arr.mean(),
        }

    return {
        "feature": name,
        "n": len(arr),
        "mean": arr.mean(),
        "median": np.median(arr),
        "std": arr.std(),
        "p25": np.percentile(arr, 25),
        "p75": np.percentile(arr, 75),
        "max": arr.max(),
        "proportion": np.nan,
    }


def load_questions(path):
    """Read the required question columns from the full Parquet dataset."""
    columns = ["title", "body_html"]
    return pd.read_parquet(path, columns=columns, engine="pyarrow")


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from the full cleaned Stack Overflow dataset."
    )
    parser.add_argument(
        "--parquet",
        required=True,
        help="Path to the cleaned Parquet file.",
    )
    parser.add_argument(
        "--out",
        default="stackoverflow_question_features.xlsx",
        help="Output Excel workbook.",
    )
    args = parser.parse_args()

    start = time.time()

    print("== 1. Reading the full Parquet dataset ==")
    df = load_questions(args.parquet)
    n = len(df)
    print(f"Loaded {n:,} questions.")

    print("\n== 2. Extracting question characteristics ==")
    features = []

    for i, row in df.iterrows():
        features.append(
            extract_features(
                row["title"],
                row["body_html"],
            )
        )

        if (i + 1) % 10000 == 0 or i + 1 == n:
            print(f"  Processed {i + 1:,}/{n:,} questions.")

    feature_df = pd.DataFrame(features)

    # Keep the original title as a convenient identifier for inspection.
    feature_df.insert(0, "title", df["title"].fillna(""))

    print("\n== 3. Computing descriptive statistics ==")

    binary_features = {
        "has_code",
        "has_error_message",
        "has_url",
        "has_list",
    }

    ordered_features = [
        "title_word_count",
        "body_word_count",
        "explanatory_text_length",
        "paragraphs",
        "has_code",
        "code_length",
        "code_text_ratio",
        "has_error_message",
        "has_url",
        "has_list",
        "question_marks",
        "uppercase_ratio",
    ]

    stats = []

    for feature in ordered_features:
        if feature not in feature_df.columns:
            continue

        values = feature_df[feature].dropna().to_numpy()

        # Code length and code-to-text ratio are summarized only
        # among questions that actually contain code.
        if feature in {"code_length", "code_text_ratio"}:
            values = feature_df.loc[
                feature_df["has_code"] == 1, feature
            ].dropna().to_numpy()

        stats.append(
            describe(
                feature,
                values,
                is_binary=feature in binary_features,
            )
        )

    stats_df = pd.DataFrame(stats)

    print("\n===== Feature distribution =====")
    for row in stats:
        if not np.isnan(row["proportion"]):
            print(
                f'{row["feature"]:28s} | '
                f'proportion={row["proportion"]:.3f}'
            )
        else:
            print(
                f'{row["feature"]:28s} | '
                f'mean={row["mean"]:.2f} '
                f'median={row["median"]:.2f} '
                f'std={row["std"]:.2f} '
                f'p25={row["p25"]:.2f} '
                f'p75={row["p75"]:.2f} '
                f'max={row["max"]:.0f}'
            )

    print("\n== 4. Saving results ==")

    # Save question-level features and descriptive statistics
    # together in one Excel workbook instead of separate CSV files.
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        feature_df.to_excel(
            writer,
            sheet_name="Question Features",
            index=False,
        )
        stats_df.to_excel(
            writer,
            sheet_name="Descriptive Statistics",
            index=False,
        )

    print(f"Saved results to: {args.out}")
    print(f"Total runtime: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
