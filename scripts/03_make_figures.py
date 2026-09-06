#!/usr/bin/env python3
"""Stage 3: create exploratory tables and figures from Stage 2 output.

This script does not clean or exclude records. It reads the question-level
feature table produced by ``02_build_question_features.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def response_distribution(
    con: duckdb.DuckDBPyConnection,
    source: Path,
    tables: Path,
    figures: Path,
) -> None:
    frame = con.execute(
        """
        WITH classified AS (
          SELECT CASE
            WHEN first_response_minutes < 1 THEN 'Under 1 min'
            WHEN first_response_minutes < 5 THEN '1-5 min'
            WHEN first_response_minutes < 15 THEN '5-15 min'
            WHEN first_response_minutes < 60 THEN '15-60 min'
            WHEN first_response_minutes < 180 THEN '1-3 h'
            WHEN first_response_minutes < 720 THEN '3-12 h'
            WHEN first_response_minutes < 1440 THEN '12-24 h'
            WHEN first_response_minutes < 10080 THEN '1-7 d'
            WHEN first_response_minutes < 43200 THEN '7-30 d'
            ELSE '30+ d'
          END AS response_bin,
          CASE
            WHEN first_response_minutes < 1 THEN 1
            WHEN first_response_minutes < 5 THEN 2
            WHEN first_response_minutes < 15 THEN 3
            WHEN first_response_minutes < 60 THEN 4
            WHEN first_response_minutes < 180 THEN 5
            WHEN first_response_minutes < 720 THEN 6
            WHEN first_response_minutes < 1440 THEN 7
            WHEN first_response_minutes < 10080 THEN 8
            WHEN first_response_minutes < 43200 THEN 9
            ELSE 10
          END AS bin_order
          FROM read_parquet(?)
          WHERE received_answer
        )
        SELECT
            response_bin,
            count(*) AS question_count,
            count(*) * 100.0 / sum(count(*)) OVER () AS percent_of_answered_questions
        FROM classified
        GROUP BY response_bin, bin_order
        ORDER BY bin_order
        """,
        [str(source)],
    ).df()
    frame.to_csv(tables / "first_response_time_bins.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(frame["response_bin"], frame["question_count"], color="#3366a8")
    ax.set_title("Distribution of Time to First Answer")
    ax.set_xlabel("Time from Question to First Answer")
    ax.set_ylabel("Number of Answered Questions")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "first_response_time_distribution.png", dpi=200)
    plt.close(fig)


def response_by_year(
    con: duckdb.DuckDBPyConnection,
    source: Path,
    tables: Path,
    figures: Path,
) -> None:
    frame = con.execute(
        """
        SELECT
            creation_year,
            count(*) AS questions,
            avg(received_answer::INTEGER) * 100 AS percent_ever_answered,
            avg(answered_within_24h::INTEGER)
                FILTER (WHERE received_answer) * 100
                AS percent_answered_within_24h_among_answered,
            median(first_response_minutes)
                FILTER (WHERE received_answer) AS median_first_response_minutes
        FROM read_parquet(?)
        GROUP BY creation_year
        ORDER BY creation_year
        """,
        [str(source)],
    ).df()
    frame.to_csv(tables / "response_by_year.csv", index=False)

    fig, left = plt.subplots(figsize=(9, 5.5))
    right = left.twinx()
    left.plot(
        frame["creation_year"],
        frame["median_first_response_minutes"],
        marker="o",
        color="#3366a8",
        label="Median first-response time",
    )
    right.plot(
        frame["creation_year"],
        frame["percent_ever_answered"],
        marker="s",
        color="#b35c1e",
        label="Ever answered",
    )
    left.set_title("Response Speed and Answer Coverage by Question Year")
    left.set_xlabel("Question Creation Year")
    left.set_ylabel("Median First-Response Time (Minutes)")
    right.set_ylabel("Questions Ever Answered (%)")
    left.grid(alpha=0.25)
    lines = left.lines + right.lines
    left.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    fig.savefig(figures / "response_trends_by_year.png", dpi=200)
    plt.close(fig)


def response_by_body_length(
    con: duckdb.DuckDBPyConnection,
    source: Path,
    tables: Path,
    figures: Path,
) -> None:
    frame = con.execute(
        """
        WITH binned AS (
          SELECT *, ntile(10) OVER (ORDER BY body_html_chars) AS body_length_decile
          FROM read_parquet(?)
        )
        SELECT
            body_length_decile,
            min(body_html_chars) AS min_body_html_chars,
            max(body_html_chars) AS max_body_html_chars,
            count(*) AS questions,
            avg(received_answer::INTEGER) * 100 AS percent_ever_answered,
            median(first_response_minutes)
                FILTER (WHERE received_answer) AS median_first_response_minutes
        FROM binned
        GROUP BY body_length_decile
        ORDER BY body_length_decile
        """,
        [str(source)],
    ).df()
    frame.to_csv(tables / "response_by_body_length_decile.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(
        frame["body_length_decile"],
        frame["median_first_response_minutes"],
        marker="o",
        color="#3366a8",
    )
    ax.set_title("Median First-Response Time by Question-Body Length")
    ax.set_xlabel("HTML-Body-Length Decile (Shorter to Longer)")
    ax.set_ylabel("Median First-Response Time (Minutes)")
    ax.set_xticks(frame["body_length_decile"])
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "response_by_body_length.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/question_response_features.parquet"),
    )
    parser.add_argument("--tables-dir", type=Path, default=Path("tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    if not args.features.exists():
        raise FileNotFoundError(
            f"Stage 2 output not found: {args.features}. "
            "Run scripts/02_build_question_features.py first."
        )
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    response_distribution(con, args.features, args.tables_dir, args.figures_dir)
    response_by_year(con, args.features, args.tables_dir, args.figures_dir)
    response_by_body_length(con, args.features, args.tables_dir, args.figures_dir)
    con.close()
    print(f"Tables: {args.tables_dir}")
    print(f"Figures: {args.figures_dir}")


if __name__ == "__main__":
    main()
