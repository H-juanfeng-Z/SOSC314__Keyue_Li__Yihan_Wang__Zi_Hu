#!/usr/bin/env python3
"""Stage 2: construct one analysis row per Stack Overflow question.

This script deliberately does not decide which records should be excluded.
It consumes the exclusion manifest produced by Keyue's preliminary screening
script (``excluded_records.csv``), applies those decisions, and constructs the
question-level variables used in the exploratory analysis.

Run from the repository root after ``simple_exclusion.ipynb`` has produced
``excluded_records.csv``::

    python scripts/02_build_question_features.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import duckdb
import pandas as pd


EXPECTED_COLUMNS = {
    "Questions.csv": 6,
    "Answers.csv": 6,
    "Tags.csv": 2,
}


def allow_large_csv_fields() -> None:
    """Raise Python's CSV field limit for long Stack Overflow posts."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def create_utf8_working_copies(raw_dir: Path, working_dir: Path) -> None:
    """Replace invalid UTF-8 bytes without excluding or otherwise filtering rows.

    The Kaggle files contain a small number of invalid UTF-8 byte sequences.
    Keyue's screening script reads them with ``encoding_errors='replace'``.
    DuckDB is stricter, so this stage persists the same decoded representation
    as valid UTF-8 before constructing variables. Any structurally malformed
    CSV row stops the pipeline instead of being silently dropped.
    """
    allow_large_csv_fields()
    working_dir.mkdir(parents=True, exist_ok=True)

    for filename, expected_fields in EXPECTED_COLUMNS.items():
        source = raw_dir / filename
        destination = working_dir / filename
        if not source.exists():
            raise FileNotFoundError(f"Required input not found: {source}")

        record_count = 0
        replacement_count = 0
        with source.open(
            "r", encoding="utf-8-sig", errors="replace", newline=""
        ) as input_stream, destination.open(
            "w", encoding="utf-8", newline=""
        ) as output_stream:
            reader = csv.reader(input_stream)
            writer = csv.writer(output_stream, lineterminator="\n")
            for record_number, row in enumerate(reader, start=1):
                if len(row) != expected_fields:
                    raise ValueError(
                        f"Malformed {filename} record {record_number}: "
                        f"expected {expected_fields} fields, found {len(row)}"
                    )
                replacement_count += sum(value.count("�") for value in row)
                writer.writerow(row)
                record_count += 1

        print(
            f"UTF-8 working copy: {filename}, "
            f"{record_count - 1} data rows, {replacement_count} replacement characters"
        )


def read_exclusion_manifest(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two sections written by the preliminary screening script."""
    answer_rows: list[dict[str, object]] = []
    tag_rows: list[dict[str, object]] = []
    section = "answers"

    with path.open(encoding="utf-8", newline="") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("# Tags"):
                section = "tags"
                continue
            if line.startswith("#"):
                continue

            fields = next(csv.reader([line]))
            if fields[0] in {"AnswerId", "QuestionId"}:
                continue

            if section == "answers":
                if len(fields) != 4:
                    raise ValueError(f"Unexpected excluded-answer row: {fields}")
                answer_rows.append(
                    {
                        "answer_id": int(fields[0]),
                        "question_id": int(fields[1]),
                        "answer_created_at": fields[2],
                        "question_created_at": fields[3],
                    }
                )
            else:
                if len(fields) != 2:
                    raise ValueError(f"Unexpected excluded-tag row: {fields}")
                tag_rows.append(
                    {"question_id": int(fields[0]), "tag": fields[1].strip()}
                )

    answers = pd.DataFrame(
        answer_rows,
        columns=[
            "answer_id",
            "question_id",
            "answer_created_at",
            "question_created_at",
        ],
    )
    tags = pd.DataFrame(tag_rows, columns=["question_id", "tag"])

    if answers["answer_id"].duplicated().any():
        raise ValueError("The exclusion manifest contains duplicate answer IDs")
    if tags.duplicated(["question_id", "tag"]).any():
        raise ValueError("The exclusion manifest contains duplicate tag rows")
    return answers, tags


def create_raw_views(con: duckdb.DuckDBPyConnection, raw_dir: Path) -> None:
    """Expose the original CSV files without applying new exclusion rules."""
    paths = {
        "raw_questions": raw_dir / "Questions.csv",
        "raw_answers": raw_dir / "Answers.csv",
        "raw_tags": raw_dir / "Tags.csv",
    }
    for table, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")
        sql_path = path.resolve().as_posix().replace("'", "''")
        con.execute(
            f"""
            CREATE TEMP VIEW {table} AS
            SELECT * FROM read_csv_auto(
                '{sql_path}', header=true, all_varchar=true,
                parallel=false, sample_size=20000
            )
            """
        )


def construct_features(
    con: duckdb.DuckDBPyConnection,
    output_path: Path,
) -> None:
    """Apply Stage 1 exclusions and construct question-level outcomes."""
    con.execute(
        """
        CREATE TEMP TABLE questions AS
        SELECT
            CAST(Id AS BIGINT) AS question_id,
            TRY_CAST(OwnerUserId AS BIGINT) AS owner_user_id,
            CAST(CreationDate AS TIMESTAMPTZ) AS question_created_at,
            CAST(Score AS INTEGER) AS question_score,
            Title AS title,
            Body AS body_html
        FROM raw_questions
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE answers AS
        SELECT
            CAST(a.Id AS BIGINT) AS answer_id,
            CAST(a.CreationDate AS TIMESTAMPTZ) AS answer_created_at,
            CAST(a.ParentId AS BIGINT) AS question_id,
            CAST(a.Score AS INTEGER) AS answer_score
        FROM raw_answers AS a
        LEFT JOIN excluded_answers AS x
          ON CAST(a.Id AS BIGINT) = x.answer_id
        WHERE x.answer_id IS NULL
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE tags AS
        SELECT DISTINCT
            CAST(t.Id AS BIGINT) AS question_id,
            lower(trim(t.Tag)) AS tag
        FROM raw_tags AS t
        LEFT JOIN excluded_tags AS x
          ON CAST(t.Id AS BIGINT) = x.question_id
         AND trim(t.Tag) = x.tag
        WHERE x.question_id IS NULL
        """
    )

    # These are interface checks, not additional screening decisions. If one
    # fails, Stage 1 and Stage 2 disagree and the pipeline stops visibly.
    invalid_answers = con.execute(
        """
        SELECT count(*)
        FROM answers AS a
        LEFT JOIN questions AS q USING (question_id)
        WHERE q.question_id IS NULL
           OR a.answer_created_at < q.question_created_at
        """
    ).fetchone()[0]
    orphan_tags = con.execute(
        """
        SELECT count(*)
        FROM tags AS t
        LEFT JOIN questions AS q USING (question_id)
        WHERE q.question_id IS NULL
        """
    ).fetchone()[0]
    if invalid_answers or orphan_tags:
        raise RuntimeError(
            "Stage 1 exclusions are incomplete: "
            f"{invalid_answers} invalid answers and {orphan_tags} orphan tags remain"
        )

    con.execute(
        """
        CREATE TEMP TABLE question_response_features AS
        WITH answer_summary AS (
            SELECT
                question_id,
                count(*) AS answer_count,
                min(answer_created_at) AS first_answer_at,
                arg_min(answer_id, answer_created_at) AS first_answer_id,
                arg_min(answer_score, answer_created_at) AS first_answer_score,
                max(answer_score) AS maximum_answer_score
            FROM answers
            GROUP BY question_id
        ),
        tag_summary AS (
            SELECT
                question_id,
                count(*) AS tag_count,
                string_agg(tag, '|' ORDER BY tag) AS tags
            FROM tags
            GROUP BY question_id
        )
        SELECT
            q.question_id,
            q.owner_user_id,
            q.question_created_at,
            q.question_score,
            q.title,
            q.body_html,
            coalesce(a.answer_count, 0) AS answer_count,
            a.first_answer_id,
            a.first_answer_at,
            a.first_answer_score,
            a.maximum_answer_score,
            a.first_answer_at IS NOT NULL AS received_answer,
            CASE WHEN a.first_answer_at IS NOT NULL THEN
                date_diff('second', q.question_created_at, a.first_answer_at) / 60.0
            END AS first_response_minutes,
            CASE WHEN a.first_answer_at IS NOT NULL THEN
                date_diff('second', q.question_created_at, a.first_answer_at) <= 86400
            END AS answered_within_24h,
            coalesce(t.tag_count, 0) AS tag_count,
            t.tags,
            length(coalesce(q.title, '')) AS title_chars,
            length(coalesce(q.body_html, '')) AS body_html_chars,
            array_length(regexp_extract_all(coalesce(q.body_html, ''), '<pre><code>'))
                AS code_block_count,
            array_length(regexp_extract_all(coalesce(q.body_html, ''), '<a '))
                AS link_count,
            array_length(regexp_extract_all(
                coalesce(q.title, '') || ' ' || coalesce(q.body_html, ''), '\\?'
            )) AS question_mark_count,
            extract(year FROM q.question_created_at)::INTEGER AS creation_year,
            extract(month FROM q.question_created_at)::INTEGER AS creation_month,
            extract(dow FROM q.question_created_at)::INTEGER AS creation_weekday_utc,
            extract(hour FROM q.question_created_at)::INTEGER AS creation_hour_utc
        FROM questions AS q
        LEFT JOIN answer_summary AS a USING (question_id)
        LEFT JOIN tag_summary AS t USING (question_id)
        """
    )
    con.execute(
        "COPY question_response_features TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
        [str(output_path)],
    )


def save_summary(
    con: duckdb.DuckDBPyConnection,
    excluded_answers: pd.DataFrame,
    excluded_tags: pd.DataFrame,
    output_path: Path,
) -> None:
    values = {
        "raw_questions": con.execute("SELECT count(*) FROM raw_questions").fetchone()[0],
        "raw_answers": con.execute("SELECT count(*) FROM raw_answers").fetchone()[0],
        "raw_tags": con.execute("SELECT count(*) FROM raw_tags").fetchone()[0],
        "excluded_answers_from_stage_1": len(excluded_answers),
        "excluded_tags_from_stage_1": len(excluded_tags),
        "retained_questions": con.execute("SELECT count(*) FROM questions").fetchone()[0],
        "retained_answers": con.execute("SELECT count(*) FROM answers").fetchone()[0],
        "retained_tags": con.execute("SELECT count(*) FROM tags").fetchone()[0],
        "questions_with_answer": con.execute(
            "SELECT count(*) FROM question_response_features WHERE received_answer"
        ).fetchone()[0],
        "questions_without_answer": con.execute(
            "SELECT count(*) FROM question_response_features WHERE NOT received_answer"
        ).fetchone()[0],
    }
    pd.DataFrame(values.items(), columns=["metric", "value"]).to_csv(
        output_path, index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw_csv"))
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path("data/interim/normalized_utf8"),
        help="Temporary UTF-8 copies; no records are excluded at this step",
    )
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=Path("data/raw_csv/excluded_records.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/question_response_features.parquet"),
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("tables/data_summary.csv")
    )
    args = parser.parse_args()

    if not args.exclusions.exists():
        raise FileNotFoundError(
            f"Stage 1 output not found: {args.exclusions}. "
            "Run the preliminary screening script first."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    excluded_answers, excluded_tags = read_exclusion_manifest(args.exclusions)
    create_utf8_working_copies(args.raw_dir, args.working_dir)
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.register("excluded_answers", excluded_answers)
    con.register("excluded_tags", excluded_tags)
    create_raw_views(con, args.working_dir)
    construct_features(con, args.output)
    save_summary(con, excluded_answers, excluded_tags, args.summary)
    print(f"Question-level data: {args.output}")
    print(f"Summary table: {args.summary}")
    con.close()


if __name__ == "__main__":
    main()
