#!/usr/bin/env python3
"""Create a deterministic question-level sample for text-model feasibility tests."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import duckdb


CODE_RE = re.compile(r"<code>(.*?)</code>", flags=re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def clean_html(value: str | None) -> str:
    return SPACE_RE.sub(" ", TAG_RE.sub(" ", html.unescape(value or ""))).strip()


def split_body(value: str | None) -> tuple[str, str, int]:
    raw = value or ""
    code_parts = [clean_html(part) for part in CODE_RE.findall(raw)]
    prose = clean_html(CODE_RE.sub(" ", raw))
    return prose, "\n".join(code_parts).strip(), len(code_parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=60000)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    frame = con.execute(
        """
        WITH archive_window AS (
            SELECT greatest(max(question_created_at), max(first_answer_at)) AS observed_through
            FROM read_parquet(?)
        )
        SELECT
            question_id, creation_year, creation_month, creation_weekday_utc,
            creation_hour_utc, title, body_html, tags, tag_count, title_chars,
            body_html_chars, code_block_count, link_count, question_mark_count,
            CASE WHEN received_answer AND first_response_minutes <= 1440 THEN 1 ELSE 0 END
                AS answered_within_24h
        FROM read_parquet(?), archive_window
        WHERE question_created_at <= observed_through - INTERVAL 24 HOURS
        ORDER BY hash(question_id, 314)
        LIMIT ?
        """,
        [str(args.features), str(args.features), args.sample_size],
    ).fetchdf()
    observed_through = con.execute(
        "SELECT greatest(max(question_created_at), max(first_answer_at)) FROM read_parquet(?)",
        [str(args.features)],
    ).fetchone()[0]
    con.close()

    with args.output.open("w", encoding="utf-8") as stream:
        for row in frame.itertuples(index=False):
            prose, code, code_element_count = split_body(row.body_html)
            record = {
                "question_id": int(row.question_id),
                "creation_year": int(row.creation_year),
                "creation_month": int(row.creation_month),
                "creation_weekday_utc": int(row.creation_weekday_utc),
                "creation_hour_utc": int(row.creation_hour_utc),
                "title": row.title or "",
                "natural_language_text": prose,
                "code_text": code,
                "tags": row.tags or "",
                "tag_count": int(row.tag_count),
                "title_chars": int(row.title_chars),
                "body_html_chars": int(row.body_html_chars),
                "code_block_count": int(row.code_block_count),
                "code_element_count": code_element_count,
                "link_count": int(row.link_count),
                "question_mark_count": int(row.question_mark_count),
                "natural_language_chars": len(prose),
                "code_chars": len(code),
                "answered_within_24h": int(row.answered_within_24h),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "rows": len(frame),
        "sampling": "Deterministic ordering by hash(question_id, 314)",
        "outcome": "At least one valid answer within 24 hours of posting",
        "censoring_rule": "Questions posted less than 24 hours before the archive's final observed timestamp are excluded.",
        "observed_through": str(observed_through),
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
