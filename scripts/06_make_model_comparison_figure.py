#!/usr/bin/env python3
"""Render the preliminary model-comparison SVG from its versioned metrics table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=Path("tables/model_comparison_metrics.csv"))
    parser.add_argument("--output", type=Path, default=Path("figures/model_comparison_roc_auc.svg"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.metrics.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 3:
        raise ValueError("This three-model figure expects exactly three rows in the metrics table.")

    width, height = 1476, 936
    left, right, top, bottom = 150, 70, 170, 230
    plot_width, plot_height = width - left - right, height - top - bottom
    y_min, y_max = 0.5, 0.8

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    labels = [
        ("Structured", "logistic regression"),
        ("TF-IDF +", "logistic regression"),
        ("MiniLM embeddings +", "logistic regression"),
    ]
    values = [float(row["roc_auc"]) for row in rows]
    colors = ["#93A8AC", "#4F81BD", "#1F4E79"]
    centers = [left + plot_width * position for position in (1 / 6, 3 / 6, 5 / 6)]
    bar_width = 190
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text { font-family: Arial, sans-serif; fill: #1f2933; } .axis { font-size: 26px; } .label { font-size: 23px; } .note { font-size: 20px; fill: #4b5563; } .value { font-size: 25px; font-weight: bold; }</style>',
        '<text x="738" y="70" text-anchor="middle" font-size="34" font-weight="bold">Preliminary text-model comparison for early response prediction</text>',
    ]
    for tick in (0.5, 0.6, 0.7, 0.8):
        y = y_position(tick)
        dash = ' stroke-dasharray="8 8"' if tick == 0.5 else ''
        color = "#555555" if tick == 0.5 else "#d9dee3"
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="{color}" stroke-width="2"{dash}/>')
        svg.append(f'<text x="{left-20}" y="{y+8:.1f}" text-anchor="end" class="axis">{tick:.1f}</text>')
    svg.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#374151" stroke-width="3"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#374151" stroke-width="3"/>',
        f'<text x="45" y="{top + plot_height/2:.1f}" class="axis" transform="rotate(-90 45 {top + plot_height/2:.1f})">ROC-AUC</text>',
    ])
    for center, label, value, color in zip(centers, labels, values, colors):
        y = y_position(value)
        svg.extend([
            f'<rect x="{center-bar_width/2:.1f}" y="{y:.1f}" width="{bar_width}" height="{height-bottom-y:.1f}" rx="3" fill="{color}"/>',
            f'<text x="{center:.1f}" y="{y-18:.1f}" text-anchor="middle" class="value">{value:.3f}</text>',
            f'<text x="{center:.1f}" y="{height-bottom+42}" text-anchor="middle" class="label">{label[0]}</text>',
            f'<text x="{center:.1f}" y="{height-bottom+70}" text-anchor="middle" class="label">{label[1]}</text>',
        ])
    svg.extend([
        f'<line x1="{width-right-370}" y1="112" x2="{width-right-300}" y2="112" stroke="#555555" stroke-width="3" stroke-dasharray="8 8"/>',
        f'<text x="{width-right-285}" y="120" class="note">Random ranking (AUC = 0.50)</text>',
        f'<text x="{width/2}" y="{height-55}" text-anchor="middle" class="note">Outcome: at least one answer within 24 hours. Temporal test set: 2016 (n = 12,738).</text>',
        '</svg>',
    ])
    args.output.write_text("\n".join(svg), encoding="utf-8")


if __name__ == "__main__":
    main()
