#!/usr/bin/env python3
"""Summarize three independent FusionRS quality-audit forms."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from itertools import combinations
from pathlib import Path

ORDINAL_FIELDS = (
    "scene_consistency_1_to_5",
    "ir_style_plausibility_1_to_5",
    "original_caption_consistency_1_to_5",
    "ir_caption_consistency_1_to_5",
)
BINARY_FIELDS = (
    "unsupported_physical_claim_0_or_1",
    "usable_for_pretraining_0_or_1",
)
ALL_FIELDS = ORDINAL_FIELDS + BINARY_FIELDS


def load_form(path: Path) -> dict[str, dict[str, int | None]]:
    rows: dict[str, dict[str, int | None]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            item_id = str(row.get("item_id", "")).strip()
            if not item_id:
                raise ValueError(f"{path}: row without item_id")
            if item_id in rows:
                raise ValueError(f"{path}: duplicate item_id {item_id}")
            parsed: dict[str, int | None] = {}
            for field in ALL_FIELDS:
                value = str(row.get(field, "")).strip()
                parsed[field] = int(value) if value else None
                if parsed[field] is not None:
                    allowed = range(1, 6) if field in ORDINAL_FIELDS else range(0, 2)
                    if parsed[field] not in allowed:
                        raise ValueError(
                            f"{path}: {item_id} has invalid {field}={value}"
                        )
            rows[item_id] = parsed
    return rows


def quadratic_weighted_kappa(
    left: dict[str, int | None],
    right: dict[str, int | None],
    categories: int,
) -> float | None:
    pairs = [
        (left[item_id], right[item_id])
        for item_id in sorted(set(left) & set(right))
        if left[item_id] is not None and right[item_id] is not None
    ]
    if not pairs:
        return None
    observed = [[0.0] * categories for _ in range(categories)]
    left_hist = [0.0] * categories
    right_hist = [0.0] * categories
    for a, b in pairs:
        assert a is not None and b is not None
        ai = a - 1 if categories == 5 else a
        bi = b - 1 if categories == 5 else b
        observed[ai][bi] += 1
        left_hist[ai] += 1
        right_hist[bi] += 1
    count = float(len(pairs))
    max_distance = float((categories - 1) ** 2)
    observed_disagreement = 0.0
    expected_disagreement = 0.0
    for i in range(categories):
        for j in range(categories):
            weight = ((i - j) ** 2) / max_distance
            observed_disagreement += weight * observed[i][j] / count
            expected_disagreement += weight * (left_hist[i] * right_hist[j]) / (
                count * count
            )
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else None
    return 1.0 - observed_disagreement / expected_disagreement


def mean_ci(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.fmean(values)
    if len(values) < 2:
        margin = 0.0
    else:
        margin = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    if len(args.annotator) != 3:
        raise ValueError("exactly three --annotator files are required")

    forms = [load_form(path) for path in args.annotator]
    item_sets = [set(form) for form in forms]
    if not all(items == item_sets[0] for items in item_sets[1:]):
        raise ValueError("annotator forms do not contain identical item IDs")

    complete_rows = [
        sum(all(values[field] is not None for field in ALL_FIELDS) for values in form.values())
        for form in forms
    ]
    field_summary = {}
    agreement = {}
    for field in ALL_FIELDS:
        values = [
            value
            for form in forms
            for value in (row[field] for row in form.values())
            if value is not None
        ]
        field_summary[field] = mean_ci(values)
        categories = 5 if field in ORDINAL_FIELDS else 2
        pair_scores = []
        for left_index, right_index in combinations(range(3), 2):
            score = quadratic_weighted_kappa(
                {item: forms[left_index][item][field] for item in item_sets[0]},
                {item: forms[right_index][item][field] for item in item_sets[0]},
                categories,
            )
            pair_scores.append(
                {
                    "annotators": [left_index + 1, right_index + 1],
                    "quadratic_weighted_kappa": score,
                }
            )
        agreement[field] = pair_scores

    result = {
        "annotators": [str(path) for path in args.annotator],
        "items": len(item_sets[0]),
        "complete_rows_by_annotator": complete_rows,
        "all_complete": all(count == len(item_sets[0]) for count in complete_rows),
        "field_summary": field_summary,
        "pairwise_agreement": agreement,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# FusionRS human-audit summary",
        "",
        f"- Items: {result['items']}",
        f"- Complete rows by annotator: {complete_rows}",
        f"- Release gate passed: {'yes' if result['all_complete'] else 'no'}",
        "",
        "| Field | N | Mean | 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for field, stats in field_summary.items():
        mean = "NA" if stats["mean"] is None else f"{stats['mean']:.3f}"
        interval = (
            "NA"
            if stats["ci95_low"] is None
            else f"[{stats['ci95_low']:.3f}, {stats['ci95_high']:.3f}]"
        )
        lines.append(f"| {field} | {stats['n']} | {mean} | {interval} |")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
