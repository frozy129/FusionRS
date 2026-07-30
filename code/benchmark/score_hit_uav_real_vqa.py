#!/usr/bin/env python3
"""Score HIT-UAV real-thermal VQA with task-aware deterministic parsing."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
REGIONS = (
    "upper-left",
    "upper-right",
    "lower-left",
    "lower-right",
    "center",
    "upper",
    "lower",
    "left",
    "right",
)
CLASSES = ("other vehicle", "bicycle", "person", "car", "none")


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def parse_prediction(row: dict) -> str | None:
    text = normalize(row.get("prediction", ""))
    question_type = row["question_type"]
    if question_type == "object_presence":
        tokens = text.split()
        for token in tokens:
            if token in {"yes", "no"}:
                return token
        return None
    if question_type == "object_count":
        match = re.search(r"\b\d+\b", text)
        if match:
            return str(int(match.group()))
        for word, value in NUMBER_WORDS.items():
            if re.search(rf"\b{word}\b", text):
                return str(value)
        return None
    if question_type == "group_location":
        canonical = (
            text.replace("top", "upper")
            .replace("bottom", "lower")
            .replace("centre", "center")
        )
        for region in REGIONS:
            if region.replace("-", " ") in canonical:
                return region
        return None
    for class_name in CLASSES:
        if class_name in text:
            return class_name
    return None


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def summarize(rows: list[dict], seed: int, iterations: int) -> dict:
    if not rows:
        return {"rows": 0, "images": 0, "accuracy": None}
    accuracy = statistics.fmean(row["correct"] for row in rows)
    by_image = defaultdict(list)
    for row in rows:
        by_image[row["benchmark_id"]].append(row["correct"])
    image_ids = sorted(by_image)
    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        sampled = [rng.choice(image_ids) for _ in image_ids]
        draws.append(
            statistics.fmean(
                value for image_id in sampled for value in by_image[image_id]
            )
        )
    return {
        "rows": len(rows),
        "images": len(image_ids),
        "accuracy": round(accuracy * 100, 4),
        "accuracy_ci95": [
            round(percentile(draws, 0.025) * 100, 4),
            round(percentile(draws, 0.975) * 100, 4),
        ],
        "parse_failure_rate": round(
            statistics.fmean(row["parsed_answer"] is None for row in rows) * 100,
            4,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    rows = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            parsed = parse_prediction(row)
            row["parsed_answer"] = parsed
            row["correct"] = float(parsed == str(row["canonical_answer"]).lower())
            rows.append(row)

    report = {
        "model_tag": rows[0].get("model_tag", "") if rows else "",
        "overall": summarize(rows, args.seed, args.bootstrap),
        "by_type": {},
        "presence_by_answer": {},
        "presence_by_class": {},
        "confusion": dict(
            Counter(
                f"{row['canonical_answer']}->{row['parsed_answer'] or 'unparsed'}"
                for row in rows
                if row["question_type"] == "object_presence"
            )
        ),
    }
    for question_type in sorted({row["question_type"] for row in rows}):
        report["by_type"][question_type] = summarize(
            [row for row in rows if row["question_type"] == question_type],
            args.seed,
            args.bootstrap,
        )
    for answer in ("yes", "no"):
        report["presence_by_answer"][answer] = summarize(
            [
                row
                for row in rows
                if row["question_type"] == "object_presence"
                and row["canonical_answer"] == answer
            ],
            args.seed,
            args.bootstrap,
        )
    for class_name in sorted(
        {
            row.get("object_class")
            for row in rows
            if row["question_type"] == "object_presence"
        }
    ):
        report["presence_by_class"][class_name] = summarize(
            [
                row
                for row in rows
                if row["question_type"] == "object_presence"
                and row.get("object_class") == class_name
            ],
            args.seed,
            args.bootstrap,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
