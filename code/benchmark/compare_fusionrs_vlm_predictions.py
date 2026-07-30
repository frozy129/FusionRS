#!/usr/bin/env python3
"""Paired image-cluster bootstrap comparison for FusionRS VLM predictions."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from score_fusionrs_benchmark import (
    bag_f1,
    normalize_answer,
    percentile,
    rouge_l_f1,
)


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("record_key")
        if not isinstance(key, str) or key in rows:
            raise ValueError(f"{path}:{line_number}: duplicate or empty record_key")
        if row.get("status") != "success":
            raise ValueError(f"{path}:{line_number}: unsuccessful prediction")
        rows[key] = row
    return rows


def score(row: dict[str, Any]) -> dict[str, float]:
    prediction = row["prediction"]
    if row["evaluation_type"] == "caption":
        references = row["references"]
        return {
            "rouge_l": max(rouge_l_f1(prediction, ref) for ref in references),
            "lexical_f1": max(
                bag_f1(prediction, ref, remove_stopwords=True)
                for ref in references
            ),
        }
    accepted = row.get("accepted_answers") or [row["canonical_answer"]]
    normalized_prediction = normalize_answer(prediction)
    return {
        "exact_match": float(
            normalized_prediction
            in {normalize_answer(answer) for answer in accepted}
        ),
        "token_f1": max(
            bag_f1(
                normalized_prediction,
                normalize_answer(answer),
                remove_stopwords=False,
            )
            for answer in accepted
        ),
    }


def paired_bootstrap(
    rows: list[dict[str, Any]],
    metric: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    by_image: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_image[row["benchmark_id"]].append(
            row["candidate_scores"][metric] - row["baseline_scores"][metric]
        )
    image_ids = sorted(by_image)
    observed = statistics.fmean(
        value for image_id in image_ids for value in by_image[image_id]
    )
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
        "candidate_minus_baseline": round(observed * 100, 4),
        "ci95": [
            round(percentile(draws, 0.025) * 100, 4),
            round(percentile(draws, 0.975) * 100, 4),
        ],
        "images": len(image_ids),
        "rows": len(rows),
    }


def summarize_group(
    rows: list[dict[str, Any]],
    metrics: tuple[str, ...],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    return {
        metric: paired_bootstrap(rows, metric, iterations, seed)
        for metric in metrics
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    baseline = read_predictions(args.baseline)
    candidate = read_predictions(args.candidate)
    if set(baseline) != set(candidate):
        raise ValueError("prediction record_key sets differ")

    paired = []
    for key in sorted(baseline):
        baseline_row = baseline[key]
        candidate_row = candidate[key]
        for field in (
            "benchmark_id",
            "evaluation_type",
            "task_type",
            "question_type",
        ):
            if baseline_row.get(field) != candidate_row.get(field):
                raise ValueError(f"{key}: mismatched {field}")
        paired.append(
            {
                **candidate_row,
                "baseline_scores": score(baseline_row),
                "candidate_scores": score(candidate_row),
            }
        )

    caption = [row for row in paired if row["evaluation_type"] == "caption"]
    vqa = [row for row in paired if row["evaluation_type"] == "vqa"]
    report: dict[str, Any] = {
        "baseline": args.baseline.stem,
        "candidate": args.candidate.stem,
        "caption": {
            "overall": summarize_group(
                caption,
                ("rouge_l", "lexical_f1"),
                args.bootstrap,
                args.seed,
            ),
            "by_task": {},
        },
        "vqa": {
            "overall": summarize_group(
                vqa,
                ("exact_match", "token_f1"),
                args.bootstrap,
                args.seed,
            ),
            "by_type": {},
        },
    }
    for task in sorted({row["task_type"] for row in caption}):
        report["caption"]["by_task"][task] = summarize_group(
            [row for row in caption if row["task_type"] == task],
            ("rouge_l", "lexical_f1"),
            args.bootstrap,
            args.seed,
        )
    for question_type in sorted({row["question_type"] for row in vqa}):
        report["vqa"]["by_type"][question_type] = summarize_group(
            [row for row in vqa if row["question_type"] == question_type],
            ("exact_match", "token_f1"),
            args.bootstrap,
            args.seed,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"baseline": report["baseline"], "candidate": report["candidate"]}))


if __name__ == "__main__":
    main()
