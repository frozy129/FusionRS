#!/usr/bin/env python3
"""Score FusionRS Caption/VQA predictions with deterministic bootstrap CIs."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import string
from collections import Counter, defaultdict
from pathlib import Path


ARTICLES = {"a", "an", "the"}
STOPWORDS = ARTICLES | {
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "this",
    "to",
    "with",
}


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def normalize_answer(text: str) -> str:
    text = str(text or "").lower()
    text = text.translate(str.maketrans({char: " " for char in string.punctuation}))
    return " ".join(token for token in text.split() if token not in ARTICLES)


def lcs_length(first: list[str], second: list[str]) -> int:
    previous = [0] * (len(second) + 1)
    for left in first:
        current = [0]
        for index, right in enumerate(second, 1):
            if left == right:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokens(prediction)
    ref_tokens = tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = lcs_length(pred_tokens, ref_tokens)
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def bag_f1(prediction: str, reference: str, remove_stopwords: bool) -> float:
    pred_tokens = tokens(prediction)
    ref_tokens = tokens(reference)
    if remove_stopwords:
        pred_tokens = [token for token in pred_tokens if token not in STOPWORDS]
        ref_tokens = [token for token in ref_tokens if token not in STOPWORDS]
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    common = sum((pred_counts & ref_counts).values())
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_ci(
    rows: list[dict], metric: str, seed: int, iterations: int
) -> list[float | None]:
    by_image: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_image[row["benchmark_id"]].append(row[metric])
    ids = sorted(by_image)
    if not ids:
        return [None, None]
    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        sampled = [rng.choice(ids) for _ in ids]
        values = [
            value for image_id in sampled for value in by_image[image_id]
        ]
        draws.append(statistics.fmean(values))
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def summarize_group(
    rows: list[dict], metrics: list[str], seed: int, iterations: int
) -> dict:
    result = {
        "rows": len(rows),
        "images": len({row["benchmark_id"] for row in rows}),
        "successes": sum(row["status"] == "success" for row in rows),
    }
    for metric in metrics:
        value = statistics.fmean(row[metric] for row in rows) if rows else 0.0
        low, high = bootstrap_ci(rows, metric, seed, iterations)
        result[metric] = round(value * 100, 4)
        result[f"{metric}_ci95"] = [
            round(low * 100, 4) if low is not None else None,
            round(high * 100, 4) if high is not None else None,
        ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    scored = []
    for row in rows:
        prediction = row.get("prediction", "") if row.get("status") == "success" else ""
        if row["evaluation_type"] == "caption":
            references = row["references"]
            row["rouge_l"] = max(
                (rouge_l_f1(prediction, ref) for ref in references),
                default=0.0,
            )
            row["lexical_f1"] = max(
                (
                    bag_f1(prediction, ref, remove_stopwords=True)
                    for ref in references
                ),
                default=0.0,
            )
        else:
            accepted = row.get("accepted_answers") or [row["canonical_answer"]]
            normalized_prediction = normalize_answer(prediction)
            row["exact_match"] = float(
                normalized_prediction
                in {normalize_answer(answer) for answer in accepted}
            )
            row["token_f1"] = max(
                (
                    bag_f1(
                        normalized_prediction,
                        normalize_answer(answer),
                        remove_stopwords=False,
                    )
                    for answer in accepted
                ),
                default=0.0,
            )
        scored.append(row)

    caption = [row for row in scored if row["evaluation_type"] == "caption"]
    vqa = [row for row in scored if row["evaluation_type"] == "vqa"]
    report = {
        "model_tag": rows[0].get("model_tag", "") if rows else "",
        "input_rows": len(rows),
        "caption": {
            "overall": summarize_group(
                caption, ["rouge_l", "lexical_f1"], args.seed, args.bootstrap
            ),
            "by_task": {},
            "by_source": {},
        },
        "vqa": {
            "overall": summarize_group(
                vqa, ["exact_match", "token_f1"], args.seed, args.bootstrap
            ),
            "by_type": {},
            "by_source": {},
        },
    }
    for task in sorted({row["task_type"] for row in caption}):
        group = [row for row in caption if row["task_type"] == task]
        report["caption"]["by_task"][task] = summarize_group(
            group, ["rouge_l", "lexical_f1"], args.seed, args.bootstrap
        )
    for source in sorted({row["dataset"] for row in caption}):
        group = [row for row in caption if row["dataset"] == source]
        report["caption"]["by_source"][source] = summarize_group(
            group, ["rouge_l", "lexical_f1"], args.seed, args.bootstrap
        )
    for question_type in sorted({row["question_type"] for row in vqa}):
        group = [row for row in vqa if row["question_type"] == question_type]
        report["vqa"]["by_type"][question_type] = summarize_group(
            group, ["exact_match", "token_f1"], args.seed, args.bootstrap
        )
    for source in sorted({row["dataset"] for row in vqa}):
        group = [row for row in vqa if row["dataset"] == source]
        report["vqa"]["by_source"][source] = summarize_group(
            group, ["exact_match", "token_f1"], args.seed, args.bootstrap
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
