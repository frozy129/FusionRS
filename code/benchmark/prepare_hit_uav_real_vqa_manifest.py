#!/usr/bin/env python3
"""Convert annotation-derived HIT-UAV VQA records to the common evaluator schema."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


NUMBER_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
}


def accepted_answers(row: dict) -> list[str]:
    answer = str(row["answer"]).strip().lower()
    answers = [answer]
    if row["answer_type"] == "integer":
        value = int(answer)
        if value in NUMBER_WORDS:
            answers.append(NUMBER_WORDS[value])
    if row["answer_type"] == "region":
        answers.extend(
            {
                "upper-left": ["upper left", "top-left", "top left"],
                "upper-right": ["upper right", "top-right", "top right"],
                "lower-left": ["lower left", "bottom-left", "bottom left"],
                "lower-right": ["lower right", "bottom-right", "bottom right"],
                "upper": ["top"],
                "lower": ["bottom"],
                "center": ["middle", "centre"],
            }.get(answer, [])
        )
    return list(dict.fromkeys(answers))


def prompt_for(row: dict) -> str:
    question = row["question"]
    question_type = row["question_type"]
    if question_type == "object_presence":
        suffix = "Answer with exactly yes or no."
    elif question_type == "object_count":
        suffix = "Answer with exactly one non-negative integer."
    elif question_type == "group_location":
        suffix = (
            "Answer with exactly one region label from: upper-left, upper, "
            "upper-right, left, center, right, lower-left, lower, lower-right."
        )
    else:
        suffix = (
            "Answer with exactly one class label from: person, car, bicycle, "
            "other vehicle, none."
        )
    return (
        "Use only the visible content of this real sensor-captured thermal "
        f"aerial image. {question} {suffix}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    opener = gzip.open if args.input.suffix == ".gz" else open
    rows = []
    missing = []
    with opener(args.input, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source = json.loads(line)
            image_locator = source["image_locator"]
            if not (args.image_root / image_locator).is_file():
                missing.append(image_locator)
                continue
            rows.append(
                {
                    **source,
                    "benchmark_id": f"HIT-UAV:{source['image_id']}",
                    "dataset": "HIT-UAV",
                    "ir_image": image_locator,
                    "canonical_answer": str(source["answer"]).lower(),
                    "accepted_answers": accepted_answers(source),
                    "prompt": prompt_for(source),
                    "max_new_tokens": 12,
                }
            )

    if missing:
        raise FileNotFoundError(f"{len(missing)} HIT-UAV images are missing")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"records": len(rows), "missing_images": 0}, indent=2))


if __name__ == "__main__":
    main()
