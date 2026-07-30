#!/usr/bin/env python3
"""Prepare a deterministic three-person audit bundle for a VQA manifest."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqa", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-type", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    by_type = defaultdict(list)
    with gzip.open(args.vqa, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_type[row["question_type"]].append(row)
    selected = []
    for question_type, rows in sorted(by_type.items()):
        rows.sort(key=lambda row: stable_key(args.seed, row["question_id"]))
        if len(rows) < args.per_type:
            raise ValueError(
                f"{question_type}: {len(rows)} rows, need {args.per_type}"
            )
        selected.extend(rows[: args.per_type])
    selected.sort(key=lambda row: row["question_id"])

    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_names = {}
    for row in selected:
        image_id = str(row["image_id"])
        if image_id in image_names:
            continue
        source = args.image_root / row["image_locator"]
        if not source.is_file():
            raise FileNotFoundError(source)
        target = image_dir / f"HIT-{image_id}.jpg"
        with Image.open(source) as image:
            image.convert("RGB").save(target, format="JPEG", quality=95)
        image_names[image_id] = str(Path("images") / target.name)

    fields = [
        "question_id",
        "image",
        "question_type",
        "question",
        "proposed_answer",
        "image_grounded_0_or_1",
        "answer_correct_0_or_1",
        "unambiguous_0_or_1",
        "comments",
    ]
    for annotator in range(1, 4):
        with (args.output_dir / f"annotator_{annotator}.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in selected:
                writer.writerow(
                    {
                        "question_id": row["question_id"],
                        "image": image_names[str(row["image_id"])],
                        "question_type": row["question_type"],
                        "question": row["question"],
                        "proposed_answer": row["answer"],
                        "image_grounded_0_or_1": "",
                        "answer_correct_0_or_1": "",
                        "unambiguous_0_or_1": "",
                        "comments": "",
                    }
                )
    summary = {
        "questions": len(selected),
        "images": len(image_names),
        "per_type": args.per_type,
        "question_types": sorted(by_type),
        "annotators": 3,
        "status": "human_review_required",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        """# HIT-UAV VQA human audit

Three reviewers independently inspect every row. Enter 0 or 1 for image
grounding, answer correctness, and ambiguity. Do not consult another
reviewer's form. A question is accepted only when at least two reviewers mark
all three fields as 1; disagreements require adjudication.
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
