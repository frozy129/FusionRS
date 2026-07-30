#!/usr/bin/env python3
"""Summarize three independent binary reviews of a VQA audit bundle."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

RATING_FIELDS = (
    "image_grounded_0_or_1",
    "answer_correct_0_or_1",
    "unambiguous_0_or_1",
)


def load_form(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for row in rows:
        question_id = row["question_id"].strip()
        if not question_id or question_id in output:
            raise ValueError(f"{path.name}: empty or duplicate question_id")
        for field in RATING_FIELDS:
            if row[field].strip() not in {"", "0", "1"}:
                raise ValueError(
                    f"{path.name}:{question_id}: {field} must be 0, 1, or blank"
                )
        output[question_id] = row
    return output


def summarize(forms: list[dict[str, dict[str, str]]]) -> dict:
    if len(forms) != 3:
        raise ValueError("exactly three annotator forms are required")
    question_ids = set(forms[0])
    if any(set(form) != question_ids for form in forms[1:]):
        raise ValueError("annotator forms contain different question IDs")

    complete = 0
    accepted = 0
    unanimous = 0
    type_counts: Counter[str] = Counter()
    accepted_types: Counter[str] = Counter()
    field_unanimous: Counter[str] = Counter()
    for question_id in sorted(question_ids):
        rows = [form[question_id] for form in forms]
        invariant_fields = (
            "image",
            "question_type",
            "question",
            "proposed_answer",
        )
        if any(
            len({row[field] for row in rows}) != 1
            for field in invariant_fields
        ):
            raise ValueError(f"{question_id}: forms disagree on fixed content")
        question_type = rows[0]["question_type"]
        type_counts[question_type] += 1
        is_complete = all(
            row[field].strip() in {"0", "1"}
            for row in rows
            for field in RATING_FIELDS
        )
        if not is_complete:
            continue
        complete += 1
        reviewer_decisions = [
            all(row[field].strip() == "1" for field in RATING_FIELDS)
            for row in rows
        ]
        is_accepted = sum(reviewer_decisions) >= 2
        if is_accepted:
            accepted += 1
            accepted_types[question_type] += 1
        if len(set(reviewer_decisions)) == 1:
            unanimous += 1
        for field in RATING_FIELDS:
            if len({row[field].strip() for row in rows}) == 1:
                field_unanimous[field] += 1

    return {
        "questions": len(question_ids),
        "complete_questions": complete,
        "status": "complete" if complete == len(question_ids) else "human_review_required",
        "accepted_questions": accepted,
        "acceptance_rate": accepted / complete if complete else None,
        "unanimous_decisions": unanimous,
        "unanimous_decision_rate": unanimous / complete if complete else None,
        "question_type_counts": dict(sorted(type_counts.items())),
        "accepted_question_type_counts": dict(sorted(accepted_types.items())),
        "field_unanimous_counts": {
            field: field_unanimous[field] for field in RATING_FIELDS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    forms = [
        load_form(args.audit_dir / f"annotator_{annotator}.csv")
        for annotator in range(1, 4)
    ]
    result = summarize(forms)
    output = args.output_json or args.audit_dir / "audit_summary.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
