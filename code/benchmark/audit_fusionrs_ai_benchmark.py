#!/usr/bin/env python3
"""Audit the AI-assisted FusionRS Caption/VQA authoring CSV files."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from score_fusionrs_benchmark import bag_f1, rouge_l_f1


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def flag(row: dict[str, str], field: str) -> bool:
    return row.get(field, "").strip() == "1"


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def caption_audit(rows: list[dict[str, str]]) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if row.get("reference_caption", "").strip()
        and flag(row, "observable_only_0_or_1")
        and flag(row, "pair_consistent_0_or_1")
    ]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in valid:
        grouped[(row["benchmark_id"], row["task_type"])].append(row)

    pairwise = []
    for items in grouped.values():
        for left, right in itertools.combinations(items, 2):
            pairwise.append(
                {
                    "exact": float(
                        normalize(left["reference_caption"])
                        == normalize(right["reference_caption"])
                    ),
                    "rouge_l": rouge_l_f1(
                        left["reference_caption"], right["reference_caption"]
                    ),
                    "lexical_f1": bag_f1(
                        left["reference_caption"],
                        right["reference_caption"],
                        remove_stopwords=True,
                    ),
                }
            )

    refs_per_item = Counter(len(items) for items in grouped.values())
    expected_items = {
        (row["benchmark_id"], task)
        for row in rows
        for task in ("semantic_description", "ir_observable_description")
    }
    return {
        "rows": len(rows),
        "benchmark_ids": len({row["benchmark_id"] for row in rows}),
        "model_ids": dict(Counter(row["model_id"] for row in rows)),
        "valid_rows": len(valid),
        "valid_rate": len(valid) / max(1, len(rows)),
        "valid_items": len(grouped),
        "expected_items": len(expected_items),
        "references_per_item": {
            str(count): frequency for count, frequency in sorted(refs_per_item.items())
        },
        "items_with_no_valid_reference": len(expected_items - set(grouped)),
        "semantic_confidence": dict(
            Counter(
                row["semantic_confidence"].strip() or "blank"
                for row in valid
                if row["task_type"] == "semantic_description"
            )
        ),
        "pairwise_valid_reference_agreement": {
            "pairs": len(pairwise),
            "normalized_exact_rate": mean([row["exact"] for row in pairwise]),
            "rouge_l": mean([row["rouge_l"] for row in pairwise]),
            "content_lexical_f1": mean(
                [row["lexical_f1"] for row in pairwise]
            ),
        },
        "valid_by_model": {
            model: {
                "rows": len(items),
                "valid": sum(row in valid for row in items),
            }
            for model, items in sorted(
                (
                    (model, [row for row in rows if row["model_id"] == model])
                    for model in {row["model_id"] for row in rows}
                )
            )
        },
    }


def vqa_audit(rows: list[dict[str, str]]) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if row.get("selected_question", "").strip()
        and row.get("canonical_answer", "").strip()
        and flag(row, "accepted_0_or_1")
        and flag(row, "unambiguous_0_or_1")
        and flag(row, "observable_in_ir_0_or_1")
    ]
    by_type = {}
    for question_type in sorted({row["question_type"] for row in rows}):
        items = [row for row in rows if row["question_type"] == question_type]
        accepted = [row for row in valid if row["question_type"] == question_type]
        answers = Counter(normalize(row["canonical_answer"]) for row in accepted)
        questions = Counter(normalize(row["selected_question"]) for row in accepted)
        by_type[question_type] = {
            "rows": len(items),
            "valid": len(accepted),
            "valid_rate": len(accepted) / max(1, len(items)),
            "unique_normalized_questions": len(questions),
            "most_common_question_rate": (
                max(questions.values()) / len(accepted) if accepted else 0.0
            ),
            "unique_normalized_answers": len(answers),
            "most_common_answers": dict(answers.most_common(10)),
        }
        if question_type == "object_presence":
            yes = answers["yes"]
            no = answers["no"]
            by_type[question_type]["yes_no_format_rate"] = (
                (yes + no) / max(1, len(accepted))
            )
            by_type[question_type]["yes_rate"] = yes / max(1, yes + no)

    return {
        "rows": len(rows),
        "benchmark_ids": len({row["benchmark_id"] for row in rows}),
        "unique_question_ids": len({row["question_id"] for row in rows}),
        "valid_rows": len(valid),
        "valid_rate": len(valid) / max(1, len(rows)),
        "rejected_rows": len(rows) - len(valid),
        "accepted_source": dict(
            Counter(row["accepted_source"].strip() or "blank" for row in valid)
        ),
        "by_type": by_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption-csv", type=Path, required=True)
    parser.add_argument("--vqa-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "annotation_type": "AI-assisted multi-model authoring and adjudication",
        "human_annotation_claim_allowed": False,
        "caption": caption_audit(read_csv(args.caption_csv)),
        "vqa": vqa_audit(read_csv(args.vqa_csv)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
