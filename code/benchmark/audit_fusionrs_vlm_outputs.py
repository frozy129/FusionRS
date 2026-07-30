#!/usr/bin/env python3
"""Audit FusionRS Caption/VQA generations for output-quality failure modes."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


IR_CUES = {
    "bright",
    "brightness",
    "contrast",
    "dark",
    "grayscale",
    "infrared",
    "intensity",
    "outline",
    "outlines",
    "texture",
    "thermal",
}
UNSUPPORTED_WORDS = {
    "reflectivity",
    "reflective",
    "emissivity",
    "emissive",
    "material",
    "materials",
    "metallic",
    "conductivity",
    "conductive",
    "absorptivity",
    "radiance",
    "radiant",
    "heat",
    "heated",
    "hot",
    "cold",
    "temperature",
    "temperatures",
}
UNSUPPORTED_PHRASES = {
    "different materials",
    "material properties",
    "material property",
    "material response",
    "surface properties",
    "surface property",
    "thermal anomaly",
    "heat signature",
    "heat signatures",
}
COLOR_WORDS = {
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "cyan",
    "magenta",
    "beige",
    "gold",
    "golden",
}
GENERIC_FAILURE_PATTERNS = (
    "unable to",
    "cannot describe",
    "can't describe",
    "no image",
    "as an ai",
    "i cannot",
    "i'm sorry",
    "i am sorry",
)


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def normalized(text: str) -> str:
    return " ".join(words(text))


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def audit_caption_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = [str(row.get("prediction") or "") for row in rows]
    normalized_predictions = [normalized(value) for value in predictions]
    token_sets = [set(words(value)) for value in predictions]
    word_counts = [len(words(value)) for value in predictions]
    counts = Counter(normalized_predictions)
    return {
        "rows": len(rows),
        "images": len({row["benchmark_id"] for row in rows}),
        "empty_rate": mean([float(not value) for value in normalized_predictions]),
        "generic_failure_rate": mean(
            [
                float(any(pattern in value for pattern in GENERIC_FAILURE_PATTERNS))
                for value in normalized_predictions
            ]
        ),
        "ir_cue_rate": mean(
            [float(bool(tokens & IR_CUES)) for tokens in token_sets]
        ),
        "unsupported_physical_claim_rate": mean(
            [
                float(
                    bool(tokens & UNSUPPORTED_WORDS)
                    or any(phrase in text for phrase in UNSUPPORTED_PHRASES)
                )
                for tokens, text in zip(token_sets, normalized_predictions)
            ]
        ),
        "rgb_color_term_rate": mean(
            [float(bool(tokens & COLOR_WORDS)) for tokens in token_sets]
        ),
        "average_words": round(mean([float(value) for value in word_counts]), 4),
        "short_output_rate_le_15": mean(
            [float(value <= 15) for value in word_counts]
        ),
        "long_output_rate_ge_25": mean(
            [float(value >= 25) for value in word_counts]
        ),
        "unique_prediction_rate": len(counts) / max(1, len(rows)),
        "most_common_prediction_rate": (
            max(counts.values()) / len(rows) if rows else 0.0
        ),
        "exact_reference_copy_rate": mean(
            [
                float(
                    normalized_prediction
                    in {normalized(reference) for reference in row["references"]}
                )
                for row, normalized_prediction in zip(
                    rows, normalized_predictions
                )
            ]
        ),
    }


def audit_vqa_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = [normalized(row.get("prediction", "")) for row in rows]
    word_counts = [len(value.split()) for value in predictions]
    return {
        "rows": len(rows),
        "images": len({row["benchmark_id"] for row in rows}),
        "empty_rate": mean([float(not value) for value in predictions]),
        "generic_failure_rate": mean(
            [
                float(any(pattern in value for pattern in GENERIC_FAILURE_PATTERNS))
                for value in predictions
            ]
        ),
        "average_words": round(mean([float(value) for value in word_counts]), 4),
        "short_answer_rate_le_5": mean(
            [float(value <= 5) for value in word_counts]
        ),
        "exact_accepted_answer_rate": mean(
            [
                float(
                    prediction
                    in {
                        normalized(answer)
                        for answer in (
                            row.get("accepted_answers")
                            or [row.get("canonical_answer", "")]
                        )
                    }
                )
                for row, prediction in zip(rows, predictions)
            ]
        ),
        "yes_no_format_rate": mean(
            [float(value in {"yes", "no"}) for value in predictions]
        )
        if rows and rows[0].get("question_type") == "object_presence"
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    caption = [
        row
        for row in rows
        if row.get("evaluation_type") == "caption"
        and row.get("status") == "success"
    ]
    vqa = [
        row
        for row in rows
        if row.get("evaluation_type") == "vqa"
        and row.get("status") == "success"
    ]

    caption_by_task: dict[str, Any] = {}
    for task in sorted({row["task_type"] for row in caption}):
        caption_by_task[task] = audit_caption_group(
            [row for row in caption if row["task_type"] == task]
        )

    vqa_by_type: dict[str, Any] = {}
    for question_type in sorted({row["question_type"] for row in vqa}):
        vqa_by_type[question_type] = audit_vqa_group(
            [row for row in vqa if row["question_type"] == question_type]
        )

    caption_by_id: dict[str, dict[str, str]] = defaultdict(dict)
    for row in caption:
        caption_by_id[row["benchmark_id"]][row["task_type"]] = normalized(
            row["prediction"]
        )
    paired = [
        tasks
        for tasks in caption_by_id.values()
        if {
            "semantic_description",
            "ir_observable_description",
        }.issubset(tasks)
    ]
    report = {
        "model_tag": rows[0].get("model_tag", "") if rows else "",
        "input_rows": len(rows),
        "caption": {
            "overall": audit_caption_group(caption),
            "by_task": caption_by_task,
            "paired_images": len(paired),
            "identical_across_prompts_rate": mean(
                [
                    float(
                        tasks["semantic_description"]
                        == tasks["ir_observable_description"]
                    )
                    for tasks in paired
                ]
            ),
            "ir_cue_prompt_delta": (
                caption_by_task["ir_observable_description"]["ir_cue_rate"]
                - caption_by_task["semantic_description"]["ir_cue_rate"]
            )
            if {
                "semantic_description",
                "ir_observable_description",
            }.issubset(caption_by_task)
            else None,
            "length_prompt_delta": (
                caption_by_task["ir_observable_description"]["average_words"]
                - caption_by_task["semantic_description"]["average_words"]
            )
            if {
                "semantic_description",
                "ir_observable_description",
            }.issubset(caption_by_task)
            else None,
        },
        "vqa": {
            "overall": audit_vqa_group(vqa),
            "by_type": vqa_by_type,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "model_tag": report["model_tag"],
                "caption_rows": len(caption),
                "vqa_rows": len(vqa),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
