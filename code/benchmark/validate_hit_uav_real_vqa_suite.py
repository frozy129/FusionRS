#!/usr/bin/env python3
"""Validate and aggregate the complete HIT-UAV real-thermal VQA suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


MODEL_TAGS = (
    "base",
    "original_only",
    "ir_aware_only",
    "task_conditioned_mixed",
    "h2rsvlm",
    "llava15",
    "llava16",
    "geochat",
    "instructblip",
)
QUESTION_TYPES = (
    "object_presence",
    "object_count",
    "group_location",
    "dominant_object_class",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=3223)
    args = parser.parse_args()

    manifest = args.root / "hit_uav_vqa_eval_manifest.jsonl"
    manifest_rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(manifest_rows) != args.expected:
        raise RuntimeError(
            f"manifest denominator {len(manifest_rows)} != {args.expected}"
        )
    presence = [
        row for row in manifest_rows if row["question_type"] == "object_presence"
    ]
    answer_counts = {
        answer: sum(row["canonical_answer"] == answer for row in presence)
        for answer in ("yes", "no")
    }
    if answer_counts != {"yes": 598, "no": 598}:
        raise RuntimeError(f"unbalanced presence answers: {answer_counts}")

    table_rows = []
    evidence = {}
    for tag in MODEL_TAGS:
        prediction_path = args.root / f"{tag}.jsonl"
        summary_path = args.root / f"{tag}.summary.json"
        metrics_path = args.root / f"{tag}.metrics.json"
        for path in (prediction_path, summary_path, metrics_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        predictions = [
            json.loads(line)
            for line in prediction_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        keys = {row["record_key"] for row in predictions}
        successes = sum(row.get("status") == "success" for row in predictions)
        errors = sum(row.get("status") != "success" for row in predictions)
        if len(predictions) != args.expected or len(keys) != args.expected:
            raise RuntimeError(
                f"{tag}: rows={len(predictions)}, unique={len(keys)}"
            )
        if successes != args.expected or errors:
            raise RuntimeError(
                f"{tag}: successes={successes}, errors={errors}"
            )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {
            "model": tag,
            "overall_accuracy": metrics["overall"]["accuracy"],
            "overall_ci_low": metrics["overall"]["accuracy_ci95"][0],
            "overall_ci_high": metrics["overall"]["accuracy_ci95"][1],
            "parse_failure_rate": metrics["overall"]["parse_failure_rate"],
            "presence_yes_accuracy": metrics["presence_by_answer"]["yes"][
                "accuracy"
            ],
            "presence_no_accuracy": metrics["presence_by_answer"]["no"][
                "accuracy"
            ],
        }
        for question_type in QUESTION_TYPES:
            row[f"{question_type}_accuracy"] = metrics["by_type"][question_type][
                "accuracy"
            ]
        table_rows.append(row)
        evidence[tag] = {
            "rows": len(predictions),
            "unique_record_keys": len(keys),
            "successes": successes,
            "errors": errors,
            "prediction_sha256": sha256(prediction_path),
            "metrics_sha256": sha256(metrics_path),
        }

    fieldnames = list(table_rows[0])
    table_path = args.root / "hit_uav_real_thermal_vqa_table.tsv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(table_rows)
    validation = {
        "valid": True,
        "manifest_rows": len(manifest_rows),
        "manifest_sha256": sha256(manifest),
        "images": len({row["benchmark_id"] for row in manifest_rows}),
        "presence_answer_counts": answer_counts,
        "models": evidence,
        "table_sha256": sha256(table_path),
    }
    (args.root / "suite_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
