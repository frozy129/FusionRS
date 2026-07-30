#!/usr/bin/env python3
"""Validate the final rc4 CLIP/downstream and primary VLM result suites."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require_finite(value: Any, label: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} is not finite: {value!r}")


def add_file(report: dict[str, Any], root: Path, path: Path, rows: int) -> None:
    report["files"].append(
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "rows": rows,
        }
    )


def validate_adaptation_json(
    report: dict[str, Any], root: Path, path: Path, expected: int
) -> None:
    doc = load_json(path)
    rows = doc.get("results")
    if not isinstance(rows, list) or len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} results")
    for index, row in enumerate(rows):
        if not isinstance(row.get("test"), dict):
            raise ValueError(f"{path}: result {index} has no test metrics")
        numeric = [
            value
            for value in row["test"].values()
            if isinstance(value, (int, float))
        ]
        if not numeric:
            raise ValueError(f"{path}: result {index} has no numeric test metric")
        for value in numeric:
            require_finite(value, f"{path}: result {index}")
    add_file(report, root, path, len(rows))


def validate_retrieval_json(
    report: dict[str, Any], root: Path, path: Path, expected: int
) -> None:
    doc = load_json(path)
    rows = doc.get("results")
    if not isinstance(rows, list) or len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} results")
    for index, row in enumerate(rows):
        require_finite(row.get("mean_recall"), f"{path}: result {index} mean_recall")
        if not isinstance(row.get("metrics"), dict) or not row["metrics"]:
            raise ValueError(f"{path}: result {index} has no retrieval metrics")
    add_file(report, root, path, len(rows))


def validate_language_json(
    report: dict[str, Any],
    root: Path,
    path: Path,
    expected: int,
    metric: str,
) -> None:
    doc = load_json(path)
    rows = doc.get("results")
    if not isinstance(rows, list) or len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} results")
    for index, row in enumerate(rows):
        require_finite(row.get(metric), f"{path}: result {index} {metric}")
    add_file(report, root, path, len(rows))


def validate_vlm_predictions(
    report: dict[str, Any],
    root: Path,
    model: str,
    expected: int,
) -> None:
    predictions = root / f"{model}.jsonl"
    summary_path = root / f"{model}.summary.json"
    metrics_path = root / f"{model}.metrics.json"
    summary = load_json(summary_path)
    if (
        summary.get("expected") != expected
        or summary.get("successful") != expected
        or summary.get("missing_or_error") != 0
    ):
        raise ValueError(f"{summary_path}: incomplete summary")

    keys: set[str] = set()
    count = 0
    with predictions.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("record_key")
            if row.get("status") != "success" or not row.get("prediction"):
                raise ValueError(
                    f"{predictions}:{line_number}: unsuccessful or empty prediction"
                )
            if not isinstance(key, str) or key in keys:
                raise ValueError(
                    f"{predictions}:{line_number}: empty or duplicate record_key"
                )
            keys.add(key)
            count += 1
    if count != expected:
        raise ValueError(f"{predictions}: expected {expected} unique predictions")
    load_json(metrics_path)
    add_file(report, root, predictions, count)
    add_file(report, root, summary_path, 1)
    add_file(report, root, metrics_path, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rc4-root",
        type=Path,
        default=Path("artifacts/rc4_experiments"),
    )
    parser.add_argument(
        "--vlm-root",
        type=Path,
        default=Path("artifacts/vlm_predictions"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rc4_root": str(args.rc4_root),
        "vlm_root": str(args.vlm_root),
        "files": [],
        "status": "valid",
    }

    for seed in (42, 3407, 2026):
        for task, filename in (
            ("hit_uav", "hit_uav_box_detection.json"),
            ("sirst_v2", "sirst_v2_segmentation.json"),
            ("irstd1k", "irstd1k_segmentation.json"),
            ("caltech_10pct", "caltech_paired_semantic_finetune.json"),
        ):
            validate_adaptation_json(
                report,
                args.rc4_root,
                args.rc4_root
                / "downstream"
                / task
                / f"seed_{seed}"
                / "eval"
                / filename,
                7,
            )

    validate_retrieval_json(
        report,
        args.rc4_root,
        args.rc4_root
        / "downstream"
        / "caltech_pair"
        / "caltech_rgbt_retrieval.json",
        7,
    )
    validate_language_json(
        report,
        args.rc4_root,
        args.rc4_root
        / "downstream"
        / "hit_language"
        / "zero_shot"
        / "hit_uav_object_presence.json",
        10,
        "mAP",
    )
    validate_language_json(
        report,
        args.rc4_root,
        args.rc4_root
        / "downstream"
        / "hit_language"
        / "linear_probe"
        / "hit_uav_linear_probe.json",
        10,
        "mAP",
    )
    validate_retrieval_json(
        report,
        args.rc4_root,
        args.rc4_root / "capacity" / "eval" / "test_580k_only_baseline.json",
        8,
    )
    validate_retrieval_json(
        report,
        args.rc4_root,
        args.rc4_root
        / "multiclip"
        / "retrieval"
        / "test_580k_only_baseline.json",
        4,
    )
    validate_retrieval_json(
        report,
        args.rc4_root,
        args.rc4_root
        / "multiclip"
        / "caltech_pair"
        / "caltech_rgbt_retrieval.json",
        4,
    )
    validate_language_json(
        report,
        args.rc4_root,
        args.rc4_root
        / "multiclip"
        / "hit_zero_shot"
        / "hit_uav_object_presence.json",
        4,
        "mAP",
    )

    for model in (
        "base",
        "original_only",
        "ir_aware_only",
        "task_conditioned_mixed",
        "h2rsvlm",
        "llava15",
        "llava16",
        "geochat",
        "instructblip",
    ):
        validate_vlm_predictions(report, args.vlm_root, model, 2991)

    report["validated_file_count"] = len(report["files"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "validated_file_count": report["validated_file_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
