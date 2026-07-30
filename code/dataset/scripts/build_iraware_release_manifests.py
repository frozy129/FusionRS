#!/usr/bin/env python3
"""Sanitize accepted and removed IR-aware caption manifests for release."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


def open_index(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def write_sanitized(
    path: Path,
    rows: list[dict],
    index: dict[str, dict],
    status: str,
    reason_field: str | None,
) -> dict:
    counts: Counter[str] = Counter()
    seen = set()
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in sorted(
            rows, key=lambda item: str(item.get("sample_id") or item.get("id"))
        ):
            sample_id = str(row.get("sample_id") or row.get("id") or "")
            if not sample_id or sample_id in seen:
                raise ValueError(f"empty or duplicate caption ID: {sample_id!r}")
            seen.add(sample_id)
            record = index.get(sample_id)
            if record is None:
                raise ValueError(f"{sample_id}: missing from canonical index")
            caption = str(row.get("caption") or row.get("text") or "").strip()
            reasons = list(row.get(reason_field, [])) if reason_field else []
            output = {
                "schema_version": "fusionrs-iraware-caption-v1",
                "sample_id": sample_id,
                "source_dataset": record["source_dataset"],
                "split": record["split"],
                "caption": caption,
                "status": status,
                "filter_reasons": reasons,
            }
            handle.write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")
            counts[record["source_dataset"]] += 1
    return {
        "rows": len(seen),
        "source_counts": dict(sorted(counts.items())),
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--accepted-v3", type=Path, required=True)
    parser.add_argument("--removed-v2", type=Path, required=True)
    parser.add_argument("--removed-v3", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    all_rows = (
        load_rows(args.accepted_v3)
        + load_rows(args.removed_v2)
        + load_rows(args.removed_v3)
    )
    wanted = {
        str(row.get("sample_id") or row.get("id"))
        for row in all_rows
    }
    index = {}
    with open_index(args.index) as handle:
        for line in handle:
            row = json.loads(line)
            if row["sample_id"] in wanted:
                index[row["sample_id"]] = row
    if set(index) != wanted:
        raise ValueError(f"{len(wanted) - len(index)} caption IDs missing from index")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "accepted_v3": write_sanitized(
            args.output_dir / "iraware_strict_v3_45913.jsonl.gz",
            load_rows(args.accepted_v3),
            index,
            "accepted_strict_v3",
            None,
        ),
        "removed_v2": write_sanitized(
            args.output_dir / "iraware_removed_v2_2659.jsonl.gz",
            load_rows(args.removed_v2),
            index,
            "removed_strict_v2",
            "strict_filter_reasons",
        ),
        "removed_v3_additional": write_sanitized(
            args.output_dir / "iraware_removed_v3_additional_44.jsonl.gz",
            load_rows(args.removed_v3),
            index,
            "removed_strict_v3",
            "strict_v3_flags",
        ),
    }
    expected = {
        "accepted_v3": 45913,
        "removed_v2": 2659,
        "removed_v3_additional": 44,
    }
    for name, count in expected.items():
        if outputs[name]["rows"] != count:
            raise ValueError(
                f"{name}: {outputs[name]['rows']} rows, expected {count}"
            )
    summary = {"outputs": outputs}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
