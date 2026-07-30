#!/usr/bin/env python3
"""Materialize an index-only FusionRS RGB workspace from upstream datasets."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
from pathlib import Path

SOURCES = ("NWPU", "RS5M", "RSICD", "RSITMD", "SkyScript")


def open_index(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def parse_source_roots(values: list[str]) -> dict[str, Path]:
    roots = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected SOURCE=/path, received {value!r}")
        source, raw_path = value.split("=", 1)
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"{source} root does not exist: {path}")
        roots[source] = path
    missing = set(SOURCES) - set(roots)
    if missing:
        raise ValueError(f"missing source roots: {sorted(missing)}")
    return roots


def materialize(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    if mode == "hardlink":
        os.link(source, target)
    elif mode == "symlink":
        target.symlink_to(source)
    elif mode == "copy":
        shutil.copy2(source, target)
    else:
        raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        action="append",
        default=[],
        help="Repeat as SOURCE=/absolute/upstream/root",
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("hardlink", "symlink", "copy"), default="symlink"
    )
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    roots = parse_source_roots(args.source_root)
    metadata_path = args.workspace / "translation_metadata.jsonl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    missing = []
    with open_index(args.index) as handle, metadata_path.open(
        "w", encoding="utf-8"
    ) as metadata:
        for line in handle:
            record = json.loads(line)
            if args.split and record["split"] != args.split:
                continue
            source_name = record["source_dataset"]
            source_path = roots[source_name] / record["source_locator"]
            if not source_path.is_file():
                missing.append(str(source_path))
                continue
            suffix = source_path.suffix.lower() or ".jpg"
            target = (
                args.workspace
                / "rgb"
                / source_name
                / f"{record['sample_id']}{suffix}"
            )
            materialize(source_path, target, args.mode)
            metadata.write(
                json.dumps(
                    {
                        "sample_id": record["sample_id"],
                        "dataset": source_name,
                        "image": str(target),
                        "filename": target.name,
                        "text": record.get("caption") or "",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            rows += 1
            if args.limit and rows >= args.limit:
                break

    summary = {
        "materialized": rows,
        "missing": len(missing),
        "missing_examples": missing[:20],
        "mode": args.mode,
        "workspace": str(args.workspace),
    }
    (args.workspace / "reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
