#!/usr/bin/env python3
"""Run the checked DiffV2IR entrypoint over a reconstructed RGB workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SOURCES = ("NWPU", "RS5M", "RSICD", "RSITMD", "SkyScript")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffv2ir-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--config-record",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "diffv2ir_generation.json",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--source", choices=SOURCES, action="append")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    record = json.loads(args.config_record.read_text(encoding="utf-8"))
    entrypoint = args.diffv2ir_repo / record["entrypoint"]
    config = args.diffv2ir_repo / record["config"]
    check_hash(entrypoint, record["entrypoint_sha256"], "entrypoint")
    check_hash(config, record["config_sha256"], "config")
    check_hash(args.checkpoint, record["checkpoint_sha256"], "checkpoint")

    metadata = args.workspace / "translation_metadata.jsonl"
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    sources = args.source or list(SOURCES)
    commands = []
    settings = record["settings"]
    for source in sources:
        input_root = args.workspace / "rgb" / source
        if not input_root.is_dir():
            raise FileNotFoundError(input_root)
        output_root = args.workspace / "ir" / source
        command = [
            args.python,
            str(entrypoint),
            "--config",
            str(config),
            "--ckpt",
            str(args.checkpoint),
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--metadata",
            str(metadata),
            "--resolution",
            str(settings["resolution"]),
            "--steps",
            str(settings["steps"]),
            "--cfg-text",
            str(settings["cfg_text"]),
            "--cfg-image",
            str(settings["cfg_image"]),
            "--cfg-seg",
            str(settings["cfg_seg"]),
            "--seed",
            str(settings["seed"]),
        ]
        if args.limit:
            command.extend(["--limit", str(args.limit)])
        commands.append(command)

    run_record = {
        "config_record": record,
        "commands": commands,
        "dry_run": args.dry_run,
    }
    (args.workspace / "diffv2ir_run_record.json").write_text(
        json.dumps(run_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for command in commands:
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=args.diffv2ir_repo, check=True)


if __name__ == "__main__":
    main()
