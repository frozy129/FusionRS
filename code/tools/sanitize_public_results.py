#!/usr/bin/env python3
"""Remove machine-local paths from JSON/JSONL/Markdown result artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PATH_KEYS = {
    "adapter_path",
    "checkpoint",
    "data_root",
    "dataset_root",
    "hit_uav_root",
    "image_path",
    "model_path",
    "models_root",
    "output_dir",
    "rc4_root",
    "test_pair_jsonl",
    "val_pair_jsonl",
    "vlm_root",
}
LOCAL_PATH = re.compile(r"/(?:root|Users)/[^\s`\"']+")


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: sanitize(item)
            for key, item in value.items()
            if key not in PATH_KEYS
        }
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        return LOCAL_PATH.sub("<machine-local-path-removed>", value)
    return value


def sanitize_file(path: Path) -> None:
    suffixes = path.suffixes
    if path.suffix == ".json":
        payload = sanitize(json.loads(path.read_text(encoding="utf-8")))
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    elif suffixes[-1:] == [".jsonl"]:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.dumps(sanitize(json.loads(line)), ensure_ascii=False))
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    elif path.suffix == ".md":
        text = LOCAL_PATH.sub("<machine-local-path-removed>", path.read_text(encoding="utf-8"))
        path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    for root in args.roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and (path.suffix in {".json", ".jsonl", ".md"}):
                sanitize_file(path)


if __name__ == "__main__":
    main()

