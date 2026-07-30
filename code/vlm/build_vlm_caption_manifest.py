#!/usr/bin/env python3
"""Build the canonical strict-v3 FusionRS VLM caption training manifest."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

PROMPT = (
    "Describe this infrared-style aerial image in one concise sentence. "
    "Report visible scene semantics, grayscale intensity, contrast, edges, "
    "texture, or structure only. Do not infer temperature, heat, material "
    "properties, emissivity, or RGB color."
)
BLOCKED_TERMS = (
    "temperature",
    "heat signature",
    "thermal signature",
    "emissivity",
    "reflectivity",
    "material property",
)


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


def materialize(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    if mode == "symlink":
        target.symlink_to(source)
    elif mode == "hardlink":
        os.link(source, target)
    elif mode == "copy":
        shutil.copy2(source, target)
    else:
        raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-captions", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--materialize", choices=("symlink", "hardlink", "copy"), default="symlink"
    )
    args = parser.parse_args()

    strict_rows = {}
    with args.strict_captions.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or row.get("id") or "").strip()
            caption = str(row.get("caption") or row.get("text") or "").strip()
            image = Path(str(row.get("image") or ""))
            if not sample_id or not caption:
                raise ValueError(
                    f"{args.strict_captions}:{line_number}: empty ID/caption"
                )
            if sample_id in strict_rows:
                raise ValueError(f"duplicate strict caption ID: {sample_id}")
            if not image.is_file():
                raise FileNotFoundError(image)
            lower = caption.lower()
            blocked = [term for term in BLOCKED_TERMS if term in lower]
            if blocked:
                raise ValueError(f"{sample_id}: blocked terms survived: {blocked}")
            strict_rows[sample_id] = {"caption": caption, "image": image}

    index_rows = {}
    with open_index(args.index) as handle:
        for line in handle:
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in strict_rows:
                index_rows[sample_id] = row
    missing = sorted(set(strict_rows) - set(index_rows))
    if missing:
        raise ValueError(f"{len(missing)} strict IDs missing from canonical index")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"train_caption_{len(strict_rows)}.jsonl"
    source_counts: Counter[str] = Counter()
    with manifest_path.open("w", encoding="utf-8") as target:
        for sample_id in sorted(strict_rows):
            strict = strict_rows[sample_id]
            index = index_rows[sample_id]
            if index["split"] != "train" or not index["vlm_train_used"]:
                raise ValueError(f"{sample_id}: not protected VLM training data")
            source = str(index["source_dataset"])
            suffix = strict["image"].suffix.lower() or ".jpg"
            image_key = Path("images") / source / f"{sample_id}{suffix}"
            materialize(
                strict["image"],
                args.output_dir / image_key,
                args.materialize,
            )
            output = {
                "sample_id": sample_id,
                "source_dataset": source,
                "task": "ir_style_caption",
                "image": image_key.as_posix(),
                "prompt": PROMPT,
                "response": strict["caption"],
                "benchmark_split": "train",
            }
            target.write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")
            source_counts[source] += 1

    summary = {
        "rows": len(strict_rows),
        "unique_sample_ids": len(strict_rows),
        "source_counts": dict(sorted(source_counts.items())),
        "manifest": manifest_path.name,
        "manifest_sha256": sha256(manifest_path),
        "materialize_mode": args.materialize,
        "assistant_only_supervision_required": True,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
