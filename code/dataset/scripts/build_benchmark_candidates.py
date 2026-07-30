#!/usr/bin/env python3
"""Build a deterministic, source-balanced held-out Caption/VQA authoring set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

SOURCES = ("NWPU", "RS5M", "RSICD", "RSITMD", "SkyScript")
QUESTION_TYPES = (
    "scene_category",
    "object_presence",
    "spatial_structure",
    "intensity_contrast",
)
CAPTION_TASKS = (
    "semantic_description",
    "ir_observable_description",
)


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def load_pairs(path: Path) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(f"{path}:{line_number}: missing sample_id")
            if row.get("benchmark_split") != "test":
                raise ValueError(f"{sample_id}: expected benchmark_split=test")
            if row.get("VLM_train_used") is True:
                raise ValueError(f"{sample_id}: held-out item was used by VLM training")
            pair_type = row.get("pair_type")
            if pair_type not in {"rgb_caption", "ir_caption"}:
                raise ValueError(f"{sample_id}: unexpected pair_type={pair_type!r}")

            record = grouped.setdefault(
                sample_id,
                {
                    "sample_id": sample_id,
                    "dataset": row.get("dataset", ""),
                    "class_name": row.get("class_name", ""),
                    "original_caption": row.get("caption") or row.get("text") or "",
                    "benchmark_split": "test",
                    "VLM_train_used": False,
                    "joint_group_id": row.get("joint_group_id", ""),
                },
            )
            for key in ("dataset", "class_name", "joint_group_id"):
                if record[key] != row.get(key, ""):
                    raise ValueError(f"{sample_id}: inconsistent {key}")
            image_key = "rgb_path" if pair_type == "rgb_caption" else "ir_path"
            if image_key in record:
                raise ValueError(f"{sample_id}: duplicate {pair_type} row")
            record[image_key] = row.get("image", "")

    required = {"rgb_path", "ir_path"}
    incomplete = [sid for sid, row in grouped.items() if not required <= row.keys()]
    if incomplete:
        raise ValueError(f"{len(incomplete)} records lack paired RGB/IR rows")
    missing_groups = [
        sid for sid, row in grouped.items() if not row["joint_group_id"]
    ]
    if missing_groups:
        raise ValueError(
            f"{len(missing_groups)} records lack joint_group_id"
        )
    return grouped


def select_source_balanced(
    grouped: dict[str, dict], per_source: int, seed: int
) -> list[dict]:
    by_source_class: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in grouped.values():
        source = row["dataset"]
        if source not in SOURCES:
            raise ValueError(f"unsupported source {source!r}")
        class_key = (
            str(row.get("class_name") or "__unlabeled__")
            if source == "NWPU"
            else "__unlabeled__"
        )
        by_source_class[source][class_key].append(row)

    selected: list[dict] = []
    used_groups: set[str] = set()
    source_order = sorted(
        SOURCES,
        key=lambda source: (
            len(
                {
                    row["joint_group_id"]
                    for rows in by_source_class[source].values()
                    for row in rows
                }
            ),
            SOURCES.index(source),
        ),
    )
    for source in source_order:
        buckets = by_source_class[source]
        available = len(
            {
                row["joint_group_id"]
                for rows in buckets.values()
                for row in rows
                if row["joint_group_id"] not in used_groups
            }
        )
        if available < per_source:
            raise ValueError(
                f"{source}: only {available} unused joint groups, "
                f"need {per_source}"
            )
        for class_name, rows in buckets.items():
            rows.sort(key=lambda row: stable_key(seed, row["sample_id"]))
        class_order = sorted(
            buckets, key=lambda name: stable_key(seed, f"{source}:{name}")
        )

        source_rows: list[dict] = []
        while len(source_rows) < per_source:
            progressed = False
            for class_name in class_order:
                while (
                    buckets[class_name]
                    and buckets[class_name][0]["joint_group_id"] in used_groups
                ):
                    buckets[class_name].pop(0)
                if buckets[class_name]:
                    row = buckets[class_name].pop(0)
                    source_rows.append(row)
                    used_groups.add(row["joint_group_id"])
                    progressed = True
                    if len(source_rows) == per_source:
                        break
            if not progressed:
                raise RuntimeError(f"{source}: selection exhausted unexpectedly")
        selected.extend(source_rows)

    selected.sort(key=lambda row: (SOURCES.index(row["dataset"]), row["sample_id"]))
    for index, row in enumerate(selected, 1):
        row["benchmark_id"] = f"FBQ-{index:04d}"
    return selected


def copy_pair(row: dict, image_dir: Path) -> tuple[str, str]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "--copy-images requires Pillow; install it with `pip install Pillow`"
        ) from exc

    image_dir.mkdir(parents=True, exist_ok=True)
    rgb_source = Path(row["rgb_path"])
    ir_source = Path(row["ir_path"])
    if not rgb_source.is_file() or not ir_source.is_file():
        raise FileNotFoundError(f"{row['sample_id']}: source image missing")
    rgb_target = image_dir / f"{row['benchmark_id']}_rgb.jpg"
    ir_target = image_dir / f"{row['benchmark_id']}_ir.jpg"
    for source, target in ((rgb_source, rgb_target), (ir_source, ir_target)):
        with Image.open(source) as image:
            image.convert("RGB").save(target, format="JPEG", quality=95)
    return str(Path("images") / rgb_target.name), str(Path("images") / ir_target.name)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_annotation_forms(output_dir: Path, selected: list[dict]) -> None:
    caption_fields = [
        "benchmark_id",
        "dataset",
        "ir_image",
        "task_type",
        "reference_caption",
        "observable_only_0_or_1",
        "comments",
    ]
    for annotator in range(1, 4):
        rows = [
            {
                "benchmark_id": row["benchmark_id"],
                "dataset": row["dataset"],
                "ir_image": row["ir_image"],
                "task_type": task_type,
                "reference_caption": "",
                "observable_only_0_or_1": "",
                "comments": "",
            }
            for row in selected
            for task_type in CAPTION_TASKS
        ]
        write_csv(
            output_dir / f"caption_reference_annotator_{annotator}.csv",
            caption_fields,
            rows,
        )

    vqa_fields = [
        "benchmark_id",
        "question_id",
        "dataset",
        "ir_image",
        "question_type",
        "question",
        "answer",
        "answer_type",
        "observable_in_ir_0_or_1",
        "comments",
    ]
    for author in range(1, 3):
        vqa_rows = []
        for row in selected:
            for question_index, question_type in enumerate(QUESTION_TYPES, 1):
                vqa_rows.append(
                    {
                        "benchmark_id": row["benchmark_id"],
                        "question_id": (
                            f"{row['benchmark_id']}-A{author}-Q{question_index}"
                        ),
                        "dataset": row["dataset"],
                        "ir_image": row["ir_image"],
                        "question_type": question_type,
                        "question": "",
                        "answer": "",
                        "answer_type": "short_text",
                        "observable_in_ir_0_or_1": "",
                        "comments": "",
                    }
                )
        write_csv(
            output_dir / f"vqa_author_{author}.csv",
            vqa_fields,
            vqa_rows,
        )

    adjudication_rows = [
        {
            "benchmark_id": row["benchmark_id"],
            "dataset": row["dataset"],
            "ir_image": row["ir_image"],
            "question_type": question_type,
            "selected_question": "",
            "canonical_answer": "",
            "accepted_author": "",
            "unambiguous_0_or_1": "",
            "observable_in_ir_0_or_1": "",
            "adjudicator_comments": "",
        }
        for row in selected
        for question_type in QUESTION_TYPES
    ]
    write_csv(
        output_dir / "vqa_adjudication.csv",
        [
            "benchmark_id",
            "dataset",
            "ir_image",
            "question_type",
            "selected_question",
            "canonical_answer",
            "accepted_author",
            "unambiguous_0_or_1",
            "observable_in_ir_0_or_1",
            "adjudicator_comments",
        ],
        adjudication_rows,
    )
    (output_dir / "ANNOTATION_GUIDE.md").write_text(
        """# FusionRS benchmark annotation guide

## General rules

- Work independently and inspect only the infrared-style image named in the form.
- Do not open the original caption, another annotator's form, or model output.
- Describe only information visible in the image.
- Do not infer temperature, heat, emissivity, reflectivity, material properties,
  or RGB color.
- Mark an item unusable in comments when the image is ambiguous or corrupted.

## Caption tasks

- `semantic_description`: describe the scene and visible objects.
- `ir_observable_description`: describe grayscale intensity, contrast, edges,
  texture, and spatial structure without physical interpretation.
- Each reference should be a complete English sentence.

## VQA authoring

Each author writes one question and short answer for every required type:

- `scene_category`
- `object_presence`
- `spatial_structure`
- `intensity_contrast`

Questions must be answerable from the infrared-style image alone. Avoid
questions whose answer is copied from wording in an existing caption.

## Adjudication

The adjudicator compares the two independently authored versions, chooses or
rewrites one canonical question/answer, and accepts only unambiguous,
image-grounded items. Test labels remain sealed until model and decoding
settings are frozen.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-pairs",
        "--test-unique-pairs",
        dest="test_pairs",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--copy-images", action="store_true")
    args = parser.parse_args()

    grouped = load_pairs(args.test_pairs)
    selected = select_source_balanced(grouped, args.per_source, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    public_rows = []
    for row in selected:
        public_row = dict(row)
        if args.copy_images:
            rgb_image, ir_image = copy_pair(row, args.output_dir / "images")
            public_row["rgb_image"] = rgb_image
            public_row["ir_image"] = ir_image
        else:
            public_row["rgb_image"] = row["rgb_path"]
            public_row["ir_image"] = row["ir_path"]
        public_row.pop("rgb_path", None)
        public_row.pop("ir_path", None)
        public_rows.append(public_row)

    write_jsonl(args.output_dir / "candidates.jsonl", public_rows)
    build_annotation_forms(args.output_dir, public_rows)
    summary = {
        "seed": args.seed,
        "per_source": args.per_source,
        "total": len(public_rows),
        "source_counts": {
            source: sum(row["dataset"] == source for row in public_rows)
            for source in SOURCES
        },
        "caption_tasks": list(CAPTION_TASKS),
        "caption_references_required": len(public_rows) * len(CAPTION_TASKS) * 3,
        "vqa_author_rows": len(public_rows) * len(QUESTION_TYPES) * 2,
        "vqa_adjudicated_rows": len(public_rows) * len(QUESTION_TYPES),
        "status": "annotation_required",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
