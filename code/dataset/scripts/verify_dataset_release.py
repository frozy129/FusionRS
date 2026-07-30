#!/usr/bin/env python3
"""Verify canonical FusionRS artifacts and benchmark authoring outputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

SOURCES = {"NWPU", "RS5M", "RSICD", "RSITMD", "SkyScript"}
BLOCKED_PHYSICAL_TERMS = {
    "temperature",
    "thermal signature",
    "heat signature",
    "emissivity",
    "reflectivity",
    "material property",
}
INDEX_FIELDS = {
    "schema_version",
    "sample_id",
    "source_dataset",
    "source_record_id",
    "source_locator",
    "source_rights_note",
    "split",
    "class_name",
    "caption",
    "caption_usable",
    "normalized_caption_key",
    "joint_group_id",
    "phash4_component_id",
    "canonical_component_representative",
    "is_canonical_component_representative",
    "is_eval_representative",
    "vlm_train_used",
    "prior_reused_protected",
    "translation",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def normalize_caption(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def normalized_caption_key(value: object) -> str | None:
    normalized = normalize_caption(value)
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def unsafe_relative_path(value: object) -> bool:
    text = str(value or "")
    return (
        not text
        or text.startswith("/")
        or bool(re.match(r"^[A-Za-z]:\\", text))
        or ".." in Path(text).parts
    )


def verify_artifacts(root: Path, version: dict) -> list[str]:
    errors = []
    for name, expected in version["canonical_artifacts"].items():
        path = root / name
        if not path.is_file():
            errors.append(f"missing canonical artifact: {name}")
            continue
        rows = count_lines(path)
        if rows != expected["rows"]:
            errors.append(f"{name}: rows {rows}, expected {expected['rows']}")
        actual_hash = sha256(path)
        if actual_hash != expected["sha256"]:
            errors.append(f"{name}: sha256 mismatch")
    return errors


def verify_package_files(root: Path) -> list[str]:
    errors = []
    manifest_path = root / "RELEASE_MANIFEST.json"
    sums_path = root / "SHA256SUMS"
    if not manifest_path.is_file() or not sums_path.is_file():
        return ["missing RELEASE_MANIFEST.json or SHA256SUMS"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("sha256", {})
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"RELEASE_MANIFEST.json", "SHA256SUMS"}
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and not path.name.endswith((".pyc", ".pyo"))
    }
    if actual != set(expected):
        for relative in sorted(actual - set(expected)):
            errors.append(f"unexpected package file: {relative}")
        for relative in sorted(set(expected) - actual):
            errors.append(f"missing package file: {relative}")
    for relative, expected_hash in expected.items():
        path = root / relative
        if path.is_file() and sha256(path) != expected_hash:
            errors.append(f"package hash mismatch: {relative}")

    sums = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        sums[relative] = digest
    if sums.get("RELEASE_MANIFEST.json") != sha256(manifest_path):
        errors.append("SHA256SUMS has wrong RELEASE_MANIFEST.json hash")
    for relative, digest in expected.items():
        if sums.get(relative) != digest:
            errors.append(f"SHA256SUMS mismatch: {relative}")
    return errors


def verify_release_artifacts(root: Path, version: dict) -> list[str]:
    errors = []
    for relative, expected in version.get("release_artifacts", {}).items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing release artifact: {relative}")
            continue
        if sha256(path) != expected["sha256"]:
            errors.append(f"{relative}: sha256 mismatch")
        if "rows" in expected:
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8") as handle:
                rows = sum(1 for _ in handle)
            if rows != expected["rows"]:
                errors.append(
                    f"{relative}: rows {rows}, expected {expected['rows']}"
                )
        if expected.get("contains_machine_specific_paths") is False:
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if "/root/" in line or re.search(r"[A-Za-z]:\\\\", line):
                        errors.append(
                            f"{relative} row {line_number}: machine-specific path"
                        )
                        break
    return errors


def verify_iraware_manifests(root: Path, version: dict) -> list[str]:
    errors = []
    expected_counts = version["expected_counts"]
    specifications = (
        (
            "manifests/iraware_strict_v3_45913.jsonl.gz",
            "accepted_strict_v3",
            expected_counts["iraware_strict_v3_train"],
        ),
        (
            "manifests/iraware_removed_v2_2659.jsonl.gz",
            "removed_strict_v2",
            expected_counts["iraware_removed_v2"],
        ),
        (
            "manifests/iraware_removed_v3_additional_44.jsonl.gz",
            "removed_strict_v3",
            expected_counts["iraware_removed_v3_additional"],
        ),
    )
    seen: set[str] = set()
    for relative, expected_status, expected_rows in specifications:
        path = root / relative
        if not path.is_file():
            errors.append(f"missing IR-aware manifest: {relative}")
            continue
        count = 0
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                count += 1
                sample_id = str(row.get("sample_id", ""))
                if not sample_id:
                    errors.append(f"{relative} row {line_number}: empty sample_id")
                elif sample_id in seen:
                    errors.append(
                        f"{relative} row {line_number}: duplicate across manifests"
                    )
                seen.add(sample_id)
                if row.get("status") != expected_status:
                    errors.append(
                        f"{relative} row {line_number}: wrong status"
                    )
                if row.get("split") != "train":
                    errors.append(
                        f"{relative} row {line_number}: non-train caption"
                    )
                if row.get("source_dataset") not in SOURCES:
                    errors.append(
                        f"{relative} row {line_number}: invalid source"
                    )
        if count != expected_rows:
            errors.append(f"{relative}: rows {count}, expected {expected_rows}")
    return errors


def verify_real_thermal_vqa(root: Path) -> list[str]:
    errors = []
    manifest = root / "manifests/hit_uav_real_thermal_vqa_test.jsonl.gz"
    summary_path = manifest.with_suffix(manifest.suffix + ".summary.json")
    if not manifest.is_file() or not summary_path.is_file():
        return ["missing HIT-UAV VQA manifest or summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ids = set()
    image_ids = set()
    type_counts: Counter[str] = Counter()
    presence_counts: Counter[str] = Counter()
    with gzip.open(manifest, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            question_id = str(row.get("question_id", ""))
            if not question_id or question_id in ids:
                errors.append(
                    f"HIT-UAV VQA row {line_number}: empty or duplicate question_id"
                )
            ids.add(question_id)
            image_ids.add(str(row.get("image_id", "")))
            question_type = str(row.get("question_type", ""))
            type_counts[question_type] += 1
            if question_type == "object_presence":
                presence_counts[str(row.get("answer", ""))] += 1
            locator = str(row.get("image_locator", ""))
            if not locator or locator.startswith("/") or ".." in Path(locator).parts:
                errors.append(
                    f"HIT-UAV VQA row {line_number}: unsafe image locator"
                )
            if row.get("label_source") != "official_bbox_annotation":
                errors.append(
                    f"HIT-UAV VQA row {line_number}: unsupported label source"
                )
    if len(ids) != summary["questions"]:
        errors.append(
            f"HIT-UAV VQA questions {len(ids)}, expected {summary['questions']}"
        )
    if len(image_ids) != summary["images"]:
        errors.append(
            f"HIT-UAV VQA images {len(image_ids)}, expected {summary['images']}"
        )
    if dict(sorted(type_counts.items())) != summary["question_type_counts"]:
        errors.append("HIT-UAV VQA question-type counts differ from summary")
    if dict(sorted(presence_counts.items())) != summary["presence_answer_counts"]:
        errors.append("HIT-UAV VQA presence counts differ from summary")
    if presence_counts["yes"] != presence_counts["no"]:
        errors.append("HIT-UAV VQA presence questions are not balanced")
    return errors


def verify_benchmark(path: Path, expected_per_source: int) -> list[str]:
    errors = []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    ids = [row.get("sample_id") for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("benchmark contains duplicate sample IDs")
    joint_groups = [row.get("joint_group_id") for row in rows]
    if not all(joint_groups):
        errors.append("benchmark contains empty joint_group_id")
    elif len(joint_groups) != len(set(joint_groups)):
        errors.append("benchmark contains duplicate joint groups")
    counts = Counter(row.get("dataset") for row in rows)
    if set(counts) != SOURCES:
        errors.append(f"benchmark source set is {sorted(counts)}")
    for source in sorted(SOURCES):
        if counts[source] != expected_per_source:
            errors.append(
                f"{source}: {counts[source]} benchmark rows, "
                f"expected {expected_per_source}"
            )
    for row in rows:
        if row.get("benchmark_split") != "test":
            errors.append(f"{row.get('sample_id')}: not from test split")
        if row.get("VLM_train_used") is True:
            errors.append(f"{row.get('sample_id')}: VLM training leakage")
        for key in ("rgb_image", "ir_image"):
            value = row.get(key)
            if not value or not (path.parent / value).is_file():
                errors.append(f"{row.get('sample_id')}: missing {key}")
    return errors


def verify_index_manifest(path: Path, version: dict) -> list[str]:
    errors = []
    sample_ids: set[str] = set()
    component_splits: dict[str, str] = {}
    split_counts: Counter[str] = Counter()
    eval_counts: Counter[str] = Counter()
    source_split_counts: Counter[tuple[str, str]] = Counter()
    caption_splits: dict[str, str] = {}
    joint_group_splits: dict[str, str] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if "/root/" in line or re.search(r"[A-Za-z]:\\\\", line):
                errors.append(f"index row {line_number}: machine-specific path")
            row = json.loads(line)
            if set(row) != INDEX_FIELDS:
                errors.append(
                    f"index row {line_number}: schema fields differ from v1"
                )
            if row.get("schema_version") != "fusionrs-index-v1":
                errors.append(f"index row {line_number}: wrong schema_version")
            if row.get("source_dataset") not in SOURCES:
                errors.append(f"index row {line_number}: invalid source")
            if row.get("split") not in {"train", "val", "test"}:
                errors.append(f"index row {line_number}: invalid split")
            caption = row.get("caption")
            expected_caption_key = normalized_caption_key(caption)
            if row.get("caption_usable") != bool(expected_caption_key):
                errors.append(
                    f"index row {line_number}: caption_usable mismatch"
                )
            if row.get("normalized_caption_key") != expected_caption_key:
                errors.append(
                    f"index row {line_number}: normalized_caption_key mismatch"
                )
            locator = str(row.get("source_locator", ""))
            if unsafe_relative_path(locator):
                errors.append(f"index row {line_number}: unsafe source_locator")
            output_key = row.get("translation", {}).get("output_key")
            if unsafe_relative_path(output_key):
                errors.append(f"index row {line_number}: unsafe translation output")
            sample_id = str(row["sample_id"])
            if sample_id in sample_ids:
                errors.append(f"duplicate index sample_id: {sample_id}")
            sample_ids.add(sample_id)
            split = str(row["split"])
            component = str(row["phash4_component_id"])
            prior_split = component_splits.setdefault(component, split)
            if prior_split != split:
                errors.append(
                    f"component {component} crosses {prior_split}/{split}"
                )
            joint_group = str(row.get("joint_group_id") or "")
            if not joint_group:
                errors.append(f"index row {line_number}: empty joint_group_id")
            else:
                prior_joint_split = joint_group_splits.setdefault(
                    joint_group, split
                )
                if prior_joint_split != split:
                    errors.append(
                        f"joint group {joint_group} crosses "
                        f"{prior_joint_split}/{split}"
                    )
            split_counts[split] += 1
            source = str(row["source_dataset"])
            source_split_counts[(source, split)] += 1
            if source != "NWPU" and row.get("class_name") is not None:
                errors.append(
                    f"index row {line_number}: unverified non-NWPU class_name"
                )
            if expected_caption_key:
                prior_caption_split = caption_splits.setdefault(
                    expected_caption_key, split
                )
                if prior_caption_split != split:
                    errors.append(
                        f"caption key crosses {prior_caption_split}/{split}"
                    )
            if row.get("is_eval_representative"):
                eval_counts[split] += 1
                if split == "train":
                    errors.append(
                        f"index row {line_number}: train eval representative"
                    )

    expected = version["expected_counts"]
    expected_splits = {
        "train": expected["train"],
        "val": expected["validation"],
        "test": expected["test"],
    }
    if len(sample_ids) != expected["manifest"]:
        errors.append(
            f"index rows/IDs {len(sample_ids)}, expected {expected['manifest']}"
        )
    if dict(split_counts) != expected_splits:
        errors.append(
            f"index split counts {dict(split_counts)}, expected {expected_splits}"
        )
    expected_eval = {
        "val": expected["validation_unique_representatives"],
        "test": expected["test_unique_representatives"],
    }
    if dict(eval_counts) != expected_eval:
        errors.append(
            f"evaluation representatives {dict(eval_counts)}, "
            f"expected {expected_eval}"
        )
    observed_sources = {
        source for source, _split in source_split_counts
    }
    if observed_sources != SOURCES:
        errors.append(f"index source set is {sorted(observed_sources)}")
    heldout_minimum = int(version.get("heldout_minimum_per_source", 500))
    for source in sorted(SOURCES):
        for split in ("val", "test"):
            count = source_split_counts[(source, split)]
            if count < heldout_minimum:
                errors.append(
                    f"{source}:{split} has {count}, below {heldout_minimum}"
                )
    return errors


def verify_annotations(benchmark_dir: Path) -> list[str]:
    errors = []
    for annotator in range(1, 4):
        path = benchmark_dir / f"caption_reference_annotator_{annotator}.csv"
        if not path.is_file():
            errors.append(f"missing {path.name}")
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        complete = sum(bool(row["reference_caption"].strip()) for row in rows)
        if complete != len(rows):
            errors.append(f"{path.name}: {complete}/{len(rows)} captions complete")

    for author in range(1, 3):
        author_path = benchmark_dir / f"vqa_author_{author}.csv"
        if not author_path.is_file():
            errors.append(f"missing {author_path.name}")

    vqa_path = benchmark_dir / "vqa_adjudication.csv"
    if not vqa_path.is_file():
        errors.append("missing vqa_adjudication.csv")
        return errors
    with vqa_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    complete = 0
    for row in rows:
        question = row["selected_question"].strip()
        answer = row["canonical_answer"].strip()
        if (
            question
            and answer
            and row["observable_in_ir_0_or_1"] == "1"
            and row["unambiguous_0_or_1"] == "1"
        ):
            complete += 1
        combined = f"{question} {answer}".lower()
        for term in BLOCKED_PHYSICAL_TERMS:
            if term in combined:
                errors.append(
                    f"{row['benchmark_id']}:{row['question_type']}: "
                    f"blocked term {term!r}"
                )
    if complete != len(rows):
        errors.append(
            f"vqa_adjudication.csv: {complete}/{len(rows)} rows complete"
        )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--index-manifest", type=Path)
    parser.add_argument("--verify-package-files", action="store_true")
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--per-source", type=int, default=100)
    parser.add_argument("--require-complete-annotations", action="store_true")
    args = parser.parse_args()

    version = json.loads(
        (args.dataset_root / "CANONICAL_VERSION.json").read_text(encoding="utf-8")
    )
    errors = []
    errors.extend(verify_release_artifacts(args.dataset_root, version))
    errors.extend(verify_iraware_manifests(args.dataset_root, version))
    errors.extend(verify_real_thermal_vqa(args.dataset_root))
    if args.verify_package_files:
        errors.extend(verify_package_files(args.dataset_root))
    if args.artifact_root:
        errors.extend(verify_artifacts(args.artifact_root, version))
    if args.index_manifest:
        errors.extend(verify_index_manifest(args.index_manifest, version))
    if args.benchmark_dir:
        errors.extend(
            verify_benchmark(
                args.benchmark_dir / "candidates.jsonl", args.per_source
            )
        )
        if args.require_complete_annotations:
            errors.extend(verify_annotations(args.benchmark_dir))

    result = {"passed": not errors, "errors": errors}
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
