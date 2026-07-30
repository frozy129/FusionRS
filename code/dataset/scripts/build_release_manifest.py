#!/usr/bin/env python3
"""Create an index-only FusionRS manifest without machine-specific paths."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import unicodedata
from pathlib import Path

SOURCE_ROOT_MARKERS = {
    "NWPU": "/datasets/NWPU-Caption/",
    "RS5M": "/datasets/RS5M_sample_500k_clean/",
    "RSICD": "/datasets/RSICD_unpacked/",
    "RSITMD": "/datasets/RSITMD/",
    "SkyScript": "/skyscript_extracted/",
}
RIGHTS_NOTES = {
    "NWPU": "CC BY-NC 4.0 upstream; index-only by default",
    "RS5M": "heterogeneous upstream rights; index-only",
    "RSICD": "no uniform redistribution license located; index-only",
    "RSITMD": "no uniform redistribution license located; index-only",
    "SkyScript": "source-specific Earth Engine terms; index-only",
}


def open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def component_id(members: list[str]) -> str:
    payload = "\n".join(sorted(members)).encode("utf-8")
    return "phash4_" + hashlib.sha256(payload).hexdigest()[:20]


def load_components(
    paths: Path | list[Path],
) -> tuple[dict[str, str], dict[str, str]]:
    if isinstance(paths, Path):
        paths = [paths]
    parent: dict[str, str] = {}
    rank: dict[str, int] = {}

    def add(sample_id: str) -> None:
        if sample_id not in parent:
            parent[sample_id] = sample_id
            rank[sample_id] = 0

    def find(sample_id: str) -> str:
        while parent[sample_id] != sample_id:
            parent[sample_id] = parent[parent[sample_id]]
            sample_id = parent[sample_id]
        return sample_id

    def union(left: str, right: str) -> None:
        left = find(left)
        right = find(right)
        if left == right:
            return
        if rank[left] < rank[right]:
            left, right = right, left
        parent[right] = left
        if rank[left] == rank[right]:
            rank[left] += 1

    for path in paths:
        assigned: set[str] = set()
        with open_text(path, "r") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                members = sorted(
                    str(item["sample_id"]) for item in row["members"]
                )
                if not members:
                    raise ValueError(f"{path}:{line_number}: empty component")
                for sample_id in members:
                    if sample_id in assigned:
                        raise ValueError(
                            f"{path}:{line_number}: repeated component member "
                            f"{sample_id}"
                        )
                    assigned.add(sample_id)
                    add(sample_id)
                for sample_id in members[1:]:
                    union(members[0], sample_id)

    groups: dict[str, list[str]] = {}
    for sample_id in parent:
        groups.setdefault(find(sample_id), []).append(sample_id)
    mapping: dict[str, str] = {}
    canonical: dict[str, str] = {}
    for members in groups.values():
        members.sort()
        cid = component_id(members)
        representative = members[0]
        for sample_id in members:
            mapping[sample_id] = cid
            canonical[sample_id] = representative
    return mapping, canonical


def source_locator(dataset: str, rgb_path: str) -> str:
    marker = SOURCE_ROOT_MARKERS[dataset]
    normalized = rgb_path.replace("\\", "/")
    if marker not in normalized:
        raise ValueError(f"{dataset}: cannot sanitize source path {rgb_path!r}")
    return normalized.split(marker, 1)[1]


def normalize_caption(caption: str) -> str:
    normalized = unicodedata.normalize("NFKC", caption).casefold().strip()
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def normalized_caption_key(caption: str) -> str:
    normalized = normalize_caption(caption)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_unique_representatives(paths: list[Path]) -> set[str]:
    representatives: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                sample_id = str(row["sample_id"])
                representatives.add(sample_id)
    return representatives


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--near-clusters", type=Path, action="append", required=True
    )
    parser.add_argument("--val-unique-pairs", type=Path)
    parser.add_argument("--test-unique-pairs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    components, canonical = load_components(args.near_clusters)
    representative_paths = [
        path
        for path in (args.val_unique_pairs, args.test_unique_pairs)
        if path is not None
    ]
    eval_representatives = load_unique_representatives(representative_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    source_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    with args.manifest.open(encoding="utf-8") as source, open_text(
        args.output, "w"
    ) as target:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            dataset = str(row["dataset"])
            caption = str(row.get("caption") or "")
            normalized_caption = normalize_caption(caption)
            cid = components.get(sample_id, f"singleton_{sample_id}")
            representative = canonical.get(sample_id, sample_id)
            output = {
                "schema_version": "fusionrs-index-v1",
                "sample_id": sample_id,
                "source_dataset": dataset,
                "source_record_id": sample_id,
                "source_locator": source_locator(dataset, str(row["rgb_path"])),
                "source_rights_note": RIGHTS_NOTES[dataset],
                "split": row["benchmark_split"],
                "class_name": (
                    row.get("class_name") or None if dataset == "NWPU" else None
                ),
                "caption": caption or None,
                "caption_usable": bool(normalized_caption),
                "normalized_caption_key": (
                    normalized_caption_key(caption) if normalized_caption else None
                ),
                "joint_group_id": str(row["joint_group_id"]),
                "phash4_component_id": cid,
                "canonical_component_representative": representative,
                "is_canonical_component_representative": (
                    sample_id == representative
                ),
                "is_eval_representative": sample_id in eval_representatives,
                "vlm_train_used": bool(row.get("VLM_train_used")),
                "prior_reused_protected": bool(row.get("prior_reused_protected")),
                "translation": {
                    "model": "DiffV2IR",
                    "output_key": f"{dataset}/{sample_id}.png",
                    "redistribution": "conditional_on_upstream_rights",
                },
            }
            target.write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")
            rows += 1
            source_counts[dataset] = source_counts.get(dataset, 0) + 1
            split = str(output["split"])
            split_counts[split] = split_counts.get(split, 0) + 1

    summary = {
        "rows": rows,
        "source_counts": dict(sorted(source_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "non_singleton_members": len(components),
        "eval_representatives": len(eval_representatives),
        "output": args.output.name,
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
