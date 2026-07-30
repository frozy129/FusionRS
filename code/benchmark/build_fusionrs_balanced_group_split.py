#!/usr/bin/env python3
"""Build a deterministic, source-balanced, radius-4 group-disjoint FusionRS split.

The source manifest is JSONL.  The near-cluster file follows the format emitted by
the pHash radius-4 audit: one JSON object per line with ``members[].sample_id``.
Samples absent from that file are treated as singleton clusters.
"""

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


SPLITS = ("train", "val", "test")
HELDOUT_SPLITS = ("val", "test")


@dataclass(frozen=True)
class Cluster:
    key: str
    indices: tuple
    size: int
    source_counts: Counter
    original_counts: Counter
    owner: str
    protected: bool


def open_text(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def read_jsonl(path):
    rows = []
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def source_of(row):
    source = row.get("dataset", row.get("source", ""))
    if not source:
        raise RuntimeError(f"row has no dataset/source field: {row.get('sample_id', '<unknown>')}")
    return str(source)


def is_true(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def stable_digest(namespace, key):
    return hashlib.sha256(f"{namespace}:{key}".encode("utf-8")).hexdigest()


def read_cluster_keys(path, rows):
    sample_to_index = {}
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise RuntimeError(f"manifest row {index} has no sample_id")
        if sample_id in sample_to_index:
            raise RuntimeError(f"duplicate sample_id in manifest: {sample_id}")
        sample_to_index[sample_id] = index

    keys = [f"singleton:{row['sample_id']}" for row in rows]
    assigned = set()
    with open_text(path) as handle:
        cluster_number = 0
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            cluster = json.loads(line)
            members = cluster.get("members")
            if not isinstance(members, list) or not members:
                raise RuntimeError(f"near cluster at line {line_number} has no members")
            key = f"near-cluster:{cluster_number}"
            cluster_number += 1
            for member in members:
                sample_id = str(member.get("sample_id", ""))
                if sample_id not in sample_to_index:
                    raise RuntimeError(
                        f"near-cluster member is absent from manifest: {sample_id}"
                    )
                index = sample_to_index[sample_id]
                if index in assigned:
                    raise RuntimeError(f"near clusters overlap at sample_id={sample_id}")
                assigned.add(index)
                keys[index] = key
    return keys


def build_clusters(rows, keys):
    grouped = defaultdict(list)
    for index, key in enumerate(keys):
        grouped[key].append(index)

    clusters = {}
    for key, indices_list in grouped.items():
        indices = tuple(indices_list)
        source_counts = Counter(source_of(rows[index]) for index in indices)
        original_counts = Counter(rows[index].get("split", "") for index in indices)
        unknown_splits = set(original_counts) - set(SPLITS)
        if unknown_splits:
            raise RuntimeError(f"cluster {key} has invalid original splits: {unknown_splits}")
        owner = max(
            SPLITS,
            key=lambda split: (original_counts[split], -SPLITS.index(split)),
        )
        protected = any(
            is_true(rows[index].get("VLM_train_used", False))
            or is_true(rows[index].get("prior_reused_protected", False))
            for index in indices
        )
        clusters[key] = Cluster(
            key=key,
            indices=indices,
            size=len(indices),
            source_counts=source_counts,
            original_counts=original_counts,
            owner=owner,
            protected=protected,
        )
    return clusters


def parse_source_minimums(sources, default_minimum, overrides):
    minimums = {source: default_minimum for source in sources}
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--source-min must be SOURCE=N, got: {item}")
        source, value = item.rsplit("=", 1)
        if source not in minimums:
            raise SystemExit(f"--source-min names an unknown source: {source}")
        try:
            minimum = int(value)
        except ValueError as exc:
            raise SystemExit(f"invalid --source-min value: {item}") from exc
        if minimum < 0:
            raise SystemExit(f"--source-min cannot be negative: {item}")
        minimums[source] = minimum
    return minimums


def assignment_score(cluster, target_split, desired_source, salt):
    if cluster.owner == target_split:
        owner_rank = 0
    elif cluster.owner == "train":
        owner_rank = 1
    else:
        owner_rank = 2
    single_source_rank = 0 if len(cluster.source_counts) == 1 else 1
    desired_purity = cluster.source_counts[desired_source] / cluster.size
    return (
        owner_rank,
        single_source_rank,
        cluster.size,
        -desired_purity,
        stable_digest(f"{salt}:quota:{target_split}:{desired_source}", cluster.key),
    )


def fill_score(cluster, target_split, salt):
    if cluster.owner == target_split:
        owner_rank = 0
    elif cluster.owner == "train":
        owner_rank = 1
    else:
        owner_rank = 2
    return (
        owner_rank,
        0 if len(cluster.source_counts) == 1 else 1,
        cluster.size,
        stable_digest(f"{salt}:fill:{target_split}", cluster.key),
    )


def add_cluster(key, split, clusters, assignments, split_sizes, split_sources):
    if key in assignments:
        raise AssertionError(f"cluster assigned twice: {key}")
    cluster = clusters[key]
    assignments[key] = split
    split_sizes[split] += cluster.size
    split_sources[split].update(cluster.source_counts)


def assign_source_minimums(
    clusters, assignments, split_sizes, split_sources, targets, minimums, salt
):
    candidates = {}
    cursors = {}
    available_rows = Counter()
    for cluster in clusters.values():
        if cluster.protected:
            continue
        available_rows.update(cluster.source_counts)

    for source, minimum in minimums.items():
        if available_rows[source] < 2 * minimum:
            raise RuntimeError(
                f"source {source} has only {available_rows[source]} unprotected rows; "
                f"at least {2 * minimum} are required for val+test"
            )
        for split in HELDOUT_SPLITS:
            key = (split, source)
            candidates[key] = sorted(
                (
                    cluster.key
                    for cluster in clusters.values()
                    if not cluster.protected and cluster.source_counts[source] > 0
                ),
                key=lambda cluster_key: assignment_score(
                    clusters[cluster_key], split, source, salt
                ),
            )
            cursors[key] = 0

    source_order = sorted(
        minimums,
        key=lambda source: (
            available_rows[source] / max(1, 2 * minimums[source]),
            source,
        ),
    )
    requirements = [
        (split, source) for source in source_order for split in HELDOUT_SPLITS
    ]

    while True:
        unmet = [
            (split, source)
            for split, source in requirements
            if split_sources[split][source] < minimums[source]
        ]
        if not unmet:
            return
        progress = False
        for split, source in unmet:
            if split_sources[split][source] >= minimums[source]:
                continue
            remaining_capacity = targets[split] - split_sizes[split]
            key = (split, source)
            choices = candidates[key]
            while cursors[key] < len(choices):
                cluster_key = choices[cursors[key]]
                cursors[key] += 1
                if cluster_key in assignments:
                    continue
                if clusters[cluster_key].size > remaining_capacity:
                    continue
                add_cluster(
                    cluster_key,
                    split,
                    clusters,
                    assignments,
                    split_sizes,
                    split_sources,
                )
                progress = True
                break
        if not progress:
            details = {
                f"{split}:{source}": minimums[source] - split_sources[split][source]
                for split, source in unmet
            }
            raise RuntimeError(
                "cannot satisfy source minimums with whole, unprotected clusters; "
                f"remaining deficits={details}"
            )


def exact_subset(candidates, clusters, target):
    """Return a deterministic whole-cluster subset whose sizes sum to target."""
    if target == 0:
        return []
    reachable = 1
    mask = (1 << (target + 1)) - 1
    previous_sum = [-1] * (target + 1)
    previous_key = [None] * (target + 1)

    for key in candidates:
        weight = clusters[key].size
        if weight > target:
            continue
        new_bits = ((reachable << weight) & mask) & ~reachable
        bits = new_bits
        while bits:
            least = bits & -bits
            total = least.bit_length() - 1
            previous_sum[total] = total - weight
            previous_key[total] = key
            bits ^= least
        reachable |= new_bits
        if (reachable >> target) & 1:
            selected = []
            total = target
            while total:
                key_at_total = previous_key[total]
                if key_at_total is None:
                    raise AssertionError("subset reconstruction failed")
                selected.append(key_at_total)
                total = previous_sum[total]
            selected.reverse()
            return selected
    raise RuntimeError(f"cannot exactly fill remaining heldout capacity={target}")


def build_assignments(rows, clusters, targets, minimums, salt):
    if sum(targets.values()) != len(rows):
        raise RuntimeError(
            f"targets sum to {sum(targets.values())}, but manifest has {len(rows)} rows"
        )
    if any(targets[split] < 0 for split in SPLITS):
        raise RuntimeError(f"split targets must be non-negative: {targets}")
    minimum_sum = sum(minimums.values())
    for split in HELDOUT_SPLITS:
        if minimum_sum > targets[split]:
            raise RuntimeError(
                f"sum of per-source minima ({minimum_sum}) exceeds {split} target "
                f"({targets[split]})"
            )

    assignments = {}
    split_sizes = Counter()
    split_sources = {split: Counter() for split in SPLITS}
    for cluster in clusters.values():
        if cluster.protected:
            add_cluster(
                cluster.key,
                "train",
                clusters,
                assignments,
                split_sizes,
                split_sources,
            )
    if split_sizes["train"] > targets["train"]:
        raise RuntimeError(
            f"protected clusters contain {split_sizes['train']} rows, exceeding train target"
        )

    assign_source_minimums(
        clusters,
        assignments,
        split_sizes,
        split_sources,
        targets,
        minimums,
        salt,
    )

    for split in HELDOUT_SPLITS:
        deficit = targets[split] - split_sizes[split]
        candidates = sorted(
            (
                cluster.key
                for cluster in clusters.values()
                if not cluster.protected and cluster.key not in assignments
            ),
            key=lambda key: fill_score(clusters[key], split, salt),
        )
        for key in exact_subset(candidates, clusters, deficit):
            add_cluster(
                key,
                split,
                clusters,
                assignments,
                split_sizes,
                split_sources,
            )

    for cluster in clusters.values():
        if cluster.key not in assignments:
            add_cluster(
                cluster.key,
                "train",
                clusters,
                assignments,
                split_sizes,
                split_sources,
            )

    assert dict(split_sizes) == targets, (dict(split_sizes), targets)
    for split in HELDOUT_SPLITS:
        for source, minimum in minimums.items():
            assert split_sources[split][source] >= minimum, (
                split,
                source,
                split_sources[split][source],
                minimum,
            )
    for cluster in clusters.values():
        if cluster.protected:
            assert assignments[cluster.key] == "train"
    return assignments


def as_triplet(row, split):
    result = dict(row)
    result["split"] = split
    result["benchmark_split"] = split
    return result


def as_pairs(row, split):
    sample_id = row["sample_id"]
    common = {
        "sample_id": sample_id,
        "id": sample_id,
        "dataset": source_of(row),
        "class_name": row.get("class_name", ""),
        "text": row.get("caption", ""),
        "caption": row.get("caption", ""),
        "benchmark_split": split,
        "VLM_train_used": is_true(row.get("VLM_train_used", False)),
    }
    return (
        {**common, "image": row.get("ir_path", ""), "pair_type": "ir_caption"},
        {**common, "image": row.get("rgb_path", ""), "pair_type": "rgb_caption"},
    )


def atomic_write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build_outputs(rows, keys, clusters, assignments):
    split_rows = {split: [] for split in SPLITS}
    manifest_rows = []
    movement = {old: Counter() for old in SPLITS}
    movement_by_source = defaultdict(lambda: {old: Counter() for old in SPLITS})

    for index, row in enumerate(rows):
        split = assignments[keys[index]]
        output_row = as_triplet(row, split)
        split_rows[split].append(output_row)
        manifest_rows.append(output_row)
        old_split = row["split"]
        movement[old_split][split] += 1
        movement_by_source[source_of(row)][old_split][split] += 1

    unique_rows = {split: [] for split in HELDOUT_SPLITS}
    seen = {split: set() for split in HELDOUT_SPLITS}
    key_by_sample_id = {
        str(row["sample_id"]): keys[index] for index, row in enumerate(rows)
    }
    for split in HELDOUT_SPLITS:
        for row in split_rows[split]:
            key = key_by_sample_id[str(row["sample_id"])]
            if key in seen[split]:
                continue
            seen[split].add(key)
            unique_rows[split].append(row)

    return manifest_rows, split_rows, unique_rows, movement, movement_by_source


def make_summary(
    rows,
    clusters,
    assignments,
    split_rows,
    unique_rows,
    movement,
    movement_by_source,
    targets,
    minimums,
    manifest_path,
    near_clusters_path,
    output_dir,
    dry_run,
):
    cluster_splits = defaultdict(set)
    for key, split in assignments.items():
        cluster_splits[key].add(split)
    cross_split = sum(len(values) > 1 for values in cluster_splits.values())
    protected_clusters = [cluster for cluster in clusters.values() if cluster.protected]
    protected_rows = sum(cluster.size for cluster in protected_clusters)
    protected_vlm_rows = sum(
        1 for row in rows if is_true(row.get("VLM_train_used", False))
    )
    protected_reused_rows = sum(
        1 for row in rows if is_true(row.get("prior_reused_protected", False))
    )

    def matrix_to_dict(matrix):
        return {
            old: {new: matrix[old][new] for new in SPLITS}
            for old in SPLITS
        }

    source_counts = {}
    for source in sorted(minimums):
        source_counts[source] = {
            split: sum(source_of(row) == source for row in split_rows[split])
            for split in SPLITS
        }

    summary = {
        "dry_run": dry_run,
        "policy": "phash64_radius4_balanced_group_split",
        "source_manifest": str(manifest_path),
        "near_clusters": str(near_clusters_path),
        "output_dir": str(output_dir),
        "rows": len(rows),
        "targets": targets,
        "counts": {split: len(split_rows[split]) for split in SPLITS},
        "source_minimums_per_heldout_split": minimums,
        "source_counts": source_counts,
        "movement_matrix": matrix_to_dict(movement),
        "movement_matrix_by_source": {
            source: matrix_to_dict(movement_by_source[source])
            for source in sorted(movement_by_source)
        },
        "clusters": {
            "total": len(clusters),
            "singletons": sum(cluster.size == 1 for cluster in clusters.values()),
            "multi_sample": sum(cluster.size > 1 for cluster in clusters.values()),
            "cross_split_after": cross_split,
        },
        "protected": {
            "clusters_forced_train": len(protected_clusters),
            "rows_in_protected_clusters": protected_rows,
            "rows_flagged_VLM_train_used": protected_vlm_rows,
            "rows_flagged_prior_reused_protected": protected_reused_rows,
        },
        "unique_heldout": {
            split: {
                "triplets": len(unique_rows[split]),
                "pairs": 2 * len(unique_rows[split]),
            }
            for split in HELDOUT_SPLITS
        },
        "rule": (
            "VLM_train_used/prior_reused_protected clusters are forced to train. "
            "Heldout source minima are satisfied first; remaining capacity is filled "
            "deterministically with whole clusters, preferring original ownership, "
            "single-source clusters, and small clusters."
        ),
    }

    assert summary["counts"] == targets
    assert cross_split == 0
    assert all(assignments[cluster.key] == "train" for cluster in protected_clusters)
    for split in HELDOUT_SPLITS:
        for source, minimum in minimums.items():
            assert source_counts[source][split] >= minimum
    return summary


def write_outputs(output_dir, manifest_rows, split_rows, unique_rows, summary):
    output_dir = Path(output_dir)
    atomic_write_jsonl(output_dir / "manifest_600k.jsonl", manifest_rows)
    for split in SPLITS:
        atomic_write_jsonl(output_dir / f"{split}_triplets.jsonl", split_rows[split])
    for split in HELDOUT_SPLITS:
        atomic_write_jsonl(
            output_dir / f"{split}_unique_triplets.jsonl", unique_rows[split]
        )
        atomic_write_jsonl(
            output_dir / f"{split}_pairs.jsonl",
            (
                pair
                for row in split_rows[split]
                for pair in as_pairs(row, split)
            ),
        )
        atomic_write_jsonl(
            output_dir / f"{split}_unique_pairs.jsonl",
            (
                pair
                for row in unique_rows[split]
                for pair in as_pairs(row, split)
            ),
        )
    atomic_write_json(output_dir / "summary.json", summary)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="source 600K JSONL manifest")
    parser.add_argument(
        "--near-clusters", required=True, help="radius-4 near-cluster JSONL[.gz]"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-train", type=int, default=580000)
    parser.add_argument("--target-val", type=int, default=10000)
    parser.add_argument("--target-test", type=int, default=10000)
    parser.add_argument("--min-per-source", type=int, default=500)
    parser.add_argument(
        "--source-min",
        action="append",
        default=[],
        metavar="SOURCE=N",
        help="override the heldout minimum for one source; repeat as needed",
    )
    parser.add_argument("--salt", default="fusionrs-balanced-radius4-v1")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate and print summary without writing"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_per_source < 0:
        raise SystemExit("--min-per-source cannot be negative")
    rows = read_jsonl(args.manifest)
    keys = read_cluster_keys(args.near_clusters, rows)
    clusters = build_clusters(rows, keys)
    sources = sorted({source_of(row) for row in rows})
    minimums = parse_source_minimums(sources, args.min_per_source, args.source_min)
    targets = {
        "train": args.target_train,
        "val": args.target_val,
        "test": args.target_test,
    }
    assignments = build_assignments(rows, clusters, targets, minimums, args.salt)

    manifest_rows, split_rows, unique_rows, movement, movement_by_source = build_outputs(
        rows, keys, clusters, assignments
    )
    summary = make_summary(
        rows,
        clusters,
        assignments,
        split_rows,
        unique_rows,
        movement,
        movement_by_source,
        targets,
        minimums,
        args.manifest,
        args.near_clusters,
        args.output_dir,
        args.dry_run,
    )
    if not args.dry_run:
        write_outputs(args.output_dir, manifest_rows, split_rows, unique_rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
