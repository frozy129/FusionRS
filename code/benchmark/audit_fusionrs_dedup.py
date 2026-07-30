#!/usr/bin/env python3
"""Audit exact and perceptual RGB-image duplicates in the FusionRS split manifest."""

import argparse
import gzip
import hashlib
import io
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn


def phash64(image):
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    coeff = dctn(np.asarray(gray, dtype=np.float32), type=2, norm="ortho")[:8, :8]
    values = coeff.reshape(-1)
    threshold = float(np.median(values[1:]))
    bits = values > threshold
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hash_one(item):
    index, path = item
    try:
        data = Path(path).read_bytes()
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            perceptual = phash64(image)
        return index, hashlib.sha256(data).hexdigest(), perceptual, width, height, ""
    except Exception as exc:
        return index, "", -1, 0, 0, f"{type(exc).__name__}: {exc}"


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = bytearray(size)

    def find(self, value):
        parent = self.parent
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(self, left, right):
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


class BKTree:
    def __init__(self):
        self.values = []
        self.children = []

    @staticmethod
    def distance(left, right):
        return (left ^ right).bit_count()

    def query(self, value, radius):
        if not self.values:
            return []
        matches = []
        stack = [0]
        while stack:
            node = stack.pop()
            distance = self.distance(value, self.values[node])
            if distance <= radius:
                matches.append((node, distance))
            low = distance - radius
            high = distance + radius
            for edge, child in self.children[node].items():
                if low <= edge <= high:
                    stack.append(child)
        return matches

    def add(self, value):
        if not self.values:
            self.values.append(value)
            self.children.append({})
            return 0
        node = 0
        while True:
            distance = self.distance(value, self.values[node])
            child = self.children[node].get(distance)
            if child is None:
                child = len(self.values)
                self.children[node][distance] = child
                self.values.append(value)
                self.children.append({})
                return child
            node = child


def load_manifest(path, max_rows):
    rows = []
    seen_ids = defaultdict(list)
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if max_rows and len(rows) >= max_rows:
                break
            row = json.loads(line)
            record = {
                "sample_id": str(row["sample_id"]),
                "split": str(row["split"]),
                "dataset": str(row["dataset"]),
                "rgb_path": str(row["rgb_path"]),
                "line_number": line_number,
            }
            seen_ids[record["sample_id"]].append(len(rows))
            rows.append(record)
    duplicate_ids = {key: value for key, value in seen_ids.items() if len(value) > 1}
    return rows, duplicate_ids


def group_summary(indices, rows):
    members = [rows[index] for index in indices]
    return {
        "size": len(indices),
        "cross_split": len({row["split"] for row in members}) > 1,
        "cross_dataset": len({row["dataset"] for row in members}) > 1,
        "members": [
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "dataset": row["dataset"],
                "rgb_path": row["rgb_path"],
            }
            for row in members
        ],
    }


def write_jsonl_gz(path, records):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--near-radius", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, duplicate_ids = load_manifest(args.manifest, args.max_rows)
    print(f"loaded_rows={len(rows)} duplicate_sample_ids={len(duplicate_ids)}", flush=True)

    sha_groups = defaultdict(list)
    phash_groups = defaultdict(list)
    errors = []
    hash_rows = [None] * len(rows)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        work = ((index, row["rgb_path"]) for index, row in enumerate(rows))
        for completed, result in enumerate(executor.map(hash_one, work, chunksize=32), 1):
            index, sha256, perceptual, width, height, error = result
            hash_rows[index] = (sha256, perceptual, width, height, error)
            if error:
                errors.append({"index": index, "sample_id": rows[index]["sample_id"], "error": error})
            else:
                sha_groups[sha256].append(index)
                phash_groups[perceptual].append(index)
            if completed % 10000 == 0 or completed == len(rows):
                elapsed = time.time() - started
                print(f"hashed={completed}/{len(rows)} elapsed_s={elapsed:.1f}", flush=True)

    with gzip.open(output_dir / "rgb_hash_manifest.tsv.gz", "wt", encoding="utf-8") as handle:
        handle.write("index\tsample_id\tsplit\tdataset\trgb_path\tsha256\tphash64\twidth\theight\terror\n")
        for index, row in enumerate(rows):
            sha256, perceptual, width, height, error = hash_rows[index]
            phash_text = "" if perceptual < 0 else f"{perceptual:016x}"
            safe_error = error.replace("\t", " ").replace("\n", " ")
            handle.write(
                f"{index}\t{row['sample_id']}\t{row['split']}\t{row['dataset']}\t"
                f"{row['rgb_path']}\t{sha256}\t{phash_text}\t{width}\t{height}\t{safe_error}\n"
            )

    exact_groups = [group_summary(indices, rows) for indices in sha_groups.values() if len(indices) > 1]
    exact_phash_groups = [group_summary(indices, rows) for indices in phash_groups.values() if len(indices) > 1]
    write_jsonl_gz(output_dir / "exact_sha256_groups.jsonl.gz", exact_groups)
    write_jsonl_gz(output_dir / "exact_phash_groups.jsonl.gz", exact_phash_groups)

    union = UnionFind(len(rows))
    for indices in phash_groups.values():
        for index in indices[1:]:
            union.union(indices[0], index)

    tree = BKTree()
    tree_representatives = []
    unique_hashes = list(phash_groups)
    near_edges = []
    for completed, perceptual in enumerate(unique_hashes, 1):
        representative = phash_groups[perceptual][0]
        for node, distance in tree.query(perceptual, args.near_radius):
            other = tree_representatives[node]
            union.union(representative, other)
            near_edges.append(
                {
                    "left": rows[representative]["sample_id"],
                    "right": rows[other]["sample_id"],
                    "distance": distance,
                }
            )
        node = tree.add(perceptual)
        if node == len(tree_representatives):
            tree_representatives.append(representative)
        if completed % 50000 == 0 or completed == len(unique_hashes):
            print(f"near_indexed={completed}/{len(unique_hashes)} edges={len(near_edges)}", flush=True)

    cluster_indices = defaultdict(list)
    for index, item in enumerate(hash_rows):
        if item[1] >= 0:
            cluster_indices[union.find(index)].append(index)
    near_clusters = [
        group_summary(indices, rows)
        for indices in cluster_indices.values()
        if len(indices) > 1
    ]
    write_jsonl_gz(output_dir / "near_phash_clusters.jsonl.gz", near_clusters)
    write_jsonl_gz(output_dir / "near_phash_edges.jsonl.gz", near_edges)
    write_jsonl_gz(
        output_dir / "duplicate_sample_ids.jsonl.gz",
        ({"sample_id": key, "indices": value} for key, value in duplicate_ids.items()),
    )
    write_jsonl_gz(output_dir / "errors.jsonl.gz", errors)

    def counts(groups):
        return {
            "groups": len(groups),
            "samples": sum(group["size"] for group in groups),
            "cross_split_groups": sum(group["cross_split"] for group in groups),
            "cross_dataset_groups": sum(group["cross_dataset"] for group in groups),
        }

    summary = {
        "manifest": args.manifest,
        "rows": len(rows),
        "unique_sample_ids": len(rows) - sum(len(value) - 1 for value in duplicate_ids.values()),
        "duplicate_sample_id_groups": len(duplicate_ids),
        "read_errors": len(errors),
        "near_radius": args.near_radius,
        "exact_sha256": counts(exact_groups),
        "exact_phash": counts(exact_phash_groups),
        "near_phash": counts(near_clusters),
        "near_edges": len(near_edges),
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
