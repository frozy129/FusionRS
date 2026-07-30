#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_KS = (1, 5, 10)


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_representatives(path: Path, split: str) -> list[dict]:
    rows = []
    with open_text(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != split or not row.get("is_eval_representative"):
                continue
            if not row.get("caption_usable") or not row.get("normalized_caption_key"):
                raise ValueError(
                    f"Evaluation representative lacks usable text: {row.get('sample_id')}"
                )
            rows.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "caption_key": str(row["normalized_caption_key"]),
                    "source_dataset": str(row["source_dataset"]),
                }
            )
    if not rows:
        raise ValueError(f"No evaluation representatives found for split={split}")
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample_id values in evaluation index")
    return rows


def normalize(matrix: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{name} contains a zero-norm row")
    return matrix / norms


def align_embeddings(path: Path, representatives: list[dict]):
    with np.load(path, allow_pickle=False) as bundle:
        required = {"sample_id", "rgb", "ir", "text"}
        missing = required - set(bundle.files)
        if missing:
            raise ValueError(f"Embedding bundle is missing keys: {sorted(missing)}")
        ids = np.asarray(bundle["sample_id"]).astype(str)
        if ids.ndim != 1:
            raise ValueError("sample_id must be a one-dimensional string array")
        if len(ids) != len(set(ids.tolist())):
            raise ValueError("Embedding bundle contains duplicate sample_id values")
        positions = {sample_id: index for index, sample_id in enumerate(ids.tolist())}
        wanted = [row["sample_id"] for row in representatives]
        absent = [sample_id for sample_id in wanted if sample_id not in positions]
        if absent:
            raise ValueError(
                f"Embedding bundle misses {len(absent)} representatives; first={absent[0]}"
            )
        order = np.asarray([positions[sample_id] for sample_id in wanted])
        arrays = {}
        for name in ("rgb", "ir", "text"):
            raw = np.asarray(bundle[name])
            if raw.shape[0] != len(ids):
                raise ValueError(
                    f"{name} row count {raw.shape[0]} does not match sample_id count {len(ids)}"
                )
            arrays[name] = normalize(raw[order], name)
    dimensions = {name: value.shape[1] for name, value in arrays.items()}
    if len(set(dimensions.values())) != 1:
        raise ValueError(f"Embedding dimensions differ: {dimensions}")
    return arrays


def recall_at_k(
    queries: np.ndarray,
    targets: np.ndarray,
    query_positive_keys: np.ndarray,
    target_positive_keys: np.ndarray,
    ks=DEFAULT_KS,
    chunk_size: int = 512,
) -> dict[str, float]:
    if len(queries) != len(query_positive_keys):
        raise ValueError("Query embeddings and positive keys have different lengths")
    if len(targets) != len(target_positive_keys):
        raise ValueError("Target embeddings and positive keys have different lengths")
    max_k = min(max(ks), len(targets))
    hits = {k: 0 for k in ks}
    for start in range(0, len(queries), chunk_size):
        stop = min(start + chunk_size, len(queries))
        scores = queries[start:stop] @ targets.T
        top = np.argpartition(-scores, kth=max_k - 1, axis=1)[:, :max_k]
        top_scores = np.take_along_axis(scores, top, axis=1)
        order = np.argsort(-top_scores, axis=1, kind="stable")
        top = np.take_along_axis(top, order, axis=1)
        positive = query_positive_keys[start:stop, None] == target_positive_keys[top]
        for k in ks:
            hits[k] += int(np.any(positive[:, : min(k, max_k)], axis=1).sum())
    return {f"R@{k}": 100.0 * hits[k] / len(queries) for k in ks}


def mean_recall(*directions: dict[str, float]) -> float:
    return float(np.mean([value for direction in directions for value in direction.values()]))


def score(index: Path, embeddings: Path, split: str, chunk_size: int) -> dict:
    representatives = load_representatives(index, split)
    arrays = align_embeddings(embeddings, representatives)
    sample_ids = np.asarray([row["sample_id"] for row in representatives])
    caption_keys = np.asarray([row["caption_key"] for row in representatives])

    ir_to_text = recall_at_k(
        arrays["ir"], arrays["text"], caption_keys, caption_keys, chunk_size=chunk_size
    )
    text_to_ir = recall_at_k(
        arrays["text"], arrays["ir"], caption_keys, caption_keys, chunk_size=chunk_size
    )
    rgb_to_ir = recall_at_k(
        arrays["rgb"], arrays["ir"], sample_ids, sample_ids, chunk_size=chunk_size
    )
    ir_to_rgb = recall_at_k(
        arrays["ir"], arrays["rgb"], sample_ids, sample_ids, chunk_size=chunk_size
    )
    text_mean = mean_recall(ir_to_text, text_to_ir)
    pair_mean = mean_recall(rgb_to_ir, ir_to_rgb)
    return {
        "protocol": "fusionrs-rc4-retrieval-v1",
        "split": split,
        "num_representatives": len(representatives),
        "positive_definitions": {
            "image_text": "normalized_caption_key multi-positive",
            "paired_view": "sample_id exact pair",
        },
        "directions": {
            "ir_to_text": ir_to_text,
            "text_to_ir": text_to_ir,
            "rgb_to_ir": rgb_to_ir,
            "ir_to_rgb": ir_to_rgb,
        },
        "text_mean": text_mean,
        "pair_mean": pair_mean,
        "overall_mean": (text_mean + pair_mean) / 2.0,
        "index_sha256": sha256(index),
        "embeddings_sha256": sha256(embeddings),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    result = score(args.index, args.embeddings, args.split, args.chunk_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
