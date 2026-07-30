#!/usr/bin/env python3
"""Create deterministic nested FusionRS training subsets for the scale ablation."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    rows = []
    raw_rows = 0
    rejected = Counter()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                raw_rows += 1
                row = json.loads(line)
                missing = [key for key in ("rgb_path", "ir_path", "caption") if not row.get(key)]
                if missing:
                    rejected["missing_" + "_".join(missing)] += 1
                    continue
                rows.append(row)
    return rows, raw_rows, rejected


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[50000, 100000, 300000])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows, raw_rows, rejected = read_jsonl(args.train_jsonl)
    if max(args.sizes) > len(rows):
        raise SystemExit(f"requested {max(args.sizes)} rows from a {len(rows)}-row source")
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "source": args.train_jsonl,
        "source_rows_raw": raw_rows,
        "source_rows_usable": len(rows),
        "rejected_rows": raw_rows - len(rows),
        "rejected_by_reason": dict(sorted(rejected.items())),
        "seed": args.seed,
        "policy": "Filter unusable triplets, then apply one deterministic shuffle; every smaller subset is a strict prefix of every larger subset.",
        "subsets": {},
    }
    previous_ids = set()
    for size in sorted(set(args.sizes)):
        subset = [rows[index] for index in order[:size]]
        sample_ids = {str(row["sample_id"]) for row in subset}
        if previous_ids and not previous_ids.issubset(sample_ids):
            raise AssertionError("subsets are not nested")
        path = output_dir / f"train_{size // 1000}k.jsonl"
        write_jsonl(path, subset)
        summary["subsets"][str(size)] = {
            "path": str(path),
            "rows": len(subset),
            "unique_sample_ids": len(sample_ids),
            "by_dataset": dict(sorted(Counter(row.get("dataset", "") for row in subset).items())),
        }
        previous_ids = sample_ids

    (output_dir / "scale_subset_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
