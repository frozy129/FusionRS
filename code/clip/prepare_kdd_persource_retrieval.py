#!/usr/bin/env python3
"""Split clean FusionRS retrieval manifests by original source dataset."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-pairs", required=True)
    parser.add_argument("--test-pairs", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    grouped = defaultdict(lambda: {"val": [], "test": []})
    for split, path in (("val", args.val_pairs), ("test", args.test_pairs)):
        for row in load(path):
            grouped[str(row.get("dataset", "unknown"))][split].append(row)

    summary = {}
    for source, splits in sorted(grouped.items()):
        summary[source] = {}
        for split, rows in splits.items():
            write(output / source / f"{split}_pairs.jsonl", rows)
            summary[source][split] = {
                "pair_rows": len(rows),
                "unique_sample_ids": len({str(row["sample_id"]) for row in rows}),
                "pair_types": dict(sorted(Counter(row["pair_type"] for row in rows).items())),
            }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
