#!/usr/bin/env python3
"""Verify that a public FusionRS index contains no blocked text/image fields."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path


BLOCKED_FIELDS = {
    "caption",
    "normalized_caption_key",
    "class_name",
    "rgb_path",
    "ir_path",
    "image_bytes",
}
ABSOLUTE_PATH = re.compile(r"^(?:/|[A-Za-z]:[\\/])")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=600000)
    args = parser.parse_args()

    counts = Counter()
    sample_ids = set()
    with gzip.open(args.index, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            leaked = BLOCKED_FIELDS.intersection(row)
            if leaked:
                raise ValueError(
                    f"row {line_number}: blocked fields present: {sorted(leaked)}"
                )
            if row.get("schema_version") != "fusionrs-public-index-v1":
                raise ValueError(f"row {line_number}: wrong schema_version")
            sample_id = row["sample_id"]
            if sample_id in sample_ids:
                raise ValueError(f"row {line_number}: duplicate {sample_id}")
            sample_ids.add(sample_id)
            if ABSOLUTE_PATH.match(row["source_locator"]):
                raise ValueError(f"row {line_number}: absolute source path")
            if ABSOLUTE_PATH.match(row["translation"]["output_key"]):
                raise ValueError(f"row {line_number}: absolute output path")
            if not row.get("source_rights_code"):
                raise ValueError(f"row {line_number}: missing rights code")
            counts[(row["source_dataset"], row["split"])] += 1

    if len(sample_ids) != args.expected_rows:
        raise ValueError(
            f"rows={len(sample_ids)}, expected={args.expected_rows}"
        )
    print(
        json.dumps(
            {
                "status": "pass",
                "rows": len(sample_ids),
                "blocked_fields_absent": sorted(BLOCKED_FIELDS),
                "source_split_counts": {
                    f"{source}:{split}": count
                    for (source, split), count in sorted(counts.items())
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
