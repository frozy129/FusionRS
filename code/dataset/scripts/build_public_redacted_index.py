#!/usr/bin/env python3
"""Build a deterministic public FusionRS index with uncleared text removed."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
from collections import Counter
from pathlib import Path


BLOCKED_FIELDS = {
    "caption",
    "caption_usable",
    "normalized_caption_key",
    "class_name",
}
SKYSCRIPT_CODE_RE = re.compile(r"_([A-Z0-9]{2})_[^/]*$")
SKYSCRIPT_RIGHTS = {
    "US": "skyscript-naip-public-domain",
    "CH": "skyscript-swissimage-source-reference",
    "DE": "skyscript-brandenburg-dl-de-by-2.0",
    "S2": "skyscript-sentinel-copernicus-notice",
    "FI": "skyscript-finland-nls-cc-by-4.0",
    "ES": "skyscript-pnoa-cc-by-4.0",
    "P3": "skyscript-skysat-rgb-cc-by-4.0",
    "L8": "skyscript-landsat8-public-domain",
    "P5": "skyscript-skysat-multispectral-cc-by-4.0",
    "L9": "skyscript-landsat9-public-domain",
}
SOURCE_RIGHTS = {
    "RS5M": "rs5m-item-rights-unconfirmed",
    "NWPU": "nwpu-captions-text-unlicensed",
    "RSICD": "rsicd-redistribution-license-not-located",
    "RSITMD": "rsitmd-redistribution-license-not-located",
}


def open_gzip_text(path: Path, mode: str):
    if "w" in mode:
        raw = path.open("wb")
        zipped = gzip.GzipFile(fileobj=raw, mode="wb", mtime=0)
        return io.TextIOWrapper(zipped, encoding="utf-8")
    return gzip.open(path, mode, encoding="utf-8")


def rights_code(row: dict) -> str:
    source = row["source_dataset"]
    if source != "SkyScript":
        return SOURCE_RIGHTS[source]
    match = SKYSCRIPT_CODE_RE.search(row["source_locator"])
    if not match:
        raise ValueError(
            f"Cannot determine SkyScript subsource from {row['source_locator']!r}"
        )
    code = match.group(1)
    if code not in SKYSCRIPT_RIGHTS:
        raise ValueError(f"Unknown SkyScript subsource code {code!r}")
    return SKYSCRIPT_RIGHTS[code]


def redact(row: dict) -> dict:
    leaked = BLOCKED_FIELDS.intersection(row)
    if leaked != BLOCKED_FIELDS:
        raise ValueError(f"Unexpected internal schema; text fields={sorted(leaked)}")
    return {
        "schema_version": "fusionrs-public-index-v1",
        "sample_id": row["sample_id"],
        "source_dataset": row["source_dataset"],
        "source_record_id": row["source_record_id"],
        "source_locator": row["source_locator"],
        "source_rights_code": rights_code(row),
        "split": row["split"],
        "joint_group_id": row["joint_group_id"],
        "phash4_component_id": row["phash4_component_id"],
        "canonical_component_representative": (
            row["canonical_component_representative"]
        ),
        "is_canonical_component_representative": (
            row["is_canonical_component_representative"]
        ),
        "is_eval_representative": row["is_eval_representative"],
        "caption_supervision_available_upstream": row["caption_usable"],
        "vlm_train_identity": row["vlm_train_used"],
        "prior_reused_protected": row["prior_reused_protected"],
        "translation": row["translation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    rights_counts = Counter()
    with open_gzip_text(args.input, "rt") as source, open_gzip_text(
        args.output, "wt"
    ) as target:
        for line_number, line in enumerate(source, start=1):
            row = redact(json.loads(line))
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            counts[(row["source_dataset"], row["split"])] += 1
            rights_counts[row["source_rights_code"]] += 1

    summary = {
        "schema_version": "fusionrs-public-index-summary-v1",
        "rows": sum(counts.values()),
        "redacted_fields": sorted(BLOCKED_FIELDS),
        "source_split_counts": {
            f"{source}:{split}": count
            for (source, split), count in sorted(counts.items())
        },
        "rights_code_counts": dict(sorted(rights_counts.items())),
    }
    summary_path = Path(str(args.output) + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
