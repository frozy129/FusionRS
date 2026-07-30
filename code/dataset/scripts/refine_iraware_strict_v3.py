#!/usr/bin/env python3
"""Remove residual physical-language and RGB-color leakage from strict-v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

PATTERNS = {
    "thermal_signature": re.compile(r"\bthermal signatures?\b", re.I),
    "heat": re.compile(r"\bheat(?:ed|ing)?\b|\bheat signatures?\b", re.I),
    "temperature": re.compile(
        r"\btemperatures?\b|\bhot(?:ter)?\b|\bcold(?:er)?\b|"
        r"\bwarm(?!-up\b)(?:er)?\b",
        re.I,
    ),
    "emissivity_radiance": re.compile(r"\bemissiv\w*\b|\bradiance\b", re.I),
    "reflectivity": re.compile(r"\breflectiv\w*\b", re.I),
    "material_property": re.compile(
        r"\bmaterials?\b|\bmetallic\b|\bconductiv\w*\b|"
        r"\bsurface propert(?:y|ies)\b",
        re.I,
    ),
    "rgb_color": re.compile(
        r"\b(?:red|green|blue|yellow|orange|purple|pink|brown|golden|gold|"
        r"turquoise)\b",
        re.I,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--removed", type=Path, required=True)
    args = parser.parse_args()
    args.accepted.parent.mkdir(parents=True, exist_ok=True)
    args.removed.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    total = kept = removed = 0
    examples = {}
    with args.input.open(encoding="utf-8") as source, args.accepted.open(
        "w", encoding="utf-8"
    ) as accepted, args.removed.open("w", encoding="utf-8") as rejected:
        for line in source:
            row = json.loads(line)
            caption = str(row.get("caption") or row.get("text") or "")
            flags = [name for name, pattern in PATTERNS.items() if pattern.search(caption)]
            total += 1
            if flags:
                removed += 1
                for flag in flags:
                    counts[flag] += 1
                    examples.setdefault(
                        flag,
                        {
                            "sample_id": row.get("sample_id") or row.get("id"),
                            "caption": caption,
                        },
                    )
                row["strict_v3_flags"] = flags
                rejected.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
            else:
                kept += 1
                accepted.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )

    summary = {
        "input_rows": total,
        "kept": kept,
        "removed": removed,
        "reason_counts": dict(sorted(counts.items())),
        "examples": examples,
        "accepted_sha256": sha256(args.accepted),
        "removed_sha256": sha256(args.removed),
    }
    summary_path = args.accepted.with_suffix(args.accepted.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
