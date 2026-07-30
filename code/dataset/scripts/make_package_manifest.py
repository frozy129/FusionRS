#!/usr/bin/env python3
"""Create deterministic hashes for the public FusionRS dataset package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXCLUDED_NAMES = {"RELEASE_MANIFEST.json", "SHA256SUMS"}
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_files(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.name not in EXCLUDED_NAMES
            and not EXCLUDED_PARTS.intersection(path.parts)
            and not path.name.endswith((".pyc", ".pyo"))
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    files = release_files(root)
    hashes = {
        path.relative_to(root).as_posix(): sha256(path)
        for path in files
    }
    manifest = {
        "schema_version": "fusionrs-package-manifest-v1",
        "package": "FusionRS-dataset",
        "release": "1.0.0-rc4",
        "sha256": hashes,
    }
    manifest_path = root / "RELEASE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sums = [
        f"{digest}  {relative}"
        for relative, digest in sorted(hashes.items())
    ]
    sums.append(f"{sha256(manifest_path)}  RELEASE_MANIFEST.json")
    (root / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "files": len(files),
                "manifest": str(manifest_path),
                "release": manifest["release"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
