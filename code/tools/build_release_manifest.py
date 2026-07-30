#!/usr/bin/env python3
"""Build deterministic SHA-256 manifests for the public repository."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


EXCLUDED = {".git", "__pycache__"}
SELF_FILES = {"RELEASE_MANIFEST.json", "SHA256SUMS"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or any(part in EXCLUDED for part in relative.parts):
            continue
        if relative.as_posix() in SELF_FILES:
            continue
        rows.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )

    manifest = {
        "release": "FusionRS-rc4-20260727",
        "file_count": len(rows),
        "files": rows,
    }
    (root / "RELEASE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (root / "SHA256SUMS").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

