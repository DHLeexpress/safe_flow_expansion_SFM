#!/usr/bin/env python3
"""Build the byte-level package inventory, excluding Git and the inventory itself."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SOURCE_MANIFEST.json"
EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__"}
EXCLUDED_FILES = {OUTPUT.name, ".DS_Store"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name in EXCLUDED_FILES:
            continue
        relative = path.relative_to(ROOT)
        if EXCLUDED_PARTS.intersection(relative.parts):
            continue
        rows.append({
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        })
    payload = {
        "status": "SAFE_FLOW_EXPANSION_SFM_PACKAGE",
        "source_lineage": "safeMPPI@e5ab47ba4971aae6c1df710c6d6864577f3728f7",
        "files": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUTPUT} with {len(rows)} files")


if __name__ == "__main__":
    main()

