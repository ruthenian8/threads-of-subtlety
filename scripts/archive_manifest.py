#!/usr/bin/env python3
"""Create a deterministic manifest for the local research archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("archive"))
    args = parser.parse_args()
    output = args.root / "manifest.json"
    files = {}
    for path in sorted(args.root.rglob("*")):
        if path.is_file() and path != output:
            files[str(path.relative_to(args.root))] = {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    value = {"schema_version": 1, "root": str(args.root), "files": files}
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    print(f"wrote {output} ({len(files)} files)")


if __name__ == "__main__":
    main()
