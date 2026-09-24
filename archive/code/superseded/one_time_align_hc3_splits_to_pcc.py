#!/usr/bin/env python3
"""Create HC3 validation/test files aligned to the PCC split membership.

This is a one-time migration helper.  It pools each inventory's existing
train, validation, and test files, locates the documents used by PCC, and
writes new validation/test artifacts in exactly the PCC document order.  The
existing split files are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Mapping, MutableMapping, Tuple

from tqdm import tqdm


INVENTORIES = (
    "deu.rst.pcc",
    "eng.erst.gum",
    "eng.rst.rstdt",
    "nld.rst.nldt",
)
REFERENCE_INVENTORY = "deu.rst.pcc"
SUFFIX = "discourse_parsed.graph_added.m3_m6_motif_dists.jsonl"
INPUT_NAMES = {
    split: f"hc3_{split}.{SUFFIX}" for split in ("train", "validation", "test")
}
TAIL_PATTERN = re.compile(
    r", 'source': (?P<source>None|'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"), "
    r"'label': (?P<label>-?\d+)}\s*$"
)


def _quoted_literal_end(line: str, start: int) -> int:
    """Return the end of a single- or double-quoted Python string literal."""
    quote = line[start]
    if quote not in ("'", '"'):
        raise ValueError("text value is not a quoted Python string")
    escaped = False
    for index in range(start + 1, len(line)):
        char = line[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == quote:
            return index
    raise ValueError("unterminated text string")


def document_key(raw_line: bytes, path: Path, line_number: int) -> str:
    """Build a stable key without parsing the multi-megabyte nested record."""
    try:
        line = raw_line.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}:{line_number}: invalid UTF-8") from error
    prefix = "{'text': "
    if not line.startswith(prefix):
        raise ValueError(
            f"{path}:{line_number}: unsupported record layout; expected {prefix!r}"
        )
    text_start = len(prefix)
    text_end = _quoted_literal_end(line, text_start)
    text_literal = line[text_start : text_end + 1]
    tail = TAIL_PATTERN.search(line)
    if tail is None:
        raise ValueError(
            f"{path}:{line_number}: could not locate trailing source and label"
        )
    identity = "\0".join(
        (text_literal, tail.group("source"), tail.group("label"))
    ).encode("utf-8")
    return hashlib.sha256(identity).hexdigest()


def read_reference_keys(path: Path) -> List[str]:
    keys: List[str] = []
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if raw_line.strip():
                keys.append(document_key(raw_line, path, line_number))
    if not keys:
        raise ValueError(f"Reference split is empty: {path}")
    return keys


Location = Tuple[Path, int, int]


def locate_reference_documents(
    input_paths: Iterable[Path],
    reference_keys: Mapping[str, List[str]],
) -> Dict[str, List[Location]]:
    slots: MutableMapping[str, Deque[Tuple[str, int]]] = defaultdict(deque)
    locations: Dict[str, List[Location]] = {
        split: [None] * len(keys)  # type: ignore[list-item]
        for split, keys in reference_keys.items()
    }
    for split, keys in reference_keys.items():
        for index, key in enumerate(keys):
            slots[key].append((split, index))
    remaining = sum(len(values) for values in slots.values())

    for path in input_paths:
        with path.open("rb") as handle:
            progress = tqdm(
                desc=f"matching {path.parent.name}/{path.name.split('.')[0]}",
                unit="doc",
                leave=False,
            )
            line_number = 0
            while remaining:
                offset = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                line_number += 1
                if not raw_line.strip():
                    continue
                key = document_key(raw_line, path, line_number)
                if slots[key]:
                    split, index = slots[key].popleft()
                    locations[split][index] = (path, offset, len(raw_line))
                    remaining -= 1
                progress.update(1)
            progress.close()
        if remaining == 0:
            break

    if remaining:
        missing = {
            split: sum(location is None for location in split_locations)
            for split, split_locations in locations.items()
        }
        raise ValueError(
            "Could not reproduce the complete PCC split from the inventory pool: "
            + ", ".join(f"{split}={count} missing" for split, count in missing.items())
        )
    return locations


def write_locations(locations: List[Location], output_path: Path, force: bool) -> None:
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite {output_path}; pass --force to replace it"
        )
    temporary = output_path.with_name(output_path.name + ".tmp")
    handles = {}
    try:
        with temporary.open("wb") as output:
            for source_path, offset, length in locations:
                handle = handles.get(source_path)
                if handle is None:
                    handle = source_path.open("rb")
                    handles[source_path] = handle
                handle.seek(offset)
                raw_line = handle.read(length)
                if len(raw_line) != length:
                    raise OSError(f"Short read from {source_path} at byte {offset}")
                output.write(raw_line)
        os.replace(temporary, output_path)
    finally:
        for handle in handles.values():
            handle.close()
        if temporary.exists():
            temporary.unlink()


def atomic_copy(source: Path, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite {destination}; pass --force to replace it"
        )
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def aligned_name(split: str, tag: str) -> str:
    return f"hc3_{split}_{tag}.{SUFFIX}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--tag", default="pcc_aligned")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_dir) / "unirst"
    reference_dir = root / f"rel-{REFERENCE_INVENTORY}"
    reference_paths = {
        split: reference_dir / INPUT_NAMES[split]
        for split in ("validation", "test")
    }
    missing = [path for path in reference_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing PCC reference files: " + ", ".join(map(str, missing)))
    reference_keys = {
        split: read_reference_keys(path) for split, path in reference_paths.items()
    }
    overlap = set(reference_keys["validation"]) & set(reference_keys["test"])
    if overlap:
        print(
            "warning: PCC validation and test share "
            f"{len(overlap)} duplicate text/source/label identities; "
            "their multiplicities and positions will be preserved"
        )

    print(
        "PCC reference: "
        f"validation={len(reference_keys['validation'])}, "
        f"test={len(reference_keys['test'])}"
    )
    for inventory in INVENTORIES:
        directory = root / f"rel-{inventory}"
        outputs = {
            split: directory / aligned_name(split, args.tag)
            for split in ("validation", "test")
        }
        if inventory == REFERENCE_INVENTORY:
            for split in ("validation", "test"):
                atomic_copy(reference_paths[split], outputs[split], args.force)
        else:
            input_paths = [directory / INPUT_NAMES[split] for split in ("train", "validation", "test")]
            missing = [path for path in input_paths if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Missing {inventory} pool files: " + ", ".join(map(str, missing))
                )
            locations = locate_reference_documents(input_paths, reference_keys)
            for split in ("validation", "test"):
                write_locations(locations[split], outputs[split], args.force)
        print(
            f"{inventory}: wrote {outputs['validation']} "
            f"and {outputs['test']}"
        )


if __name__ == "__main__":
    main()
