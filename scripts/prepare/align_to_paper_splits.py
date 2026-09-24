#!/usr/bin/env python3
"""Align UniRST HC3 parses to the author-released paper splits.

The reference files are the HC3 train/validation/test artifacts from
``zaemyung/ToS-Dataset`` at revision
``f94c7f19a7ff99ba6996533910903a63e952ce1d``.  Only the reference
``text``, ``source``, and ``label`` fields define membership; the released
DMRST parses and motif distributions are never copied into the UniRST data.

The script streams the large one-record-per-line corpora, records byte
locations for matching UniRST documents, and writes split files in the exact
reference order.  Existing source files are never modified.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

from tqdm import tqdm


REFERENCE_REVISION = "f94c7f19a7ff99ba6996533910903a63e952ce1d"
INVENTORIES = (
    "deu.rst.pcc",
    "eng.erst.gum",
    "eng.rst.rstdt",
    "nld.rst.nldt",
)
SPLITS = ("train", "validation", "test")
REFERENCE_NAMES = {
    split: f"hc3_{split}.discourse_parsed.graph_added.motif_dists.jsonl"
    for split in SPLITS
}
REFERENCE_SHA256 = {
    "train": "db46253183f541d560bde5b170a2e4e2b2b7f1c39ba7d733bdae42923af94ded",
    "validation": "352fedc8c83e7b2d069fd65df671b80d7281309684b6c1914b0f2897c50296c8",
    "test": "43fe588ecd25a7c3c2188e7a96deb19cc5207f0d96b8cd4906142396c99b9608",
}
SOURCE_PATTERN = re.compile(
    r"^hc3_.+_(?:human|machine)\.discourse_parsed\.jsonl$"
)
TAIL_PATTERN = re.compile(
    r", 'source': (?P<source>None|'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"), "
    r"'label': (?P<label>-?\d+)}\s*$"
)


@dataclass(frozen=True)
class Identity:
    key: str
    text_sha256: str
    text: str
    source: Optional[str]
    label: int


@dataclass(frozen=True)
class Location:
    path: Path
    offset: int
    length: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quoted_literal_end(line: str, start: int) -> int:
    quote = line[start]
    if quote not in ("'", '\"'):
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


def extract_identity(raw_line: bytes, path: Path, line_number: int) -> Identity:
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
    text = ast.literal_eval(line[text_start : text_end + 1])
    if not isinstance(text, str):
        raise ValueError(f"{path}:{line_number}: text is not a string")

    tail = TAIL_PATTERN.search(line)
    if tail is None:
        raise ValueError(
            f"{path}:{line_number}: could not locate trailing source and label"
        )
    source = ast.literal_eval(tail.group("source"))
    label = int(tail.group("label"))
    payload = json.dumps(
        [text, source, label], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return Identity(
        key=hashlib.sha256(payload).hexdigest(),
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        text=text,
        source=source,
        label=label,
    )


def read_reference_splits(
    reference_dir: Path,
) -> Tuple[Dict[str, List[Identity]], Dict[str, str]]:
    identities: Dict[str, List[Identity]] = {}
    checksums: Dict[str, str] = {}
    for split in SPLITS:
        path = reference_dir / REFERENCE_NAMES[split]
        if not path.is_file():
            raise FileNotFoundError(f"Missing reference split: {path}")
        checksum = sha256_file(path)
        if checksum != REFERENCE_SHA256[split]:
            raise ValueError(
                f"Reference checksum mismatch for {path}: expected "
                f"{REFERENCE_SHA256[split]}, found {checksum}"
            )
        checksums[split] = checksum
        split_identities: List[Identity] = []
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(
                tqdm(handle, desc=f"reading paper {split}", unit="doc"), 1
            ):
                if raw_line.strip():
                    split_identities.append(
                        extract_identity(raw_line, path, line_number)
                    )
        if not split_identities:
            raise ValueError(f"Reference split is empty: {path}")
        identities[split] = split_identities
    return identities, checksums


def read_identity_reference_splits(
    reference_dir: Path,
) -> Tuple[Dict[str, List[Identity]], Dict[str, str]]:
    """Read parse-independent identities from an existing aligned inventory."""
    identities: Dict[str, List[Identity]] = {}
    checksums: Dict[str, str] = {}
    for split in SPLITS:
        path = reference_dir / f"hc3_{split}.discourse_parsed.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing aligned reference split: {path}")
        checksums[split] = sha256_file(path)
        split_identities = []
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(
                tqdm(handle, desc=f"reading aligned {split}", unit="doc"), 1
            ):
                if raw_line.strip():
                    split_identities.append(
                        extract_identity(raw_line, path, line_number)
                    )
        if not split_identities:
            raise ValueError(f"Aligned reference split is empty: {path}")
        identities[split] = split_identities
    return identities, checksums


def source_files(source_dir: Path) -> List[Path]:
    files = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and SOURCE_PATTERN.match(path.name)
    )
    if not files:
        raise FileNotFoundError(f"No post-parse HC3 source files in {source_dir}")
    return files


def locate_documents(
    paths: Iterable[Path],
    reference: Mapping[str, List[Identity]],
) -> Tuple[Dict[str, List[Location]], int]:
    slots: MutableMapping[str, Deque[Tuple[str, int]]] = defaultdict(deque)
    locations: Dict[str, List[Optional[Location]]] = {
        split: [None] * len(reference[split]) for split in SPLITS
    }
    for split in SPLITS:
        for index, identity in enumerate(reference[split]):
            slots[identity.key].append((split, index))

    remaining = sum(len(values) for values in slots.values())
    scanned = 0
    for path in paths:
        with path.open("rb") as handle:
            progress = tqdm(desc=f"matching {path.parent.name}/{path.stem}", unit="doc")
            line_number = 0
            while remaining:
                offset = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                line_number += 1
                if not raw_line.strip():
                    continue
                scanned += 1
                identity = extract_identity(raw_line, path, line_number)
                if slots[identity.key]:
                    split, index = slots[identity.key].popleft()
                    locations[split][index] = Location(path, offset, len(raw_line))
                    remaining -= 1
                progress.update(1)
            progress.close()
        if remaining == 0:
            break

    if remaining:
        missing = {
            split: sum(location is None for location in locations[split])
            for split in SPLITS
        }
        raise ValueError(
            "Could not recover every paper-split identity: "
            + ", ".join(f"{split}={count} missing" for split, count in missing.items())
        )

    return {
        split: [location for location in locations[split] if location is not None]
        for split in SPLITS
    }, scanned


def write_split(locations: List[Location], destination: Path, force: bool) -> str:
    if destination.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite {destination}; pass --force to replace it"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    handles: Dict[Path, object] = {}
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as output:
            for location in tqdm(locations, desc=f"writing {destination.name}", unit="doc"):
                handle = handles.get(location.path)
                if handle is None:
                    handle = location.path.open("rb")
                    handles[location.path] = handle
                handle.seek(location.offset)
                raw_line = handle.read(location.length)
                if len(raw_line) != location.length:
                    raise OSError(f"Short read from {location.path} at {location.offset}")
                output.write(raw_line)
                digest.update(raw_line)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        for handle in handles.values():
            handle.close()
        if temporary.exists():
            temporary.unlink()
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_manifest(path: Path, reference: Mapping[str, List[Identity]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for split in SPLITS:
            for position, identity in enumerate(reference[split]):
                record = {
                    "split": split,
                    "position": position,
                    "text": identity.text,
                    "source": identity.source,
                    "label": identity.label,
                    "text_sha256": identity.text_sha256,
                    "identity_sha256": identity.key,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def duplicate_audit(reference: Mapping[str, List[Identity]]) -> Dict[str, object]:
    split_counts = {
        split: Counter(identity.key for identity in reference[split])
        for split in SPLITS
    }
    overlaps = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            shared = set(split_counts[left]) & set(split_counts[right])
            overlaps[f"{left}__{right}"] = {
                "unique_identities": len(shared),
                "rows_in_left": sum(split_counts[left][key] for key in shared),
                "rows_in_right": sum(split_counts[right][key] for key in shared),
            }
    return {
        "within_split_duplicate_rows": {
            split: sum(count - 1 for count in split_counts[split].values())
            for split in SPLITS
        },
        "cross_split_overlaps": overlaps,
    }


def read_identity_keys(path: Path) -> Tuple[List[str], str]:
    keys: List[str] = []
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(
            tqdm(handle, desc=f"verifying {path.parent.name}/{path.name}", unit="doc"),
            1,
        ):
            if not raw_line.strip():
                continue
            digest.update(raw_line)
            keys.append(extract_identity(raw_line, path, line_number).key)
    return keys, digest.hexdigest()


def dropped_from_subsequence(reference: List[str], observed: List[str]) -> List[str]:
    dropped: List[str] = []
    observed_index = 0
    for key in reference:
        if observed_index < len(observed) and key == observed[observed_index]:
            observed_index += 1
        else:
            dropped.append(key)
    if observed_index != len(observed):
        raise ValueError("Graph-added identities are not a subsequence of the paper split")
    return dropped


def verify_existing_outputs(
    output_root: Path,
    inventories: Iterable[str],
    reference: Mapping[str, List[Identity]],
) -> None:
    audit_path = output_root / "hc3_paper_alignment_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Missing alignment audit: {audit_path}")
    with audit_path.open(encoding="utf-8") as handle:
        audit = json.load(handle)

    reference_keys = {
        split: [identity.key for identity in reference[split]] for split in SPLITS
    }
    common_graph_keys: Dict[str, List[str]] = {}
    for inventory in inventories:
        directory = output_root / f"rel-{inventory}"
        graph_audit = {}
        for split in SPLITS:
            parsed_path = directory / f"hc3_{split}.discourse_parsed.jsonl"
            graph_path = directory / f"hc3_{split}.discourse_parsed.graph_added.jsonl"
            parsed_keys, parsed_checksum = read_identity_keys(parsed_path)
            if parsed_keys != reference_keys[split]:
                raise ValueError(
                    f"Post-parse identity/order mismatch for {inventory}/{split}"
                )
            graph_keys, graph_checksum = read_identity_keys(graph_path)
            dropped = dropped_from_subsequence(reference_keys[split], graph_keys)
            if split in common_graph_keys and graph_keys != common_graph_keys[split]:
                raise ValueError(
                    f"Graph-added identity/order mismatch across inventories for {split}"
                )
            common_graph_keys.setdefault(split, graph_keys)
            graph_audit[split] = {
                "post_parse_count": len(parsed_keys),
                "graph_added_count": len(graph_keys),
                "empty_or_failed_parse_rows_removed": len(dropped),
                "post_parse_sha256": parsed_checksum,
                "graph_added_sha256": graph_checksum,
                "removed_identity_sha256": dropped,
            }
        audit["inventories"][inventory]["graph_added"] = graph_audit

    audit["graph_added_identity_order_identical_across_inventories"] = True
    write_json_atomic(audit_path, audit)
    print(f"verified post-parse and graph-added outputs; updated {audit_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    reference_group = parser.add_mutually_exclusive_group(required=True)
    reference_group.add_argument("--reference-dir", type=Path)
    reference_group.add_argument(
        "--identity-reference-dir",
        type=Path,
        help=(
            "Use text/source/label identities from another already aligned "
            "inventory instead of the released DMRST files."
        ),
    )
    parser.add_argument("--source-root", type=Path, default=Path("data/unirst"))
    parser.add_argument("--output-root", type=Path, default=Path("data/unirst2"))
    parser.add_argument(
        "--inventory",
        action="append",
        help="Align one inventory; repeat as needed. Defaults to all four.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing post-parse and graph-added outputs and update the audit.",
    )
    args = parser.parse_args()

    inventories = tuple(args.inventory or INVENTORIES)
    if args.identity_reference_dir:
        reference, reference_checksums = read_identity_reference_splits(
            args.identity_reference_dir
        )
    else:
        reference, reference_checksums = read_reference_splits(args.reference_dir)
    if args.verify_only:
        verify_existing_outputs(args.output_root, inventories, reference)
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    generic_reference = args.identity_reference_dir is not None
    manifest_root = (
        args.output_root / f"rel-{inventories[0]}"
        if generic_reference and len(inventories) == 1
        else args.output_root
    )
    manifest_name = (
        "hc3_split_identities.jsonl"
        if generic_reference
        else "hc3_paper_split_identities.jsonl"
    )
    write_manifest(manifest_root / manifest_name, reference)

    audit: Dict[str, object] = {
        "reference": {
            "repository": (
                None if generic_reference else "zaemyung/ToS-Dataset"
            ),
            "revision": None if generic_reference else REFERENCE_REVISION,
            "identity_reference_dir": (
                str(args.identity_reference_dir.resolve())
                if generic_reference
                else None
            ),
            "checksums": reference_checksums,
            "counts": {split: len(reference[split]) for split in SPLITS},
            "duplicates": duplicate_audit(reference),
        },
        "identity_fields": ["text", "source", "label"],
        "inventories": {},
    }

    for inventory in inventories:
        source_dir = args.source_root / f"rel-{inventory}"
        output_dir = args.output_root / f"rel-{inventory}"
        paths = source_files(source_dir)
        locations, scanned = locate_documents(paths, reference)
        output_checksums = {}
        for split in SPLITS:
            destination = output_dir / f"hc3_{split}.discourse_parsed.jsonl"
            output_checksums[split] = write_split(
                locations[split], destination, args.force
            )
        audit["inventories"][inventory] = {
            "source_files": [str(path) for path in paths],
            "source_documents_scanned": scanned,
            "output_counts": {split: len(locations[split]) for split in SPLITS},
            "output_sha256": output_checksums,
        }
        audit_name = (
            "hc3_split_alignment_audit.json"
            if generic_reference
            else "hc3_paper_alignment_audit.json"
        )
        write_json_atomic(manifest_root / audit_name, audit)
        print(f"aligned {inventory}")

    write_json_atomic(manifest_root / audit_name, audit)


if __name__ == "__main__":
    main()
