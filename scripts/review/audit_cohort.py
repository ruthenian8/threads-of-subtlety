#!/usr/bin/env python3
"""Audit UniRST2 cohort identity, provenance, parser completeness, and dependence.

This stage is intentionally read-only with respect to source data and model
artifacts.  It writes an auditable manifest and sensitivity-cluster files for
the cross-inventory complementarity analysis.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from tqdm import tqdm


INVENTORIES = (
    "deu.rst.pcc",
    "eng.erst.gum.new",
    "eng.rst.rstdt.new",
    "nld.rst.nldt",
)
SPLITS = ("train", "validation", "test")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_MASK64 = (1 << 64) - 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        atomic_text(path, "")
        return
    fields = list(rows[0])
    output = []
    import io

    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def read_record_header(line: bytes) -> Mapping[str, Any]:
    """Read identity fields without evaluating the large nested tree mapping."""
    value = line.decode("utf-8").strip()
    if not value:
        raise ValueError("empty record")
    if value.startswith("{\""):
        # JSON records are uncommon but cheap to handle when encountered.
        record = json.loads(value)
        complete, _reason, scene_count = parser_status(record)
        return {
            "text": str(record.get("text") or ""),
            "source": str(record.get("source", record.get("src", "unknown"))),
            "label": int(record.get("label")),
            "parser_complete": complete,
            "scene_count": scene_count,
        }
    prefix = "{'text': "
    if not value.startswith(prefix):
        raise ValueError("unexpected record prefix")
    start = len(prefix)
    quote = value[start]
    if quote not in "'\"":
        raise ValueError("text is not a quoted string")
    index = start + 1
    escaped = False
    while index < len(value):
        character = value[index]
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == quote:
            break
        index += 1
    if index >= len(value):
        raise ValueError("unterminated text string")
    text = ast.literal_eval(value[start : index + 1])
    source_marker = b", 'source': "
    label_marker = b", 'label': "
    source_start = line.rfind(source_marker)
    label_start = line.rfind(label_marker)
    if source_start < 0 or label_start < source_start:
        raise ValueError("missing source/label")
    source_token = line[source_start + len(source_marker) : label_start]
    source = ast.literal_eval(source_token.decode("utf-8"))
    label = int(line[label_start + len(label_marker) :].split(b"}", 1)[0])
    # Graph/motif output has already passed the empty-tree filter.  These
    # lexical checks catch truncated or incomplete records without evaluating
    # every nested graph object.
    complete = (
        b"'scene_discourse_trees':" in line
        and b"'motif_dists': {'m3':" in line
        and b"'parsed':" in line
        and b"'edus':" in line
    )
    return {
        "text": text,
        "source": source,
        "label": label,
        "parser_complete": complete,
        "scene_count": line.count(b"'parsed':"),
    }


def source_domain(source: str) -> str:
    return re.sub(r"_(?:human|llm|gpt4|machine|para)$", "", source) or source


def identity_hash(text: str, source: str, label: int) -> str:
    payload = json.dumps(
        [text, source, label], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parser_status(record: Mapping[str, Any]) -> Tuple[bool, str, int]:
    trees = record.get("scene_discourse_trees")
    if isinstance(trees, Mapping):
        scenes = [value for value in trees.values() if isinstance(value, Mapping)]
    elif isinstance(trees, Sequence) and not isinstance(trees, (str, bytes)):
        scenes = [value for value in trees if isinstance(value, Mapping)]
    else:
        return False, "missing_scene_discourse_trees", 0
    if not scenes:
        return False, "no_scenes", 0
    for scene_index, scene in enumerate(scenes):
        edus = scene.get("edus")
        if not isinstance(edus, Mapping) or not edus:
            return False, f"scene_{scene_index}_missing_edus", len(scenes)
        if len(edus) > 1 and not str(scene.get("parsed") or "").strip():
            return False, f"scene_{scene_index}_missing_parse", len(scenes)
        motif_dists = scene.get("motif_dists")
        if not isinstance(motif_dists, Mapping) or not isinstance(motif_dists.get("m3"), Mapping):
            return False, f"scene_{scene_index}_missing_m3_distribution", len(scenes)
    return True, "ok", len(scenes)


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def token_signature(text: str, hashes: int = 16) -> Tuple[int, ...]:
    normalized = " ".join(_WORD_RE.findall(text.casefold()))
    tokens = normalized.split()
    if not tokens:
        return tuple([_MASK64] * hashes)
    if len(tokens) < 3:
        shingles = {normalized}
    else:
        shingles = {
            " ".join(tokens[index : index + 3])
            for index in range(len(tokens) - 2)
        }
    values = [
        int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")
        for shingle in shingles
    ]
    result = []
    for index in range(hashes):
        salt = (0x9E3779B97F4A7C15 * (index + 1)) & _MASK64
        result.append(min((value ^ salt) & _MASK64 for value in values))
    return tuple(result)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def near_duplicate_clusters(
    rows: Sequence[Mapping[str, Any]], threshold: float
) -> List[Dict[str, Any]]:
    """Build conservative exact-normalized and MinHash near-duplicate groups."""
    if not rows:
        return []
    signatures = [token_signature(str(row["text"])) for row in rows]
    union = UnionFind(len(rows))
    normalized: Dict[str, int] = {}
    for index, row in enumerate(rows):
        text = " ".join(_WORD_RE.findall(str(row["text"]).casefold()))
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        previous = normalized.get(key)
        if previous is not None:
            union.union(index, previous)
        else:
            normalized[key] = index

    # Four bands of four MinHash values give candidates without an O(n^2) pass.
    buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = defaultdict(list)
    for index, signature in enumerate(signatures):
        for band in range(4):
            key = (band, signature[band * 4 : band * 4 + 4])
            if len(buckets[key]) < 256:
                buckets[key].append(index)
    for candidates in buckets.values():
        if len(candidates) < 2:
            continue
        for position, left in enumerate(candidates):
            for right in candidates[position + 1 :]:
                agreement = sum(a == b for a, b in zip(signatures[left], signatures[right]))
                if agreement / float(len(signatures[left])) >= threshold:
                    union.union(left, right)

    groups: Dict[int, List[int]] = defaultdict(list)
    for index in range(len(rows)):
        groups[union.find(index)].append(index)
    result = []
    for members in groups.values():
        if len(members) < 2:
            continue
        result.append(
            {
                "cluster_id": hashlib.sha256(
                    ":".join(sorted(rows[index]["identity_hash"] for index in members)).encode()
                ).hexdigest()[:16],
                "method": "identity_or_minhash",
                "estimated_jaccard_threshold": threshold,
                "members": [
                    {
                        "split": rows[index]["split"],
                        "row": rows[index]["row"],
                        "identity_hash": rows[index]["identity_hash"],
                        "source": rows[index]["source"],
                        "label": rows[index]["label"],
                    }
                    for index in members
                ],
            }
        )
    return sorted(result, key=lambda value: value["cluster_id"])


def load_reference(
    dataset_root: Path, inventory: str, splits: Sequence[str]
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    all_rows: Dict[str, List[Dict[str, Any]]] = {}
    audit_rows: List[Dict[str, Any]] = []
    for split in splits:
        # Feature matrices are built from this post-filter, motif-bearing file.
        # The unfiltered parsed file is counted separately to expose rows
        # removed by ``filter_empty_trees`` in the graph stage.
        path = dataset_root / f"rel-{inventory}" / f"hc3_{split}.discourse_parsed.graph_added.m3_motif_dists.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        total = count_lines(path)
        rows: List[Dict[str, Any]] = []
        occurrences: Counter[str] = Counter()
        with path.open("rb") as handle:
            for row_number, line in enumerate(
                tqdm(handle, total=total, desc=f"audit {inventory}/{split}", unit="doc")
            ):
                if not line.strip():
                    continue
                try:
                    record = read_record_header(line)
                    text = str(record.get("text") or "")
                    source = str(record.get("source", record.get("src", "unknown")))
                    label = int(record.get("label"))
                    if label not in (0, 1):
                        raise ValueError(f"invalid label {label}")
                    digest = identity_hash(text, source, label)
                    occurrence = occurrences[digest]
                    occurrences[digest] += 1
                    complete = bool(record["parser_complete"])
                    reason = "ok" if complete else "incomplete_serialized_tree"
                    scene_count = int(record["scene_count"])
                    row = {
                        "split": split,
                        "row": row_number,
                        "identity_hash": digest,
                        "identity_id": f"{digest}:{occurrence:06d}",
                        "text": text,
                        "source": source,
                        "domain": source_domain(source),
                        "label": label,
                        "parser_complete": complete,
                        "parser_status": reason,
                        "scene_count": scene_count,
                    }
                except Exception as error:  # retain malformed-row evidence
                    row = {
                        "split": split,
                        "row": row_number,
                        "identity_hash": "",
                        "identity_id": "",
                        "text": "",
                        "source": "",
                        "domain": "",
                        "label": "",
                        "parser_complete": False,
                        "parser_status": f"malformed:{type(error).__name__}:{error}",
                        "scene_count": 0,
                    }
                rows.append(row)
                audit_rows.append({key: value for key, value in row.items() if key != "text"})
        all_rows[split] = rows
    return all_rows, audit_rows


def verify_matrices(
    feature_dir: Path,
    inventories: Sequence[str],
    reference: Mapping[str, Sequence[Mapping[str, Any]]],
    verify_hashes: bool,
) -> List[Dict[str, Any]]:
    rows = []
    for inventory in inventories:
        for split in SPLITS:
            matrix_path = feature_dir / inventory / f"hc3_{split}.features.npz"
            provenance_path = matrix_path.with_suffix(".provenance.json")
            if not matrix_path.exists() or not provenance_path.exists():
                raise FileNotFoundError(f"missing matrix/provenance: {matrix_path}")
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            source_path = Path(provenance["source_path"])
            if not source_path.exists():
                raise FileNotFoundError(source_path)
            source_hash_ok = (not verify_hashes) or sha256_file(source_path) == provenance["source_sha256"]
            artifact_hash_ok = (not verify_hashes) or sha256_file(matrix_path) == provenance["artifact_sha256"]
            if not source_hash_ok or not artifact_hash_ok:
                raise ValueError(f"provenance hash mismatch: {matrix_path}")
            with np.load(matrix_path, allow_pickle=False) as matrix:
                expected = reference[split]
                expected_hashes = np.asarray([row["identity_hash"] for row in expected])
                expected_ids = np.asarray([row["identity_id"] for row in expected])
                expected_labels = np.asarray([row["label"] for row in expected], dtype=np.int8)
                expected_sources = np.asarray([row["source"] for row in expected])
                checks = {
                    "identity_hash": np.array_equal(matrix["identity_hash"], expected_hashes),
                    "identity_id": np.array_equal(matrix["identity_id"], expected_ids),
                    "label": np.array_equal(matrix["y"], expected_labels),
                    "source": np.array_equal(matrix["source"], expected_sources),
                }
                if not all(checks.values()):
                    raise ValueError(f"row alignment mismatch for {inventory}/{split}: {checks}")
                rows.append(
                    {
                        "inventory": inventory,
                        "split": split,
                        "records": int(len(matrix["y"])),
                        "source_hash_verified": source_hash_ok,
                        "artifact_hash_verified": artifact_hash_ok,
                        "alignment_verified": True,
                        "catalog_path": str(
                            json.loads(
                                (feature_dir / inventory / "feature_schema.json").read_text(encoding="utf-8")
                            )["catalog"]["catalog_path"]
                        ),
                    }
                )
    return rows


def verify_catalogs(motif_root: Path, feature_dir: Path, inventories: Sequence[str]) -> List[Dict[str, Any]]:
    result = []
    for inventory in inventories:
        root = motif_root / inventory
        catalog_path = root / "hc3_M3_motifs.json"
        selection_path = root / "hc3_selected-motif-hashes.generated.json"
        provenance_path = root / "hc3_M3_provenance.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        selected = json.loads(selection_path.read_text(encoding="utf-8"))
        hashes = list(selected.get("m3") or [])
        if set(hashes) != set(catalog) or len(hashes) != len(set(hashes)):
            raise ValueError(f"catalog/selection mismatch for {inventory}")
        schema = json.loads((feature_dir / inventory / "feature_schema.json").read_text(encoding="utf-8"))
        schema_catalog = schema["catalog"]
        if schema_catalog["catalog_sha256"] != sha256_file(catalog_path):
            raise ValueError(f"schema catalog hash mismatch for {inventory}")
        if schema_catalog["selection_manifest_sha256"] != sha256_file(selection_path):
            raise ValueError(f"schema selection hash mismatch for {inventory}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        source_checks = []
        for source in provenance.get("source_files", []):
            source_file = Path(source["path"])
            source_checks.append(source_file.exists() and sha256_file(source_file) == source["sha256"])
        if not all(source_checks):
            raise ValueError(f"motif provenance source hash mismatch for {inventory}")
        result.append(
            {
                "inventory": inventory,
                "motif_count": len(hashes),
                "catalog_sha256": sha256_file(catalog_path),
                "selection_manifest_sha256": sha256_file(selection_path),
                "motif_provenance_sha256": sha256_file(provenance_path),
                "source_files_verified": len(source_checks),
            }
        )
    return result


def split_summary(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    identities = {split: [row["identity_hash"] for row in values if row["identity_hash"]] for split, values in rows.items()}
    summary = {
        split: {
            "rows": len(values),
            "unique_identities": len(set(identities[split])),
            "duplicate_rows": len(values) - len(set(identities[split])),
            "parser_incomplete_rows": sum(not row["parser_complete"] for row in values),
        }
        for split, values in rows.items()
    }
    summary["cross_split_overlap"] = {
        "train_validation": len(set(identities["train"]) & set(identities["validation"])),
        "train_test": len(set(identities["train"]) & set(identities["test"])),
        "validation_test": len(set(identities["validation"]) & set(identities["test"])),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/unirst2"))
    parser.add_argument("--feature-dir", type=Path, default=Path("artifacts/canonical/feature_matrices"))
    parser.add_argument("--motif-root", type=Path, default=Path("data/motifs2"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/review"))
    parser.add_argument("--inventories", nargs="+", default=list(INVENTORIES))
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.90)
    parser.add_argument("--skip-source-hashes", action="store_true")
    args = parser.parse_args()
    if set(args.inventories) != set(INVENTORIES):
        raise ValueError("stage 1 requires all four inventories for alignment")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference, audit_rows = load_reference(args.dataset_root, args.inventories[0], SPLITS)
    write_csv(args.output_dir / "identity_audit.csv", audit_rows)
    pre_graph_rows = {
        split: count_lines(
            args.dataset_root
            / f"rel-{args.inventories[0]}"
            / f"hc3_{split}.discourse_parsed.jsonl"
        )
        for split in SPLITS
    }
    write_csv(
        args.output_dir / "parse_failure_audit.csv",
        [
            {
                "split": split,
                "unfiltered_parsed_rows": pre_graph_rows[split],
                "motif_input_rows": len(reference[split]),
                "rows_removed_by_empty_tree_filter": pre_graph_rows[split] - len(reference[split]),
                "incomplete_motif_input_rows": sum(
                    not row["parser_complete"] for row in reference[split]
                ),
                "policy": "graph stage removes empty/NONE trees before motif extraction",
            }
            for split in SPLITS
        ],
    )
    catalog_rows = verify_catalogs(args.motif_root, args.feature_dir, args.inventories)
    matrix_rows = verify_matrices(
        args.feature_dir,
        args.inventories,
        reference,
        verify_hashes=not args.skip_source_hashes,
    )
    combined_rows = [
        row
        for split in SPLITS
        for row in reference[split]
        if row["identity_hash"]
    ]
    clusters = near_duplicate_clusters(combined_rows, args.near_duplicate_threshold)
    with (args.output_dir / "near_duplicate_clusters.jsonl.tmp").open("w", encoding="utf-8") as handle:
        for cluster in clusters:
            handle.write(json.dumps(cluster, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(args.output_dir / "near_duplicate_clusters.jsonl.tmp", args.output_dir / "near_duplicate_clusters.jsonl")

    summary = {
        "stage": "unirst2_complementarity_cohort_audit",
        "dataset_root": str(args.dataset_root.resolve()),
        "feature_dir": str(args.feature_dir.resolve()),
        "motif_root": str(args.motif_root.resolve()),
        "inventories": list(args.inventories),
        "splits": list(SPLITS),
        "reference_inventory": args.inventories[0],
        "split_summary": split_summary(reference),
        "pre_graph_filter_rows": pre_graph_rows,
        "catalogs": catalog_rows,
        "matrices": matrix_rows,
        "near_duplicate": {
            "method": "exact normalized text plus 16-hash token-3-shingle MinHash, four bands",
            "threshold": args.near_duplicate_threshold,
            "clusters": len(clusters),
            "clustered_rows": sum(len(cluster["members"]) for cluster in clusters),
        },
        "policy": {
            "primary_split": "original released HC3 membership",
            "test_status": "exploratory because existing test outcomes were inspected",
            "identity": "sha256(canonical JSON [text, source, label])",
            "parser_complete_definition": "nonempty scenes and EDUs; multi-EDU scenes require a parsed tree",
        },
    }
    summary["manifest_sha256_inputs"] = {
        "identity_audit": sha256_file(args.output_dir / "identity_audit.csv"),
        "parse_failure_audit": sha256_file(args.output_dir / "parse_failure_audit.csv"),
        "near_duplicate_clusters": sha256_file(args.output_dir / "near_duplicate_clusters.jsonl"),
    }
    atomic_json(args.output_dir / "cohort_manifest.json", summary)
    print(json.dumps({"rows": sum(len(values) for values in reference.values()), "clusters": len(clusters)}, indent=2))


if __name__ == "__main__":
    main()
