#!/usr/bin/env python3
"""Build aligned relation, structural, and M3 feature matrices."""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
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
STRUCTURAL_FEATURES = (
    "text_chars",
    "scene_count",
    "edu_count",
    "node_count",
    "edge_count",
)
OTHER_RELATION = "__OTHER__"


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_record(serialized: str) -> Mapping[str, Any]:
    serialized = serialized.strip()
    if serialized.startswith('{"'):
        record = json.loads(serialized)
    else:
        record = ast.literal_eval(serialized)
    if not isinstance(record, Mapping):
        raise ValueError("Dataset record is not a mapping")
    return record


def source_domain(source: str) -> str:
    domain = re.sub(r"_(?:human|llm|gpt4|machine|para)$", "", source)
    return domain or source


def base_identity(text: str, source: str, label: int) -> str:
    payload = json.dumps(
        [text, source, label], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def graph_links(graph: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    links = graph.get("links")
    if links is None:
        links = graph.get("edges")
    return links or ()


def aggregate_document(
    record: Mapping[str, Any], motif_dimension: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int], np.ndarray]:
    trees = record.get("scene_discourse_trees") or {}
    if isinstance(trees, Mapping):
        scenes = [scene for scene in trees.values() if isinstance(scene, Mapping)]
    elif isinstance(trees, Sequence):
        scenes = [scene for scene in trees if isinstance(scene, Mapping)]
    else:
        raise ValueError("scene_discourse_trees is neither a mapping nor a sequence")

    raw_total = np.zeros(motif_dimension, dtype=np.float64)
    depth_total = np.zeros(motif_dimension, dtype=np.float64)
    relation_counts: Counter[str] = Counter()
    edu_count = 0
    node_count = 0
    edge_count = 0

    for scene in scenes:
        distribution = (scene.get("motif_dists") or {}).get("m3")
        if not isinstance(distribution, Mapping):
            raise ValueError("Scene is missing its M3 distribution")
        raw = np.asarray(distribution.get("raw", ()), dtype=np.float64)
        wad = np.asarray(distribution.get("wad", ()), dtype=np.float64)
        if raw.shape != (motif_dimension,) or wad.shape != raw.shape:
            raise ValueError(
                f"Malformed M3 vectors: raw={raw.shape}, wad={wad.shape}, "
                f"expected={(motif_dimension,)}"
            )
        present = (raw > 0) & np.isfinite(wad) & (wad >= 0)
        raw_total += raw
        depth_total[present] += raw[present] * wad[present]

        edu_count += len(scene.get("edus") or {})
        graph = scene.get("graph_dict") or {}
        nodes = graph.get("nodes") or ()
        links = graph_links(graph)
        node_count += len(nodes)
        edge_count += len(links)
        for edge in links:
            label = edge.get("label_0")
            if label is not None and label != "/":
                relation_counts[str(label)] += 1

    presence = (raw_total > 0).astype(np.uint8)
    motif_frequency = np.zeros(motif_dimension, dtype=np.float32)
    total_occurrences = raw_total.sum()
    if total_occurrences > 0:
        motif_frequency = (raw_total / total_occurrences).astype(np.float32)
    motif_depth = np.zeros(motif_dimension, dtype=np.float32)
    present = raw_total > 0
    motif_depth[present] = (depth_total[present] / raw_total[present]).astype(
        np.float32
    )
    structural = np.asarray(
        [
            len(str(record.get("text") or "")),
            len(scenes),
            edu_count,
            node_count,
            edge_count,
        ],
        dtype=np.float32,
    )
    return presence, motif_frequency, motif_depth, dict(relation_counts), structural


def count_records(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def load_split_rows(
    path: Path, motif_dimension: int, description: str
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "labels": [],
        "sources": [],
        "domains": [],
        "identity_hashes": [],
        "identity_occurrences": [],
        "identity_ids": [],
        "presence": [],
        "frequency": [],
        "depth": [],
        "relation_counts": [],
        "structural": [],
    }
    occurrence_counts: Counter[str] = Counter()
    input_digest = hashlib.sha256()
    total = count_records(path)
    with path.open("rb") as handle:
        for line_number, serialized in enumerate(
            tqdm(handle, total=total, desc=description, unit="doc"), 1
        ):
            input_digest.update(serialized)
            if not serialized.strip():
                continue
            record = read_record(serialized.decode("utf-8"))
            label = int(record.get("label"))
            if label not in (0, 1):
                raise ValueError(f"{path}:{line_number}: invalid label {label}")
            source = str(record.get("source", record.get("src", "unknown")))
            identity_hash = base_identity(
                str(record.get("text") or ""), source, label
            )
            occurrence = occurrence_counts[identity_hash]
            occurrence_counts[identity_hash] += 1
            values = aggregate_document(record, motif_dimension)

            result["labels"].append(label)
            result["sources"].append(source)
            result["domains"].append(source_domain(source))
            result["identity_hashes"].append(identity_hash)
            result["identity_occurrences"].append(occurrence)
            result["identity_ids"].append(f"{identity_hash}:{occurrence:06d}")
            for key, value in zip(
                ("presence", "frequency", "depth", "relation_counts", "structural"),
                values,
            ):
                result[key].append(value)

    result["input_sha256"] = input_digest.hexdigest()
    result["records"] = len(result["labels"])
    return result


def relation_vocabulary(rows: Mapping[str, Any]) -> List[str]:
    return sorted(
        {
            relation
            for counts in rows["relation_counts"]
            for relation in counts
        }
    )


def relation_matrix(
    rows: Mapping[str, Any], vocabulary: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    index = {relation: i for i, relation in enumerate(vocabulary)}
    other_index = len(vocabulary)
    matrix = np.zeros((rows["records"], len(vocabulary) + 1), dtype=np.float32)
    totals = np.zeros(rows["records"], dtype=np.int32)
    for row_index, counts in enumerate(rows["relation_counts"]):
        total = sum(counts.values())
        totals[row_index] = total
        if total == 0:
            continue
        for relation, count in counts.items():
            column = index.get(relation, other_index)
            matrix[row_index, column] += count / total
    return matrix, totals


def save_matrix(
    path: Path,
    inventory: str,
    split: str,
    rows: Mapping[str, Any],
    relation_names: Sequence[str],
    motif_hashes: Sequence[str],
) -> None:
    relations, relation_totals = relation_matrix(rows, relation_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                inventory=np.asarray(inventory),
                split=np.asarray(split),
                document_index=np.arange(rows["records"], dtype=np.int32),
                identity_id=np.asarray(rows["identity_ids"]),
                identity_hash=np.asarray(rows["identity_hashes"]),
                identity_occurrence=np.asarray(
                    rows["identity_occurrences"], dtype=np.int32
                ),
                source=np.asarray(rows["sources"]),
                domain=np.asarray(rows["domains"]),
                y=np.asarray(rows["labels"], dtype=np.int8),
                X_relation_frequency=relations,
                relation_total=np.asarray(relation_totals, dtype=np.int32),
                relation_names=np.asarray(list(relation_names) + [OTHER_RELATION]),
                X_structural=np.asarray(rows["structural"], dtype=np.float32),
                structural_names=np.asarray(STRUCTURAL_FEATURES),
                X_m3_presence=np.asarray(rows["presence"], dtype=np.uint8),
                X_m3_frequency=np.asarray(rows["frequency"], dtype=np.float32),
                X_m3_depth=np.asarray(rows["depth"], dtype=np.float32),
                m3_hashes=np.asarray(motif_hashes),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_catalog(motif_root: Path, inventory: str) -> Tuple[List[str], Dict[str, Any]]:
    inventory_root = motif_root / inventory
    catalog_path = inventory_root / "hc3_M3_motifs.json"
    manifest_path = inventory_root / "hc3_selected-motif-hashes.generated.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    motif_hashes = list(manifest.get("m3") or ())
    if not motif_hashes or set(motif_hashes) != set(catalog):
        raise ValueError(f"M3 manifest/catalog mismatch for {inventory}")
    provenance = {
        "catalog_path": str(catalog_path.resolve()),
        "catalog_sha256": sha256_file(catalog_path),
        "selection_manifest_path": str(manifest_path.resolve()),
        "selection_manifest_sha256": sha256_file(manifest_path),
        "motif_count": len(motif_hashes),
    }
    return motif_hashes, provenance


def schema_record(
    inventory: str,
    motif_hashes: Sequence[str],
    relation_names: Sequence[str],
    catalog: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "inventory": inventory,
        "catalog": dict(catalog),
        "identity": {
            "base": "sha256(canonical JSON [text, source, label])",
            "identity_id": "<base>:<zero-based occurrence within split>",
        },
        "relation_frequency": {
            "names": list(relation_names) + [OTHER_RELATION],
            "vocabulary_source": "training split only",
            "structural_edge_label_excluded": "/",
            "normalization": "document relation count / all document relation edges",
        },
        "structural": {"names": list(STRUCTURAL_FEATURES)},
        "m3": {
            "hashes": list(motif_hashes),
            "presence": "1 iff at least one occurrence across document scenes",
            "frequency": "document occurrence count / all M3 occurrences",
            "depth": (
                "occurrence-weighted scene-normalized depth; 0 when absent, "
                "disambiguated by X_m3_presence"
            ),
        },
    }


def build_inventory(args: argparse.Namespace, inventory: str) -> List[Dict[str, Any]]:
    motif_hashes, catalog = load_catalog(Path(args.motif_root), inventory)
    output_root = Path(args.output_dir) / inventory
    output_root.mkdir(parents=True, exist_ok=True)
    requested = tuple(split for split in SPLITS if split in args.splits)
    if not requested:
        return []

    schema_path = output_root / "feature_schema.json"
    train_rows = None
    if "train" in requested:
        train_output = output_root / "hc3_train.features.npz"
        if train_output.exists() and schema_path.exists() and not args.force:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            relation_names = [
                name
                for name in schema["relation_frequency"]["names"]
                if name != OTHER_RELATION
            ]
        else:
            train_path = (
                Path(args.root)
                / f"rel-{inventory}"
                / "hc3_train.discourse_parsed.graph_added.m3_motif_dists.jsonl"
            )
            train_rows = load_split_rows(
                train_path, len(motif_hashes), f"{inventory}/train"
            )
            relation_names = relation_vocabulary(train_rows)
            atomic_write_json(
                schema_path,
                schema_record(inventory, motif_hashes, relation_names, catalog),
            )
    elif schema_path.exists():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        relation_names = [
            name
            for name in schema["relation_frequency"]["names"]
            if name != OTHER_RELATION
        ]
    else:
        raise FileNotFoundError(
            f"{schema_path} is required when building without the training split"
        )

    artifacts = []
    for split in requested:
        source_path = (
            Path(args.root)
            / f"rel-{inventory}"
            / f"hc3_{split}.discourse_parsed.graph_added.m3_motif_dists.jsonl"
        )
        output_path = output_root / f"hc3_{split}.features.npz"
        if output_path.exists() and not args.force:
            print(f"skipping existing {output_path}")
            provenance_path = output_path.with_suffix(".provenance.json")
            if not provenance_path.exists():
                raise FileNotFoundError(provenance_path)
            artifacts.append(
                json.loads(provenance_path.read_text(encoding="utf-8"))
            )
            continue
        rows = train_rows if split == "train" else load_split_rows(
            source_path, len(motif_hashes), f"{inventory}/{split}"
        )
        save_matrix(
            output_path,
            inventory,
            split,
            rows,
            relation_names,
            motif_hashes,
        )
        record = {
            "inventory": inventory,
            "split": split,
            "records": rows["records"],
            "source_path": str(source_path.resolve()),
            "source_sha256": rows["input_sha256"],
            "artifact_path": str(output_path.resolve()),
            "artifact_sha256": sha256_file(output_path),
            "shapes": {
                "relation_frequency": [rows["records"], len(relation_names) + 1],
                "structural": [rows["records"], len(STRUCTURAL_FEATURES)],
                "m3": [rows["records"], len(motif_hashes)],
            },
        }
        atomic_write_json(
            output_path.with_suffix(".provenance.json"), record
        )
        artifacts.append(record)
        print(f"wrote {output_path}")
        if split == "train":
            train_rows = None
    return artifacts


def verify_alignment(
    output_dir: Path, inventories: Iterable[str]
) -> Dict[str, Any]:
    inventories = tuple(inventories)
    audit: Dict[str, Any] = {
        "reference_inventory": inventories[0],
        "compared_fields": ["identity_id", "label"],
        "splits": {},
    }
    for split in SPLITS:
        reference = None
        for inventory in inventories:
            path = output_dir / inventory / f"hc3_{split}.features.npz"
            with np.load(path, allow_pickle=False) as matrix:
                identity_ids = matrix["identity_id"]
                labels = matrix["y"]
            if reference is None:
                reference = (identity_ids, labels)
            elif not (
                np.array_equal(identity_ids, reference[0])
                and np.array_equal(labels, reference[1])
            ):
                raise ValueError(f"Identity alignment failed for {inventory}/{split}")
        audit["splits"][split] = {
            "rows": int(len(reference[0])),
            "exact_across_all_inventories": True,
        }
        print(f"verified aligned identities: {split}")
    return audit


def split_identity_audit(output_dir: Path, inventory: str) -> Dict[str, Any]:
    identities = {}
    audit: Dict[str, Any] = {"reference_inventory": inventory, "splits": {}}
    for split in SPLITS:
        path = output_dir / inventory / f"hc3_{split}.features.npz"
        with np.load(path, allow_pickle=False) as matrix:
            values = matrix["identity_hash"]
        identities[split] = set(values.tolist())
        audit["splits"][split] = {
            "rows": int(len(values)),
            "unique_text_source_label_identities": int(len(identities[split])),
            "duplicate_rows": int(len(values) - len(identities[split])),
        }
    audit["cross_split_unique_identity_overlap"] = {
        "train_validation": len(identities["train"] & identities["validation"]),
        "train_test": len(identities["train"] & identities["test"]),
        "validation_test": len(identities["validation"] & identities["test"]),
    }
    audit["policy"] = (
        "Original released split membership is preserved; overlaps are "
        "reported rather than removed."
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/unirst2")
    parser.add_argument("--motif-root", default="data/motifs2")
    parser.add_argument("--output-dir", default="artifacts/work/raw_feature_matrices")
    parser.add_argument("--inventories", nargs="+", default=list(INVENTORIES))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    summary_path = Path(args.output_dir) / "manifest.json"
    summary: Dict[str, Any] = {
        "dataset_root": str(Path(args.root).resolve()),
        "motif_root": str(Path(args.motif_root).resolve()),
        "artifacts": [],
    }
    for inventory in args.inventories:
        summary["artifacts"].extend(build_inventory(args, inventory))
        atomic_write_json(summary_path, summary)

    if set(args.inventories) == set(INVENTORIES) and set(args.splits) == set(SPLITS):
        summary["alignment_audit"] = verify_alignment(
            Path(args.output_dir), args.inventories
        )
        summary["split_identity_audit"] = split_identity_audit(
            Path(args.output_dir), args.inventories[0]
        )
        atomic_write_json(summary_path, summary)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
