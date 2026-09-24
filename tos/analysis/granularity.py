"""Shared I/O, feature, and graph helpers for granularity comparisons."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import networkx as nx
import numpy as np


INVENTORIES = (
    "deu.rst.pcc",
    "eng.erst.gum.new",
    "eng.rst.rstdt.new",
    "nld.rst.nldt",
)
SPLITS = ("train", "validation", "test")
METRICS = ("roc_auc", "average_precision", "balanced_accuracy", "f1", "macro_f1")
GRANULARITY_DIRS = {
    "deu.rst.pcc": "pcc_nldt",
    "eng.erst.gum.new": "erst",
    "eng.rst.rstdt.new": "rstdt",
    "nld.rst.nldt": "pcc_nldt",
}
_SHARED_PROJECTION_DIRS = frozenset(("deu.rst.pcc", "nld.rst.nldt"))


def granularity_dir(inventory: str) -> str:
    try:
        return GRANULARITY_DIRS[inventory]
    except KeyError as error:
        raise ValueError(f"Unknown canonical inventory: {inventory}") from error


def coarse_feature_path(root: Path, inventory: str, split: str) -> Path:
    directory = granularity_dir(inventory)
    if inventory in _SHARED_PROJECTION_DIRS:
        return root / directory / "feature_matrices" / inventory / f"hc3_{split}.features.npz"
    return root / directory / "features" / f"hc3_{split}.coarse.features.npz"


def coarse_lr_prediction_path(root: Path, inventory: str) -> Path:
    directory = granularity_dir(inventory)
    if inventory in _SHARED_PROJECTION_DIRS:
        return root / directory / "models" / inventory / "coarse_all" / "lr" / "predictions.npz"
    return root / directory / "models" / "m3_frequency" / "lr" / "predictions.npz"


def coarse_results_path(root: Path, inventory: str) -> Path:
    return root / granularity_dir(inventory) / "results.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, output.getvalue())


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {key: np.asarray(values[key]) for key in values.files}


def validate_pair(
    fine: Mapping[str, np.ndarray],
    coarse: Mapping[str, np.ndarray],
    label: str,
    mass_key: str = "X_m3_frequency",
) -> None:
    for key in ("identity_id", "identity_hash", "y"):
        if not np.array_equal(fine[key], coarse[key]):
            raise ValueError(f"Fine/coarse {key} mismatch for {label}")
    if not np.allclose(fine[mass_key].sum(axis=1), coarse[mass_key].sum(axis=1), atol=2e-6):
        raise ValueError(f"Fine/coarse feature mass mismatch for {label}")


def aggregate_wad(
    fine_frequency: np.ndarray,
    fine_wad: np.ndarray,
    assignment: np.ndarray,
    coarse_frequency: np.ndarray,
) -> np.ndarray:
    """Aggregate motif depth using fine-motif frequency as its weight."""
    if fine_frequency.shape != fine_wad.shape:
        raise ValueError("Fine MF and WAD matrices differ in shape")
    if fine_frequency.shape[1] != len(assignment):
        raise ValueError("Motif assignment width does not match fine feature width")
    if coarse_frequency.ndim != 2 or fine_frequency.shape[0] != coarse_frequency.shape[0]:
        raise ValueError("Fine/coarse MF matrices differ in row count")
    if np.any(assignment < 0) or np.any(assignment >= coarse_frequency.shape[1]):
        raise ValueError("Motif assignment contains an invalid coarse index")
    numerator = np.zeros(coarse_frequency.shape, dtype=np.float32)
    for fine_index, coarse_index in enumerate(assignment):
        numerator[:, coarse_index] += fine_frequency[:, fine_index] * fine_wad[:, fine_index]
    return np.divide(
        numerator,
        coarse_frequency,
        out=np.zeros_like(numerator),
        where=coarse_frequency > 0,
    )


def interleave_mf_wad(frequency: np.ndarray, wad: np.ndarray) -> np.ndarray:
    """Interleave each motif's frequency and weighted average depth columns."""
    frequency = np.asarray(frequency, dtype=np.float32)
    wad = np.asarray(wad, dtype=np.float32)
    if frequency.ndim != 2 or frequency.shape != wad.shape:
        raise ValueError(f"Malformed MF/WAD matrices: {frequency.shape}, {wad.shape}")
    result = np.empty((frequency.shape[0], frequency.shape[1] * 2), dtype=np.float32)
    result[:, 0::2] = frequency
    result[:, 1::2] = wad
    if not np.isfinite(result).all():
        raise ValueError("MF+WAD features contain non-finite values")
    return result


def identity_clean_mask(train_hashes: np.ndarray, test_hashes: np.ndarray) -> np.ndarray:
    train_values = {str(value) for value in train_hashes}
    seen = set()
    mask = np.zeros(len(test_hashes), dtype=bool)
    for index, value in enumerate(test_hashes):
        value = str(value)
        if value not in train_values and value not in seen:
            mask[index] = True
            seen.add(value)
    return mask


def assignment_from_pairs(
    fine_hashes: Sequence[Any],
    coarse_hashes: Sequence[Any],
    pairs: Sequence[tuple[Any, Any]],
    source: str,
) -> np.ndarray:
    fine_index = {str(value): index for index, value in enumerate(fine_hashes)}
    coarse_index = {str(value): index for index, value in enumerate(coarse_hashes)}
    if len(fine_index) != len(fine_hashes) or len(coarse_index) != len(coarse_hashes):
        raise ValueError(f"Duplicate hashes in feature ordering: {source}")
    assignment = np.full(len(fine_hashes), -1, dtype=np.int64)
    for fine_hash, coarse_hash in pairs:
        fine_key, coarse_key = str(fine_hash), str(coarse_hash)
        if fine_key not in fine_index or coarse_key not in coarse_index:
            raise ValueError(f"Unknown motif hash in mapping from {source}: {fine_key}")
        index = fine_index[fine_key]
        target = coarse_index[coarse_key]
        if assignment[index] not in (-1, target):
            raise ValueError(f"Conflicting coarse assignments in {source}: {fine_key}")
        assignment[index] = target
    if np.any(assignment < 0):
        raise ValueError(f"Incomplete motif mapping from {source}")
    return assignment


def graph_from_data(value: Mapping[str, Any]) -> nx.DiGraph:
    graph = nx.node_link_graph(dict(value))
    if not graph.is_directed() or graph.is_multigraph() or len(graph) != 3:
        raise ValueError("Expected a directed simple three-node motif")
    return graph


def coarsen_graph(
    graph: nx.DiGraph,
    relation_map: Mapping[str, str],
    context: str = "motif",
) -> nx.DiGraph:
    result = graph.copy()
    for _, _, attributes in result.edges(data=True):
        relation = str(attributes.get("label_0"))
        if relation == "/":
            continue
        if relation not in relation_map:
            raise ValueError(f"No coarse mapping for relation {relation!r} in {context}")
        attributes["label_0"] = relation_map[relation]
    return result


def relation_labeled_isomorphic(left: nx.DiGraph, right: nx.DiGraph) -> bool:
    return nx.is_isomorphic(
        left,
        right,
        edge_match=lambda left_edge, right_edge: left_edge.get("label_0") == right_edge.get("label_0"),
    )


def relation_signature(graph: nx.DiGraph) -> str:
    return "|".join(sorted(
        str(attributes.get("label_0"))
        for _, _, attributes in graph.edges(data=True)
        if attributes.get("label_0") != "/"
    )) or "(structural only)"
