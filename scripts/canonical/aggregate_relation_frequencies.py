#!/usr/bin/env python3
"""Aggregate corpus-level relation frequencies from canonical feature matrices.

Each canonical row stores relation proportions and a total relation-edge count.
Multiplying those values recovers integer edge counts per document; this script
validates that recovery before pooling counts over train, validation, and test.
The resulting shares are edge-weighted (not an average of document-level
proportions), and exclude the structural "/" edge as in the feature builder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INVENTORIES = (
    "deu.rst.pcc",
    "eng.erst.gum.new",
    "eng.rst.rstdt.new",
    "nld.rst.nldt",
)
SPLITS = ("train", "validation", "test")
STRUCTURAL_EDGE = "/"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_exact_counts(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as matrix:
        names = matrix["relation_names"].astype(str).tolist()
        proportions = np.asarray(matrix["X_relation_frequency"], dtype=np.float64)
        totals = np.asarray(matrix["relation_total"], dtype=np.int64)
        identities = matrix["identity_id"].astype(str)
        inventory = str(matrix["inventory"].item())
        split = str(matrix["split"].item())

    if proportions.ndim != 2 or proportions.shape != (len(identities), len(names)):
        raise ValueError(f"{path}: malformed relation matrix {proportions.shape}")
    if totals.shape != (len(identities),):
        raise ValueError(f"{path}: malformed relation totals {totals.shape}")
    if np.any(totals < 0) or not np.isfinite(proportions).all() or np.any(proportions < 0):
        raise ValueError(f"{path}: invalid relation proportions or totals")

    reconstructed = np.rint(proportions * totals[:, None]).astype(np.int64)
    row_sums = reconstructed.sum(axis=1)
    if not np.array_equal(row_sums, totals):
        bad = np.flatnonzero(row_sums != totals)[:5].tolist()
        raise ValueError(f"{path}: normalized proportions do not reconstruct edge totals at rows {bad}")
    if inventory != path.parent.name or split != path.name.split("_")[1].split(".")[0]:
        raise ValueError(f"{path}: embedded identity ({inventory}, {split}) disagrees with its path")
    return names, reconstructed, identities, totals


def summarize_matrix_root(matrix_root: Path, output_root: Path) -> None:
    all_rows: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    inventory_summaries: list[dict[str, Any]] = []
    reference_identities: dict[str, np.ndarray] = {}

    for inventory in INVENTORIES:
        split_counts: dict[str, dict[str, int]] = {}
        inventory_names: list[str] | None = None
        total_documents = 0
        for split in SPLITS:
            matrix_path = matrix_root / inventory / f"hc3_{split}.features.npz"
            if not matrix_path.is_file():
                raise FileNotFoundError(matrix_path)
            names, counts, identities, totals = load_exact_counts(matrix_path)
            if inventory_names is None:
                inventory_names = names
            elif names != inventory_names:
                raise ValueError(f"{inventory}/{split}: relation vocabulary differs across splits")
            if inventory == INVENTORIES[0]:
                reference_identities[split] = identities
            elif not np.array_equal(identities, reference_identities[split]):
                raise ValueError(f"{inventory}/{split}: row identities differ from {INVENTORIES[0]}")

            split_counts[split] = {
                name: int(value)
                for name, value in zip(names, counts.sum(axis=0))
            }
            total_documents += len(identities)
            input_records.append({
                "path": str(matrix_path),
                "sha256": sha256_file(matrix_path),
                "inventory": inventory,
                "split": split,
                "documents": int(len(identities)),
                "relation_edges": int(totals.sum()),
            })

        assert inventory_names is not None
        if STRUCTURAL_EDGE in inventory_names:
            raise ValueError(f"{inventory}: structural edge label '/' unexpectedly present")
        inventory_totals = {
            name: sum(split_counts[split][name] for split in SPLITS)
            for name in inventory_names
        }
        edge_denominator = sum(inventory_totals.values())
        if edge_denominator == 0:
            raise ValueError(f"{inventory}: no relation edges were counted")

        for name in inventory_names:
            count = inventory_totals[name]
            all_rows.append({
                "inventory": inventory,
                "relation": name,
                "train_count": split_counts["train"][name],
                "validation_count": split_counts["validation"][name],
                "test_count": split_counts["test"][name],
                "total_count": count,
                "total_edge_denominator": edge_denominator,
                "frequency": count / edge_denominator,
                "frequency_percent": 100 * count / edge_denominator,
                "document_denominator": total_documents,
            })
        inventory_summaries.append({
            "inventory": inventory,
            "documents": total_documents,
            "relation_types": len(inventory_names),
            "relation_edges": edge_denominator,
            "denominator": "all labeled relation edges pooled over HC3 train, validation, and test; '/' excluded",
        })

    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "overall_relation_frequencies.csv"
    temporary_csv = csv_path.with_name(csv_path.name + ".tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = list(all_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    os.replace(temporary_csv, csv_path)

    for summary in inventory_summaries:
        inventory = summary["inventory"]
        rows = [row for row in all_rows if row["inventory"] == inventory]
        rows.sort(key=lambda row: row["total_count"])
        height = max(5.5, 0.28 * len(rows) + 1.5)
        fig, ax = plt.subplots(figsize=(10, height), constrained_layout=True)
        ax.barh([row["relation"] for row in rows], [row["frequency_percent"] for row in rows], color="#527a9e")
        ax.set_xlabel("Share of all labeled relation edges (%)")
        ax.set_ylabel("Relation")
        ax.set_title(f"{inventory}: pooled relation frequencies\nN={summary['documents']:,} documents; {summary['relation_edges']:,} relation edges")
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        fig.savefig(output_root / f"{inventory}_relation_frequency_histogram.png", dpi=240)
        plt.close(fig)

    atomic_json(output_root / "manifest.json", {
        "analysis": "pooled_canonical_relation_edge_frequencies",
        "aggregation": "integer edge counts reconstructed per document from X_relation_frequency * relation_total and validated against row totals",
        "frequency_denominator": "all labeled relation edges pooled over train, validation, and test; structural '/' edge excluded",
        "cohort": "aligned canonical HC3 identity cohort; counts are descriptive and not label-conditioned",
        "inputs": input_records,
        "inventories": inventory_summaries,
        "output_csv": str(csv_path),
        "histograms": [f"{item['inventory']}_relation_frequency_histogram.png" for item in inventory_summaries],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, default=Path("artifacts/canonical/feature_matrices"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/canonical/relation_frequencies"))
    args = parser.parse_args()
    summarize_matrix_root(args.matrix_root, args.output_root)
    print(f"Wrote pooled relation frequencies and histograms to {args.output_root}")


if __name__ == "__main__":
    main()
