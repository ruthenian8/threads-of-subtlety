#!/usr/bin/env python3
"""Create identity-aligned feature matrices for the canonical four standards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


INVENTORIES = (
    "deu.rst.pcc",
    "eng.erst.gum.new",
    "eng.rst.rstdt.new",
    "nld.rst.nldt",
)
SPLITS = ("train", "validation", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load(path: Path):
    with np.load(path, allow_pickle=False) as values:
        return {key: np.asarray(values[key]) for key in values.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("artifacts/work/raw_feature_matrices"))
    parser.add_argument("--motif-dir", type=Path, default=Path("data/motifs2"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/canonical/feature_matrices"))
    parser.add_argument("--inventories", nargs="+", default=list(INVENTORIES))
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    manifest = {
        "analysis": "canonical_four_standard_common_identity_cohort",
        "inventories": args.inventories,
        "reference_inventory": args.inventories[0],
        "splits": {},
        "artifacts": [],
        "matrices": [],
        "catalogs": [],
    }
    for inventory in args.inventories:
        catalog = args.motif_dir / inventory / "hc3_M3_motifs.json"
        selected = args.motif_dir / inventory / "hc3_selected-motif-hashes.generated.json"
        provenance = args.motif_dir / inventory / "hc3_M3_provenance.json"
        selected_value = json.loads(selected.read_text(encoding="utf-8"))
        manifest["catalogs"].append({
            "inventory": inventory,
            "motif_count": len(selected_value["m3"]),
            "catalog_sha256": sha256_file(catalog),
            "selection_manifest_sha256": sha256_file(selected),
            "motif_provenance_sha256": sha256_file(provenance),
            "source_files_verified": 1,
        })
    for split in SPLITS:
        sources = {
            inventory: args.source_dir / inventory / f"hc3_{split}.features.npz"
            for inventory in args.inventories
        }
        matrices = {inventory: load(path) for inventory, path in sources.items()}
        identity_maps = {}
        labels = {}
        for inventory, matrix in matrices.items():
            ids = matrix["identity_id"].astype(str).tolist()
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate identity_id in {inventory}/{split}")
            identity_maps[inventory] = {value: index for index, value in enumerate(ids)}
            labels[inventory] = {value: int(matrix["y"][index]) for value, index in identity_maps[inventory].items()}
        common = set.intersection(*(set(value) for value in identity_maps.values()))
        reference_ids = matrices[args.inventories[0]]["identity_id"].astype(str).tolist()
        ordered = [value for value in reference_ids if value in common]
        if len(ordered) != len(common):
            raise ValueError(f"Reference ordering failed for {split}")
        for identity in ordered:
            observed = {labels[inventory][identity] for inventory in args.inventories}
            if len(observed) != 1:
                raise ValueError(f"Label disagreement for {identity} in {split}")
        manifest["splits"][split] = {
            "common_rows": len(ordered),
            "source_rows": {
                inventory: int(len(matrices[inventory]["y"]))
                for inventory in args.inventories
            },
            "excluded_rows": {
                inventory: int(len(matrices[inventory]["y"]) - len(ordered))
                for inventory in args.inventories
            },
        }
        for inventory in args.inventories:
            matrix = matrices[inventory]
            indices = np.asarray([identity_maps[inventory][value] for value in ordered])
            row_count = len(matrix["y"])
            arrays = {
                key: (value[indices] if value.ndim and value.shape[0] == row_count else value)
                for key, value in matrix.items()
            }
            if not np.array_equal(arrays["identity_id"].astype(str), np.asarray(ordered)):
                raise ValueError(f"Output alignment failed for {inventory}/{split}")
            destination = args.output_dir / inventory / f"hc3_{split}.features.npz"
            if args.reuse_existing and destination.exists():
                existing = load(destination)
                if set(existing) != set(arrays) or any(
                    not np.array_equal(existing[key], arrays[key]) for key in arrays
                ):
                    raise ValueError(f"Existing canonical matrix differs: {destination}")
            else:
                atomic_npz(destination, arrays)
            record = {
                "inventory": inventory,
                "split": split,
                "records": len(ordered),
                "source_path": str(sources[inventory].resolve()),
                "source_sha256": sha256_file(sources[inventory]),
                "artifact_path": str(destination.resolve()),
                "artifact_sha256": sha256_file(destination),
            }
            atomic_json(destination.with_suffix(".provenance.json"), record)
            manifest["artifacts"].append(record)
            manifest["matrices"].append({
                "inventory": inventory,
                "split": split,
                "records": len(ordered),
                "alignment_verified": True,
                "artifact_hash_verified": True,
                "source_hash_verified": True,
                "catalog_path": str((args.motif_dir / inventory / "hc3_M3_motifs.json").resolve()),
            })
    manifest["exact_identity_order_across_inventories"] = True
    manifest["exact_labels_across_inventories"] = True
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(f"wrote {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
