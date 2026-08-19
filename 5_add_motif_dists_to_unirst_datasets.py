"""Add the paper's motif distributions to UniRST graph outputs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import glob
import os
import random
import re
from typing import Dict, List, Optional

import numpy as np
from tqdm.auto import tqdm

from tos.tos_dataset import DiscourseMotifDists, Document, ToSDataset
from tos.tos_utils import (
    load_json,
    resolve_selected_motif_hashes,
    validate_selected_motif_hashes,
)


random.seed(42)


_WORKER_MOTIFS: Optional[Dict[int, List]] = None


def _init_motif_worker(motifs: Dict[int, List]) -> None:
    """Initialize motif collections once in each process-pool worker."""
    global _WORKER_MOTIFS
    _WORKER_MOTIFS = motifs


def _add_motif_dist_to_document_worker(document: Document) -> Document:
    if _WORKER_MOTIFS is None:
        raise RuntimeError("motif worker was not initialized")
    return add_motif_dist_to_document(
        document,
        _WORKER_MOTIFS[3],
        _WORKER_MOTIFS[6],
        _WORKER_MOTIFS[9],
    )


def add_motif_dist_to_document(
    document: Document,
    m3_motifs: List,
    m6_motifs: List,
    m9_motifs: List,
) -> Document:
    for tree in document.scene_discourse_trees.values():
        if tree is None:
            continue
        root_label = f"span_1-{len(tree.edus)}"
        tree.motif_dists = {
            "m3": DiscourseMotifDists(
                **ToSDataset.calculate_motif_distribution(
                    tree.graph_networkx, m3_motifs, root_label
                )
            ),
            "m6": DiscourseMotifDists(
                **ToSDataset.calculate_motif_distribution(
                    tree.graph_networkx, m6_motifs, root_label
                )
            ),
            "m9": DiscourseMotifDists(
                **ToSDataset.calculate_motif_distribution(
                    tree.graph_networkx, m9_motifs, root_label
                )
            ),
        }
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/unirst")
    parser.add_argument("--motif-dir", default=None)
    parser.add_argument(
        "--relinventory",
        default=None,
        help="Process one RST inventory with its standard-specific motif set.",
    )
    parser.add_argument("--dataset-name", default="hc3-mage")
    parser.add_argument(
        "--selected-hashes",
        default=None,
        help="Selection manifest; generated manifest is preferred by default.",
    )
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    inventories = [args.relinventory] if args.relinventory else [None]
    if args.relinventory is None:
        inventory_dirs = sorted(glob.glob(os.path.join(args.root, "rel-*")))
        inventories = [
            os.path.basename(path)[len("rel-") :]
            for path in inventory_dirs
            if os.path.isdir(path)
        ] or [None]

    for relinventory in inventories:
        inventory_name = (
            re.sub(r"[^A-Za-z0-9_.-]+", "_", relinventory)
            if relinventory
            else None
        )
        motif_dir = (
            os.path.join(args.motif_dir, inventory_name)
            if args.motif_dir and args.relinventory is None and inventory_name
            else args.motif_dir
            or (os.path.join("data/motifs", inventory_name) if inventory_name else "data/motifs")
        )
        selected_manifest = resolve_selected_motif_hashes(
            motif_dir, args.dataset_name, args.selected_hashes
        )
        selected = load_json(selected_manifest)
        validate_selected_motif_hashes(
            motif_dir, selected, dataset_name=args.dataset_name
        )
        motifs = {
            size: ToSDataset.load_motifs(
                os.path.join(
                    motif_dir, f"{args.dataset_name}_M{size}_motifs.json"
                ),
                selected[f"m{size}"],
            )
            for size in (3, 6, 9)
        }

        inventory_dir = f"rel-{inventory_name}" if inventory_name else "rel-*"
        files = glob.glob(
            os.path.join(
                args.root, inventory_dir, "*.discourse_parsed.graph_added.jsonl"
            )
        )
        for file_path in sorted(files):
            dataset = ToSDataset.load_document_corpus(file_path)
            # Motifs are sent once per worker through the initializer instead of
            # being serialized in every document task.
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_init_motif_worker,
                initargs=(motifs,),
            ) as executor:
                dataset = list(
                    tqdm(
                        executor.map(
                            _add_motif_dist_to_document_worker,
                            dataset,
                            chunksize=1,
                        ),
                        total=len(dataset),
                        desc=os.path.basename(file_path),
                    )
                )
            output_path = f"{file_path[:-6]}.motif_dists.jsonl"
            ToSDataset.save_dataset_as_jsonl(dataset, output_path)
            print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
