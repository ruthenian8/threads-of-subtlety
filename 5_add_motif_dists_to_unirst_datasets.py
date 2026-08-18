"""Add the paper's motif distributions to UniRST graph outputs."""

from __future__ import annotations

import argparse
import glob
import os
import random
from typing import Dict, List

import numpy as np
from tqdm.contrib.concurrent import process_map

from tos.tos_dataset import DiscourseMotifDists, Document, ToSDataset
from tos.tos_utils import load_json


random.seed(42)


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
    parser.add_argument("--motif-dir", default="data/motifs")
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    selected = load_json(
        os.path.join(args.motif_dir, "hc3-mage_selected-motif-hashes.json")
    )
    motifs = {
        size: ToSDataset.load_motifs(
            os.path.join(args.motif_dir, f"hc3-mage_M{size}_motifs.json"),
            selected[f"m{size}"],
        )
        for size in (3, 6, 9)
    }

    files = glob.glob(
        os.path.join(args.root, "rel-*", "*.discourse_parsed.graph_added.jsonl")
    )
    for file_path in sorted(files):
        dataset = ToSDataset.load_document_corpus(file_path)
        dataset = process_map(
            add_motif_dist_to_document,
            dataset,
            [motifs[3]] * len(dataset),
            [motifs[6]] * len(dataset),
            [motifs[9]] * len(dataset),
            max_workers=args.workers,
            chunksize=1,
        )
        output_path = f"{file_path[:-6]}.motif_dists.jsonl"
        ToSDataset.save_dataset_as_jsonl(dataset, output_path)
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

