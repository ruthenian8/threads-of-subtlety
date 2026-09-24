"""Add discourse graphs to UniRST outputs for every relation inventory."""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
from typing import Any, Dict

import networkx as nx
from tqdm.auto import tqdm

from tos.tos_dataset import Document, ToSDataset


def filter_empty_trees(sample: Dict[str, Any]) -> Dict[str, Any]:
    kept_indices = [
        int(scene_idx)
        for scene_idx, tree in sample["scene_discourse_trees"].items()
        if tree is not None and tree["parsed"] not in {"NONE", ""}
    ]
    sample["scenes"] = [sample["scenes"][idx] for idx in kept_indices]
    sample["scene_discourse_trees"] = {
        str(new_idx): sample["scene_discourse_trees"][str(old_idx)]
        for new_idx, old_idx in enumerate(kept_indices)
    }
    return sample


def read_record(serialized: str) -> Dict[str, Any]:
    """Read either strict JSON or the project's legacy Python-literal rows."""
    value = json.loads(serialized) if serialized.startswith('{"') else ast.literal_eval(serialized)
    if not isinstance(value, dict):
        raise ValueError("dataset row is not a mapping")
    return value


def count_records(file_path: str) -> int:
    with open(file_path, "rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def add_graphs(file_path: str, force: bool = False) -> None:
    output_path = f"{file_path[:-6]}.graph_added.jsonl"
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0 and not force:
        print(f"skipping existing {output_path}")
        return

    temporary_path = f"{output_path}.partial"
    with open(file_path, encoding="utf-8") as source, open(
        temporary_path, "w", encoding="utf-8"
    ) as destination:
        lines = tqdm(
            source,
            total=count_records(file_path),
            desc=os.path.basename(file_path),
            unit="doc",
        )
        for document_index, line in enumerate(lines, start=1):
            serialized = line.strip()
            if not serialized:
                continue
            sample = filter_empty_trees(read_record(serialized))
            if not sample["scene_discourse_trees"]:
                continue
            for tree in sample["scene_discourse_trees"].values():
                graph = ToSDataset.create_graph_from_const_format(tree["parsed"])
                tree["graph_dict"] = nx.json_graph.node_link_data(graph)
                tree["graph_networkx"] = None
            document = Document(**sample)
            destination.write(f"{ToSDataset.document_to_dict(document)}\n")
            if document_index % 100 == 0:
                destination.flush()
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary_path, output_path)
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/unirst")
    parser.add_argument(
        "--dataset-name",
        default=None,
        help=(
            "Only process files beginning with this dataset name; for example, "
            "'human_detectors'. The default processes every parsed dataset."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        help=(
            "Only process one canonical split. This is useful when per-domain "
            "parsed shards and aligned split files share the same directory."
        ),
    )
    parser.add_argument(
        "--relinventory",
        help=(
            "Only process one relation inventory directory, for example "
            "'eng.rst.rstdt.new'."
        ),
    )
    parser.add_argument(
        "--force", action="store_true", help="Rebuild existing graph-added files."
    )
    args = parser.parse_args()
    if args.split:
        if not args.dataset_name:
            parser.error("--split requires --dataset-name")
        prefix = f"{args.dataset_name}_{args.split}"
    else:
        prefix = f"{args.dataset_name}_*" if args.dataset_name else "*"
    inventory_directory = (
        f"rel-{args.relinventory}" if args.relinventory else "rel-*"
    )
    files = glob.glob(
        os.path.join(
            args.root, inventory_directory, f"{prefix}.discourse_parsed.jsonl"
        )
    )
    for file_path in sorted(files):
        add_graphs(file_path, force=args.force)


if __name__ == "__main__":
    main()
