"""Add discourse graphs to UniRST outputs for every relation inventory."""

from __future__ import annotations

import argparse
import glob
import os
from typing import Any, Dict

import networkx as nx

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


def add_graphs(file_path: str, force: bool = False) -> None:
    output_path = f"{file_path[:-6]}.graph_added.jsonl"
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0 and not force:
        print(f"skipping existing {output_path}")
        return

    temporary_path = f"{output_path}.partial"
    with open(file_path, encoding="utf-8") as source, open(
        temporary_path, "w", encoding="utf-8"
    ) as destination:
        for document_index, line in enumerate(source, start=1):
            serialized = line.strip()
            if not serialized:
                continue
            sample = filter_empty_trees(eval(serialized))
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
        "--force", action="store_true", help="Rebuild existing graph-added files."
    )
    args = parser.parse_args()
    files = glob.glob(os.path.join(args.root, "rel-*", "*.discourse_parsed.jsonl"))
    for file_path in sorted(files):
        add_graphs(file_path, force=args.force)


if __name__ == "__main__":
    main()
