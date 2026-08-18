"""Add discourse graphs to UniRST outputs for every relation inventory."""

from __future__ import annotations

import argparse
import glob
import os
from typing import Any, Dict, List

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


def add_graphs(file_path: str) -> None:
    output_path = f"{file_path[:-6]}.graph_added.jsonl"
    documents: List[Document] = []
    with open(file_path, encoding="utf-8") as handle:
        for line in handle:
            sample = filter_empty_trees(eval(line.strip()))
            if not sample["scene_discourse_trees"]:
                continue
            for tree in sample["scene_discourse_trees"].values():
                graph = ToSDataset.create_graph_from_const_format(tree["parsed"])
                tree["graph_dict"] = nx.json_graph.node_link_data(graph)
                tree["graph_networkx"] = None
            documents.append(Document(**sample))
    with open(output_path, "w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(f"{document.model_dump(mode='json')}\n")
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/unirst")
    args = parser.parse_args()
    files = glob.glob(os.path.join(args.root, "rel-*", "*.discourse_parsed.jsonl"))
    for file_path in sorted(files):
        add_graphs(file_path)


if __name__ == "__main__":
    main()

