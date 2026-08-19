import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import re
from glob import glob

import networkx as nx
from tqdm.auto import tqdm

from tos.tos_dataset import ToSDataset
from tos.tos_utils import (
    index_graph_motifs,
    load_graph_motifs,
    observed_double_motifs,
    save_graph_motifs,
)


motif_buckets = {}


def init_motif_worker(motifs):
    global motif_buckets
    motif_buckets = index_graph_motifs(motifs)


def extract_double_motifs(sample):
    return observed_double_motifs(sample, motif_buckets)


def extract_inventory(args, relinventory: str) -> None:
    inventory_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", relinventory)
    motif_dir = (
        os.path.join(args.motif_dir, inventory_name)
        if args.motif_dir and args.relinventory is None
        else args.motif_dir or os.path.join("data/motifs", inventory_name)
    )
    motif_path = os.path.join(motif_dir, f"{args.dataset_name}_M3_motifs.json")
    motifs_three = [
        nx.convert_node_labels_to_integers(motif)
        for motif in load_graph_motifs(motif_path).values()
    ]
    file_paths = sorted(
        glob(
            os.path.join(
                args.root,
                f"rel-{inventory_name}",
                "*.discourse_parsed.graph_added.jsonl",
            )
        )
    )
    if not file_paths:
        raise FileNotFoundError(
            "No graph-added files found for inventory "
            f"{relinventory!r} below {args.root!r}"
        )
    dataset = ToSDataset.load_datasets(file_paths, max_per_file=args.max_per_file)
    all_graphs = [
        tree.graph_networkx
        for document in dataset
        for tree in document.scene_discourse_trees.values()
    ]
    double_motifs = {}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_motif_worker,
        initargs=(motifs_three,),
    ) as executor:
        for result in tqdm(
            executor.map(extract_double_motifs, all_graphs, chunksize=args.chunksize),
            total=len(all_graphs),
            desc=f"extracting double motifs ({relinventory})",
        ):
            double_motifs.update(result)
    save_graph_motifs(
        6, double_motifs.values(), args.dataset_name, output_dir=motif_dir
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/unirst")
    parser.add_argument("--motif-dir", default=None)
    parser.add_argument("--dataset-name", default="hc3-mage")
    parser.add_argument("--relinventory", default=None)
    parser.add_argument("--max-per-file", type=int, default=15000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunksize", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.chunksize < 1:
        parser.error("--chunksize must be at least 1")

    relinventories = [args.relinventory] if args.relinventory else sorted(
        os.path.basename(path)[len("rel-") :]
        for path in glob(os.path.join(args.root, "rel-*"))
        if os.path.isdir(path)
    )
    if not relinventories:
        raise FileNotFoundError(
            f"No rel-* inventory directories found below {args.root!r}"
        )
    for relinventory in relinventories:
        extract_inventory(args, relinventory)


if __name__ == "__main__":
    main()
