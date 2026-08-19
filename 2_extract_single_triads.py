import argparse
import os
import re
from glob import glob
from itertools import combinations
from multiprocessing import Manager, Pool

import networkx as nx
from tqdm import tqdm

from tos.tos_dataset import ToSDataset
from tos.tos_utils import is_isomorphic_multiple, save_graph_motifs

shared_list = None


def init_globals(manager_list):
    """Initializer for each child process to set the global shared_list."""
    global shared_list
    shared_list = manager_list


def worker_function(G):
    """
    Checks if 'item' is in the shared_list.
    If not found, appends it.
    Returns a message for demonstration.
    """
    for SG in (G.subgraph(s).copy() for s in combinations(G, 3)):
        if len(list(nx.isolates(SG))):
            continue
        if is_isomorphic_multiple(shared_list, SG):
            continue
        shared_list.append(SG)
    return "P"


def extract_inventory(args, relinventory: str) -> None:
    inventory_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", relinventory)
    inventory_dir = f"rel-{inventory_name}"
    file_paths = sorted(
        glob(
            os.path.join(
                args.root, inventory_dir, "*.discourse_parsed.graph_added.jsonl"
            )
        )
    )
    if not file_paths:
        raise FileNotFoundError(
            f"No graph-added files found for inventory {relinventory!r} below {args.root!r}"
        )
    dataset = ToSDataset.load_datasets(file_paths, max_per_file=args.max_per_file)

    all_graphs = [
        tree.graph_networkx
        for document in dataset
        for tree in document.scene_discourse_trees.values()
    ]
    print(f"{relinventory}: no. of all_graphs: {len(all_graphs)}")

    with Manager() as manager:
        manager_list = manager.list()
        pool_kwargs = {"initializer": init_globals, "initargs": (manager_list,)}
        if args.workers is not None:
            pool_kwargs["processes"] = args.workers
        with Pool(**pool_kwargs) as pool:
            list(tqdm(pool.imap(worker_function, all_graphs), total=len(all_graphs)))
        motifs = list(manager_list)
        print(f"{relinventory}: len: {len(motifs)}")

    motif_dir = (
        os.path.join(args.motif_dir, inventory_name)
        if args.motif_dir and args.relinventory is None
        else args.motif_dir or os.path.join("data/motifs", inventory_name)
    )
    save_graph_motifs(3, motifs, args.dataset_name, output_dir=motif_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="data/unirst",
        help="Dataset root containing rel-*/...graph_added.jsonl files.",
    )
    parser.add_argument("--motif-dir", default=None)
    parser.add_argument("--dataset-name", default="hc3-mage")
    parser.add_argument(
        "--relinventory",
        default=None,
        help="Process only one RST inventory for a standard-specific motif set.",
    )
    parser.add_argument("--max-per-file", type=int, default=15000)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    if args.relinventory:
        relinventories = [args.relinventory]
    else:
        relinventories = sorted(
            os.path.basename(path)[len("rel-") :]
            for path in glob(os.path.join(args.root, "rel-*"))
            if os.path.isdir(path)
        )
    if not relinventories:
        raise FileNotFoundError(f"No rel-* inventory directories found below {args.root!r}")
    for relinventory in relinventories:
        extract_inventory(args, relinventory)


if __name__ == "__main__":
    main()
