import argparse
import os
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="data/unirst",
        help="Dataset root containing rel-*/...graph_added.jsonl files.",
    )
    parser.add_argument("--motif-dir", default="data/motifs")
    parser.add_argument("--dataset-name", default="hc3-mage")
    parser.add_argument("--max-per-file", type=int, default=15000)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    file_paths = sorted(
        glob(os.path.join(args.root, "rel-*", "*.discourse_parsed.graph_added.jsonl"))
    )
    if not file_paths:
        raise FileNotFoundError(
            f"No graph-added UniRST files found below {args.root!r}"
        )
    dataset = ToSDataset.load_datasets(file_paths, max_per_file=args.max_per_file)

    motif_size = 3

    all_graphs = []
    for document in dataset:
        for tree in document.scene_discourse_trees.values():
            all_graphs.append(tree.graph_networkx)
    print(f"no. of all_graphs: {len(all_graphs)}")

    with Manager() as manager:
        manager_list = manager.list()
        pool_kwargs = {"initializer": init_globals, "initargs": (manager_list,)}
        if args.workers is not None:
            pool_kwargs["processes"] = args.workers
        with Pool(**pool_kwargs) as pool:
            results = list(
                tqdm(pool.imap(worker_function, all_graphs), total=len(all_graphs))
            )
        motifs = list(manager_list)
        print("Shared list:", motifs)
        print("len:", len(motifs))

    save_graph_motifs(
        motif_size,
        motifs,
        args.dataset_name,
        output_dir=args.motif_dir,
    )


if __name__ == "__main__":
    main()
