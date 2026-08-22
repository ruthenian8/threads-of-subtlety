"""Extract per-inventory M3 catalogs from UniRST discourse graphs."""

import argparse
import os
import re
from glob import glob
from multiprocessing import Pool

from tqdm import tqdm

from tos.tos_dataset import ToSDataset
from tos.tos_utils import (
    connected_three_node_subgraphs,
    deduplicate_graph_motifs,
    save_graph_motifs,
)


def worker_function(G):
    """Return motifs unique within one graph; the parent merges all workers."""
    return deduplicate_graph_motifs(connected_three_node_subgraphs(G))


def resolve_dataset_root(args) -> str:
    """Resolve the UniRST graph root below the configurable data directory."""
    return args.root or os.path.join(args.data_dir, "unirst")


def resolve_motif_dir(args, relinventory: str) -> str:
    """Resolve one inventory's catalog directory.

    By default catalogs live in ``data/motifs/<inventory>``.  Preserve the
    existing single-inventory convention where an explicit ``--motif-dir`` is
    the exact output directory rather than a parent directory.
    """
    inventory_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", relinventory)
    motif_root = args.motif_dir or os.path.join(args.data_dir, "motifs")
    if args.relinventory is not None and args.motif_dir:
        return args.motif_dir
    return os.path.join(motif_root, inventory_name)


def discover_graph_files(
    dataset_root: str, inventory_dir: str, dataset_name: str
):
    """Return only the corpus shards represented by the catalog name."""
    prefixes = ("hc3", "mage") if dataset_name == "hc3-mage" else (dataset_name,)
    return sorted(
        {
            path
            for prefix in prefixes
            for path in glob(
                os.path.join(
                    dataset_root,
                    inventory_dir,
                    f"{prefix}_*.discourse_parsed.graph_added.jsonl",
                )
            )
        }
    )


def extract_inventory(args, relinventory: str) -> None:
    inventory_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", relinventory)
    inventory_dir = f"rel-{inventory_name}"
    dataset_root = resolve_dataset_root(args)
    file_paths = discover_graph_files(
        dataset_root, inventory_dir, args.dataset_name
    )
    if not file_paths:
        raise FileNotFoundError(
            f"No graph-added files found for inventory {relinventory!r} "
            f"below {dataset_root!r}"
        )
    dataset = ToSDataset.load_datasets(file_paths, max_per_file=args.max_per_file)

    all_graphs = [
        tree.graph_networkx
        for document in dataset
        for tree in document.scene_discourse_trees.values()
    ]
    print(f"{relinventory}: no. of all_graphs: {len(all_graphs)}")

    with Pool(processes=args.workers) as pool:
        motifs = deduplicate_graph_motifs(
            motif
            for graph_motifs in tqdm(
                pool.imap(worker_function, all_graphs, chunksize=args.chunksize),
                total=len(all_graphs),
                desc=f"extracting single motifs ({relinventory})",
            )
            for motif in graph_motifs
        )
    print(f"{relinventory}: len: {len(motifs)}")

    motif_dir = resolve_motif_dir(args, relinventory)
    save_graph_motifs(3, motifs, args.dataset_name, output_dir=motif_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Top-level data directory containing unirst/ and motifs/.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Override the default <data-dir>/unirst dataset root.",
    )
    parser.add_argument(
        "--motif-dir",
        default=None,
        help=(
            "Override <data-dir>/motifs. With --relinventory, this is the "
            "exact inventory output directory."
        ),
    )
    parser.add_argument("--dataset-name", default="hc3")
    parser.add_argument(
        "--relinventory",
        default=None,
        help="Process only one RST inventory for a standard-specific motif set.",
    )
    parser.add_argument("--max-per-file", type=int, default=15000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=8)
    args = parser.parse_args()

    if args.chunksize < 1:
        parser.error("--chunksize must be at least 1")

    if args.relinventory:
        relinventories = [args.relinventory]
    else:
        dataset_root = resolve_dataset_root(args)
        relinventories = sorted(
            os.path.basename(path)[len("rel-") :]
            for path in glob(os.path.join(dataset_root, "rel-*"))
            if os.path.isdir(path)
        )
    if not relinventories:
        raise FileNotFoundError(
            f"No rel-* inventory directories found below "
            f"{resolve_dataset_root(args)!r}"
        )
    for relinventory in relinventories:
        extract_inventory(args, relinventory)


if __name__ == "__main__":
    main()
