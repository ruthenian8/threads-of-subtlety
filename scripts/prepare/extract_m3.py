"""Extract per-inventory M3 catalogs from UniRST discourse graphs."""

import argparse
import ast
import hashlib
import json
import os
import platform
import re
import sys
from glob import glob
from itertools import combinations, permutations
from multiprocessing import Pool

import networkx as nx
from tqdm import tqdm

from tos.tos_utils import (
    node_link_graph_compat,
    save_graph_motifs,
)


def canonical_m3_signature(graph, nodes):
    """Return an exact isomorphism key for a three-node edge-labeled digraph."""
    signatures = []
    for ordered_nodes in permutations(nodes):
        signature = []
        for source in ordered_nodes:
            for target in ordered_nodes:
                edge = graph.get_edge_data(source, target)
                signature.append(
                    "\x00" if edge is None else "\x01" + str(edge["label_0"])
                )
        signatures.append(tuple(signature))
    return min(signatures)


def worker_function(graph):
    """Return exact, unique M3 signatures and representatives from one graph."""
    undirected = graph.to_undirected(as_view=True)
    seen_node_sets = set()
    motifs = {}
    for center in undirected:
        for left, right in combinations(undirected.neighbors(center), 2):
            nodes = frozenset((center, left, right))
            if nodes in seen_node_sets:
                continue
            seen_node_sets.add(nodes)
            signature = canonical_m3_signature(graph, tuple(nodes))
            if signature not in motifs:
                motifs[signature] = graph.subgraph(nodes).copy()
    return list(motifs.items())


def iter_graph_dicts(serialized):
    """Yield graph dictionaries without evaluating irrelevant document fields."""
    marker = "'graph_dict': "
    terminator = ", 'graph_networkx':"
    position = 0
    while True:
        marker_at = serialized.find(marker, position)
        if marker_at < 0:
            return
        value_at = marker_at + len(marker)
        value_end = serialized.find(terminator, value_at)
        if value_end < 0:
            raise ValueError("graph_dict has no graph_networkx terminator")
        literal = serialized[value_at:value_end]
        if literal != "None":
            yield ast.literal_eval(literal)
        position = value_end + len(terminator)


def iter_graphs(file_paths, max_per_file, stats):
    """Stream only graph dictionaries, avoiding full document evaluation."""
    for file_path in file_paths:
        print(file_path)
        file_documents = 0
        with open(file_path, encoding="utf-8") as handle:
            for line in handle:
                if max_per_file and file_documents >= max_per_file:
                    break
                serialized = line.strip()
                if not serialized:
                    continue
                file_documents += 1
                stats["document_count"] += 1
                for graph_dict in iter_graph_dicts(serialized):
                    stats["scene_count"] += 1
                    yield node_link_graph_compat(graph_dict)
        print(f"loaded: {file_documents} samples")


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
    dataset_root: str,
    inventory_dir: str,
    dataset_name: str,
    split: str = None,
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
                    (
                        f"{prefix}_{split}.discourse_parsed.graph_added.jsonl"
                        if split
                        else f"{prefix}_*.discourse_parsed.graph_added.jsonl"
                    ),
                )
            )
        }
    )


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(
    motif_dir: str,
    args,
    relinventory: str,
    file_paths,
    document_count: int,
    scene_count: int,
    motif_count: int,
) -> str:
    catalog_path = os.path.join(motif_dir, f"{args.dataset_name}_M3_motifs.json")
    provenance_path = os.path.join(
        motif_dir, f"{args.dataset_name}_M3_provenance.json"
    )
    record = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "relinventory": relinventory,
        "dataset_root": os.path.abspath(resolve_dataset_root(args)),
        "source_files": [
            {"path": os.path.abspath(path), "sha256": sha256_file(path)}
            for path in file_paths
        ],
        "document_count": document_count,
        "scene_count": scene_count,
        "motif_count": motif_count,
        "max_per_file": args.max_per_file or None,
        "catalog": {
            "path": os.path.abspath(catalog_path),
            "sha256": sha256_file(catalog_path),
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "networkx_version": nx.__version__,
        },
        "identity": {
            "wl_edge_attr": "label_0",
            "exact_isomorphism_edge_attr": "label_0",
            "node_attributes_used": [],
            "deduplication": "exact canonical M3 permutation signature",
        },
    }
    temporary_path = provenance_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, provenance_path)
    return provenance_path


def extract_inventory(args, relinventory: str) -> None:
    inventory_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", relinventory)
    inventory_dir = f"rel-{inventory_name}"
    dataset_root = resolve_dataset_root(args)
    file_paths = discover_graph_files(
        dataset_root, inventory_dir, args.dataset_name, split=args.split
    )
    if not file_paths:
        raise FileNotFoundError(
            f"No graph-added files found for inventory {relinventory!r} "
            f"below {dataset_root!r}"
        )
    stats = {"document_count": 0, "scene_count": 0}
    all_graphs = iter_graphs(file_paths, args.max_per_file, stats)
    if args.workers == 1:
        graph_motif_sets = map(worker_function, all_graphs)
        motif_sets = tqdm(
            graph_motif_sets,
            desc=f"extracting single motifs ({relinventory})",
            unit="scene",
        )
        motifs_by_signature = {}
        for graph_motifs in motif_sets:
            for signature, motif in graph_motifs:
                motifs_by_signature.setdefault(signature, motif)
    else:
        with Pool(processes=args.workers) as pool:
            motif_sets = tqdm(
                pool.imap(
                    worker_function, all_graphs, chunksize=args.chunksize
                ),
                desc=f"extracting single motifs ({relinventory})",
                unit="scene",
            )
            motifs_by_signature = {}
            for graph_motifs in motif_sets:
                for signature, motif in graph_motifs:
                    motifs_by_signature.setdefault(signature, motif)
    motifs = list(motifs_by_signature.values())
    print(f"{relinventory}: no. of all_graphs: {stats['scene_count']}")
    print(f"{relinventory}: len: {len(motifs)}")

    motif_dir = resolve_motif_dir(args, relinventory)
    save_graph_motifs(3, motifs, args.dataset_name, output_dir=motif_dir)
    provenance_path = write_provenance(
        motif_dir,
        args,
        relinventory,
        file_paths,
        document_count=stats["document_count"],
        scene_count=stats["scene_count"],
        motif_count=len(motifs),
    )
    print(f"{relinventory}: wrote provenance {provenance_path}")


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
        "--split",
        choices=("train", "validation", "test"),
        default="train",
        help="Restrict catalog discovery to one canonical split.",
    )
    parser.add_argument(
        "--relinventory",
        default=None,
        help="Process only one RST inventory for a standard-specific motif set.",
    )
    parser.add_argument(
        "--max-per-file",
        type=int,
        default=0,
        help="Maximum documents per input file; 0 uses every document.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes; 1 avoids duplicating model-library memory.",
    )
    parser.add_argument("--chunksize", type=int, default=8)
    args = parser.parse_args()

    if args.chunksize < 1:
        parser.error("--chunksize must be at least 1")
    if args.max_per_file < 0:
        parser.error("--max-per-file cannot be negative")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

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
