"""Extract per-inventory M6 catalogs and finalize an M3+M6 manifest."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
import re
from glob import glob
from typing import Iterable, List, Mapping

import networkx as nx
from tqdm.auto import tqdm

from tos.tos_dataset import ToSDataset
from tos.tos_utils import (
    index_graph_motifs,
    load_graph_motifs,
    observed_double_motifs,
    save_graph_motifs,
    write_selected_motif_hashes,
)


motif_buckets = {}


def init_motif_worker(motifs):
    global motif_buckets
    motif_buckets = index_graph_motifs(motifs)


def extract_double_motifs(sample):
    return observed_double_motifs(sample, motif_buckets)


def select_motif_hashes(
    scene_support: Mapping[str, int],
    shard_support: Mapping[str, int],
    min_scene_support: int = 1,
    min_shard_support: int = 1,
    coverage: float = 1.0,
) -> List[str]:
    """Select frequent, broadly observed motifs in deterministic order.

    Minimum-support filters are applied first. ``coverage`` then retains the
    smallest frequency-ranked prefix accounting for that proportion of the
    surviving scene support. Ties are resolved by motif hash.
    """
    if min_scene_support < 1:
        raise ValueError("min_scene_support must be at least 1")
    if min_shard_support < 1:
        raise ValueError("min_shard_support must be at least 1")
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be greater than 0 and at most 1")

    eligible = sorted(
        (
            motif_hash
            for motif_hash, support in scene_support.items()
            if support >= min_scene_support
            and shard_support.get(motif_hash, 0) >= min_shard_support
        ),
        key=lambda motif_hash: (-scene_support[motif_hash], motif_hash),
    )
    if coverage == 1 or not eligible:
        return eligible

    target_support = coverage * sum(scene_support[h] for h in eligible)
    selected = []
    accumulated_support = 0
    for motif_hash in eligible:
        selected.append(motif_hash)
        accumulated_support += scene_support[motif_hash]
        if accumulated_support >= target_support:
            break
    return selected


def write_support_report(
    motif_dir: str,
    dataset_name: str,
    relinventory: str,
    scene_support: Mapping[str, int],
    shard_support: Mapping[str, int],
    selected_hashes: Iterable[str],
    total_scenes: int,
    total_shards: int,
    min_scene_support: int,
    min_shard_support: int,
    coverage: float,
) -> str:
    """Persist selection inputs so a pruned manifest is reproducible."""
    selected = set(selected_hashes)
    report = {
        "dataset_name": dataset_name,
        "relinventory": relinventory,
        "total_scenes": total_scenes,
        "total_shards": total_shards,
        "selection": {
            "min_scene_support": min_scene_support,
            "min_shard_support": min_shard_support,
            "coverage": coverage,
            "selected_m6": len(selected),
            "available_m6": len(scene_support),
        },
        "motifs": {
            motif_hash: {
                "scene_support": scene_support[motif_hash],
                "shard_support": shard_support.get(motif_hash, 0),
                "selected": motif_hash in selected,
            }
            for motif_hash in sorted(scene_support)
        },
    }
    report_path = os.path.join(
        motif_dir, f"{dataset_name}_M6_support.json"
    )
    temporary_path = f"{report_path}.tmp"
    os.makedirs(motif_dir, exist_ok=True)
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    os.replace(temporary_path, report_path)
    return report_path


def resolve_dataset_root(args) -> str:
    return args.root or os.path.join(args.data_dir, "unirst")


def resolve_motif_dir(args, relinventory: str) -> str:
    inventory_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", relinventory)
    motif_root = args.motif_dir or os.path.join(args.data_dir, "motifs")
    if args.relinventory is not None and args.motif_dir:
        return args.motif_dir
    return os.path.join(motif_root, inventory_name)


def discover_graph_files(
    dataset_root: str, inventory_dir: str, dataset_name: str
):
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
    motif_dir = resolve_motif_dir(args, relinventory)
    motif_path = os.path.join(motif_dir, f"{args.dataset_name}_M3_motifs.json")
    motifs_three = [
        nx.convert_node_labels_to_integers(motif)
        for motif in load_graph_motifs(motif_path).values()
    ]
    file_paths = discover_graph_files(
        resolve_dataset_root(args),
        f"rel-{inventory_name}",
        args.dataset_name,
    )
    if not file_paths:
        raise FileNotFoundError(
            "No graph-added files found for inventory "
            f"{relinventory!r} below {resolve_dataset_root(args)!r}"
        )
    double_motifs = {}
    scene_support = Counter()
    shard_support = Counter()
    total_scenes = 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_motif_worker,
        initargs=(motifs_three,),
    ) as executor:
        for file_path in file_paths:
            dataset = ToSDataset.load_datasets(
                [file_path], max_per_file=args.max_per_file
            )
            graphs = [
                tree.graph_networkx
                for document in dataset
                for tree in document.scene_discourse_trees.values()
                if tree is not None
            ]
            total_scenes += len(graphs)
            motifs_in_shard = set()
            for result in tqdm(
                executor.map(
                    extract_double_motifs,
                    graphs,
                    chunksize=args.chunksize,
                ),
                total=len(graphs),
                desc=(
                    f"extracting M6 ({relinventory}: "
                    f"{os.path.basename(file_path)})"
                ),
            ):
                motif_hashes = result.keys()
                scene_support.update(motif_hashes)
                motifs_in_shard.update(motif_hashes)
                double_motifs.update(result)
            shard_support.update(motifs_in_shard)

    selected_m6_hashes = select_motif_hashes(
        scene_support,
        shard_support,
        min_scene_support=args.min_scene_support,
        min_shard_support=args.min_shard_support,
        coverage=args.coverage,
    )
    if not selected_m6_hashes:
        raise ValueError(
            f"M6 selection for {relinventory!r} is empty; relax "
            "--min-scene-support, --min-shard-support, or --coverage"
        )
    save_graph_motifs(
        6, double_motifs.values(), args.dataset_name, output_dir=motif_dir
    )
    manifest_path = write_selected_motif_hashes(
        motif_dir,
        dataset_name=args.dataset_name,
        sizes=(3, 6),
        selected_hashes_by_size={"m6": selected_m6_hashes},
    )
    report_path = write_support_report(
        motif_dir=motif_dir,
        dataset_name=args.dataset_name,
        relinventory=relinventory,
        scene_support=scene_support,
        shard_support=shard_support,
        selected_hashes=selected_m6_hashes,
        total_scenes=total_scenes,
        total_shards=len(file_paths),
        min_scene_support=args.min_scene_support,
        min_shard_support=args.min_shard_support,
        coverage=args.coverage,
    )
    print(
        f"{relinventory}: selected {len(selected_m6_hashes)} of "
        f"{len(double_motifs)} M6 motifs; wrote {manifest_path} and "
        f"{report_path}"
    )


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
            "exact inventory catalog directory."
        ),
    )
    parser.add_argument("--dataset-name", default="hc3")
    parser.add_argument("--relinventory", default=None)
    parser.add_argument("--max-per-file", type=int, default=15000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument(
        "--min-scene-support",
        type=int,
        default=1,
        help=(
            "Keep M6 motifs observed in at least this many scenes before "
            "coverage pruning (default: 1, no scene-support pruning)."
        ),
    )
    parser.add_argument(
        "--min-shard-support",
        type=int,
        default=1,
        help=(
            "Keep M6 motifs observed in at least this many input shards "
            "before coverage pruning (default: 1, no shard pruning)."
        ),
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=1.0,
        help=(
            "After support filtering, retain the smallest frequency-ranked "
            "M6 set covering this fraction of scene support (0 < value <= 1; "
            "default: 1)."
        ),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.chunksize < 1:
        parser.error("--chunksize must be at least 1")
    if args.min_scene_support < 1:
        parser.error("--min-scene-support must be at least 1")
    if args.min_shard_support < 1:
        parser.error("--min-shard-support must be at least 1")
    if not 0 < args.coverage <= 1:
        parser.error("--coverage must be greater than 0 and at most 1")

    dataset_root = resolve_dataset_root(args)
    relinventories = [args.relinventory] if args.relinventory else sorted(
        os.path.basename(path)[len("rel-") :]
        for path in glob(os.path.join(dataset_root, "rel-*"))
        if os.path.isdir(path)
    )
    if not relinventories:
        raise FileNotFoundError(
            f"No rel-* inventory directories found below {dataset_root!r}"
        )
    for relinventory in relinventories:
        extract_inventory(args, relinventory)


if __name__ == "__main__":
    main()
