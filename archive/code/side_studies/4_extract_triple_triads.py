"""Build size-nine motif definitions from the size-three and size-six sets."""

from __future__ import annotations

import argparse
import os
import re
from glob import glob

from tqdm.auto import tqdm

from tos.tos_utils import (
    iter_composed_triple_motifs,
    load_graph_motifs,
    prepare_double_docking_motifs,
    prepare_single_docking_variants,
    save_graph_motifs,
    write_selected_motif_hashes,
)


def resolve_motif_dir(args, relinventory: str) -> str:
    inventory_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", relinventory)
    motif_root = args.motif_dir or "data/motifs"
    return (
        args.motif_dir
        if args.relinventory is not None and args.motif_dir
        else os.path.join(motif_root, inventory_name)
    )


def extract_inventory(args, relinventory: str) -> None:
    motif_dir = resolve_motif_dir(args, relinventory)
    single_motifs_path = os.path.join(
        motif_dir, f"{args.dataset_name}_M3_motifs.json"
    )
    double_motifs_path = os.path.join(
        motif_dir, f"{args.dataset_name}_M6_motifs.json"
    )
    single_motifs = load_graph_motifs(single_motifs_path)
    double_motifs = load_graph_motifs(double_motifs_path)
    print(f"{relinventory}: no. of single motifs: {len(single_motifs)}")
    print(f"{relinventory}: no. of double motifs: {len(double_motifs)}")

    double_motifs_prepared = prepare_double_docking_motifs(
        double_motifs.values()
    )
    single_motifs_prepared = prepare_single_docking_variants(
        single_motifs.values()
    )
    print(
        f"{relinventory}: no. of prepared double motifs: "
        f"{len(double_motifs_prepared)}"
    )
    print(
        f"{relinventory}: no. of single docking variants: "
        f"{len(single_motifs_prepared)}"
    )

    triple_motifs_candidates = {}
    results = iter_composed_triple_motifs(
        double_motifs_prepared,
        single_motifs_prepared,
        args.workers,
        args.chunksize,
    )
    for result in tqdm(
        results,
        total=len(double_motifs_prepared),
        desc=f"composing triple motifs ({relinventory})",
    ):
        triple_motifs_candidates.update(result)
    print(
        f"{relinventory}: len of triple triangular motifs candidates: "
        f"{len(triple_motifs_candidates)}"
    )
    save_graph_motifs(
        9,
        triple_motifs_candidates.values(),
        args.dataset_name,
        show_tracking=args.show_tracking,
        output_dir=motif_dir,
    )
    manifest_path = write_selected_motif_hashes(
        motif_dir, dataset_name=args.dataset_name
    )
    print(f"{relinventory}: wrote selected motif manifest {manifest_path}")


def discover_relinventories(args):
    if args.relinventory:
        return [args.relinventory]

    relinventories = sorted(
        os.path.basename(path)[len("rel-") :]
        for path in glob(os.path.join(args.root, "rel-*"))
        if os.path.isdir(path)
    )
    if relinventories:
        return relinventories

    motif_root = args.motif_dir or "data/motifs"
    return sorted(
        os.path.basename(path)
        for path in glob(os.path.join(motif_root, "*"))
        if os.path.isdir(path)
        and os.path.isfile(
            os.path.join(path, f"{args.dataset_name}_M3_motifs.json")
        )
        and os.path.isfile(
            os.path.join(path, f"{args.dataset_name}_M6_motifs.json")
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="data/unirst",
        help="Dataset root whose rel-* directories identify RST inventories.",
    )
    parser.add_argument(
        "--motif-dir",
        default=None,
        help=(
            "Motif root for all inventories, or the exact inventory directory "
            "when --relinventory is set."
        ),
    )
    parser.add_argument("--dataset-name", default="hc3-mage")
    parser.add_argument(
        "--relinventory",
        default=None,
        help="Process only one RST inventory.",
    )
    parser.add_argument("--show-tracking", action="store_true")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--chunksize", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.chunksize < 1:
        parser.error("--chunksize must be at least 1")

    relinventories = discover_relinventories(args)
    if not relinventories:
        motif_root = args.motif_dir or "data/motifs"
        raise FileNotFoundError(
            f"No rel-* inventory directories found below {args.root!r}, and no "
            f"inventory M3/M6 motif directories found below {motif_root!r}"
        )
    for relinventory in relinventories:
        extract_inventory(args, relinventory)


if __name__ == "__main__":
    main()
