"""Build size-nine motif definitions from the size-three and size-six sets."""

from __future__ import annotations

import argparse
import os

from tqdm.auto import tqdm

from tos.tos_utils import (
    iter_composed_triple_motifs,
    load_graph_motifs,
    prepare_double_docking_motifs,
    prepare_single_docking_variants,
    save_graph_motifs,
    write_selected_motif_hashes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motif-dir", default="data/motifs")
    parser.add_argument("--dataset-name", default="hc3-mage")
    parser.add_argument("--show-tracking", action="store_true")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--chunksize", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.chunksize < 1:
        parser.error("--chunksize must be at least 1")

    single_motifs_path = os.path.join(
        args.motif_dir, f"{args.dataset_name}_M3_motifs.json"
    )
    double_motifs_path = os.path.join(
        args.motif_dir, f"{args.dataset_name}_M6_motifs.json"
    )
    single_motifs = load_graph_motifs(single_motifs_path)
    double_motifs = load_graph_motifs(double_motifs_path)
    print(f"no. of single motifs: {len(single_motifs)}")
    print(f"no. of double motifs: {len(double_motifs)}")

    double_motifs_prepared = prepare_double_docking_motifs(
        double_motifs.values()
    )
    single_motifs_prepared = prepare_single_docking_variants(
        single_motifs.values()
    )
    print(f"no. of prepared double motifs: {len(double_motifs_prepared)}")
    print(f"no. of single docking variants: {len(single_motifs_prepared)}")

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
        desc="composing triple motifs",
    ):
        triple_motifs_candidates.update(result)
    print(
        "len of triple triangular motifs candidates: "
        f"{len(triple_motifs_candidates)}"
    )
    save_graph_motifs(
        9,
        triple_motifs_candidates.values(),
        args.dataset_name,
        show_tracking=args.show_tracking,
        output_dir=args.motif_dir,
    )
    manifest_path = write_selected_motif_hashes(
        args.motif_dir, dataset_name=args.dataset_name
    )
    print(f"wrote selected motif manifest {manifest_path}")


if __name__ == "__main__":
    main()
