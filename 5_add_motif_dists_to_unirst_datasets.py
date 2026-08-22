"""Add per-inventory M3 and M6 distributions to UniRST graph outputs."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import glob
from itertools import islice
import os
import random
import re
from typing import Dict, Iterable, Iterator, List, Optional

from tqdm.auto import tqdm

from tos.tos_dataset import (
    DiscourseMotifDists,
    Document,
    MotifCatalogIndex,
    ToSDataset,
)
from tos.tos_utils import (
    load_json,
    resolve_selected_motif_hashes,
    validate_selected_motif_hashes,
)


random.seed(42)


MOTIF_SIZES = (3, 6)
MOTIF_GROUP_NAMES = tuple(f"m{size}" for size in MOTIF_SIZES)
MOTIF_SET_NAME = "_".join(MOTIF_GROUP_NAMES)


_WORKER_MOTIFS: Optional[Dict[str, MotifCatalogIndex]] = None


def _init_motif_worker(motifs: Dict[str, MotifCatalogIndex]) -> None:
    """Initialize motif collections once in each process-pool worker."""
    global _WORKER_MOTIFS
    _WORKER_MOTIFS = motifs


def _add_motif_dist_to_document_batch_worker(
    documents: List[Document],
) -> List[Document]:
    if _WORKER_MOTIFS is None:
        raise RuntimeError("motif worker was not initialized")
    processed = []
    for document in documents:
        document = add_motif_dist_to_document(document, _WORKER_MOTIFS)
        # graph_dict is the persisted representation. Avoid sending duplicate
        # NetworkX objects back to the parent process.
        for tree in document.scene_discourse_trees.values():
            if tree is not None:
                tree.graph_networkx = None
        processed.append(document)
    return processed


def add_motif_dist_to_document(
    document: Document,
    motif_groups,
) -> Document:
    for tree in document.scene_discourse_trees.values():
        if tree is None:
            continue
        root_label = f"span_1-{len(tree.edus)}"
        if all(
            isinstance(group, MotifCatalogIndex)
            for group in motif_groups.values()
        ):
            distributions = ToSDataset.calculate_observed_m3_m6_distributions(
                tree.graph_networkx,
                motif_groups,
                root_label,
            )
        else:
            # Retain the generic path for callers supplying arbitrary groups.
            distributions = ToSDataset.calculate_motif_distributions(
                tree.graph_networkx,
                motif_groups,
                root_label,
            )
        tree.motif_dists = {
            name: DiscourseMotifDists(**distribution)
            for name, distribution in distributions.items()
        }
    return document


def _batched(documents: Iterable[Document], batch_size: int):
    iterator = iter(documents)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            return
        yield batch


def iter_processed_documents(
    executor: ProcessPoolExecutor,
    documents: Iterable[Document],
    batch_size: int,
    max_pending_batches: int,
) -> Iterator[Document]:
    """Process bounded batches while preserving the input document order."""
    batches = iter(_batched(documents, batch_size))
    pending = deque()
    for batch in islice(batches, max_pending_batches):
        pending.append(
            executor.submit(_add_motif_dist_to_document_batch_worker, batch)
        )

    while pending:
        for document in pending.popleft().result():
            yield document
        try:
            batch = next(batches)
        except StopIteration:
            continue
        pending.append(
            executor.submit(_add_motif_dist_to_document_batch_worker, batch)
        )


def iter_processed_documents_as_completed(
    executor: ProcessPoolExecutor,
    documents: Iterable[Document],
    batch_size: int,
    max_pending_batches: int,
) -> Iterator[Document]:
    """Keep workers busy by yielding batches as soon as they complete.

    Document order is not semantically relevant to the generated training
    corpus.  Consuming completion order avoids a single complex document
    blocking the parent while every other worker is idle.
    """
    batches = iter(_batched(documents, batch_size))
    pending = set()
    for batch in islice(batches, max_pending_batches):
        pending.add(
            executor.submit(_add_motif_dist_to_document_batch_worker, batch)
        )

    while pending:
        completed, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
            processed_batch = future.result()
            try:
                batch = next(batches)
            except StopIteration:
                pass
            else:
                pending.add(
                    executor.submit(
                        _add_motif_dist_to_document_batch_worker, batch
                    )
                )
            yield from processed_batch


def count_documents(file_path: str) -> int:
    with open(file_path) as handle:
        return sum(1 for line in handle if line.strip())


def output_path_for(file_path: str) -> str:
    # Include the feature-set name so an existing M3+M6+M9 output cannot be
    # mistaken for a completed M3+M6 artifact and skipped.
    return f"{os.path.splitext(file_path)[0]}.{MOTIF_SET_NAME}_motif_dists.jsonl"


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
) -> List[str]:
    prefixes = ("hc3", "mage") if dataset_name == "hc3-mage" else (dataset_name,)
    return sorted(
        {
            path
            for prefix in prefixes
            for path in glob.glob(
                os.path.join(
                    dataset_root,
                    inventory_dir,
                    f"{prefix}_*.discourse_parsed.graph_added.jsonl",
                )
            )
        }
    )


def pending_output_files(file_paths: Iterable[str], force: bool):
    pending = []
    for file_path in sorted(file_paths):
        output_path = output_path_for(file_path)
        if os.path.exists(output_path) and not force:
            print(f"skipping existing {output_path}")
            continue
        pending.append((file_path, output_path))
    return pending


def write_motif_distributions(
    executor: ProcessPoolExecutor,
    file_path: str,
    output_path: str,
    chunksize: int,
    workers: int,
) -> None:
    temporary_path = f"{output_path}.partial"
    documents = ToSDataset.iter_document_corpus(file_path)
    processed = iter_processed_documents_as_completed(
        executor,
        documents,
        batch_size=chunksize,
        max_pending_batches=max(1, workers * 2),
    )
    with open(temporary_path, "w") as handle:
        for document_index, document in enumerate(
            tqdm(
                processed,
                total=count_documents(file_path),
                desc=os.path.basename(file_path),
            ),
            start=1,
        ):
            handle.write(f"{ToSDataset.document_to_dict(document)}\n")
            if document_index % chunksize == 0:
                handle.flush()
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, output_path)
    print(f"wrote {output_path}")


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
    parser.add_argument(
        "--relinventory",
        default=None,
        help="Process one RST inventory with its standard-specific motif set.",
    )
    parser.add_argument("--dataset-name", default="hc3")
    parser.add_argument(
        "--selected-hashes",
        default=None,
        help="Selection manifest; generated manifest is preferred by default.",
    )
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument(
        "--chunksize",
        type=int,
        default=8,
        help="Documents per worker task.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute outputs that already exist.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.chunksize < 1:
        parser.error("--chunksize must be at least 1")

    dataset_root = resolve_dataset_root(args)
    if args.relinventory is not None:
        inventories = [args.relinventory]
    else:
        inventory_dirs = sorted(glob.glob(os.path.join(dataset_root, "rel-*")))
        inventories = [
            os.path.basename(path)[len("rel-") :]
            for path in inventory_dirs
            if os.path.isdir(path)
        ]
    if not inventories:
        raise FileNotFoundError(
            f"No rel-* inventory directories found below {dataset_root!r}"
        )

    for relinventory in inventories:
        inventory_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", relinventory)
        motif_dir = resolve_motif_dir(args, relinventory)
        inventory_dir = f"rel-{inventory_name}"
        files = discover_graph_files(
            dataset_root,
            inventory_dir,
            args.dataset_name,
        )
        pending_files = pending_output_files(files, args.force)
        if not pending_files:
            continue

        selected_manifest = resolve_selected_motif_hashes(
            motif_dir, args.dataset_name, args.selected_hashes
        )
        selected = load_json(selected_manifest)
        validate_selected_motif_hashes(
            motif_dir,
            selected,
            dataset_name=args.dataset_name,
            sizes=MOTIF_SIZES,
        )
        motifs = {
            group_name: ToSDataset.prepare_motif_catalog_index(
                ToSDataset.load_motifs(
                    os.path.join(
                        motif_dir, f"{args.dataset_name}_M{size}_motifs.json"
                    ),
                    selected[f"m{size}"],
                )
            )
            for size, group_name in zip(MOTIF_SIZES, MOTIF_GROUP_NAMES)
        }

        # Keep one pool alive for every shard in this relation inventory. Motif
        # metadata is initialized once per worker and reused throughout.
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_motif_worker,
            initargs=(motifs,),
        ) as executor:
            for file_path, output_path in pending_files:
                write_motif_distributions(
                    executor,
                    file_path,
                    output_path,
                    args.chunksize,
                    args.workers,
                )


if __name__ == "__main__":
    main()
