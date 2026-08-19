"""Add the paper's motif distributions to UniRST graph outputs."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
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
    MotifMetadata,
    ToSDataset,
)
from tos.tos_utils import (
    load_json,
    resolve_selected_motif_hashes,
    validate_selected_motif_hashes,
)


random.seed(42)


_WORKER_MOTIFS: Optional[Dict[int, List[MotifMetadata]]] = None


def _init_motif_worker(motifs: Dict[int, List[MotifMetadata]]) -> None:
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
        document = add_motif_dist_to_document(
            document,
            _WORKER_MOTIFS[3],
            _WORKER_MOTIFS[6],
            _WORKER_MOTIFS[9],
        )
        # graph_dict is the persisted representation. Avoid sending duplicate
        # NetworkX objects back to the parent process.
        for tree in document.scene_discourse_trees.values():
            if tree is not None:
                tree.graph_networkx = None
        processed.append(document)
    return processed


def add_motif_dist_to_document(
    document: Document,
    m3_motifs: List,
    m6_motifs: List,
    m9_motifs: List,
) -> Document:
    for tree in document.scene_discourse_trees.values():
        if tree is None:
            continue
        root_label = f"span_1-{len(tree.edus)}"
        distributions = ToSDataset.calculate_motif_distributions(
            tree.graph_networkx,
            {"m3": m3_motifs, "m6": m6_motifs, "m9": m9_motifs},
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


def count_documents(file_path: str) -> int:
    with open(file_path) as handle:
        return sum(1 for line in handle if line.strip())


def output_path_for(file_path: str) -> str:
    return f"{os.path.splitext(file_path)[0]}.motif_dists.jsonl"


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
    processed = iter_processed_documents(
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
    parser.add_argument("--root", default="data/unirst")
    parser.add_argument("--motif-dir", default=None)
    parser.add_argument(
        "--relinventory",
        default=None,
        help="Process one RST inventory with its standard-specific motif set.",
    )
    parser.add_argument("--dataset-name", default="hc3-mage")
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

    inventories = [args.relinventory] if args.relinventory else [None]
    if args.relinventory is None:
        inventory_dirs = sorted(glob.glob(os.path.join(args.root, "rel-*")))
        inventories = [
            os.path.basename(path)[len("rel-") :]
            for path in inventory_dirs
            if os.path.isdir(path)
        ] or [None]

    for relinventory in inventories:
        inventory_name = (
            re.sub(r"[^A-Za-z0-9_.-]+", "_", relinventory)
            if relinventory
            else None
        )
        motif_dir = (
            os.path.join(args.motif_dir, inventory_name)
            if args.motif_dir and args.relinventory is None and inventory_name
            else args.motif_dir
            or (
                os.path.join("data/motifs", inventory_name)
                if inventory_name
                else "data/motifs"
            )
        )
        inventory_dir = f"rel-{inventory_name}" if inventory_name else "rel-*"
        files = glob.glob(
            os.path.join(
                args.root, inventory_dir, "*.discourse_parsed.graph_added.jsonl"
            )
        )
        pending_files = pending_output_files(files, args.force)
        if not pending_files:
            continue

        selected_manifest = resolve_selected_motif_hashes(
            motif_dir, args.dataset_name, args.selected_hashes
        )
        selected = load_json(selected_manifest)
        validate_selected_motif_hashes(
            motif_dir, selected, dataset_name=args.dataset_name
        )
        motifs = {
            size: ToSDataset.prepare_motif_metadata(
                ToSDataset.load_motifs(
                    os.path.join(
                        motif_dir, f"{args.dataset_name}_M{size}_motifs.json"
                    ),
                    selected[f"m{size}"],
                )
            )
            for size in (3, 6, 9)
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
