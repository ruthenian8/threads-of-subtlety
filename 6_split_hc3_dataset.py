"""Split parsed HC3 corpora into train/validation/test sets.

The UniRST workflow stores one set of files below ``data/unirst/rel-*`` for
each relation inventory.  Splitting is therefore performed independently for
each inventory; mixing inventories would produce incompatible motif vectors.
The legacy ``data/hc3`` layout remains supported when no UniRST files are
present.
"""

import argparse
import os
import random
import re
from collections import defaultdict
from glob import glob
from typing import List, Optional, Tuple

from tos.tos_dataset import ToSDataset


def _safe_inventory_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _inventory_dirs(data_dir: str, relinventory: Optional[str]) -> List[Tuple[Optional[str], str]]:
    root = os.path.join(data_dir, "unirst")
    if relinventory:
        path = os.path.join(root, f"rel-{_safe_inventory_name(relinventory)}")
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Relation inventory directory not found: {path}")
        return [(relinventory, path)]
    paths = sorted(glob(os.path.join(root, "rel-*")))
    return [(os.path.basename(path)[4:], path) for path in paths if os.path.isdir(path)]


def _source_files(directory: str, dataset_name: str) -> List[str]:
    # Do not accidentally consume split outputs on a second invocation.
    pattern = os.path.join(
        directory,
        f"{dataset_name}_[A-Za-z0-9_.-]*.discourse_parsed.graph_added.m3_m6_motif_dists.jsonl",
    )
    return sorted(
        path
        for path in glob(pattern)
        if not re.search(r"_(train|validation|valid|test)\.discourse_parsed", os.path.basename(path))
    )


def split_inventory(
    files: List[str],
    output_dir: str,
    dataset_name: str,
    valid_size: int,
    test_size: int,
    seed: int,
    output_suffix: str = "m3_m6_motif_dists",
) -> Tuple[int, int, int]:
    rng = random.Random(seed)
    train, valid, test = [], [], []
    domains = defaultdict(list)
    for file_path in files:
        print(file_path)
        basename = os.path.basename(file_path)
        match = re.match(
            rf"{re.escape(dataset_name)}_(.+)_(?:human|machine)\.discourse_parsed",
            basename,
        )
        # Human and machine files are two shards of one HC3 domain.  They
        # must share one quota so validation/test sizes remain per-domain.
        domain = match.group(1) if match else os.path.splitext(basename)[0]
        domains[domain].extend(ToSDataset.load_document_corpus(file_path))

    for domain, dataset in sorted(domains.items()):
        rng.shuffle(dataset)
        valid.extend(dataset[:valid_size])
        test.extend(dataset[valid_size : valid_size + test_size])
        train.extend(dataset[valid_size + test_size :])

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)
    os.makedirs(output_dir, exist_ok=True)
    stem = f"{dataset_name}_{{}}.discourse_parsed.graph_added.{output_suffix}.jsonl"
    ToSDataset.save_dataset_as_jsonl(train, os.path.join(output_dir, stem.format("train")))
    ToSDataset.save_dataset_as_jsonl(valid, os.path.join(output_dir, stem.format("validation")))
    ToSDataset.save_dataset_as_jsonl(test, os.path.join(output_dir, stem.format("test")))
    return len(train), len(valid), len(test)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset-name", default="hc3")
    parser.add_argument("--relinventory", help="split only this UniRST relation inventory")
    parser.add_argument("--valid-size-per-domain", type=int, default=200)
    parser.add_argument("--test-size-per-domain", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    inventories = _inventory_dirs(args.data_dir, args.relinventory)
    if inventories:
        for inventory, directory in inventories:
            files = _source_files(directory, args.dataset_name)
            if not files:
                print(f"No motif-distribution sources found in {directory}; skipping")
                continue
            counts = split_inventory(
                files,
                directory,
                args.dataset_name,
                args.valid_size_per_domain,
                args.test_size_per_domain,
                args.seed,
                output_suffix="m3_m6_motif_dists",
            )
            print(f"{inventory}: train={counts[0]}, valid={counts[1]}, test={counts[2]}")
        return

    # Legacy data/hc3 fallback.
    legacy_dir = os.path.join(args.data_dir, "hc3")
    files = [
        path
        for path in glob(
            os.path.join(legacy_dir, f"{args.dataset_name}_*.graph_added.motif_dists.jsonl")
        )
        if not re.search(
            r"_(train|validation|valid|test)\.discourse_parsed",
            os.path.basename(path),
        )
    ]
    if not files:
        raise FileNotFoundError("No UniRST or legacy HC3 motif-distribution files found")
    counts = split_inventory(
        files,
        legacy_dir,
        args.dataset_name,
        args.valid_size_per_domain,
        args.test_size_per_domain,
        args.seed,
        output_suffix="motif_dists",
    )
    print(f"train={counts[0]}, valid={counts[1]}, test={counts[2]}")


if __name__ == "__main__":
    main()
