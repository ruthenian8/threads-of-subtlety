import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from glob import glob
from itertools import combinations

import networkx as nx
from tqdm.auto import tqdm

from tos.tos_dataset import ToSDataset
from tos.tos_utils import is_motif_present, load_graph_motifs, save_graph_motifs

motifs_three = []


def init_motif_worker(motifs):
    global motifs_three
    motifs_three = motifs


def extract_double_motifs(sample):
    G = sample

    present_motifs = []
    for motif in motifs_three:
        if is_motif_present(G, motif):
            present_motifs.append(motif)

    present_double_motifs = []
    relabels = [
        {"a": 0},
        {"b": 0},
        {"c": 0},
        {"a": 1},
        {"b": 1},
        {"c": 1},
        {"a": 2},
        {"b": 2},
        {"c": 2},
    ]
    for motif_a, motif_b in combinations(present_motifs, 2):
        nx.relabel_nodes(motif_b, {0: "a", 1: "b", 2: "c"}, copy=False)
        for label in relabels:
            re_motif_b = nx.relabel_nodes(motif_b, label, copy=True)
            double_motif = nx.compose(motif_a, re_motif_b)
            if is_motif_present(G, double_motif):
                present_double_motifs.append(double_motif)

    unique_double_motifs = {}
    for motif in present_double_motifs:
        sg_hash = nx.weisfeiler_lehman_graph_hash(motif, edge_attr="label_0")
        unique_double_motifs[sg_hash] = motif
    return unique_double_motifs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/unirst")
    parser.add_argument("--motif-dir", default="data/motifs")
    parser.add_argument("--dataset-name", default="hc3-mage")
    parser.add_argument("--max-per-file", type=int, default=15000)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    single_triads_path = os.path.join(
        args.motif_dir, f"{args.dataset_name}_M3_motifs.json"
    )
    motifs_three = load_graph_motifs(single_triads_path).values()
    motifs_three = [nx.convert_node_labels_to_integers(m) for m in motifs_three]
    print(f"no. of motifs_three: {len(motifs_three)}")

    file_paths = sorted(
        glob(os.path.join(args.root, "rel-*", "*.discourse_parsed.graph_added.jsonl"))
    )
    if not file_paths:
        raise FileNotFoundError(
            f"No graph-added UniRST files found below {args.root!r}"
        )
    dataset = ToSDataset.load_datasets(file_paths, max_per_file=args.max_per_file)

    all_graphs = []
    for document in dataset:
        for tree in document.scene_discourse_trees.values():
            all_graphs.append(tree.graph_networkx)
    print(f"no. of all_graphs: {len(all_graphs)}")

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_motif_worker,
        initargs=(motifs_three,),
    ) as executor:
        results = list(
            tqdm(
                executor.map(extract_double_motifs, all_graphs, chunksize=1),
                total=len(all_graphs),
                desc="extracting double motifs",
            )
        )

    double_motifs = {}
    for res in results:
        for sg_hash, motif in res.items():
            double_motifs[sg_hash] = motif
    print(len(double_motifs))

    save_graph_motifs(
        6,
        double_motifs.values(),
        args.dataset_name,
        output_dir=args.motif_dir,
    )
