import json
import os
from typing import Dict

import evaluate
import networkx as nx
import numpy as np
from rich.progress import track
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
except (FileNotFoundError, OSError, RuntimeError):
    # Recent evaluate versions no longer ship the metric scripts locally.  Do
    # not make importing the preprocessing pipeline depend on network access.
    metric = None


def split_list_into_n_chunks(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def is_motif_present(G, motif):
    DiGM = nx.algorithms.isomorphism.DiGraphMatcher(
        G, motif, edge_match=lambda e1, e2: e1["label_0"] == e2["label_0"]
    )
    if next(DiGM.subgraph_isomorphisms_iter(), "EMPTY") == "EMPTY":
        return False
    return True


def is_isomorphic_multiple(graphs, candidate_graph) -> bool:
    for motif in graphs:
        DiGM = nx.algorithms.isomorphism.DiGraphMatcher(
            motif,
            candidate_graph,
            edge_match=lambda e1, e2: e1["label_0"] == e2["label_0"],
        )
        if DiGM.is_isomorphic():
            return True
    return False


def save_graph_motifs(
    n_nodes,
    graphs,
    dataset_name,
    show_tracking=False,
    output_dir="data/motifs",
):
    # sanity check to remove isomorphic graphs
    non_iso_graphs = []
    for graph in track(
        graphs,
        description="Checking isomorphism",
        disable=not show_tracking,
        total=len(graphs),
    ):
        if not is_isomorphic_multiple(non_iso_graphs, graph):
            non_iso_graphs.append(graph)

    non_iso_dict = {}
    for G in track(
        non_iso_graphs,
        description="Converting to dict",
        disable=not show_tracking,
        total=len(non_iso_graphs),
    ):
        iso_hash = nx.weisfeiler_lehman_graph_hash(G, edge_attr="label_0")
        G_dict = nx.json_graph.node_link_data(G)
        non_iso_dict[iso_hash] = G_dict

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{dataset_name}_M{n_nodes}_motifs.json"), "w") as f:
        json.dump(non_iso_dict, f, indent=2)


def write_selected_motif_hashes(
    motif_dir: str,
    dataset_name: str = "hc3-mage",
    sizes=(3, 6, 9),
    output_path: str = None,
) -> str:
    """Write a selection manifest matching the currently saved motif files.

    Regenerated motifs have different Weisfeiler-Lehman hashes from the
    checked-in corpus, so the old curated manifest cannot safely be reused.
    The generated manifest deliberately selects every motif; users can curate
    it afterward without changing the motif files.
    """
    manifest = {}
    for size in sizes:
        motif_path = os.path.join(motif_dir, f"{dataset_name}_M{size}_motifs.json")
        with open(motif_path, encoding="utf-8") as handle:
            motifs = json.load(handle)
        manifest[f"m{size}"] = sorted(motifs)

    output_path = output_path or os.path.join(
        motif_dir, f"{dataset_name}_selected-motif-hashes.generated.json"
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    temporary_path = f"{output_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    os.replace(temporary_path, output_path)
    return output_path


def resolve_selected_motif_hashes(
    motif_dir: str, dataset_name: str = "hc3-mage", manifest_path: str = None
) -> str:
    """Prefer a generated manifest, falling back to the checked-in one."""
    if manifest_path:
        return manifest_path
    generated_path = os.path.join(
        motif_dir, f"{dataset_name}_selected-motif-hashes.generated.json"
    )
    if os.path.exists(generated_path):
        return generated_path
    return os.path.join(motif_dir, f"{dataset_name}_selected-motif-hashes.json")


def validate_selected_motif_hashes(
    motif_dir: str,
    selected_hashes: Dict[str, list],
    dataset_name: str = "hc3-mage",
    sizes=(3, 6, 9),
) -> None:
    """Fail early with an actionable error when a manifest is stale."""
    missing_by_size = {}
    for size in sizes:
        motif_path = os.path.join(motif_dir, f"{dataset_name}_M{size}_motifs.json")
        with open(motif_path, encoding="utf-8") as handle:
            available = set(json.load(handle))
        selected = selected_hashes.get(f"m{size}")
        if selected is None:
            missing_by_size[f"m{size}"] = ["<missing manifest key>"]
            continue
        missing = sorted(set(selected) - available)
        if missing:
            missing_by_size[f"m{size}"] = missing
    if missing_by_size:
        details = ", ".join(
            f"{size}: {len(hashes)} missing" for size, hashes in missing_by_size.items()
        )
        raise ValueError(
            "Selected motif hashes do not match the motif files ("
            f"{details}). Regenerate the selection manifest with "
            "4_extract_triple_triads.py or pass --selected-hashes with a "
            "matching manifest."
        )


def load_graph_motifs(path: str) -> Dict[str, nx.Graph]:
    with open(path, "r") as f:
        motifs = json.load(f)
    return {k: nx.json_graph.node_link_graph(v) for k, v in motifs.items()}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    if metric is not None:
        return metric.compute(predictions=predictions, references=labels)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
    }


def load_json(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    return data
