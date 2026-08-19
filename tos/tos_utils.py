import json
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
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


def connected_three_node_subgraphs(graph):
    """Yield each connected induced three-node subgraph exactly once.

    Every connected three-node undirected graph has a node adjacent to the
    other two, so enumerating neighbor pairs avoids the cubic all-node scan.
    Direction and edge labels are retained in the induced subgraph.
    """
    undirected = graph.to_undirected(as_view=True)
    seen = set()
    for center in undirected:
        for left, right in combinations(undirected.neighbors(center), 2):
            nodes = frozenset((center, left, right))
            if nodes in seen:
                continue
            seen.add(nodes)
            yield graph.subgraph(nodes).copy()


def deduplicate_graph_motifs(graphs):
    """Deduplicate motifs with WL hash buckets and exact collision checks."""
    buckets = {}
    unique = []
    for graph in graphs:
        motif_hash = nx.weisfeiler_lehman_graph_hash(graph, edge_attr="label_0")
        bucket = buckets.setdefault(motif_hash, [])
        if is_isomorphic_multiple(bucket, graph):
            continue
        bucket.append(graph)
        unique.append(graph)
    return unique


def index_graph_motifs(graphs):
    """Index motifs by WL hash for constant-time candidate bucketing."""
    buckets = {}
    for graph in graphs:
        motif_hash = nx.weisfeiler_lehman_graph_hash(graph, edge_attr="label_0")
        buckets.setdefault(motif_hash, []).append(graph)
    return buckets


def observed_double_motifs(graph, motif_buckets):
    """Return observed composites of two distinct M3 motif types.

    This is equivalent to composing every pair of present M3 types at all
    nine possible attachment points and testing induced-subgraph presence,
    but it starts from actual three-node occurrences in ``graph``.
    """
    occurrences = []
    for candidate in connected_three_node_subgraphs(graph):
        motif_hash = nx.weisfeiler_lehman_graph_hash(
            candidate, edge_attr="label_0"
        )
        bucket = motif_buckets.get(motif_hash, ())
        if bucket and is_isomorphic_multiple(bucket, candidate):
            occurrences.append((motif_hash, frozenset(candidate), candidate))

    occurrences_by_node = {}
    for occurrence in occurrences:
        for node in occurrence[1]:
            occurrences_by_node.setdefault(node, []).append(occurrence)

    double_motifs = {}
    for shared_occurrences in occurrences_by_node.values():
        for (left_hash, left_nodes, left), (
            right_hash,
            right_nodes,
            right,
        ) in combinations(shared_occurrences, 2):
            if left_hash == right_hash or len(left_nodes & right_nodes) != 1:
                continue
            nodes = left_nodes | right_nodes
            composite = nx.compose(left, right)
            # GraphMatcher.subgraph_isomorphisms_iter() uses induced subgraphs.
            # Reject occurrence pairs with additional cross-edges accordingly.
            if graph.subgraph(nodes).number_of_edges() != composite.number_of_edges():
                continue
            motif_hash = nx.weisfeiler_lehman_graph_hash(
                composite, edge_attr="label_0"
            )
            double_motifs[motif_hash] = composite
    return double_motifs


def prepare_double_docking_motifs(graphs):
    """Return triangular M6 motifs with an immutable docking-point label."""
    prepared = []
    for graph in graphs:
        integer_graph = nx.convert_node_labels_to_integers(graph)
        if integer_graph.number_of_edges() != 6:
            continue
        for node in integer_graph:
            if (
                integer_graph.in_degree(node) == 2
                and integer_graph.out_degree(node) == 0
            ):
                prepared.append(
                    nx.relabel_nodes(
                        integer_graph, {node: "docking_point"}, copy=True
                    )
                )
                break
    return prepared


def prepare_single_docking_variants(graphs):
    """Return every valid immutable docking variant of triangular M3 motifs."""
    variants = []
    for graph in graphs:
        if graph.number_of_edges() != 3:
            continue
        integer_graph = nx.convert_node_labels_to_integers(graph)
        named_graph = nx.relabel_nodes(
            integer_graph, {0: "a", 1: "b", 2: "c"}, copy=True
        )
        for node in named_graph:
            in_degree = named_graph.in_degree(node)
            out_degree = named_graph.out_degree(node)
            if (in_degree == 1 and out_degree == 1) or (
                in_degree == 0 and out_degree == 2
            ):
                variants.append(
                    nx.relabel_nodes(
                        named_graph, {node: "docking_point"}, copy=True
                    )
                )
    return variants


def compose_triple_motifs(double_motif, single_variants):
    """Compose one M6 motif with all M3 docking variants, keyed by WL hash."""
    candidates = {}
    for single_motif in single_variants:
        triple_motif = nx.compose(double_motif, single_motif)
        motif_hash = nx.weisfeiler_lehman_graph_hash(
            triple_motif, edge_attr="label_0"
        )
        candidates[motif_hash] = triple_motif
    return candidates


_single_docking_variants = ()


def _init_triple_motif_worker(single_variants):
    global _single_docking_variants
    _single_docking_variants = single_variants


def _compose_triple_motif_worker(double_motif):
    return compose_triple_motifs(double_motif, _single_docking_variants)


def iter_composed_triple_motifs(
    double_motifs, single_variants, workers=1, chunksize=8
):
    """Yield exact M9 candidates while keeping M3 data resident per worker."""
    if workers == 1:
        for double_motif in double_motifs:
            yield compose_triple_motifs(double_motif, single_variants)
        return

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_triple_motif_worker,
        initargs=(single_variants,),
    ) as executor:
        yield from executor.map(
            _compose_triple_motif_worker, double_motifs, chunksize=chunksize
        )


def save_graph_motifs(
    n_nodes,
    graphs,
    dataset_name,
    show_tracking=False,
    output_dir="data/motifs",
):
    # Restrict exact isomorphism checks to WL-hash collision buckets instead
    # of comparing every candidate against the full retained collection.
    graph_list = list(graphs)
    non_iso_graphs = deduplicate_graph_motifs(
        track(
            graph_list,
            description="Checking isomorphism",
            disable=not show_tracking,
            total=len(graph_list),
        )
    )

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
