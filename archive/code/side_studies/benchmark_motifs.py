"""Compare reference and optimized motif operations on deterministic graphs."""

from __future__ import annotations

import argparse
from itertools import combinations
import os
import statistics
import time

import networkx as nx
import numpy as np

from tos.tos_dataset import ToSDataset
from tos.tos_utils import (
    compose_triple_motifs,
    connected_three_node_subgraphs,
    deduplicate_graph_motifs,
    index_graph_motifs,
    is_motif_present,
    is_isomorphic_multiple,
    iter_composed_triple_motifs,
    load_graph_motifs,
    observed_double_motifs,
    prepare_double_docking_motifs,
    prepare_single_docking_variants,
)


EDGE_LABELS = ("/", "elaboration", "contrast", "joint")


def make_graph(node_count: int, seed: int = 42) -> nx.DiGraph:
    rng = np.random.default_rng(seed)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(node_count))
    for node in range(1, node_count):
        parent = int(rng.integers(0, node))
        graph.add_edge(node, parent, label_0=EDGE_LABELS[node % len(EDGE_LABELS)])
    for _ in range(node_count // 2):
        left, right = rng.choice(node_count, size=2, replace=False)
        graph.add_edge(int(left), int(right), label_0=EDGE_LABELS[int(left) % 4])
    return graph


def reference_single_motifs(graph: nx.DiGraph):
    motifs = []
    for nodes in combinations(graph, 3):
        candidate = graph.subgraph(nodes).copy()
        if list(nx.isolates(candidate)):
            continue
        if not is_isomorphic_multiple(motifs, candidate):
            motifs.append(candidate)
    return motifs


def optimized_single_motifs(graph: nx.DiGraph):
    return deduplicate_graph_motifs(connected_three_node_subgraphs(graph))


def reference_double_motifs(graph, motifs):
    present = [motif for motif in motifs if is_motif_present(graph, motif)]
    candidates = {}
    for left, right in combinations(present, 2):
        left = nx.convert_node_labels_to_integers(left)
        right = nx.convert_node_labels_to_integers(right)
        named_right = nx.relabel_nodes(
            right, {0: "a", 1: "b", 2: "c"}, copy=True
        )
        for source in ("a", "b", "c"):
            for target in (0, 1, 2):
                candidate = nx.compose(
                    left,
                    nx.relabel_nodes(named_right, {source: target}, copy=True),
                )
                if is_motif_present(graph, candidate):
                    motif_hash = nx.weisfeiler_lehman_graph_hash(
                        candidate, edge_attr="label_0"
                    )
                    candidates[motif_hash] = candidate
    return candidates


def optimized_double_motifs(graph, motifs):
    return observed_double_motifs(graph, index_graph_motifs(motifs))


def sequential_triple_motifs(double_motifs, single_variants):
    candidates = {}
    for double_motif in double_motifs:
        candidates.update(compose_triple_motifs(double_motif, single_variants))
    return candidates


def parallel_triple_motifs(double_motifs, single_variants, workers, chunksize):
    candidates = {}
    for result in iter_composed_triple_motifs(
        double_motifs, single_variants, workers=workers, chunksize=chunksize
    ):
        candidates.update(result)
    return candidates


def reference_distribution(graph, motifs, root_label):
    diameter = nx.diameter(graph.to_undirected())
    hist = np.zeros(len(motifs), dtype=float)
    wad = np.zeros(len(motifs), dtype=float)
    for index, motif in enumerate(motifs):
        matcher = nx.algorithms.isomorphism.DiGraphMatcher(
            graph,
            motif,
            edge_match=lambda left, right: left["label_0"] == right["label_0"],
        )
        depths = []
        for mapping in matcher.subgraph_isomorphisms_iter():
            depths.append(
                np.mean(
                    [
                        nx.shortest_path_length(
                            graph.to_undirected(), root_label, node
                        )
                        for node in mapping
                    ]
                )
            )
        hist[index] = len(depths)
        wad[index] = np.mean(depths) if depths else -1
    total = np.sum(hist)
    frequencies = hist / total if total > 0 else hist
    wad = wad / diameter
    wad[wad < 0] = -1
    return {"raw": hist.tolist(), "mf": frequencies.tolist(), "wad": wad.tolist()}


def assert_same_motifs(reference, optimized) -> None:
    assert len(reference) == len(optimized)
    assert all(is_isomorphic_multiple(optimized, motif) for motif in reference)


def assert_same_distribution(reference, optimized) -> None:
    for key in ("raw", "mf", "wad"):
        np.testing.assert_allclose(reference[key], optimized[key], rtol=0, atol=1e-12)


def median_runtime(function, repeats: int):
    durations = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        durations.append(time.perf_counter() - started)
    return statistics.median(durations), result


def report(label: str, reference_time: float, optimized_time: float) -> None:
    print(
        f"{label:24s} reference={reference_time:8.4f}s "
        f"optimized={optimized_time:8.4f}s "
        f"speedup={reference_time / optimized_time:7.2f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=40)
    parser.add_argument("--motifs", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--m6-nodes", type=int, default=14)
    parser.add_argument("--m6-repeats", type=int, default=1)
    parser.add_argument("--motif-dir", default="data/motifs")
    parser.add_argument("--dataset-name", default="hc3-mage")
    parser.add_argument("--m9-double-motifs", type=int, default=200)
    parser.add_argument("--m9-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--m9-chunksize", type=int, default=8)
    parser.add_argument("--m9-repeats", type=int, default=1)
    parser.add_argument("--skip-m9", action="store_true")
    args = parser.parse_args()

    graph = make_graph(args.nodes)
    reference_time, reference_motifs = median_runtime(
        lambda: reference_single_motifs(graph), args.repeats
    )
    optimized_time, optimized_motifs = median_runtime(
        lambda: optimized_single_motifs(graph), args.repeats
    )
    assert_same_motifs(reference_motifs, optimized_motifs)
    report("single-motif extraction", reference_time, optimized_time)

    motifs = reference_motifs[: args.motifs]
    reference_time, reference_result = median_runtime(
        lambda: reference_distribution(graph, motifs, 0), args.repeats
    )
    optimized_time, optimized_result = median_runtime(
        lambda: ToSDataset.calculate_motif_distribution(graph, motifs, 0),
        args.repeats,
    )
    assert_same_distribution(reference_result, optimized_result)
    report("motif distributions", reference_time, optimized_time)

    m6_graph = make_graph(args.m6_nodes)
    motif_catalog = {}
    for motif in optimized_single_motifs(m6_graph):
        motif_hash = nx.weisfeiler_lehman_graph_hash(
            motif, edge_attr="label_0"
        )
        motif_catalog[motif_hash] = motif
    m3_motifs = list(motif_catalog.values())
    reference_time, reference_result = median_runtime(
        lambda: reference_double_motifs(m6_graph, m3_motifs),
        args.m6_repeats,
    )
    optimized_time, optimized_result = median_runtime(
        lambda: optimized_double_motifs(m6_graph, m3_motifs),
        args.m6_repeats,
    )
    assert set(reference_result) == set(optimized_result)
    report("double-motif extraction", reference_time, optimized_time)

    if args.skip_m9:
        return
    motif_catalogs = {
        size: list(
            load_graph_motifs(
                os.path.join(
                    args.motif_dir,
                    f"{args.dataset_name}_M{size}_motifs.json",
                )
            ).values()
        )
        for size in (3, 6, 9)
    }
    reference_time, reference_result = median_runtime(
        lambda: {
            size: reference_distribution(graph, motifs, 0)
            for size, motifs in motif_catalogs.items()
        },
        1,
    )
    motif_metadata = {
        size: ToSDataset.prepare_motif_metadata(motifs)
        for size, motifs in motif_catalogs.items()
    }
    optimized_time, optimized_result = median_runtime(
        lambda: ToSDataset.calculate_motif_distributions(
            graph,
            {f"m{size}": metadata for size, metadata in motif_metadata.items()},
            0,
        ),
        1,
    )
    for size, distribution in reference_result.items():
        assert_same_distribution(distribution, optimized_result[f"m{size}"])
    report("all-motif distributions", reference_time, optimized_time)

    single_motifs = motif_catalogs[3]
    double_motifs = motif_catalogs[6]
    single_variants = prepare_single_docking_variants(single_motifs)
    prepared_double = prepare_double_docking_motifs(double_motifs)[
        : args.m9_double_motifs
    ]
    reference_time, reference_result = median_runtime(
        lambda: sequential_triple_motifs(prepared_double, single_variants),
        args.m9_repeats,
    )
    optimized_time, optimized_result = median_runtime(
        lambda: parallel_triple_motifs(
            prepared_double,
            single_variants,
            args.m9_workers,
            args.m9_chunksize,
        ),
        args.m9_repeats,
    )
    assert set(reference_result) == set(optimized_result)
    report("triple-motif composition", reference_time, optimized_time)


if __name__ == "__main__":
    main()
