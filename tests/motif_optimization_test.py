import unittest
from itertools import combinations

import networkx as nx
import numpy as np

from tos.tos_dataset import ToSDataset
from tos.tos_utils import (
    compose_triple_motifs,
    connected_three_node_subgraphs,
    deduplicate_graph_motifs,
    index_graph_motifs,
    is_isomorphic_multiple,
    is_motif_present,
    iter_composed_triple_motifs,
    node_link_graph_compat,
    observed_double_motifs,
    prepare_double_docking_motifs,
    prepare_single_docking_variants,
)


def reference_motifs(graph):
    motifs = []
    for nodes in combinations(graph, 3):
        candidate = graph.subgraph(nodes).copy()
        if list(nx.isolates(candidate)):
            continue
        if not is_isomorphic_multiple(motifs, candidate):
            motifs.append(candidate)
    return motifs


def reference_distribution(graph, motifs, root):
    diameter = nx.diameter(graph.to_undirected())
    hist = np.zeros(len(motifs), dtype=float)
    wad = np.zeros(len(motifs), dtype=float)
    for index, motif in enumerate(motifs):
        matcher = nx.algorithms.isomorphism.DiGraphMatcher(
            graph,
            motif,
            edge_match=lambda left, right: left["label_0"] == right["label_0"],
        )
        depths = [
            np.mean(
                [
                    nx.shortest_path_length(graph.to_undirected(), root, node)
                    for node in mapping
                ]
            )
            for mapping in matcher.subgraph_isomorphisms_iter()
        ]
        hist[index] = len(depths)
        wad[index] = np.mean(depths) if depths else -1
    total = np.sum(hist)
    frequencies = hist / total if total else hist
    wad = wad / diameter
    wad[wad < 0] = -1
    return {"raw": hist.tolist(), "mf": frequencies.tolist(), "wad": wad.tolist()}


def reference_double_motifs(graph, motifs):
    """Corrected version of the original Cartesian M6 implementation."""
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
                attached_right = nx.relabel_nodes(
                    named_right, {source: target}, copy=True
                )
                candidate = nx.compose(left, attached_right)
                if is_motif_present(graph, candidate):
                    motif_hash = nx.weisfeiler_lehman_graph_hash(
                        candidate, edge_attr="label_0"
                    )
                    candidates[motif_hash] = candidate
    return candidates


def reference_triple_motifs(single_motifs, double_motifs):
    """Immutable sequential equivalent of the original M9 implementation."""
    prepared_double = []
    for graph in double_motifs:
        if graph.number_of_edges() != 6:
            continue
        for node in graph:
            if graph.in_degree(node) == 2 and graph.out_degree(node) == 0:
                prepared_double.append(
                    nx.relabel_nodes(graph, {node: "docking_point"}, copy=True)
                )
                break

    prepared_single = []
    for graph in single_motifs:
        if graph.number_of_edges() != 3:
            continue
        graph = nx.convert_node_labels_to_integers(graph)
        named_graph = nx.relabel_nodes(
            graph, {0: "a", 1: "b", 2: "c"}, copy=True
        )
        for node in named_graph:
            in_degree = named_graph.in_degree(node)
            out_degree = named_graph.out_degree(node)
            if (in_degree == 1 and out_degree == 1) or (
                in_degree == 0 and out_degree == 2
            ):
                prepared_single.append(
                    nx.relabel_nodes(
                        named_graph, {node: "docking_point"}, copy=True
                    )
                )

    candidates = {}
    for double_motif in prepared_double:
        candidates.update(compose_triple_motifs(double_motif, prepared_single))
    return candidates


class MotifOptimizationTest(unittest.TestCase):
    def test_node_link_loader_accepts_edges_and_links_keys(self):
        graph = nx.DiGraph()
        graph.add_edge(0, 1, label_0="joint")
        links_data = nx.json_graph.node_link_data(graph)
        edge_key = "links" if "links" in links_data else "edges"
        other_key = "edges" if edge_key == "links" else "links"
        alternate_data = dict(links_data)
        alternate_data[other_key] = alternate_data.pop(edge_key)

        for data in (links_data, alternate_data):
            loaded = node_link_graph_compat(data)
            self.assertTrue(
                nx.is_isomorphic(
                    graph,
                    loaded,
                    edge_match=lambda left, right: left["label_0"]
                    == right["label_0"],
                )
            )

    def setUp(self):
        self.graph = nx.DiGraph()
        edges = [
            (1, 0, "/"),
            (2, 0, "joint"),
            (3, 1, "contrast"),
            (4, 1, "/"),
            (5, 2, "elaboration"),
            (4, 2, "joint"),
        ]
        self.graph.add_edges_from(
            (left, right, {"label_0": label}) for left, right, label in edges
        )

    def test_connected_enumeration_matches_cubic_reference(self):
        reference = reference_motifs(self.graph)
        optimized = deduplicate_graph_motifs(
            connected_three_node_subgraphs(self.graph)
        )
        self.assertEqual(len(reference), len(optimized))
        self.assertTrue(
            all(is_isomorphic_multiple(optimized, motif) for motif in reference)
        )

    def test_connected_enumeration_matches_random_graphs(self):
        for seed in range(10):
            graph = nx.gnp_random_graph(10, 0.2, seed=seed, directed=True)
            for left, right in graph.edges:
                graph[left][right]["label_0"] = ("/", "joint", "contrast")[
                    (left + right) % 3
                ]
            reference = reference_motifs(graph)
            optimized = deduplicate_graph_motifs(
                connected_three_node_subgraphs(graph)
            )
            self.assertEqual(len(reference), len(optimized), msg=f"seed={seed}")
            self.assertTrue(
                all(is_isomorphic_multiple(optimized, motif) for motif in reference),
                msg=f"seed={seed}",
            )

    def test_distribution_matches_reference(self):
        motifs = reference_motifs(self.graph)
        reference = reference_distribution(self.graph, motifs, 0)
        optimized = ToSDataset.calculate_motif_distribution(
            self.graph, motifs, 0
        )
        for key in ("raw", "mf", "wad"):
            np.testing.assert_allclose(reference[key], optimized[key], atol=1e-12)

    def test_combined_pruned_distributions_match_reference(self):
        motifs = reference_motifs(self.graph)
        impossible = nx.DiGraph()
        impossible.add_edges_from(
            [
                ("a", "b", {"label_0": "missing-relation"}),
                ("b", "c", {"label_0": "/"}),
            ]
        )
        groups = {
            "present": ToSDataset.prepare_motif_metadata(motifs),
            "pruned": ToSDataset.prepare_motif_metadata([impossible]),
        }

        optimized = ToSDataset.calculate_motif_distributions(
            self.graph, groups, 0
        )
        references = {
            "present": reference_distribution(self.graph, motifs, 0),
            "pruned": reference_distribution(self.graph, [impossible], 0),
        }
        for group_name in groups:
            for key in ("raw", "mf", "wad"):
                np.testing.assert_allclose(
                    references[group_name][key],
                    optimized[group_name][key],
                    atol=1e-12,
                )

    def test_occurrence_driven_m3_m6_matches_graphmatcher(self):
        m3_catalog = {}
        for motif in deduplicate_graph_motifs(
            connected_three_node_subgraphs(self.graph)
        ):
            motif_hash = nx.weisfeiler_lehman_graph_hash(
                motif, edge_attr="label_0"
            )
            m3_catalog[motif_hash] = motif
        m6_catalog = observed_double_motifs(
            self.graph, index_graph_motifs(m3_catalog.values())
        )
        motif_groups = {
            "m3": list(m3_catalog.values()),
            "m6": list(m6_catalog.values()),
        }

        reference = ToSDataset.calculate_motif_distributions(
            self.graph, motif_groups, 0
        )
        indexed = {
            name: ToSDataset.prepare_motif_catalog_index(motifs)
            for name, motifs in motif_groups.items()
        }
        optimized = ToSDataset.calculate_observed_m3_m6_distributions(
            self.graph, indexed, 0
        )

        for group_name in motif_groups:
            for key in ("raw", "mf", "wad"):
                np.testing.assert_allclose(
                    reference[group_name][key],
                    optimized[group_name][key],
                    atol=1e-12,
                )

    def test_pruning_matches_reference_on_random_graphs(self):
        labels = ("/", "joint", "contrast")
        for seed in range(3):
            graph = nx.gnp_random_graph(8, 0.16, seed=seed, directed=True)
            graph.add_edges_from((node, node - 1) for node in range(1, 8))
            for left, right in graph.edges:
                graph[left][right]["label_0"] = labels[(left + right) % 3]

            motifs = []
            for motif_size in (3, 5):
                for nodes in combinations(graph, motif_size):
                    candidate = graph.subgraph(nodes).copy()
                    if nx.is_connected(candidate.to_undirected()):
                        motifs.append(candidate)
                    if len(motifs) >= 16:
                        break
            impossible = nx.DiGraph()
            impossible.add_edges_from(
                [
                    (0, 1, {"label_0": "unavailable"}),
                    (1, 2, {"label_0": "/"}),
                ]
            )
            motifs.append(impossible)

            reference = reference_distribution(graph, motifs, 0)
            optimized = ToSDataset.calculate_motif_distribution(
                graph, ToSDataset.prepare_motif_metadata(motifs), 0
            )
            for key in ("raw", "mf", "wad"):
                np.testing.assert_allclose(
                    reference[key], optimized[key], atol=1e-12
                )

    def test_observed_double_motifs_match_cartesian_reference(self):
        for seed in range(6):
            graph = nx.gnp_random_graph(8, 0.16, seed=seed, directed=True)
            for left, right in graph.edges:
                graph[left][right]["label_0"] = ("/", "joint", "contrast")[
                    (left + right) % 3
                ]
            # Saved motif catalogs retain one graph per WL hash, so mirror
            # that contract before comparing the two extraction paths.
            motif_catalog = {}
            for motif in deduplicate_graph_motifs(
                connected_three_node_subgraphs(graph)
            ):
                motif_hash = nx.weisfeiler_lehman_graph_hash(
                    motif, edge_attr="label_0"
                )
                motif_catalog[motif_hash] = motif
            motifs = list(motif_catalog.values())
            reference = reference_double_motifs(graph, motifs)
            optimized = observed_double_motifs(
                graph, index_graph_motifs(motifs)
            )
            self.assertEqual(set(reference), set(optimized), msg=f"seed={seed}")

    def test_triple_composition_matches_reference_without_mutation(self):
        single = nx.DiGraph()
        single.add_edges_from(
            [
                (1, 0, {"label_0": "/"}),
                (2, 0, {"label_0": "joint"}),
                (1, 2, {"label_0": "contrast"}),
            ]
        )
        double = nx.DiGraph()
        double.add_edges_from(
            [
                (1, 0, {"label_0": "/"}),
                (2, 0, {"label_0": "joint"}),
                (1, 2, {"label_0": "contrast"}),
                (3, 1, {"label_0": "/"}),
                (4, 1, {"label_0": "joint"}),
                (3, 4, {"label_0": "contrast"}),
            ]
        )
        original_single_nodes = set(single)
        original_double_nodes = set(double)
        reference = reference_triple_motifs([single], [double])

        prepared_double = prepare_double_docking_motifs([double])
        prepared_single = prepare_single_docking_variants([single])
        optimized = {}
        for motif in prepared_double:
            optimized.update(compose_triple_motifs(motif, prepared_single))
        parallel = {}
        for result in iter_composed_triple_motifs(
            prepared_double, prepared_single, workers=2, chunksize=1
        ):
            parallel.update(result)

        self.assertEqual(set(reference), set(optimized))
        self.assertEqual(set(reference), set(parallel))
        self.assertEqual(original_single_nodes, set(single))
        self.assertEqual(original_double_nodes, set(double))
        self.assertNotIn("docking_point", single)
        self.assertNotIn("docking_point", double)


if __name__ == "__main__":
    unittest.main()
