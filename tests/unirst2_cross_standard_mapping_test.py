import importlib.util
from pathlib import Path
import sys
import unittest

import networkx as nx


_path = Path(__file__).parents[1] / "scripts/canonical/map_cross_standard_motifs.py"
_spec = importlib.util.spec_from_file_location("unirst2_cross_standard_mapping", _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)


def graph(relation="joint", satellite="Satellite"):
    value = nx.DiGraph()
    value.add_node("a", label_0="Nucleus")
    value.add_node("b", label_0=satellite)
    value.add_node("r", label_0="root")
    value.add_edge("a", "r", label_0="/")
    value.add_edge("b", "r", label_0="/")
    value.add_edge("b", "a", label_0=relation)
    return value


class UniRST2CrossStandardMappingTest(unittest.TestCase):
    def test_exact_isomorphism_ignores_node_ids(self):
        left = graph()
        right = nx.relabel_nodes(graph(), {"a": "x", "b": "y", "r": "z"})
        self.assertTrue(_module.exact_isomorphic(left, right))

    def test_structural_family_ignores_relation_name_only(self):
        self.assertFalse(_module.exact_isomorphic(graph("joint"), graph("list")))
        self.assertTrue(_module.structural_isomorphic(graph("joint"), graph("list")))
        self.assertFalse(
            _module.structural_isomorphic(graph("joint"), graph("joint", "Nucleus"))
        )

    def test_pipeline_identity_ignores_node_nuclearity(self):
        self.assertTrue(
            _module.pipeline_isomorphic(graph("joint"), graph("joint", "Nucleus"))
        )

    def test_union_find_forms_transitive_components(self):
        union = _module.UnionFind(4)
        union.union(0, 1)
        union.union(1, 2)
        self.assertEqual(union.groups(), [[0, 1, 2], [3]])

    def test_repeated_hash_collision_is_reported(self):
        members = [
            {"motif_hash": "same", "inventory": "a"},
            {"motif_hash": "same", "inventory": "b"},
        ]
        result = _module.verify_repeated_hashes(
            members,
            {0: "pipeline-1", 1: "pipeline-1"},
            {0: "nuclearity-1", 1: "nuclearity-2"},
        )
        self.assertEqual(result["hash_collision_count"], 0)
        self.assertEqual(result["nuclearity_ambiguity_count"], 1)
        self.assertTrue(result["repeated_hashes_all_pipeline_isomorphic"])


if __name__ == "__main__":
    unittest.main()
