import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import networkx as nx

from tos.tos_utils import save_graph_motifs, write_selected_motif_hashes


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "3_extract_double_triads.py"
SPEC = importlib.util.spec_from_file_location("extract_double_triads", SCRIPT_PATH)
DOUBLE_TRIADS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOUBLE_TRIADS)


class DoubleMotifSelectionTest(unittest.TestCase):
    def test_support_filters_and_coverage_are_combined(self):
        scene_support = {"a": 100, "b": 50, "c": 25, "d": 5, "e": 1000}
        shard_support = {"a": 3, "b": 2, "c": 2, "d": 2, "e": 1}

        selected = DOUBLE_TRIADS.select_motif_hashes(
            scene_support,
            shard_support,
            min_scene_support=10,
            min_shard_support=2,
            coverage=0.75,
        )

        # e fails shard support and d fails scene support. a+b are the
        # smallest ranked prefix covering 75% of a+b+c support.
        self.assertEqual(selected, ["a", "b"])

    def test_unpruned_defaults_retain_all_in_deterministic_order(self):
        self.assertEqual(
            DOUBLE_TRIADS.select_motif_hashes(
                {"b": 1, "a": 2}, {"a": 1, "b": 1}
            ),
            ["a", "b"],
        )

    def test_manifest_accepts_one_pruned_group(self):
        graph_a = nx.DiGraph()
        graph_a.add_edge(0, 1, label_0="a")
        graph_b = nx.DiGraph()
        graph_b.add_edge(0, 1, label_0="b")
        with tempfile.TemporaryDirectory() as temporary_dir:
            save_graph_motifs(3, [graph_a], "sample", output_dir=temporary_dir)
            save_graph_motifs(
                6, [graph_a, graph_b], "sample", output_dir=temporary_dir
            )
            m6_path = Path(temporary_dir) / "sample_M6_motifs.json"
            m6_hashes = sorted(json.loads(m6_path.read_text()))

            manifest_path = write_selected_motif_hashes(
                temporary_dir,
                dataset_name="sample",
                sizes=(3, 6),
                selected_hashes_by_size={"m6": [m6_hashes[0]]},
            )
            manifest = json.loads(Path(manifest_path).read_text())

            self.assertEqual(len(manifest["m3"]), 1)
            self.assertEqual(manifest["m6"], [m6_hashes[0]])


if __name__ == "__main__":
    unittest.main()
