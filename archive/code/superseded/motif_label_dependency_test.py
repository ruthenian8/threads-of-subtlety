import unittest

import numpy as np

import importlib.util
import sys
from pathlib import Path


_path = Path(__file__).parents[1] / "10_analyze_motif_label_dependency.py"
_spec = importlib.util.spec_from_file_location("motif_dependency", _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)


class MotifLabelDependencyTest(unittest.TestCase):
    def test_document_aggregation_recomputes_frequency_and_wad(self):
        record = {
            "text": "abc",
            "label": 1,
            "source": "medicine_human",
            "scene_discourse_trees": {
                "0": {"edus": {"a": "x"}, "graph_dict": {"nodes": [1], "links": [1]},
                      "motif_dists": {"m3": {"raw": [2, 0], "wad": [0.5, -1]},
                                       "m6": {"raw": [0], "wad": [-1]}}},
                "1": {"edus": {"a": "x", "b": "y"}, "graph_dict": {"nodes": [1, 2], "links": []},
                      "motif_dists": {"m3": {"raw": [1, 1], "wad": [1.0, 0.25]},
                                       "m6": {"raw": [0], "wad": [-1]}}},
            },
        }
        vector, structural = _module._aggregate_record(record, {"m3": 2, "m6": 1})
        # m3 feature order is mf, wad per motif.
        np.testing.assert_allclose(vector[:4], [3 / 4, (2 * .5 + 1 * 1) / 3, 1 / 4, .25])
        self.assertEqual(structural["scene_count"], 2)
        self.assertEqual(structural["edu_count"], 3)
        self.assertEqual(_module._source_group("medicine_human"), "medicine")
        self.assertEqual(_module._source_group("medicine_llm"), "medicine")

    def test_config_feature_dimensions(self):
        corpus = _module.Corpus(
            np.zeros((3, 6)), np.array([0, 1, 0]),
            np.array(["a", "b", "a"]), np.array(["a_human", "b_llm", "a_llm"]),
            ["m3:0:h:mf", "m3:0:h:wad", "m6:0:h:mf", "m6:0:h:wad", "m6:1:h:mf", "m6:1:h:wad"],
            3, np.ones((3, 5)),
        )
        self.assertEqual(_module._config_matrix(corpus, "m3")[0].shape[1], 2)
        self.assertEqual(_module._config_matrix(corpus, "m6")[0].shape[1], 4)
        self.assertEqual(_module._config_matrix(corpus, "structural")[0].shape[1], 5)
        self.assertEqual(_module._config_matrix(corpus, "combined")[0].shape[1], 11)


if __name__ == "__main__":
    unittest.main()
