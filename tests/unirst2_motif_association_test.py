import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


_path = Path(__file__).parents[1] / "scripts/canonical/analyze_motif_associations.py"
_spec = importlib.util.spec_from_file_location("unirst2_motif_associations", _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)


class UniRST2MotifAssociationTest(unittest.TestCase):
    def test_prevalence_difference(self):
        y = np.asarray([0, 0, 1, 1], dtype=np.int8)
        presence = np.asarray([[0, 1], [0, 0], [1, 1], [0, 1]], dtype=float)
        weights = np.ones((1, 4), dtype=np.int16)
        actual = _module.prevalence_difference(weights, y, presence)[0]
        np.testing.assert_allclose(actual, [0.5, 0.5])

    def test_conditional_depth_excludes_absence(self):
        y = np.asarray([0, 0, 1, 1], dtype=np.int8)
        depth = np.asarray([[0], [2], [0], [4]], dtype=float)
        presence = np.asarray([[0], [1], [0], [1]], dtype=float)
        weights = np.ones((1, 4), dtype=np.int16)
        _, human_n, generated_n = _module.hedges_g(
            weights, y, depth, presence > 0
        )
        self.assertEqual(human_n[0, 0], 1)
        self.assertEqual(generated_n[0, 0], 1)

    def test_candidate_selection_uses_relative_representation_magnitude(self):
        hashes = np.asarray(["a", "b", "c"])
        coefficients = {
            "presence": np.asarray([10.0, 2.0, 0.0]),
            "frequency": np.asarray([0.0, 1.0, 0.2]),
            "depth": np.asarray([0.0, 0.0, 4.0]),
        }
        selected, scores = _module.select_candidates(hashes, coefficients, 3)
        self.assertEqual(list(hashes[selected]), ["a", "b", "c"])
        np.testing.assert_allclose(scores, [1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
