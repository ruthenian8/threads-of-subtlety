import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


_path = Path(__file__).parents[1] / "scripts/canonical/run_robustness_analysis.py"
_spec = importlib.util.spec_from_file_location("unirst2_robustness", _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)


class UniRST2RobustnessTest(unittest.TestCase):
    def test_overlap_removed_before_duplicate_consolidation(self):
        train = np.asarray(["a", "z"])
        test = np.asarray(["a", "a", "b", "b", "c"])
        y = np.asarray([0, 0, 1, 1, 0], dtype=np.int8)
        domain = np.asarray(["x", "x", "y", "y", "x"])
        cohort = _module.sensitivity_cohort(train, test, y, domain)
        np.testing.assert_array_equal(cohort["overlap_mask"], [1, 1, 0, 0, 0])
        self.assertEqual(list(cohort["unique_hashes"]), ["b", "c"])
        self.assertEqual([list(group) for group in cohort["groups"]], [[2, 3], [4]])

    def test_consolidation_averages_repeated_predictions(self):
        values = np.asarray([0.2, 0.4, 0.8])
        groups = [np.asarray([0, 1]), np.asarray([2])]
        np.testing.assert_allclose(
            _module.consolidate_vector(values, groups), [0.3, 0.8]
        )


if __name__ == "__main__":
    unittest.main()
