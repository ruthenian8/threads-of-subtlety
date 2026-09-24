import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.metrics import balanced_accuracy_score


_path = Path(__file__).parents[1] / "scripts/canonical/evaluate_feature_models.py"
_spec = importlib.util.spec_from_file_location("unirst2_feature_models", _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)


class UniRST2FeatureModelTest(unittest.TestCase):
    def test_threshold_maximizes_validation_balanced_accuracy_with_ties(self):
        labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8)
        probabilities = np.asarray([0.1, 0.2, 0.2, 0.7, 0.8, 0.9])
        threshold = _module.select_threshold(labels, probabilities)
        score = balanced_accuracy_score(labels, probabilities >= threshold)
        candidates = np.unique(
            np.r_[
                np.nextafter(probabilities.min(), -np.inf),
                np.nextafter(probabilities, np.inf),
            ]
        )
        expected = max(
            balanced_accuracy_score(labels, probabilities >= candidate)
            for candidate in candidates
        )
        self.assertAlmostEqual(score, expected)

    def test_matched_parameter_grids(self):
        labels = np.asarray([0, 0, 0, 1, 1], dtype=np.int8)
        lr_grid = _module.parameter_grid("lr", labels)
        xgb_grid = _module.parameter_grid("xgb", labels)
        self.assertEqual([candidate["C"] for candidate in lr_grid], [0.01, 0.1, 1.0, 10.0])
        self.assertEqual(len(xgb_grid), 4)
        self.assertTrue(
            all(candidate["scale_pos_weight"] == 1.5 for candidate in xgb_grid)
        )


if __name__ == "__main__":
    unittest.main()
