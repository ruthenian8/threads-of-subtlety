import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


_path = Path(__file__).parents[1] / "scripts/canonical/compare_paired_performance.py"
_spec = importlib.util.spec_from_file_location("unirst2_paired_performance", _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)


class UniRST2PairedPerformanceTest(unittest.TestCase):
    def test_weighted_metrics_match_sklearn_with_ties(self):
        y = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.int8)
        probability = np.asarray([0.1, 0.2, 0.2, 0.7, 0.9, 0.8])
        threshold = 0.6
        weights = np.asarray([[1, 2, 3, 1, 2, 1]], dtype=np.int16)
        actual = _module.weighted_metric_batch(
            y, probability, threshold, weights
        )[0]
        expanded_y = np.repeat(y, weights[0])
        expanded_prediction = np.repeat(probability >= threshold, weights[0])
        expected = np.asarray(
            [
                roc_auc_score(y, probability, sample_weight=weights[0]),
                average_precision_score(y, probability, sample_weight=weights[0]),
                balanced_accuracy_score(
                    expanded_y,
                    expanded_prediction,
                ),
            ]
        )
        np.testing.assert_allclose(actual[:3], expected)
        np.testing.assert_allclose(
            actual[3:],
            [
                f1_score(expanded_y, expanded_prediction, pos_label=1),
                f1_score(expanded_y, expanded_prediction, average="macro"),
            ],
        )

    def test_cluster_bootstrap_keeps_duplicate_identity_weights_together(self):
        identity = np.asarray(["a", "a", "b", "c", "d"])
        y = np.asarray([0, 0, 0, 1, 1], dtype=np.int8)
        clusters = _module.identity_clusters(identity, y)
        weights = _module.bootstrap_row_weights(
            np.random.default_rng(7), clusters, 50
        )
        np.testing.assert_array_equal(weights[:, 0], weights[:, 1])
        # Count each identity once: two clusters are sampled within each label.
        np.testing.assert_array_equal(weights[:, 0] + weights[:, 2], np.full(50, 2))
        np.testing.assert_array_equal(weights[:, 3] + weights[:, 4], np.full(50, 2))

    def test_holm_adjustment_is_monotone_in_sorted_order(self):
        raw = [0.04, 0.001, 0.02, 0.3]
        adjusted = _module.holm_adjust(raw)
        order = np.argsort(raw)
        ordered = np.asarray(adjusted)[order]
        self.assertTrue(np.all(np.diff(ordered) >= 0))
        self.assertTrue(all(a >= p for a, p in zip(adjusted, raw)))


if __name__ == "__main__":
    unittest.main()
