import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


_path = Path(__file__).parents[1] / "scripts/canonical/compare_rf_granularity.py"
_spec = importlib.util.spec_from_file_location("unirst2_rf_granularity", _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)


class UniRST2RFGranularityTest(unittest.TestCase):
    def test_mf_and_wad_are_interleaved_per_motif(self):
        frequency = np.asarray([[0.25, 0.75]], dtype=np.float32)
        wad = np.asarray([[0.2, 0.8]], dtype=np.float32)
        np.testing.assert_array_equal(
            _module.mf_wad_features(frequency, wad),
            np.asarray([[0.25, 0.2, 0.75, 0.8]], dtype=np.float32),
        )

    def test_coarse_wad_is_occurrence_weighted(self):
        fine_frequency = np.asarray([[0.25, 0.75, 0.0]], dtype=np.float32)
        fine_wad = np.asarray([[0.2, 0.8, 0.0]], dtype=np.float32)
        assignment = np.asarray([0, 0, 1], dtype=np.int64)
        coarse_frequency = np.asarray([[1.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(
            _module.aggregate_wad(
                fine_frequency, fine_wad, assignment, coarse_frequency
            ),
            np.asarray([[0.65, 0.0]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
