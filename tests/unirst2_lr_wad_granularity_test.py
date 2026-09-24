import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


_path = Path(__file__).parents[1] / "scripts/canonical/compare_lr_wad_granularity.py"
_spec = importlib.util.spec_from_file_location("unirst2_lr_wad_granularity", _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)


class UniRST2LRWADGranularityTest(unittest.TestCase):
    def test_coarse_wad_is_frequency_weighted_and_zero_when_absent(self):
        fine_frequency = np.asarray([[0.25, 0.75, 0.0]], dtype=np.float32)
        fine_wad = np.asarray([[0.2, 0.8, 0.7]], dtype=np.float32)
        assignment = np.asarray([0, 0, 1], dtype=np.int64)
        coarse_frequency = np.asarray([[1.0, 0.0]], dtype=np.float32)
        coarse_wad = _module.aggregate_wad(
            fine_frequency, fine_wad, assignment, coarse_frequency
        )
        np.testing.assert_allclose(coarse_wad, [[0.65, 0.0]])

    def test_interleaved_features_match_dataset_motif_order(self):
        frequency = np.asarray([[0.3, 0.7]], dtype=np.float32)
        wad = np.asarray([[0.1, 0.9]], dtype=np.float32)
        np.testing.assert_array_equal(
            _module.interleave_mf_wad(frequency, wad),
            np.asarray([[0.3, 0.1, 0.7, 0.9]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
