import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


_path = Path(__file__).parents[1] / "scripts/canonical/visualize_results.py"
_spec = importlib.util.spec_from_file_location("unirst2_visualization", _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)


class UniRST2VisualizationTest(unittest.TestCase):
    def test_f1_matrix_has_fixed_inventory_model_order(self):
        rows = []
        value = 0.0
        for inventory in _module.INVENTORIES:
            for model in _module.MODELS:
                for config in _module.CONFIGS:
                    rows.append({"inventory": inventory, "model": model, "feature_config": config, "test_f1": str(value)})
                    value += 1
        matrix = _module.f1_matrix(rows, robust=False)
        self.assertEqual(matrix.shape, (8, 5))
        np.testing.assert_array_equal(matrix[0], np.arange(5))
        np.testing.assert_array_equal(matrix[-1], np.arange(35, 40))

    def test_relation_label_is_compact(self):
        self.assertEqual(_module.relation_label("nodes=N|S; relations=joint|joint"), "joint×joint")


if __name__ == "__main__":
    unittest.main()
