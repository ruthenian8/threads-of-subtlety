import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "4_extract_triple_triads.py"
SPEC = importlib.util.spec_from_file_location("extract_triple_triads", SCRIPT_PATH)
TRIPLE_TRIADS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRIPLE_TRIADS)


class TripleInventoryTest(unittest.TestCase):
    def test_discovers_sorted_dataset_inventories(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "unirst"
            (root / "rel-nld.rst.nldt").mkdir(parents=True)
            (root / "rel-eng.erst.gum").mkdir()
            args = SimpleNamespace(
                root=str(root),
                motif_dir=None,
                dataset_name="hc3-mage",
                relinventory=None,
            )

            self.assertEqual(
                TRIPLE_TRIADS.discover_relinventories(args),
                ["eng.erst.gum", "nld.rst.nldt"],
            )

    def test_falls_back_to_complete_motif_directories(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            motif_root = Path(temporary_dir) / "motifs"
            complete = motif_root / "deu.rst.pcc"
            incomplete = motif_root / "eng.rst.rstdt"
            complete.mkdir(parents=True)
            incomplete.mkdir()
            (complete / "hc3-mage_M3_motifs.json").touch()
            (complete / "hc3-mage_M6_motifs.json").touch()
            (incomplete / "hc3-mage_M3_motifs.json").touch()
            args = SimpleNamespace(
                root=str(Path(temporary_dir) / "missing"),
                motif_dir=str(motif_root),
                dataset_name="hc3-mage",
                relinventory=None,
            )

            self.assertEqual(
                TRIPLE_TRIADS.discover_relinventories(args), ["deu.rst.pcc"]
            )

    def test_resolves_root_and_single_inventory_output_paths(self):
        all_args = SimpleNamespace(motif_dir="custom", relinventory=None)
        single_args = SimpleNamespace(
            motif_dir=os.path.join("custom", "eng.erst.gum"),
            relinventory="eng.erst.gum",
        )
        default_single_args = SimpleNamespace(
            motif_dir=None, relinventory="eng.erst.gum"
        )

        self.assertEqual(
            TRIPLE_TRIADS.resolve_motif_dir(all_args, "eng.erst.gum"),
            os.path.join("custom", "eng.erst.gum"),
        )
        self.assertEqual(
            TRIPLE_TRIADS.resolve_motif_dir(single_args, "eng.erst.gum"),
            os.path.join("custom", "eng.erst.gum"),
        )
        self.assertEqual(
            TRIPLE_TRIADS.resolve_motif_dir(
                default_single_args, "eng.erst.gum"
            ),
            os.path.join("data", "motifs", "eng.erst.gum"),
        )


if __name__ == "__main__":
    unittest.main()
