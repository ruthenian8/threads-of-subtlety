"""Batch preprocessing preserves supplied EDUs and scene-level outcomes."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "batch_preparation",
    Path(__file__).resolve().parents[1] / "0_prepare_and_parse_datasets_unirst.py",
)
preparation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preparation)


def scene(key, status="ok"):
    return {"scene_key": key, "status": status, "edus": [key], "error": "bad segmentation" if status != "ok" else None}


def single(edus):
    return {"parsed": edus[0]}


class BatchPreparationTest(unittest.TestCase):
    def setUp(self):
        self.parser = SimpleNamespace(
            from_edus=mock.Mock(side_effect=single),
            from_edus_batch=mock.Mock(
                side_effect=lambda documents, batch_size: [single(edus) for edus in documents]
            ),
        )
        self.adapter = SimpleNamespace(
            relation_inventories=("pcc", "nldt"),
            make_parser=mock.Mock(return_value=self.parser),
            to_constituency_format=lambda result, count: result["parsed"],
            release_segmenter=mock.Mock(),
        )

    def test_batches_match_serial_outputs_and_keep_segmentation_errors(self):
        scenes = [scene("a"), scene("b"), scene("bad", "error"),
                  scene("c"), scene("d"), scene("e"), scene("f")]
        expected = preparation.parse_predictions(self.adapter, scenes)
        self.parser.from_edus.reset_mock()
        callback = mock.Mock()
        actual = preparation.parse_predictions(
            self.adapter, scenes, batch_size=2, on_inventory_complete=callback
        )
        self.assertEqual(actual, expected)
        self.parser.from_edus.assert_not_called()
        self.assertEqual(self.parser.from_edus_batch.call_args_list[:4], [
            mock.call([["a"], ["b"]], batch_size=2),
            mock.call([["c"]], batch_size=2),
            mock.call([["d"], ["e"]], batch_size=2),
            mock.call([["f"]], batch_size=2),
        ])
        self.assertEqual(callback.call_count, 2)

    def test_failed_batch_retries_individually_without_losing_other_scenes(self):
        self.parser.from_edus_batch.side_effect = RuntimeError("batch failure")
        self.parser.from_edus.side_effect = [single(["a"]), ValueError("bad tree")]
        with mock.patch.object(preparation.tqdm, "write") as warning:
            result = preparation.parse_predictions(
                self.adapter, [scene("a"), scene("b")],
                inventories=("pcc",), batch_size=2,
            )
        self.assertEqual(result["a"]["pcc"]["parsed"], "a")
        self.assertEqual(result["b"]["pcc"]["error"], "ValueError: bad tree")
        self.assertEqual(self.parser.from_edus.call_count, 2)
        warning.assert_called_once()

    def test_result_count_mismatch_aborts_before_completion_callback(self):
        self.parser.from_edus_batch.return_value = []
        self.parser.from_edus_batch.side_effect = None
        callback = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "result count"):
            preparation.parse_predictions(
                self.adapter, [scene("a")], batch_size=2,
                on_inventory_complete=callback,
            )
        callback.assert_not_called()
        self.parser.from_edus.assert_not_called()

    def test_unsupported_api_fails_clearly(self):
        del self.parser.from_edus_batch
        with self.assertRaisesRegex(RuntimeError, "from_edus_batch"):
            preparation.parse_predictions(self.adapter, [scene("a")], batch_size=2)

    def test_default_stays_serial_and_sizes_must_be_positive(self):
        preparation.parse_predictions(self.adapter, [scene("a")])
        self.assertEqual(self.parser.from_edus.call_count, 2)
        self.parser.from_edus_batch.assert_not_called()
        with self.assertRaisesRegex(ValueError, "positive"):
            preparation.parse_predictions(self.adapter, [], batch_size=0)
        with mock.patch.object(sys, "argv", ["prepare", "--batch-size", "0"]):
            with self.assertRaises(SystemExit):
                preparation.parse_args()

    def test_run_group_forwards_batch_size(self):
        doc = preparation.make_document_spec("hc3", "train", 0, "a", "source", 1, ["a"])
        with mock.patch.object(preparation, "load_or_create_segmentation_cache", return_value=[scene("a")]) as segment, \
             mock.patch.object(preparation, "parse_predictions", return_value={}) as parse, \
             mock.patch.object(preparation, "write_group_outputs"):
            preparation.run_group(
                [doc], self.adapter, "/missing/segments", "/missing/outputs",
                "sample", "sample", False, "separate", batch_size=4,
            )
        self.assertEqual(parse.call_args.kwargs["batch_size"], 4)
        self.assertEqual(segment.call_args.kwargs["batch_size"], 4)

    def test_cli_forwards_batch_size_for_all_dataset_routes(self):
        argv = ["prepare", "--batch-size", "4", "--human-detectors-path", "hd.json",
                "--storyscope-dataset", "storyscope", "--storyscope-splits", "test"]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(preparation, "EttinSceneSplitter"), \
             mock.patch.object(preparation, "UniRSTAdapter"), \
             mock.patch.object(preparation, "hc3_groups", return_value={"finance": [{}]}), \
             mock.patch.object(preparation, "human_detectors_group", return_value=[{}]), \
             mock.patch.object(preparation, "storyscope_group", return_value=[{}]), \
             mock.patch.object(preparation, "mage_group", return_value=[{}]), \
             mock.patch.object(preparation, "load_dataset", side_effect=lambda name, **kw: [{}] if "split" in kw else {}), \
             mock.patch.object(preparation, "run_group") as run:
            preparation.main()
        self.assertEqual(run.call_count, 6)
        self.assertTrue(all(call.kwargs["batch_size"] == 4 for call in run.call_args_list))


if __name__ == "__main__":
    unittest.main()
