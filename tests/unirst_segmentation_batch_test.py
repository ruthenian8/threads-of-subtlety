"""Segmentation batching uses only the encoder and keeps cache resume semantics."""
import os
import pickle
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import tos.unirst as unirst


class Predictor:
    def __init__(self):
        self.calls = []
        self.model = SimpleNamespace(encoder=self.encode)
        self.tokenizer = self.tokenize_offsets

    @staticmethod
    def tokenize_offsets(text, **kwargs):
        return {
            "input_ids": list(range(len(text.split()))),
            "offset_mapping": [(m.start(), m.end()) for m in re.finditer(r"\S+", text)],
        }

    def tokenize(self, data):
        return SimpleNamespace(
            input_sentences=[list(range(len(words))) for words in data.input_sentences],
            entity_ids=None, entity_position_ids=None, edu_breaks=data.edu_breaks,
            sent_breaks=None, dataset_index=[0] * len(data.input_sentences),
        )

    def encode(self, sentences, *args, **kwargs):
        assert kwargs["is_test"]
        self.calls.append([len(sentence) for sentence in sentences])
        return None, None, None, [[1, len(sentence) - 1] for sentence in sentences]


class SegmentationBatchTest(unittest.TestCase):
    def setUp(self):
        self.predictor = Predictor()
        self.adapter = unirst.UniRSTAdapter(
            parser_factory=lambda **kwargs: SimpleNamespace(predictor=self.predictor)
        )
        self.texts = ["alpha beta gamma", "Hi", "naïve beta gamma delta epsilon",
                      "alpha  beta gamma delta"]

    def test_matches_serial_boundaries_with_mixed_lengths_and_short_scenes(self):
        expected = [self.adapter.segment_scene(text) for text in self.texts]
        self.predictor.calls.clear()
        actual = self.adapter.segment_scenes(self.texts, batch_size=4)
        self.assertEqual(actual, expected)
        self.assertEqual(self.predictor.calls, [[3, 5, 4]])
        self.assertEqual(actual[1]["edus"], ["Hi"])
        self.assertEqual(actual[2]["edus"], ["naïve beta", "gamma delta epsilon"])

    def test_bounded_batches_and_short_only_scenes(self):
        self.adapter.segment_scenes(self.texts, batch_size=2)
        self.assertEqual(self.predictor.calls, [[3], [5, 4]])
        self.predictor.calls.clear()
        self.adapter.segment_scenes(["Hi", "one two"], batch_size=2)
        self.assertEqual(self.predictor.calls, [])
        self.assertEqual(self.adapter.segment_scenes([]), [])
        with self.assertRaises(ValueError):
            self.adapter.segment_scenes(self.texts, batch_size=0)

    def test_invalid_encoder_result_count_is_rejected(self):
        self.predictor.model.encoder = lambda *args, **kw: (None, None, None, [[]])
        with self.assertRaisesRegex(RuntimeError, "invalid segmentation batch"):
            self.adapter.segment_scenes(["one two three", "four five six"])

    def specs(self):
        return [{"scene_key": str(i), "text": text} for i, text in enumerate(
            ["alpha beta gamma"] * 5
        )]

    def test_checkpoints_and_resume_skip_completed_prefix_and_full_cache(self):
        specs = self.specs()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "segments.pkl")
            original = unirst._atomic_pickle_dump
            def interrupt_after_checkpoint(value, destination):
                original(value, destination)
                raise KeyboardInterrupt()
            with mock.patch.object(unirst, "_atomic_pickle_dump", side_effect=interrupt_after_checkpoint):
                with self.assertRaises(KeyboardInterrupt):
                    unirst.load_or_create_segmentation_cache(
                        self.adapter, specs, path, batch_size=4, checkpoint_interval=2
                    )
            with open(path, "rb") as handle:
                self.assertEqual(len(pickle.load(handle)["scenes"]), 2)
            self.predictor.calls.clear()
            result = unirst.load_or_create_segmentation_cache(
                self.adapter, specs, path, batch_size=4, checkpoint_interval=2
            )
            self.assertEqual(self.predictor.calls, [[3, 3], [3]])
            self.assertEqual([r["scene_key"] for r in result], ["0", "1", "2", "3", "4"])
            self.predictor.calls.clear()
            self.assertEqual(
                unirst.load_or_create_segmentation_cache(self.adapter, specs, path, batch_size=8),
                result,
            )
            self.assertEqual(self.predictor.calls, [])

    def test_bad_scene_is_isolated_by_single_scene_retry(self):
        specs = [{"scene_key": "good", "text": "alpha beta gamma"},
                 {"scene_key": "bad", "text": ""}]
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(unirst.tqdm, "write") as warning:
            result = unirst.load_or_create_segmentation_cache(
                self.adapter, specs, os.path.join(directory, "segments.pkl"), batch_size=2
            )
        self.assertEqual([r["status"] for r in result], ["ok", "error"])
        self.assertEqual(result[1]["edus"], [])
        warning.assert_called_once()

    def test_wrong_record_count_cannot_be_saved_as_complete(self):
        self.adapter.segment_scenes = mock.Mock(return_value=[])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "segments.pkl")
            with self.assertRaisesRegex(RuntimeError, "result count"):
                unirst.load_or_create_segmentation_cache(
                    self.adapter, self.specs(), path, batch_size=3
                )
            self.assertFalse(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
