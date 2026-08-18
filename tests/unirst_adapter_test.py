import os
import re
import tempfile
import unittest
from types import SimpleNamespace

from tos.unirst import (
    UniRSTAdapter,
    load_or_create_segmentation_cache,
)


class Unit:
    def __init__(self, text=None, left=None, right=None, nuclearity=None, relation=None):
        self.text = text
        self.left = left
        self.right = right
        self.nuclearity = nuclearity
        self.relation = relation


class FakeParser:
    calls = 0
    segmentation_calls = 0

    def __init__(self, **kwargs):
        self.relinventory = kwargs["relinventory"]

    def __call__(self, text):
        FakeParser.calls += 1
        return {
            "rst": [
                Unit(
                    left=Unit(text="First EDU."),
                    right=Unit(text="Second EDU."),
                    nuclearity="NS",
                    relation="elaboration",
                )
            ]
        }

    def segment_edus(self, text):
        FakeParser.segmentation_calls += 1
        return ["First EDU.", "Second EDU."]

    def from_edus(self, edus):
        return {
            "rst": [
                Unit(
                    left=Unit(text=edus[0]),
                    right=Unit(text=edus[1]),
                    nuclearity="SN",
                    relation="contrast",
                )
            ]
        }


class FakeEncoder:
    def __init__(self):
        self.calls = 0

    def __call__(
        self,
        input_sentences,
        entity_ids,
        entity_position_ids,
        edu_breaks,
        sent_breaks,
        is_test,
        dataset_index,
    ):
        self.calls += 1
        if not is_test:
            raise AssertionError("The segmenter must run in inference mode")
        return None, None, None, [[2, 5]]


class FakeTokenizer:
    def convert_ids_to_tokens(self, input_ids):
        return ["▁First", "▁EDU", ".", "▁Second", "▁EDU", "."]


class FakePredictor:
    def __init__(self):
        self.model = SimpleNamespace(encoder=FakeEncoder())
        self.tokenizer = FakeTokenizer()

    def tokenize(self, data):
        return SimpleNamespace(
            input_sentences=[list(range(6))],
            entity_ids=None,
            entity_position_ids=None,
            edu_breaks=data.edu_breaks,
            sent_breaks=None,
            dataset_index=[0],
        )

    def build_offset_converter_from_words(self, text, tokens, offsets):
        positions = list(range(len(text) + 1))
        return positions, positions

    def remap_tree_offsets(self, unit, positions, originals, text):
        unit.text = text[unit.start : unit.end]


class FakeInternalParser:
    calls = 0

    def __init__(self, **kwargs):
        self.predictor = FakePredictor()

    def __call__(self, text):
        FakeInternalParser.calls += 1
        raise AssertionError("Full RST parsing must not run during segmentation")


class UniRSTAdapterTest(unittest.TestCase):
    def setUp(self):
        FakeParser.calls = 0
        FakeParser.segmentation_calls = 0
        FakeInternalParser.calls = 0

    def make_adapter(self):
        return UniRSTAdapter(
            relation_inventories=("deu.rst.pcc",),
            parser_factory=FakeParser,
        )

    def test_extract_and_convert_tree(self):
        adapter = self.make_adapter()
        result = FakeParser(relinventory="deu.rst.pcc")("ignored")
        self.assertEqual(adapter.extract_edus(result), ["First EDU.", "Second EDU."])
        parsed = adapter.to_constituency_format(result, 2)
        self.assertEqual(
            re.findall(r"\(([^ ]+)\)", parsed),
            ["1:Nucleus=span:1,2:Satellite=elaboration:2"],
        )
        self.assertTrue(parsed.startswith("(1:"))

    def test_cache_is_reused(self):
        adapter = self.make_adapter()
        scenes = [{"scene_key": "test:0", "text": "First EDU. Second EDU."}]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "segments.pkl")
            first = load_or_create_segmentation_cache(adapter, scenes, path)
            calls_after_first = FakeParser.calls
            second = load_or_create_segmentation_cache(adapter, scenes, path)
        self.assertEqual(first[0]["edus"], ["First EDU.", "Second EDU."])
        self.assertEqual(first, second)
        self.assertEqual(FakeParser.calls, calls_after_first)
        self.assertEqual(FakeParser.calls, 0)
        self.assertEqual(FakeParser.segmentation_calls, 1)

    def test_segmentation_does_not_invoke_full_parser(self):
        adapter = self.make_adapter()

        segmented = adapter.segment_scene("First EDU. Second EDU.")

        self.assertEqual(segmented["edus"], ["First EDU.", "Second EDU."])
        self.assertEqual(FakeParser.segmentation_calls, 1)
        self.assertEqual(FakeParser.calls, 0)

    def test_installed_api_path_stops_after_encoder_segmentation(self):
        adapter = UniRSTAdapter(parser_factory=FakeInternalParser)

        segmented = adapter.segment_scene("First EDU. Second EDU.")

        self.assertEqual(segmented["edus"], ["First EDU.", "Second EDU."])
        self.assertEqual(adapter.segmenter().predictor.model.encoder.calls, 1)
        self.assertEqual(FakeInternalParser.calls, 0)

    def test_one_word_edu_advances_alignment_cursor(self):
        boundaries = UniRSTAdapter._align_predicted_segments(
            ["Formats", "that allow older cards"],
            ["Formats", "that", "allow", "older", "cards"],
        )

        self.assertEqual(boundaries, [1, 5])

    def test_du_converter_replacement_handles_one_word_edu(self):
        fixed = UniRSTAdapter._fix_segmented_strings(
            ["Formats", "that allow older cards"],
            ["Formats", "that", "allow", "older", "cards"],
        )

        self.assertEqual(fixed, ["Formats", "that allow older cards"])

    def test_production_parser_installs_bounded_du_converter(self):
        from isanlp_rst.utils.du_converter import DUConverter

        original = DUConverter.__dict__["fix_segmented_strings"]
        try:
            UniRSTAdapter._install_du_converter_alignment_fix()

            fixed = DUConverter.fix_segmented_strings(
                ["Formats", "that allow older cards"],
                ["Formats", "that", "allow", "older", "cards"],
            )
        finally:
            DUConverter.fix_segmented_strings = original

        self.assertEqual(fixed, ["Formats", "that allow older cards"])

    def test_alignment_rejects_boundary_inside_gold_token(self):
        with self.assertRaisesRegex(ValueError, "ends inside a gold token"):
            UniRSTAdapter._align_predicted_segments(
                ["Format"],
                ["Formats"],
            )

    def test_single_edu_is_marked_none(self):
        result = {"rst": [Unit(text="Only EDU.")]}
        self.assertEqual(UniRSTAdapter.to_constituency_format(result, 1), "NONE")


if __name__ == "__main__":
    unittest.main()
