import os
import re
import tempfile
import unittest

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


class UniRSTAdapterTest(unittest.TestCase):
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

    def test_single_edu_is_marked_none(self):
        result = {"rst": [Unit(text="Only EDU.")]}
        self.assertEqual(UniRSTAdapter.to_constituency_format(result, 1), "NONE")


if __name__ == "__main__":
    unittest.main()

