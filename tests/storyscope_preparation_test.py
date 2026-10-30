"""StoryScope ingestion keeps every model story as an independent document."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "0_prepare_and_parse_datasets_unirst.py"
SPEC = importlib.util.spec_from_file_location("unirst_preparation", SCRIPT)
preparation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preparation)


class FakeSplitter:
    def split(self, text):
        return [text]


def row(prompt_id, suffix):
    return {
        "prompt_id": prompt_id,
        **{
            column: f"{model} story {suffix}"
            for column, model in preparation.STORYSCOPE_MODEL_COLUMNS
        },
    }


class StoryScopePreparationTest(unittest.TestCase):
    def test_each_model_column_becomes_a_separate_machine_entry(self):
        dataset = {"test": [row(17, "one")]}
        documents = preparation.storyscope_group(
            dataset, "test", FakeSplitter(), 1, 0, 10
        )
        self.assertEqual(len(documents), 5)
        self.assertEqual(
            [document["source"] for document in documents],
            [
                f"storyscope::{model}::prompt-17"
                for _, model in preparation.STORYSCOPE_MODEL_COLUMNS
            ],
        )
        self.assertEqual([document["document_index"] for document in documents], list(range(5)))
        self.assertEqual({document["label"] for document in documents}, {0})
        self.assertEqual(
            [document["text"] for document in documents],
            [f"{model} story one" for _, model in preparation.STORYSCOPE_MODEL_COLUMNS],
        )
        self.assertTrue(all(document["scenes"] == [document["text"]] for document in documents))
        output_paths = preparation.expected_group_output_paths(
            documents, "data/unirst", "storyscope_test_gpu0",
            ("deu.rst.pcc",), "separate",
        )
        self.assertEqual(
            output_paths,
            [
                "data/unirst/rel-deu.rst.pcc/"
                "storyscope_test_gpu0_machine.discourse_parsed.jsonl"
            ],
        )

    def test_shards_are_disjoint_and_short_or_missing_stories_are_skipped(self):
        rows = [row(index, str(index)) for index in range(4)]
        rows[0]["story_gemini"] = None
        rows[1]["story_claude"] = "tiny"
        dataset = {"test": rows}
        first = preparation.storyscope_group(dataset, "test", FakeSplitter(), 2, 0, 10)
        second = preparation.storyscope_group(dataset, "test", FakeSplitter(), 2, 1, 10)
        self.assertEqual((len(first), len(second)), (8, 10))
        self.assertEqual(
            {document["source"] for document in first}
            & {document["source"] for document in second},
            set(),
        )

    def test_missing_model_column_fails_instead_of_silently_dropping_model(self):
        incomplete = row(1, "one")
        del incomplete["story_kimi"]
        with self.assertRaisesRegex(ValueError, "story_kimi"):
            preparation.storyscope_group(
                {"test": [incomplete]}, "test", FakeSplitter(), 1, 0, 10
            )

    def test_cli_routes_storyscope_to_its_own_output_shard(self):
        dataset = [row(9, "first"), row(10, "second")]
        args = [
            "0_prepare_and_parse_datasets_unirst.py",
            "--skip-hc3", "--skip-mage",
            "--storyscope-dataset", "jjrussell10/storyscope",
            "--storyscope-splits", "dev",
            "--storyscope-batch-prompts", "1",
        ]
        with mock.patch.object(sys, "argv", args), \
             mock.patch.object(preparation, "load_dataset", return_value=dataset) as load, \
             mock.patch.object(preparation, "SceneSplitter", return_value=FakeSplitter()), \
             mock.patch.object(preparation, "UniRSTAdapter"), \
             mock.patch.object(preparation, "run_group") as run:
            preparation.main()
        load.assert_called_once_with("jjrussell10/storyscope", split="dev")
        self.assertEqual(run.call_count, 2)
        for batch_index, call in enumerate(run.call_args_list):
            self.assertEqual(len(call.args[0]), 5)
            name = f"storyscope_dev_gpu0_batch{batch_index:04d}"
            self.assertEqual(call.args[4:6], (name, name))


if __name__ == "__main__":
    unittest.main()
