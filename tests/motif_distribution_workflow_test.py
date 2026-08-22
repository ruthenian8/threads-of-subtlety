import importlib.util
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import tempfile
import unittest

import networkx as nx

from tos.tos_dataset import Document, SceneDiscourseTree, ToSDataset


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "5_add_motif_dists_to_unirst_datasets.py"
)
SPEC = importlib.util.spec_from_file_location(
    "add_unirst_motif_distributions", SCRIPT_PATH
)
WORKFLOW = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = WORKFLOW
SPEC.loader.exec_module(WORKFLOW)


class ImmediateFuture:
    def __init__(self, executor, result):
        self.executor = executor
        self.value = result

    def result(self):
        self.executor.outstanding -= 1
        return self.value


class ImmediateExecutor:
    def __init__(self):
        self.submitted = []
        self.outstanding = 0
        self.max_outstanding = 0

    def submit(self, function, batch):
        self.submitted.append(list(batch))
        self.outstanding += 1
        self.max_outstanding = max(self.max_outstanding, self.outstanding)
        return ImmediateFuture(self, list(batch))


class MotifDistributionWorkflowTest(unittest.TestCase):
    def test_graph_discovery_respects_catalog_corpus(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            inventory_dir = Path(temporary_dir) / "rel-eng.rst.rstdt"
            inventory_dir.mkdir()
            hc3 = inventory_dir / "hc3_source_human.discourse_parsed.graph_added.jsonl"
            mage = inventory_dir / "mage_train_human.discourse_parsed.graph_added.jsonl"
            other = inventory_dir / "other_human.discourse_parsed.graph_added.jsonl"
            for path in (hc3, mage, other):
                path.touch()

            self.assertEqual(
                WORKFLOW.discover_graph_files(
                    temporary_dir, "rel-eng.rst.rstdt", "hc3"
                ),
                [str(hc3)],
            )
            self.assertEqual(
                WORKFLOW.discover_graph_files(
                    temporary_dir, "rel-eng.rst.rstdt", "hc3-mage"
                ),
                sorted((str(hc3), str(mage))),
            )

    def test_batched_processing_is_bounded_and_ordered(self):
        executor = ImmediateExecutor()
        processed = list(
            WORKFLOW.iter_processed_documents(
                executor,
                range(11),
                batch_size=3,
                max_pending_batches=2,
            )
        )

        self.assertEqual(processed, list(range(11)))
        self.assertEqual(
            [len(batch) for batch in executor.submitted], [3, 3, 3, 2]
        )
        self.assertLessEqual(executor.max_outstanding, 2)

    def test_empty_output_is_atomically_finalized(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            input_path = Path(temporary_dir) / "sample.graph_added.jsonl"
            output_path = Path(temporary_dir) / "sample.m3_m6_motif_dists.jsonl"
            input_path.touch()

            WORKFLOW.write_motif_distributions(
                ImmediateExecutor(),
                str(input_path),
                str(output_path),
                chunksize=2,
                workers=1,
            )

            self.assertTrue(output_path.is_file())
            self.assertFalse(Path(f"{output_path}.partial").exists())
            self.assertEqual(output_path.read_text(), "")

    def test_process_pool_streams_a_document_to_final_output(self):
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                ("span_1-1", "span_1-3", {"label_0": "/"}),
                ("span_2-3", "span_1-3", {"label_0": "/"}),
                ("span_1-1", "span_2-3", {"label_0": "joint"}),
            ]
        )
        tree = SceneDiscourseTree(
            text="one two three",
            tokenized=[],
            segments=[],
            edus={
                "span_1-1": "one",
                "span_2-2": "two",
                "span_3-3": "three",
            },
            parsed="",
            graph_dict=nx.json_graph.node_link_data(graph),
            graph_networkx=None,
            motif_dists=None,
        )
        document = Document(
            text="one two three",
            scenes=["one two three"],
            scene_discourse_trees={0: tree},
            source="test",
            label=0,
        )
        motif_metadata = ToSDataset.prepare_motif_metadata([graph.copy()])
        motifs = {name: motif_metadata for name in ("m3", "m6")}

        with tempfile.TemporaryDirectory() as temporary_dir:
            input_path = Path(temporary_dir) / "sample.graph_added.jsonl"
            output_path = Path(temporary_dir) / "sample.m3_m6_motif_dists.jsonl"
            input_path.write_text(f"{ToSDataset.document_to_dict(document)}\n")
            with ProcessPoolExecutor(
                max_workers=2,
                initializer=WORKFLOW._init_motif_worker,
                initargs=(motifs,),
            ) as executor:
                WORKFLOW.write_motif_distributions(
                    executor,
                    str(input_path),
                    str(output_path),
                    chunksize=1,
                    workers=2,
                )

            result = ToSDataset.load_document_corpus(str(output_path))[0]
            result_tree = result.scene_discourse_trees[0]
            self.assertEqual(set(result_tree.motif_dists), {"m3", "m6"})
            self.assertEqual(result_tree.motif_dists["m3"].raw, [1.0])
            self.assertFalse(Path(f"{output_path}.partial").exists())

    def test_output_path_keeps_the_input_stem(self):
        self.assertEqual(
            WORKFLOW.output_path_for("sample.discourse_parsed.graph_added.jsonl"),
            "sample.discourse_parsed.graph_added.m3_m6_motif_dists.jsonl",
        )

    def test_existing_outputs_are_skipped_unless_forced(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            input_path = Path(temporary_dir) / "sample.graph_added.jsonl"
            output_path = Path(WORKFLOW.output_path_for(str(input_path)))
            input_path.touch()
            output_path.touch()

            self.assertEqual(
                WORKFLOW.pending_output_files([str(input_path)], force=False), []
            )
            self.assertEqual(
                WORKFLOW.pending_output_files([str(input_path)], force=True),
                [(str(input_path), str(output_path))],
            )


if __name__ == "__main__":
    unittest.main()
