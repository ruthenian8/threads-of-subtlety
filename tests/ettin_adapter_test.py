"""Ettin integration tests use real Dipo trees and offline model doubles."""
import importlib.util
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

import torch
from dipo.rst.data.tree import RstTree

from tos.ettin import EttinAdapter, EttinParser, EttinSceneSplitter, GUM, MODELS, RSTDT, legacy_result
from tos.tos_dataset import ToSDataset
from tos.unirst import UniRSTAdapter, load_or_create_segmentation_cache


def load_script(filename):
    path = Path(__file__).resolve().parents[1] / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tokenizer:
    is_fast = True
    cls_token_id = 31
    sep_token_id = 30
    pad_token_id = 29
    bos_token_id = None
    eos_token_id = None
    model_max_length = 64
    def __call__(self, text, **kwargs):
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(spans))), "offset_mapping": spans}


class Model:
    def __init__(self, relation="joint-list"):
        self.config = SimpleNamespace(
            split_cjk_tokens=False, relation_types=((relation, "multinuc"),)
        )
        self.device = torch.device("cpu")
        self.tokenizer = Tokenizer()
        self.segmenter = SimpleNamespace(predict_breaks=lambda x: [len(x) - 1])
        self.layer_norm = lambda x: x
        self.encoded = []
        self.supplied = []

    def eval(self):
        return self

    def _encode_tokens(self, ids):
        self.encoded.append(ids.tolist())
        return torch.zeros((len(ids), 4))

    def predict(self, tree):
        self.supplied.append(list(tree.edu_strings))
        return RstTree.from_parsing_actions(
            [(i, "NN", self.config.relation_types[0][0])
             for i in range(1, len(tree.edus))],
            tree.edu_strings, relation_types=self.config.relation_types,
        )


class EttinAdapterTest(unittest.TestCase):
    def test_scene_splitter_preserves_fitting_document_and_falls_back_for_longer(self):
        model = Model()
        model.max_length = 5
        model.config.encoder_window_size = None
        adapter = EttinAdapter(model_loader=lambda *a, **kw: model)
        splitter = EttinSceneSplitter(adapter)
        document = "one\n\ntwo three"
        self.assertEqual(splitter.split(document), [document])
        fallback = mock.Mock()
        fallback.split.return_value = ["one two", "three four"]
        with mock.patch("tos.ettin.SceneSplitter", return_value=fallback):
            self.assertEqual(splitter.split("one two three four"), ["one two", "three four"])
        fallback.split.assert_called_once_with("one two three four")

    def test_native_conversion_keeps_nested_spans_nuclearity_and_labels(self):
        edus = ["First.", "Second.", "Third."]
        tree = RstTree.from_parsing_actions(
            [(2, "NS", "causal-result"), (1, "NN", "joint-list")], edus
        )
        result = legacy_result(tree, edus)
        parsed = UniRSTAdapter.to_constituency_format(result, 3)
        self.assertEqual(parsed, "(1:Nucleus=span:2,3:Satellite=causal-result:3) "
                                 "(1:Nucleus=joint-list:1,2:Nucleus=joint-list:2)")
        graph = ToSDataset.create_graph_from_const_format(parsed)
        self.assertTrue(graph.number_of_edges() > 0)

    def test_rejects_changed_edus_incomplete_trees_and_repeated_splits(self):
        tree = RstTree.from_parsing_actions([(1, "NN", "joint")], ["a", "b"])
        with self.assertRaisesRegex(ValueError, "text or order"):
            legacy_result(tree, ["b", "a"])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            legacy_result(SimpleNamespace(edu_strings=["a", "b"], parsing_actions=lambda: []), ["a", "b"])
        bad = SimpleNamespace(edu_strings=["a", "b", "c"],
                              parsing_actions=lambda: [(1, "NN", "joint"), (1, "NS", "cause")])
        with self.assertRaisesRegex(ValueError, "repeated"):
            legacy_result(bad, ["a", "b", "c"])

    def test_segmentation_retains_dipo_heading_paragraph_policy_without_decoding(self):
        model = Model()
        model.predict = mock.Mock(side_effect=AssertionError("must not decode"))
        parser = EttinParser(model)
        text = "Introduction\n\nAlpha beta gamma\n\nDelta epsilon zeta"
        self.assertEqual(parser.segment_edus(text), [
            "Introduction", "Alpha beta gamma", "Delta epsilon zeta"
        ])
        model.predict.assert_not_called()
        self.assertEqual(len(model.encoded), 1)

    def test_fixed_edus_do_not_run_the_segmenter(self):
        model = Model()
        model.segmenter.predict_breaks = mock.Mock(side_effect=AssertionError("must not resegment"))
        result = EttinParser(model).from_edus([" alpha  beta ", "gamma"])
        self.assertEqual(model.supplied, [["alpha beta", "gamma"]])
        self.assertEqual(UniRSTAdapter.extract_edus(result), ["alpha beta", "gamma"])
        model.segmenter.predict_breaks.assert_not_called()
        one = EttinParser(model).from_edus(["one EDU"])
        self.assertEqual(UniRSTAdapter.extract_edus(one), ["one EDU"])

    def test_loader_uses_pinned_models_and_devices(self):
        loader = mock.Mock(side_effect=lambda *args, **kw: Model())
        adapter = EttinAdapter(model_loader=loader, cuda_device=2)
        adapter.make_parser(GUM)
        adapter.make_parser(RSTDT)
        self.assertEqual(loader.call_args_list, [
            mock.call(MODELS[key][0], revision=MODELS[key][1], device="cuda:2")
            for key in (GUM, RSTDT)
        ])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            EttinAdapter(segment_relinventory="deu.rst.pcc")

    def test_cache_provenance_and_complete_cache_reuse(self):
        model = Model()
        adapter = EttinAdapter(model_loader=lambda *a, **kw: model)
        specs = [{"scene_key": "a", "text": "alpha beta gamma"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "segments.pkl")
            result = load_or_create_segmentation_cache(adapter, specs, path)
            import pickle
            with open(path, "rb") as handle:
                cache = pickle.load(handle)
            self.assertTrue(cache["metadata"]["segmentation_provenance"]["break_at_paragraphs"])
            self.assertEqual(cache["metadata"]["hf_model_name"], MODELS[GUM][0])
            adapter.segment_scene = mock.Mock(side_effect=AssertionError("cache should be reused"))
            self.assertEqual(load_or_create_segmentation_cache(adapter, specs, path), result)

    def test_output_provenance_blocks_mixed_settings_and_unidentified_outputs(self):
        adapter = EttinAdapter()
        with tempfile.TemporaryDirectory() as tmp:
            adapter.ensure_output_provenance(tmp)
            adapter.ensure_output_provenance(tmp)
            with self.assertRaisesRegex(ValueError, "provenance differs"):
                EttinAdapter(segment_relinventory=RSTDT).ensure_output_provenance(tmp)
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "rel-old"))
            with self.assertRaisesRegex(ValueError, "lack Ettin"):
                adapter.ensure_output_provenance(tmp)

    def test_both_models_receive_identical_cached_edus_and_feed_graph_stage(self):
        prep = load_script("0_prepare_and_parse_datasets_unirst.py")
        graph_stage = load_script("1_add_graphs_to_unirst_datasets.py")
        motif_stage = load_script("2_extract_single_triads.py")
        distribution_stage = load_script("5_add_motif_dists_to_unirst_datasets.py")
        models = {}
        def loader(repo, **kwargs):
            model = Model("joint-list" if repo == MODELS[GUM][0] else "joint")
            models.setdefault(repo, []).append(model)
            return model
        adapter = EttinAdapter(model_loader=loader)
        text = "Introduction\n\nAlpha beta gamma\n\nDelta epsilon zeta"
        docs = [prep.make_document_spec("hc3", "train", 0, text, "test", 1, [text])]
        with tempfile.TemporaryDirectory() as tmp:
            adapter.ensure_output_provenance(tmp)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prep.run_group(docs, adapter, os.path.join(tmp, "segments"), tmp,
                               "hc3_test", "hc3_test", False, "both", batch_size=2)
            paired_path = Path(tmp) / "paired/hc3_test_human.jsonl"
            self.assertTrue(paired_path.is_file())
            received = [models[MODELS[key][0]][-1].supplied for key in (GUM, RSTDT)]
            self.assertEqual(received[0], received[1])
            self.assertEqual(len(received[0][0]), 3)
            for key in (GUM, RSTDT):
                path = Path(tmp) / f"rel-{key}/hc3_test_human.discourse_parsed.jsonl"
                graph_stage.add_graphs(str(path))
                added = path.with_name(path.name.replace(".jsonl", ".graph_added.jsonl"))
                document = next(ToSDataset.iter_document_corpus(str(added)))
                tree = next(iter(document.scene_discourse_trees.values()))
                graph = ToSDataset.create_graph_from_const_format(tree.parsed)
                motifs = motif_stage.worker_function(graph)
                self.assertTrue(motifs)
                enriched = distribution_stage.add_motif_dist_to_document(
                    document, {"m3": motifs}
                )
                self.assertTrue(enriched.scene_discourse_trees[0].motif_dists)

    def test_ood_uses_shared_segmentation_cache_for_both_models(self):
        import pandas as pd
        ood = load_script("9_prepare_and_parse_ood_sets.py")
        models = []
        def loader(*args, **kwargs):
            model = Model()
            models.append(model)
            return model
        adapter = EttinAdapter(model_loader=loader)
        text = "Introduction\n\nAlpha beta gamma\n\nDelta epsilon zeta"
        frame = pd.DataFrame([{"text": text, "src": "test", "label": 1}])
        splitter = SimpleNamespace(split=lambda value: [value])
        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "shared.pkl")
            gum = ood.process_unirst_df(frame, "test", adapter, GUM, splitter, {}, cache)
            rstdt = ood.process_unirst_df(frame, "test", adapter, RSTDT, splitter, {}, cache)
        self.assertEqual(len(models), 3)  # One segmenter; two parsers.
        self.assertEqual(models[1].supplied, models[2].supplied)
        self.assertFalse(models[1].encoded or models[2].encoded)
        for documents in (gum, rstdt):
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].label, 1)
            self.assertEqual(len(documents[0].scene_discourse_trees[0].edus), 3)

    def test_cli_defaults_are_isolated_and_use_approved_segmenter(self):
        prep = load_script("0_prepare_and_parse_datasets_unirst.py")
        import sys
        with mock.patch.object(sys, "argv", ["prepare"]):
            args = prep.parse_args()
        self.assertEqual(args.output_dir, "data/ettin/unirst")
        self.assertEqual(args.segment_relinventory, GUM)
        self.assertEqual(tuple(args.relinventories), (GUM, RSTDT))

    def test_actual_dipo_model_path_matches_public_segmentation_and_keeps_edus(self):
        from transformers import ModernBertConfig, ModernBertModel
        from dipo.rst.parsers.dmrst.configuration_dmrst import DMRSTConfig, _SegmentationConfig
        from dipo.rst.parsers.dmrst.modeling_dmrst import DMRSTParser

        torch.manual_seed(42)
        encoder_config = ModernBertConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=64,
            local_attention=16, global_attn_every_n_layers=2,
            pad_token_id=29, cls_token_id=31, sep_token_id=30,
            reference_compile=False,
        )
        encoder_config._attn_implementation = "eager"
        encoder = ModernBertModel(encoder_config)
        config = DMRSTConfig(
            train_dir="unused", dev_dir="unused", model_name="offline-tiny",
            relation_types=[("joint", "multinuc"), ("elaboration", "rst")],
            freeze_embeddings=False, freeze_encoder_layers=0,
            segmentation=_SegmentationConfig(scheme="BIE", loss="crf"),
        )
        with mock.patch(
            "dipo.rst.parsers.dmrst.modeling_dmrst.load_encoder_and_tokenizer",
            return_value=(encoder, Tokenizer(), 64),
        ):
            native = DMRSTParser(config)
        parser = EttinParser(native)
        text = "Introduction\n\nAlpha beta gamma\n\nDelta epsilon zeta"
        edus = parser.segment_edus(text)
        expected = UniRSTAdapter._normalise_edus(native.predict_from_text(text).edu_strings)
        self.assertEqual(edus, expected)
        result = parser.from_edus(edus)
        self.assertEqual(UniRSTAdapter.extract_edus(result), edus)
        self.assertTrue(UniRSTAdapter.to_constituency_format(result, len(edus)))


if __name__ == "__main__":
    unittest.main()
