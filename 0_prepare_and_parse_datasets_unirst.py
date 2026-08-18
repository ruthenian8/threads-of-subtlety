"""Prepare HC3/MAGE with one shared UniRST segmentation and four inventories.

Example:
    python 0_prepare_and_parse_datasets_unirst.py \
        --total-gpus 1 --gpu-id 0 --output-dir data/unirst

Run this script once per GPU for MAGE when using multiple GPUs.  Each worker
owns a separate segmentation pickle and output shard.
"""

from __future__ import annotations

import argparse
import gc
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence

from datasets import load_dataset
from rich.progress import track
from sentsplit.segment import SentSplit
import torch
from transformers import AutoTokenizer

from tos.tos_dataset import Document, SceneDiscourseTree
from tos.tos_utils import split_list_into_n_chunks
from tos.unirst import (
    DEFAULT_REL_INVENTORIES,
    UniRSTAdapter,
    build_scene_lookup,
    load_or_create_segmentation_cache,
    write_jsonl_documents,
)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


class SceneSplitter:
    def __init__(self, max_tokens_margin: int = 20):
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
        self.max_tokens = self.tokenizer.model_max_length - max_tokens_margin
        self.sent_splitter = SentSplit("en")

    def split(self, document: str) -> List[str]:
        paragraphs = document.split("\n\n")
        scenes: List[str] = []
        current_scene = ""
        current_token_count = 0

        for paragraph in paragraphs:
            token_count = len(self.tokenizer.tokenize(paragraph))
            if token_count > self.max_tokens:
                sentences = self.sent_splitter.segment(paragraph, strip_spaces=False)
                for sentence in sentences:
                    token_count = len(self.tokenizer.tokenize(sentence))
                    if current_token_count + token_count > self.max_tokens:
                        if current_scene.strip():
                            scenes.append(current_scene.strip())
                        current_scene = sentence
                        current_token_count = token_count
                    else:
                        current_scene += sentence
                        current_token_count += token_count
                if current_scene.strip():
                    scenes.append(current_scene.strip())
                current_scene = ""
                current_token_count = 0
                continue

            if current_token_count + token_count > self.max_tokens:
                if current_scene.strip():
                    scenes.append(current_scene.strip())
                current_scene = paragraph
                current_token_count = token_count
            else:
                if current_scene:
                    current_scene += "\n\n"
                current_scene += paragraph
                current_token_count += token_count

        if current_scene.strip():
            scenes.append(current_scene.strip())
        return scenes


def hc3_groups(dataset_name: str, splitter: SceneSplitter, min_char_len: int) -> Dict[str, List[Dict[str, Any]]]:
    raw_dataset = load_dataset(dataset_name, name="all")["train"]
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    document_index = 0
    for sample in raw_dataset:
        try:
            human_answer = sample["human_answers"][0].strip()
            llm_answer = sample["chatgpt_answers"][0].strip()
        except (IndexError, KeyError):
            continue
        if len(human_answer) < min_char_len or len(llm_answer) < min_char_len:
            continue
        source = sample["source"]
        for text, suffix, label in (
            (human_answer, "human", 1),
            (llm_answer, "llm", 0),
        ):
            grouped[source].append(
                make_document_spec(
                    dataset="hc3",
                    split="train",
                    document_index=document_index,
                    text=text,
                    source=f"{source}_{suffix}",
                    label=label,
                    scenes=splitter.split(text),
                )
            )
            document_index += 1
    return grouped


def mage_group(
    raw_dataset: Any,
    split: str,
    splitter: SceneSplitter,
    total_gpus: int,
    gpu_id: int,
) -> List[Dict[str, Any]]:
    chunks = list(split_list_into_n_chunks(range(len(raw_dataset[split])), total_gpus))
    target_indices = chunks[gpu_id]
    return [
        make_document_spec(
            dataset="mage",
            split=split,
            document_index=doc_idx,
            text=raw_dataset[split][doc_idx]["text"],
            source=raw_dataset[split][doc_idx]["src"],
            label=raw_dataset[split][doc_idx]["label"],
            scenes=splitter.split(raw_dataset[split][doc_idx]["text"]),
        )
        for doc_idx in target_indices
    ]


def make_document_spec(
    dataset: str,
    split: str,
    document_index: int,
    text: str,
    source: str,
    label: int,
    scenes: Sequence[str],
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "split": split,
        "document_index": document_index,
        "text": text,
        "source": source,
        "label": label,
        "scenes": list(scenes),
    }


def make_scene_specs(documents: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    scene_specs = []
    for document in documents:
        for scene_index, text in enumerate(document["scenes"]):
            scene_specs.append(
                {
                    "scene_key": ":".join(
                        [
                            document["dataset"],
                            document["split"],
                            str(document["document_index"]),
                            str(scene_index),
                        ]
                    ),
                    "dataset": document["dataset"],
                    "split": document["split"],
                    "document_index": document["document_index"],
                    "scene_index": scene_index,
                    "text": text,
                }
            )
    return scene_specs


def scene_tree_from_record(record: Mapping[str, Any], parsed: str) -> SceneDiscourseTree:
    edus = {
        f"span_{index}-{index}": edu
        for index, edu in enumerate(record["edus"], start=1)
    }
    return SceneDiscourseTree(
        text=record["text"],
        tokenized=record["tokenized"],
        segments=record["segments"],
        edus=edus,
        parsed=parsed,
        graph_dict=None,
        graph_networkx=None,
        motif_dists=None,
    )


def document_output(
    document: Mapping[str, Any],
    scene_records: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    relinventory: str,
) -> Document:
    trees = {}
    for scene_index, scene_text in enumerate(document["scenes"]):
        key = ":".join(
            [
                document["dataset"],
                document["split"],
                str(document["document_index"]),
                str(scene_index),
            ]
        )
        record = scene_records[key]
        prediction = predictions[key][relinventory]
        trees[scene_index] = scene_tree_from_record(record, prediction["parsed"])
    return Document(
        text=document["text"],
        scenes=document["scenes"],
        scene_discourse_trees=trees,
        source=document["source"],
        label=document["label"],
    )


def paired_output(
    document: Mapping[str, Any],
    scene_records: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    scene_predictions = {}
    for scene_index, _ in enumerate(document["scenes"]):
        key = ":".join(
            [
                document["dataset"],
                document["split"],
                str(document["document_index"]),
                str(scene_index),
            ]
        )
        record = scene_records[key]
        scene_predictions[str(scene_index)] = {
            "scene_key": key,
            "text": record["text"],
            "edus": record["edus"],
            "segmentation_status": record["status"],
            "segmentation_error": record["error"],
            "predictions": predictions[key],
        }
    return {
        "text": document["text"],
        "scenes": document["scenes"],
        "scene_predictions": scene_predictions,
        "source": document["source"],
        "label": document["label"],
    }


def parse_predictions(
    adapter: UniRSTAdapter,
    scenes: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    predictions: Dict[str, Dict[str, Any]] = {
        scene["scene_key"]: {} for scene in scenes
    }
    for relinventory in adapter.relation_inventories:
        parser = adapter.make_parser(relinventory)
        try:
            for scene in track(
                scenes,
                description=f"parsing with {relinventory}",
                transient=True,
            ):
                if scene["status"] != "ok":
                    predictions[scene["scene_key"]][relinventory] = {
                        "status": "segmentation_error",
                        "parsed": "NONE",
                        "error": scene["error"],
                    }
                    continue
                try:
                    result = parser.from_edus(scene["edus"])
                    parsed = adapter.to_constituency_format(result, len(scene["edus"]))
                    predictions[scene["scene_key"]][relinventory] = {
                        "status": "ok",
                        "parsed": parsed,
                        "error": None,
                    }
                except Exception as exc:
                    predictions[scene["scene_key"]][relinventory] = {
                        "status": "error",
                        "parsed": "NONE",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
        finally:
            del parser
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    return predictions


def write_group_outputs(
    documents: Sequence[Mapping[str, Any]],
    scenes: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    output_dir: str,
    prefix: str,
    inventories: Sequence[str],
    output_mode: str,
) -> None:
    scene_lookup = build_scene_lookup(scenes)
    for relinventory in inventories:
        if output_mode not in {"separate", "both"}:
            break
        inventory_dir = os.path.join(output_dir, f"rel-{safe_name(relinventory)}")
        for label, label_name in ((1, "human"), (0, "machine")):
            selected = [doc for doc in documents if doc["label"] == label]
            if not selected:
                continue
            output_documents = [
                document_output(doc, scene_lookup, predictions, relinventory)
                for doc in selected
            ]
            write_jsonl_documents(
                output_documents,
                os.path.join(
                    inventory_dir,
                    f"{prefix}_{label_name}.discourse_parsed.jsonl",
                ),
            )

    if output_mode not in {"paired", "both"}:
        return
    paired_dir = os.path.join(output_dir, "paired")
    os.makedirs(paired_dir, exist_ok=True)
    for label, label_name in ((1, "human"), (0, "machine")):
        selected = [doc for doc in documents if doc["label"] == label]
        if not selected:
            continue
        path = os.path.join(paired_dir, f"{prefix}_{label_name}.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for document in selected:
                handle.write(
                    f"{paired_output(document, scene_lookup, predictions)}\n"
                )


def run_group(
    documents: Sequence[Mapping[str, Any]],
    adapter: UniRSTAdapter,
    segments_dir: str,
    output_dir: str,
    prefix: str,
    cache_name: str,
    force_segmentation: bool,
    output_mode: str,
) -> None:
    if not documents:
        return
    scene_specs = make_scene_specs(documents)
    cache_path = os.path.join(segments_dir, f"{cache_name}.pkl")
    scenes = load_or_create_segmentation_cache(
        adapter, scene_specs, cache_path, force=force_segmentation
    )
    adapter.release_segmenter()
    predictions = parse_predictions(adapter, scenes)
    write_group_outputs(
        documents,
        scenes,
        predictions,
        output_dir,
        prefix,
        adapter.relation_inventories,
        output_mode,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-gpus", type=int, default=1)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output-dir", default="data/unirst")
    parser.add_argument("--segments-dir", default=None)
    parser.add_argument("--hc3-dataset", default="Hello-SimpleAI/HC3")
    parser.add_argument("--mage-dataset", default="yaful/MAGE")
    parser.add_argument("--min-char-len", type=int, default=10)
    parser.add_argument("--segment-relinventory", default="deu.rst.pcc")
    parser.add_argument("--relinventories", nargs="+", default=list(DEFAULT_REL_INVENTORIES))
    parser.add_argument("--hf-model-name", default="tchewik/isanlp_rst_v3")
    parser.add_argument("--hf-model-version", default="unirst")
    parser.add_argument("--output-mode", choices=("separate", "paired", "both"), default="both")
    parser.add_argument("--force-segmentation", action="store_true")
    parser.add_argument("--skip-hc3", action="store_true")
    parser.add_argument("--skip-mage", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu_id < 0 or args.gpu_id >= args.total_gpus:
        raise ValueError("--gpu-id must be in [0, --total-gpus)")
    segments_dir = args.segments_dir or os.path.join(args.output_dir, "segments-deu.rst.pcc")
    splitter = SceneSplitter()
    adapter = UniRSTAdapter(
        segment_relinventory=args.segment_relinventory,
        relation_inventories=args.relinventories,
        hf_model_name=args.hf_model_name,
        hf_model_version=args.hf_model_version,
        cuda_device=args.gpu_id if torch.cuda.is_available() else -1,
    )

    # HC3 is not sharded across workers, so only the designated worker may
    # process it during a multi-GPU MAGE run.  This avoids duplicate parsing
    # and concurrent writes to the shared HC3 cache and output files.
    if not args.skip_hc3 and args.gpu_id == 0:
        for source, documents in hc3_groups(args.hc3_dataset, splitter, args.min_char_len).items():
            run_group(
                documents,
                adapter,
                segments_dir,
                args.output_dir,
                f"hc3_{safe_name(source)}",
                f"hc3_{safe_name(source)}",
                args.force_segmentation,
                args.output_mode,
            )

    if not args.skip_mage:
        raw_mage = load_dataset(args.mage_dataset)
        for split in ("train", "validation", "test"):
            documents = mage_group(
                raw_mage, split, splitter, args.total_gpus, args.gpu_id
            )
            run_group(
                documents,
                adapter,
                segments_dir,
                args.output_dir,
                f"mage_{split}_{args.gpu_id}",
                f"mage_{split}_gpu{args.gpu_id}",
                args.force_segmentation,
                args.output_mode,
            )


if __name__ == "__main__":
    main()
