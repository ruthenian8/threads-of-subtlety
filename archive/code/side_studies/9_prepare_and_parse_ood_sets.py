"""Parse MAGE OOD documents and add UniRST motif distributions.

Motif catalogs are now generated independently for every relation inventory,
so an inventory-specific motif directory can be selected with ``--relinventory``.
The old ``data/mage`` input/output layout remains the default.
"""

import argparse
import gc
import os
import random
import re
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd
from rich.progress import track
from tqdm.auto import tqdm

from tos.tos_dataset import Document, SceneDiscourseTree, ToSDataset
from tos.unirst import (
    SceneSplitter,
    UniRSTAdapter,
    build_scene_lookup,
    load_or_create_segmentation_cache,
)
from tos.tos_utils import (
    load_json,
    resolve_selected_motif_hashes,
    validate_selected_motif_hashes,
)

random.seed(42)


def _safe_inventory_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def load_unirst_motifs(motif_dir, dataset_name, motif_sizes):
    manifest_path = resolve_selected_motif_hashes(motif_dir, dataset_name=dataset_name)
    selected = load_json(manifest_path)
    validate_selected_motif_hashes(
        motif_dir, selected, dataset_name=dataset_name, sizes=motif_sizes
    )
    return {
        size: ToSDataset.load_motifs(
            os.path.join(motif_dir, f"{dataset_name}_M{size}_motifs.json"),
            selected[f"m{size}"],
        )
        for size in motif_sizes
    }


def make_document_specs(df, split: str, splitter: SceneSplitter) -> List[Dict[str, Any]]:
    """Convert an OOD frame to script-0-compatible document specifications."""
    documents = []
    for document_index, row in tqdm(
        df.iterrows(), total=len(df), desc=f"preparing MAGE {split}", unit="document"
    ):
        text = row["text"]
        documents.append(
            {
                "dataset": "mage",
                "split": split,
                "document_index": int(document_index),
                "text": text,
                "source": row.get("src"),
                "label": row.get("label"),
                "scenes": splitter.split(text),
            }
        )
    return documents


def make_scene_specs(documents: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    scene_specs = []
    for document in documents:
        for scene_index, text in enumerate(document["scenes"]):
            scene_specs.append(
                {
                    "scene_key": ":".join(
                        (
                            document["dataset"],
                            document["split"],
                            str(document["document_index"]),
                            str(scene_index),
                        )
                    ),
                    "dataset": document["dataset"],
                    "split": document["split"],
                    "document_index": document["document_index"],
                    "scene_index": scene_index,
                    "text": text,
                }
            )
    return scene_specs


def parse_predictions(adapter, scenes, relinventory):
    """Mirror script 0: record per-scene errors instead of aborting the run."""
    predictions = {}
    parser = adapter.make_parser(relinventory)
    try:
        for scene in tqdm(
            scenes,
            desc=f"parsing {relinventory}",
            unit="scene",
            total=len(scenes),
        ):
            if scene["status"] != "ok":
                predictions[scene["scene_key"]] = {
                    "status": "segmentation_error",
                    "parsed": "NONE",
                    "error": scene["error"],
                }
                continue
            try:
                result = parser.from_edus(scene["edus"])
                predictions[scene["scene_key"]] = {
                    "status": "ok",
                    "parsed": adapter.to_constituency_format(result, len(scene["edus"])),
                    "error": None,
                }
            except Exception as exc:
                predictions[scene["scene_key"]] = {
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


def feature_documents(documents, scenes, predictions, motifs):
    """Assemble graph/motif documents, filtering failed trees like script 1."""
    scene_lookup = build_scene_lookup(scenes)
    output = []
    skipped_scenes = 0
    for document_spec in documents:
        kept_scenes = []
        trees = {}
        for scene_index, scene_text in enumerate(document_spec["scenes"]):
            scene_key = ":".join(
                (
                    document_spec["dataset"],
                    document_spec["split"],
                    str(document_spec["document_index"]),
                    str(scene_index),
                )
            )
            record = scene_lookup[scene_key]
            prediction = predictions[scene_key]
            if record["status"] != "ok" or prediction["parsed"] in {"NONE", ""}:
                skipped_scenes += 1
                continue
            edus = {
                f"span_{index}-{index}": edu
                for index, edu in enumerate(record["edus"], start=1)
            }
            trees[len(trees)] = SceneDiscourseTree(
                text=scene_text,
                tokenized=record["tokenized"],
                segments=record["segments"],
                edus=edus,
                parsed=prediction["parsed"],
                graph_dict=None,
                graph_networkx=None,
                motif_dists=None,
            )
            kept_scenes.append(scene_text)
        if not trees:
            continue
        document = Document(
            text=document_spec["text"],
            scenes=kept_scenes,
            scene_discourse_trees=trees,
            source=document_spec["source"],
            label=document_spec["label"],
        )
        document = ToSDataset.add_discourse_graphs_to_document(document)
        document = ToSDataset.add_motif_distributions_to_document(
            document,
            **{f"m{size}_motifs": motifs[size] for size in motifs},
        )
        output.append(document)
    return output, skipped_scenes


def process_unirst_df(
    df,
    split,
    adapter,
    relinventory,
    splitter,
    motifs,
    cache_path,
    force_segmentation=False,
):
    documents = make_document_specs(df, split, splitter)
    scene_specs = make_scene_specs(documents)
    scenes = load_or_create_segmentation_cache(
        adapter,
        scene_specs,
        cache_path,
        force=force_segmentation,
        progress_desc=f"segmenting mage_{split}",
    )
    adapter.release_segmenter()
    predictions = parse_predictions(adapter, scenes, relinventory)
    parsed, skipped_scenes = feature_documents(documents, scenes, predictions, motifs)
    failed_documents = len(documents) - len(parsed)
    tqdm.write(
        f"mage_{split}: skipped {skipped_scenes} failed/empty scenes and "
        f"{failed_documents} documents without a valid tree"
    )
    return parsed


def process_df(df, tos_dataset):
    dataset = []
    for _, row in track(df.iterrows(), total=len(df), description="Processing..."):
        document = tos_dataset.parse_document_discourse(
            document=row["text"], source=row.get("src"), label=row.get("label"),
            filter_none=True, add_graph=True, add_motif_dists=True,
        )
        if document is not None:
            dataset.append(document)
    return dataset


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--input-dir", help="directory containing MAGE OOD CSV files")
    parser.add_argument("--output-dir", help="where parsed OOD JSONL files are written")
    parser.add_argument(
        "--parser-dir",
        default="/home/ubuntu/Development/DMRST_Parser",
        help="DMRST parser checkout (defaults to the historical location)",
    )
    parser.add_argument("--motif-dir")
    parser.add_argument("--relinventory")
    parser.add_argument("--segment-relinventory", default="deu.rst.pcc")
    parser.add_argument("--segments-dir")
    parser.add_argument("--hf-model-name", default="tchewik/isanlp_rst_v3")
    parser.add_argument("--hf-model-version", default="unirst")
    parser.add_argument("--force-segmentation", action="store_true")
    parser.add_argument("--force-parsing", action="store_true")
    parser.add_argument("--dataset-name")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--motif-sizes")
    args = parser.parse_args(argv)

    input_dir = args.input_dir or os.path.join(args.data_dir, "mage")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            os.path.join(args.data_dir, "unirst", f"rel-{_safe_inventory_name(args.relinventory)}")
            if args.relinventory
            else input_dir
        )
    motif_dir = args.motif_dir
    if motif_dir is None:
        motif_dir = os.path.join(args.data_dir, "motifs")
        if args.relinventory:
            motif_dir = os.path.join(motif_dir, _safe_inventory_name(args.relinventory))
    dataset_name = args.dataset_name or ("hc3" if args.relinventory else "hc3-mage")
    motif_sizes = tuple(
        int(size) for size in (args.motif_sizes or ("3,6" if args.relinventory else "3,6,9")).split(",") if size
    )

    input_paths = {
        "gpt": os.path.join(input_dir, "test_ood_set_gpt.csv"),
        "gpt_para": os.path.join(input_dir, "test_ood_set_gpt_para.csv"),
    }
    if args.relinventory:
        adapter = UniRSTAdapter(
            segment_relinventory=args.segment_relinventory,
            relation_inventories=(args.relinventory,),
            hf_model_name=args.hf_model_name,
            hf_model_version=args.hf_model_version,
            cuda_device=args.gpu_id if args.gpu_id is not None else -1,
        )
        splitter = SceneSplitter()
        motifs = load_unirst_motifs(motif_dir, dataset_name, motif_sizes)
        tos_dataset = None
    else:
        adapter = splitter = motifs = None
        tos_dataset = ToSDataset(
            dmrst_parser_dir=args.parser_dir,
            batch_size=args.batch_size,
            gpu_id=args.gpu_id,
            motif_dir=motif_dir,
            dataset_name=dataset_name,
            motif_sizes=motif_sizes,
        )
    os.makedirs(output_dir, exist_ok=True)
    output_suffix = "m3_m6_motif_dists" if args.relinventory else "motif_dists"
    segments_dir = args.segments_dir or os.path.join(
        args.data_dir,
        "unirst",
        f"segments-{_safe_inventory_name(args.segment_relinventory)}",
    )
    for name, input_path in input_paths.items():
        if not os.path.exists(input_path):
            raise FileNotFoundError(input_path)
        output_path = os.path.join(
            output_dir,
            f"test_ood_set_{name}.discourse_parsed.graph_added.{output_suffix}.jsonl",
        )
        if os.path.exists(output_path) and not (
            args.force_segmentation or args.force_parsing
        ):
            print(f"skipping {name}: output exists at {output_path}")
            continue
        df = pd.read_csv(input_path, header=0)
        if args.relinventory:
            cache_path = os.path.join(segments_dir, f"mage_ood_{name}.pkl")
            parsed = process_unirst_df(
                df,
                name,
                adapter,
                args.relinventory,
                splitter,
                motifs,
                cache_path,
                force_segmentation=args.force_segmentation,
            )
        else:
            parsed = process_df(df, tos_dataset)
        ToSDataset.save_dataset_as_jsonl(parsed, output_path)
        print(f"wrote {len(parsed)} documents to {output_path}")


if __name__ == "__main__":
    main()
