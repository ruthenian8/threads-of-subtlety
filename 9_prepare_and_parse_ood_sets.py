"""Parse MAGE OOD documents and add UniRST motif distributions.

Motif catalogs are now generated independently for every relation inventory,
so an inventory-specific motif directory can be selected with ``--relinventory``.
The old ``data/mage`` input/output layout remains the default.
"""

import argparse
import os
import random
import re
from collections import OrderedDict

import pandas as pd
from rich.progress import track
from sentsplit.segment import SentSplit
from transformers import AutoTokenizer

from tos.tos_dataset import Document, ToSDataset
from tos.unirst import UniRSTAdapter
from tos.tos_utils import (
    load_json,
    resolve_selected_motif_hashes,
    validate_selected_motif_hashes,
)

random.seed(42)


def _safe_inventory_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


class SceneSplitter:
    """Use the same scene boundaries as the UniRST preparation workflow."""

    def __init__(self, max_tokens_margin=20):
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
        self.max_tokens = self.tokenizer.model_max_length - max_tokens_margin
        self.sent_splitter = SentSplit("en")

    def split(self, document):
        scenes, current, current_tokens = [], "", 0
        for paragraph in document.split("\n\n"):
            paragraph_tokens = len(self.tokenizer.tokenize(paragraph))
            if paragraph_tokens > self.max_tokens:
                pieces = self.sent_splitter.segment(paragraph, strip_spaces=False)
                for piece in pieces:
                    piece_tokens = len(self.tokenizer.tokenize(piece))
                    if current_tokens + piece_tokens > self.max_tokens:
                        if current.strip():
                            scenes.append(current.strip())
                        current, current_tokens = piece, piece_tokens
                    else:
                        current += piece
                        current_tokens += piece_tokens
                if current.strip():
                    scenes.append(current.strip())
                current, current_tokens = "", 0
                continue
            if current_tokens + paragraph_tokens > self.max_tokens:
                if current.strip():
                    scenes.append(current.strip())
                current, current_tokens = paragraph, paragraph_tokens
            else:
                current = f"{current}\n\n{paragraph}" if current else paragraph
                current_tokens += paragraph_tokens
        if current.strip():
            scenes.append(current.strip())
        return scenes


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


def parse_unirst_document(text, source, label, adapter, parser, splitter, motifs):
    scenes = splitter.split(text)
    trees = OrderedDict()
    kept_scenes = []
    for scene_text in scenes:
        segmented = adapter.segment_scene(scene_text)
        if len(segmented["edus"]) < 2:
            continue
        result = parser.from_edus(segmented["edus"])
        trees[len(kept_scenes)] = adapter.scene_tree(
            scene_text, segmented["edus"], result
        )
        kept_scenes.append(scene_text)
    if not trees:
        return None
    document = Document(
        text=text,
        scenes=kept_scenes,
        scene_discourse_trees=trees,
        source=source,
        label=label,
    )
    document = ToSDataset.add_discourse_graphs_to_document(document)
    document = ToSDataset.add_motif_distributions_to_document(
        document,
        **{f"m{size}_motifs": motifs[size] for size in motifs},
    )
    return document


def process_df(df, tos_dataset=None, *, adapter=None, parser=None, splitter=None, motifs=None):
    dataset = []
    for _, row in track(df.iterrows(), total=len(df), description="Processing..."):
        if adapter is not None:
            document = parse_unirst_document(
                row["text"], row.get("src"), row.get("label"),
                adapter, parser, splitter, motifs,
            )
        else:
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
            segment_relinventory=args.relinventory,
            relation_inventories=(args.relinventory,),
            cuda_device=args.gpu_id if args.gpu_id is not None else -1,
        )
        unirst_parser = adapter.make_parser(args.relinventory)
        splitter = SceneSplitter()
        motifs = load_unirst_motifs(motif_dir, dataset_name, motif_sizes)
        tos_dataset = None
    else:
        adapter = unirst_parser = splitter = motifs = None
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
    for name, input_path in input_paths.items():
        if not os.path.exists(input_path):
            raise FileNotFoundError(input_path)
        df = pd.read_csv(input_path, header=0)
        parsed = process_df(
            df, tos_dataset, adapter=adapter, parser=unirst_parser,
            splitter=splitter, motifs=motifs,
        )
        output_path = os.path.join(
            output_dir,
            f"test_ood_set_{name}.discourse_parsed.graph_added.{output_suffix}.jsonl",
        )
        ToSDataset.save_dataset_as_jsonl(parsed, output_path)
        print(f"wrote {len(parsed)} documents to {output_path}")


if __name__ == "__main__":
    main()
