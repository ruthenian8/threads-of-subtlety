import os
import argparse
import json

import evaluate
import torch
from accelerate import Accelerator
from rich.progress import track
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LongformerForSequenceClassification

from tos.tos_dataset import LongformerDataCollator, LongformerDataset
from tos.tos_models import LongformerWithMotifsForSequenceClassification


def _evaluation_assets(model_path, base_model_path=None):
    """Find training-time backbone/tokenizer metadata near a checkpoint."""
    checkpoint = os.path.abspath(model_path)
    candidates = [checkpoint, os.path.dirname(checkpoint), os.path.dirname(os.path.dirname(checkpoint))]
    resolved_backbone = base_model_path
    for directory in candidates:
        metadata_path = os.path.join(directory, "backbone.json")
        if resolved_backbone is None and os.path.exists(metadata_path):
            with open(metadata_path) as handle:
                resolved_backbone = json.load(handle).get("base_model_path")
            break
    resolved_backbone = resolved_backbone or "allenai/longformer-base-4096"
    tokenizer_source = resolved_backbone
    for directory in candidates:
        candidate = os.path.join(directory, "tokenizer")
        if os.path.isdir(candidate):
            tokenizer_source = candidate
            break
    return resolved_backbone, tokenizer_source


def evaluate_longformer(
    testset_name: str,
    model_path: str,
    add_motif: bool,
    data_dir: str = "data",
    relinventory: str = None,
    motif_sizes=(3, 6),
    base_model_path: str = None,
):
    accelerator = Accelerator()
    accelerator.print(f"\n\n------{testset_name}-------")

    backbone_path, tokenizer_path = _evaluation_assets(model_path, base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    metric = evaluate.combine(
        ["accuracy", "f1", "precision", "recall", "BucketHeadP65/confusion_matrix"]
    )

    test_dataset = LongformerDataset(
        split=testset_name,
        shuffle=False,
        saved_dir=data_dir,
        data_dir=data_dir,
        relinventory=relinventory,
        motif_sizes=motif_sizes,
    )

    if add_motif:
        model = LongformerWithMotifsForSequenceClassification(
            base_model_path=backbone_path,
            motif_dims=test_dataset.motif_dims
        )
        state_dict = load_file(os.path.join(model_path, "model.safetensors"))
        model.load_state_dict(state_dict)
    else:
        model = LongformerForSequenceClassification.from_pretrained(
            model_path, num_labels=2
        )

    data_loader = DataLoader(
        test_dataset,
        batch_size=32,
        collate_fn=LongformerDataCollator(
            tokenizer=tokenizer, padding="longest", add_motif=add_motif
        ),
    )

    model, data_loader = accelerator.prepare(model, data_loader)

    for data in track(data_loader, total=len(data_loader), description="Evaluating..."):
        targets = data["labels"]
        predictions = model(**data).logits

        predictions = torch.argmax(predictions, dim=1)
        all_predictions, all_targets = accelerator.gather_for_metrics(
            (predictions, targets)
        )
        metric.add_batch(predictions=all_predictions, references=all_targets)

    accelerator.print(metric.evaluation_modules[0].__len__())
    accelerator.print(metric.compute())
    accelerator.print("-----------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--relinventory")
    parser.add_argument("--motif", action="store_true")
    parser.add_argument("--motif-sizes", default="3,6")
    parser.add_argument("--base-model-path")
    args = parser.parse_args()
    motif_sizes = tuple(int(size) for size in args.motif_sizes.split(",") if size)
    evaluate_longformer(
        args.split,
        model_path=args.model_path,
        add_motif=args.motif,
        data_dir=args.data_dir,
        relinventory=args.relinventory,
        motif_sizes=motif_sizes,
        base_model_path=args.base_model_path,
    )
