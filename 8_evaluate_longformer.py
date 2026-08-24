import os
import argparse
import json
import re

import evaluate
import torch
from accelerate import Accelerator
from rich.progress import track
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, LongformerForSequenceClassification

from tos.tos_dataset import LongformerDataCollator, LongformerDataset
from tos.tos_models import LongformerWithMotifsForSequenceClassification


class IndexedDataset(Dataset):
    """Attach stable dataset indices without changing cached dataset records."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index]


def _indexed_collator(collator):
    def collate(indexed_features):
        sample_indices, features = zip(*indexed_features)
        batch = collator(list(features))
        batch["sample_indices"] = torch.tensor(sample_indices, dtype=torch.long)
        return batch

    return collate


def _default_predictions_path(model_path, split, relinventory):
    inventory = (
        f".{re.sub(r'[^A-Za-z0-9_.-]+', '_', relinventory)}"
        if relinventory
        else ""
    )
    return os.path.join(model_path, f"predictions.{split}{inventory}.jsonl")


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
    predictions_path: str = None,
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
        IndexedDataset(test_dataset),
        batch_size=32,
        collate_fn=_indexed_collator(
            LongformerDataCollator(
                tokenizer=tokenizer, padding="longest", add_motif=add_motif
            )
        ),
    )

    model, data_loader = accelerator.prepare(model, data_loader)

    predictions_path = predictions_path or _default_predictions_path(
        model_path, testset_name, relinventory
    )
    partial_path = f"{predictions_path}.partial"
    prediction_handle = None
    written_indices = set()
    completed = False
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(os.path.abspath(predictions_path)), exist_ok=True)
        prediction_handle = open(partial_path, "w", encoding="utf-8")
    try:
        for data in track(data_loader, total=len(data_loader), description="Evaluating..."):
            sample_indices = data.pop("sample_indices")
            targets = data["labels"]
            predictions = torch.argmax(model(**data).logits, dim=1)

            all_indices, all_predictions, all_targets = accelerator.gather_for_metrics(
                (sample_indices, predictions, targets)
            )
            metric.add_batch(predictions=all_predictions, references=all_targets)

            if accelerator.is_main_process:
                records = sorted(
                    zip(
                        all_indices.detach().cpu().tolist(),
                        all_predictions.detach().cpu().tolist(),
                        all_targets.detach().cpu().tolist(),
                    )
                )
                for sample_index, predicted_label, target_label in records:
                    if sample_index in written_indices:
                        continue
                    sample = test_dataset[sample_index]
                    prediction_handle.write(
                        json.dumps(
                            {
                                "sample_index": sample_index,
                                "predicted_label": predicted_label,
                                "target_label": target_label,
                                "text": sample["text"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written_indices.add(sample_index)
                prediction_handle.flush()
        completed = True
    finally:
        if prediction_handle is not None:
            prediction_handle.flush()
            os.fsync(prediction_handle.fileno())
            prediction_handle.close()

    accelerator.wait_for_everyone()
    results = metric.compute()
    if accelerator.is_main_process and completed:
        if len(written_indices) != len(test_dataset):
            raise RuntimeError(
                f"Persisted {len(written_indices)} predictions for "
                f"{len(test_dataset)} evaluation samples"
            )
        os.replace(partial_path, predictions_path)
        accelerator.print(f"Predictions saved to {predictions_path}")

    accelerator.print(metric.evaluation_modules[0].__len__())
    accelerator.print(results)
    accelerator.print("-----------------------")
    return results, predictions_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--relinventory")
    parser.add_argument("--motif", action="store_true")
    parser.add_argument("--motif-sizes", default="3,6")
    parser.add_argument("--base-model-path")
    parser.add_argument(
        "--predictions-path",
        help="JSONL output path (defaults to the model checkpoint directory)",
    )
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
        predictions_path=args.predictions_path,
    )
