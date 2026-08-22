import argparse
import json
import os
import random

from transformers import (
    AutoTokenizer,
    LongformerForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from tos.tos_dataset import LongformerDataCollator, LongformerDataset
from tos.tos_models import LongformerWithMotifsForSequenceClassification
from tos.tos_utils import compute_metrics

random.seed(42)


def _save_training_assets(output_dir, tokenizer, base_model_path):
    """Persist the tokenizer/backbone identity beside Trainer checkpoints."""
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    with open(os.path.join(output_dir, "backbone.json"), "w") as handle:
        json.dump({"base_model_path": base_model_path}, handle, indent=2)


def train_longformer_plain(
    model_name: str,
    base_model_path: str = "allenai/longformer-base-4096",
    num_labels=2,
    data_dir: str = "data",
    relinventory: str = None,
    motif_sizes=(3, 6),
):
    model = LongformerForSequenceClassification.from_pretrained(
        base_model_path, num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, use_fast=True
    )
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = LongformerDataset(
        split="train", shuffle=True, saved_dir=data_dir,
        data_dir=data_dir, relinventory=relinventory, motif_sizes=motif_sizes,
    )
    print(f"train_dataset loaded with {len(train_dataset)} samples.")

    valid_dataset = LongformerDataset(
        split="valid", shuffle=True, saved_dir=data_dir,
        data_dir=data_dir, relinventory=relinventory, motif_sizes=motif_sizes,
    )
    print(f"valid_dataset loaded with {len(valid_dataset)} samples.")

    training_args = TrainingArguments(
        # use_mps_device=True,
        eval_steps=100,
        save_steps=100,
        logging_steps=10,
        fp16=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        load_best_model_at_end=True,
        logging_dir=f"./models/{model_name}",  # directory for storing logs
        logging_strategy="steps",
        num_train_epochs=4,  # total number of training epochs
        output_dir=f"./results/{model_name}",  # output directory
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        remove_unused_columns=False,
        save_total_limit=3,
        weight_decay=4e-5,
    )
    _save_training_assets(training_args.output_dir, tokenizer, base_model_path)

    trainer = Trainer(
        args=training_args,
        data_collator=LongformerDataCollator(
            tokenizer=tokenizer, padding="longest", add_motif=False
        ),
        eval_dataset=valid_dataset,
        model=model,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return model


def train_longformer_motif(
    model_name: str,
    base_model_path: str = "allenai/longformer-base-4096",
    num_labels=2,
    data_dir: str = "data",
    relinventory: str = None,
    motif_sizes=(3, 6),
):
    train_dataset = LongformerDataset(
        split="train", shuffle=True, saved_dir=data_dir,
        data_dir=data_dir, relinventory=relinventory, motif_sizes=motif_sizes,
    )
    model = LongformerWithMotifsForSequenceClassification(
        base_model_path, num_labels=num_labels, motif_dims=train_dataset.motif_dims
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, use_fast=True
    )

    print(f"train_dataset loaded with {len(train_dataset)} samples.")

    valid_dataset = LongformerDataset(
        split="valid", shuffle=True, saved_dir=data_dir,
        data_dir=data_dir, relinventory=relinventory, motif_sizes=motif_sizes,
    )
    print(f"valid_dataset loaded with {len(valid_dataset)} samples.")

    training_args = TrainingArguments(
        eval_steps=100,
        save_steps=100,
        logging_steps=10,
        fp16=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        load_best_model_at_end=True,
        logging_dir=f"./models/{model_name}",  # directory for storing logs
        logging_strategy="steps",
        num_train_epochs=4,  # total number of training epochs
        output_dir=f"./results/{model_name}",  # output directory
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        remove_unused_columns=False,
        save_total_limit=3,
        weight_decay=4e-5,
    )
    _save_training_assets(training_args.output_dir, tokenizer, base_model_path)

    trainer = Trainer(
        args=training_args,
        data_collator=LongformerDataCollator(
            tokenizer=tokenizer, padding="longest", add_motif=True
        ),
        eval_dataset=valid_dataset,
        model=model,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="longformer_base_motif")
    parser.add_argument("--base-model-path", default="allenai/longformer-base-4096")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--relinventory")
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--motif-sizes", default="3,6")
    args = parser.parse_args()
    motif_sizes = tuple(int(size) for size in args.motif_sizes.split(",") if size)
    train_fn = train_longformer_plain if args.plain else train_longformer_motif
    train_fn(
        model_name=args.model_name,
        base_model_path=args.base_model_path,
        data_dir=args.data_dir,
        relinventory=args.relinventory,
        motif_sizes=motif_sizes,
    )
