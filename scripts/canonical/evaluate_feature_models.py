#!/usr/bin/env python3
"""Evaluate matched structural, relation, and M3 models on aligned HC3 splits.

Every candidate is fitted on the released training split. Hyperparameters and
the classification threshold are selected on validation only, after which the
selected (unchanged) model is evaluated once on test. Results, predictions,
and fitted estimators are checkpointed after every inventory/config/model run.
"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier


INVENTORIES = (
    "deu.rst.pcc",
    "eng.erst.gum.new",
    "eng.rst.rstdt.new",
    "nld.rst.nldt",
)
CONFIGS = (
    "structural",
    "relation_frequency",
    "m3_presence",
    "m3_frequency",
    "m3_depth",
)
MODELS = ("lr", "xgb")
MATRIX_KEYS = {
    "structural": ("X_structural", "structural_names"),
    "relation_frequency": ("X_relation_frequency", "relation_names"),
    "m3_presence": ("X_m3_presence", "m3_hashes"),
    "m3_frequency": ("X_m3_frequency", "m3_hashes"),
    "m3_depth": ("X_m3_depth", "m3_hashes"),
}


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_dump_model(path: Path, model: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        joblib.dump(model, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_feature_split(
    feature_root: Path, inventory: str, split: str, config: str
) -> Dict[str, np.ndarray]:
    matrix_key, names_key = MATRIX_KEYS[config]
    path = feature_root / inventory / f"hc3_{split}.features.npz"
    with np.load(path, allow_pickle=False) as matrix:
        result = {
            "X": np.asarray(matrix[matrix_key], dtype=np.float32),
            "y": np.asarray(matrix["y"], dtype=np.int8),
            "identity_id": np.asarray(matrix["identity_id"]),
            "identity_hash": np.asarray(matrix["identity_hash"]),
            "source": np.asarray(matrix["source"]),
            "domain": np.asarray(matrix["domain"]),
            "feature_names": np.asarray(matrix[names_key]),
        }
    if not np.isfinite(result["X"]).all():
        raise ValueError(f"Non-finite values in {path}:{matrix_key}")
    return result


def parameter_grid(kind: str, y_train: np.ndarray) -> List[Dict[str, Any]]:
    if kind == "lr":
        return [{"C": value} for value in (0.01, 0.1, 1.0, 10.0)]
    negatives = int(np.count_nonzero(y_train == 0))
    positives = int(np.count_nonzero(y_train == 1))
    scale_pos_weight = negatives / max(positives, 1)
    return [
        {
            "max_depth": depth,
            "min_child_weight": child_weight,
            "n_estimators": 300,
            "scale_pos_weight": scale_pos_weight,
        }
        for depth, child_weight in itertools.product((2, 4), (1, 5))
    ]


def make_model(kind: str, params: Mapping[str, Any], seed: int, n_jobs: int):
    if kind == "lr":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(params["C"]),
                class_weight="balanced",
                max_iter=5000,
                random_state=seed,
            ),
        )
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=int(params["max_depth"]),
        min_child_weight=float(params["min_child_weight"]),
        n_estimators=int(params["n_estimators"]),
        scale_pos_weight=float(params["scale_pos_weight"]),
        random_state=seed,
        n_jobs=n_jobs,
    )


def probabilities(model: Any, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)


def select_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    order = np.argsort(probability, kind="stable")
    sorted_y = y[order]
    sorted_probability = probability[order]
    negatives = int(np.count_nonzero(y == 0))
    positives = int(np.count_nonzero(y == 1))
    if not negatives or not positives:
        raise ValueError("Threshold selection requires both labels")

    # Initially every row is predicted positive. Moving a threshold past a
    # tied probability changes all rows in that tie together.
    true_negatives = 0
    true_positives = positives
    candidates: List[Tuple[float, float]] = [
        (float(np.nextafter(sorted_probability[0], -np.inf)), 0.5)
    ]
    index = 0
    while index < len(y):
        end = index + 1
        while (
            end < len(y)
            and sorted_probability[end] == sorted_probability[index]
        ):
            end += 1
        group = sorted_y[index:end]
        true_negatives += int(np.count_nonzero(group == 0))
        true_positives -= int(np.count_nonzero(group == 1))
        threshold = float(np.nextafter(sorted_probability[index], np.inf))
        score = 0.5 * (
            true_negatives / negatives + true_positives / positives
        )
        candidates.append((threshold, score))
        index = end
    best_score = max(score for _, score in candidates)
    tied = [threshold for threshold, score in candidates if score == best_score]
    return float(tied[len(tied) // 2])


def metrics(
    y: np.ndarray, probability: np.ndarray, threshold: float
) -> Dict[str, Any]:
    prediction = (probability >= threshold).astype(np.int8)
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "accuracy": float(accuracy_score(y, prediction)),
        "recall_generated": float(
            recall_score(y, prediction, pos_label=0, zero_division=0)
        ),
        "recall_human": float(
            recall_score(y, prediction, pos_label=1, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }


def fit_selected_model(
    kind: str,
    train: Mapping[str, np.ndarray],
    validation: Mapping[str, np.ndarray],
    seed: int,
    n_jobs: int,
    description: str,
) -> Tuple[Any, Dict[str, Any], np.ndarray, float]:
    candidates = []
    best: Tuple[float, float, str, Any, Dict[str, Any], np.ndarray] | None = None
    grid = parameter_grid(kind, train["y"])
    for params in tqdm(grid, desc=description, unit="candidate", leave=False):
        model = make_model(kind, params, seed, n_jobs)
        model.fit(train["X"], train["y"])
        validation_probability = probabilities(model, validation["X"])
        roc_auc = float(roc_auc_score(validation["y"], validation_probability))
        average_precision = float(
            average_precision_score(validation["y"], validation_probability)
        )
        candidate = {
            "parameters": dict(params),
            "validation_roc_auc": roc_auc,
            "validation_average_precision": average_precision,
        }
        candidates.append(candidate)
        # The serialized parameters make exact ties deterministic.
        rank = (roc_auc, average_precision, json.dumps(params, sort_keys=True))
        if best is None or rank[:2] > best[:2] or (
            rank[:2] == best[:2] and rank[2] < best[2]
        ):
            best = (*rank, model, candidate, validation_probability)
    if best is None:
        raise ValueError("Empty hyperparameter grid")
    _, _, _, model, selected, validation_probability = best
    threshold = select_threshold(validation["y"], validation_probability)
    selection = {
        "primary_metric": "validation_roc_auc",
        "tie_break_metric": "validation_average_precision",
        "selected_parameters": selected["parameters"],
        "selected_validation_roc_auc": selected["validation_roc_auc"],
        "selected_validation_average_precision": selected[
            "validation_average_precision"
        ],
        "validation_selected_threshold": threshold,
        "candidates": candidates,
    }
    return model, selection, validation_probability, threshold


@lru_cache(maxsize=None)
def provenance_hashes(feature_root: Path, inventory: str) -> Dict[str, str]:
    result = {}
    for split in ("train", "validation", "test"):
        path = feature_root / inventory / f"hc3_{split}.features.npz"
        result[split] = sha256_file(path)
    return result


def run_experiment(
    args: argparse.Namespace,
    inventory: str,
    config: str,
    kind: str,
) -> Dict[str, Any]:
    feature_root = Path(args.feature_dir)
    result_dir = Path(args.output_dir) / inventory / config / kind
    metrics_path = result_dir / "metrics.json"
    predictions_path = result_dir / "predictions.npz"
    model_path = result_dir / "model.joblib"
    current_hashes = provenance_hashes(feature_root, inventory)
    if (
        not args.force
        and metrics_path.exists()
        and predictions_path.exists()
        and model_path.exists()
    ):
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            existing.get("seed") == args.seed
            and existing.get("feature_matrix_sha256") == current_hashes
        ):
            print(f"skipping complete {inventory}/{config}/{kind}")
            return existing
        print(f"rebuilding stale {inventory}/{config}/{kind}")

    splits = {
        split: load_feature_split(feature_root, inventory, split, config)
        for split in ("train", "validation", "test")
    }
    names = splits["train"]["feature_names"]
    for split in ("validation", "test"):
        if not np.array_equal(names, splits[split]["feature_names"]):
            raise ValueError(f"Feature mismatch for {inventory}/{config}/{split}")

    model, selection, validation_probability, threshold = fit_selected_model(
        kind,
        splits["train"],
        splits["validation"],
        args.seed,
        args.n_jobs,
        f"{inventory} {config}/{kind}",
    )
    # This is the sole access to test features during a run, after all model
    # and threshold choices have been fixed from train/validation.
    test_probability = probabilities(model, splits["test"]["X"])
    result: Dict[str, Any] = {
        "inventory": inventory,
        "feature_config": config,
        "model": kind,
        "seed": args.seed,
        "protocol": {
            "fit": "train only",
            "model_selection": "validation ROC-AUC; AP tie-break",
            "threshold_selection": "validation balanced accuracy",
            "test_usage": "single evaluation after all choices were fixed",
        },
        "records": {
            split: int(len(splits[split]["y"])) for split in splits
        },
        "feature_count": int(splits["train"]["X"].shape[1]),
        "feature_matrix_sha256": current_hashes,
        "selection": selection,
        "validation_metrics": metrics(
            splits["validation"]["y"], validation_probability, threshold
        ),
        "test_metrics": metrics(splits["test"]["y"], test_probability, threshold),
    }
    atomic_save_npz(
        predictions_path,
        validation_identity_id=splits["validation"]["identity_id"],
        validation_identity_hash=splits["validation"]["identity_hash"],
        validation_y=splits["validation"]["y"],
        validation_probability=validation_probability,
        test_identity_id=splits["test"]["identity_id"],
        test_identity_hash=splits["test"]["identity_hash"],
        test_y=splits["test"]["y"],
        test_probability=test_probability,
        threshold=np.asarray(threshold),
    )
    atomic_dump_model(model_path, model)
    result["artifacts"] = {
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": sha256_file(predictions_path),
        "model": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
    }
    atomic_write_json(metrics_path, result)
    return result


def discover_results(output_dir: Path) -> List[Dict[str, Any]]:
    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in output_dir.glob("*/*/*/metrics.json")
    ]
    return sorted(
        results,
        key=lambda value: (
            value["inventory"], value["feature_config"], value["model"]
        ),
    )


def write_summary(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "inventory",
        "feature_config",
        "model",
        "feature_count",
        "selected_parameters",
        "threshold",
        "validation_roc_auc",
        "validation_average_precision",
        "validation_balanced_accuracy",
        "validation_f1",
        "validation_macro_f1",
        "test_roc_auc",
        "test_average_precision",
        "test_balanced_accuracy",
        "test_f1",
        "test_macro_f1",
        "test_accuracy",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for result in results:
        validation = result["validation_metrics"]
        test = result["test_metrics"]
        writer.writerow(
            {
                "inventory": result["inventory"],
                "feature_config": result["feature_config"],
                "model": result["model"],
                "feature_count": result["feature_count"],
                "selected_parameters": json.dumps(
                    result["selection"]["selected_parameters"], sort_keys=True
                ),
                "threshold": result["selection"][
                    "validation_selected_threshold"
                ],
                "validation_roc_auc": validation["roc_auc"],
                "validation_average_precision": validation["average_precision"],
                "validation_balanced_accuracy": validation["balanced_accuracy"],
                "validation_f1": validation["f1"],
                "validation_macro_f1": validation["macro_f1"],
                "test_roc_auc": test["roc_auc"],
                "test_average_precision": test["average_precision"],
                "test_balanced_accuracy": test["balanced_accuracy"],
                "test_f1": test["f1"],
                "test_macro_f1": test["macro_f1"],
                "test_accuracy": test["accuracy"],
            }
        )
    atomic_write_text(path, output.getvalue())


def checkpoint_index(output_dir: Path, feature_dir: Path) -> None:
    results = discover_results(output_dir)
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "feature_dir": str(feature_dir.resolve()),
            "selection_protocol": "train fit, validation selection, test once",
            "results": results,
        },
    )
    write_summary(output_dir / "summary.csv", results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-dir", default="artifacts/canonical/feature_matrices"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/canonical/model_comparison"
    )
    parser.add_argument(
        "--inventories",
        nargs="+",
        default=list(INVENTORIES),
        help="Inventory names below the feature directory.",
    )
    parser.add_argument(
        "--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS)
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=min(os.cpu_count() or 1, 4))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "manifest.json"
    experiments = list(
        itertools.product(args.inventories, args.configs, args.models)
    )
    for inventory, config, kind in tqdm(
        experiments, desc="matched feature models", unit="experiment"
    ):
        run_experiment(args, inventory, config, kind)
        checkpoint_index(
            output_dir,
            Path(args.feature_dir),
        )
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
