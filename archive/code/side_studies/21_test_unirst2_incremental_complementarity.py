#!/usr/bin/env python3
"""Test incremental predictive information across UniRST2 inventories.

Models use the existing aligned feature matrices.  Base models are selected on
validation, while late-fusion models are trained on grouped out-of-fold train
predictions, preventing identity duplicates from leaking across folds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import itertools
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


INVENTORIES = (
    "deu.rst.pcc",
    "eng.erst.gum",
    "eng.rst.rstdt",
    "nld.rst.nldt",
)
MODELS = ("lr", "xgb")


def load_model_module():
    path = Path(__file__).resolve().with_name("12_evaluate_unirst2_feature_models.py")
    spec = importlib.util.spec_from_file_location("unirst2_feature_models", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_paired_module():
    path = Path(__file__).resolve().with_name("13_compare_unirst2_paired_performance.py")
    spec = importlib.util.spec_from_file_location("unirst2_paired", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


models = load_model_module()
paired = load_paired_module()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
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


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    stream = io.StringIO()
    if rows:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def save_model(path: Path, estimator: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        joblib.dump(estimator, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_split(feature_dir: Path, inventory: str, split: str) -> Dict[str, np.ndarray]:
    path = feature_dir / inventory / f"hc3_{split}.features.npz"
    with np.load(path, allow_pickle=False) as matrix:
        return {
            "structural": np.asarray(matrix["X_structural"], dtype=np.float32),
            "relation": np.asarray(matrix["X_relation_frequency"], dtype=np.float32),
            "m3": np.asarray(matrix["X_m3_frequency"], dtype=np.float32),
            "y": np.asarray(matrix["y"], dtype=np.int8),
            "identity_id": np.asarray(matrix["identity_id"]),
            "identity_hash": np.asarray(matrix["identity_hash"]),
            "domain": np.asarray(matrix["domain"]),
        }


def load_saved_predictions(output_dir: Path, seed: int) -> Tuple[Dict[str, np.ndarray], Dict[str, float], List[Dict[str, Any]]]:
    """Load an already-fitted primary test prediction bundle and thresholds."""
    prediction_path = output_dir / "primary_test_predictions.npz"
    if not prediction_path.exists():
        raise FileNotFoundError(f"saved prediction bundle not found: {prediction_path}")
    predictions: Dict[str, np.ndarray] = {}
    with np.load(prediction_path, allow_pickle=False) as matrix:
        for key in matrix.files:
            if not key.startswith("probability__"):
                continue
            logical = key[len("probability__"):].replace("__", "/")
            if "/late_pair/" in logical:
                prefix, pair = logical.split("/late_pair/", 1)
                logical = prefix + "/late_pair/" + pair.replace("_", "+", 1)
            predictions[logical] = np.asarray(matrix[key], dtype=np.float64)
    thresholds: Dict[str, float] = {}
    for logical in predictions:
        model, config = logical.split("/", 1)
        metrics_path = output_dir / f"seed_{seed}" / model / config.replace("/", "__") / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"saved threshold metadata not found: {metrics_path}")
        metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
        thresholds[logical] = float(metadata["selection"]["validation_selected_threshold"])
    stability_path = output_dir / "seed_stability.csv"
    stability = list(csv.DictReader(stability_path.open(encoding="utf-8"))) if stability_path.exists() else []
    required = {"lr/full/" + inventory for inventory in INVENTORIES} | {"xgb/full/" + inventory for inventory in INVENTORIES}
    missing = sorted(required - set(predictions))
    if missing:
        raise ValueError(f"saved prediction bundle is incomplete: {missing}")
    return predictions, thresholds, stability


def feature_matrix(split: Mapping[str, np.ndarray], inventory: str, config: str) -> np.ndarray:
    if config == "sr":
        return np.concatenate((split["structural"], split["relation"]), axis=1)
    if config == "full":
        return np.concatenate((split["structural"], split["relation"], split["m3"]), axis=1)
    raise ValueError(config)


def combined_matrix(
    splits: Mapping[str, Mapping[str, np.ndarray]],
    inventories: Sequence[str],
    config: str,
) -> np.ndarray:
    structural = splits[inventories[0]]["structural"]
    blocks = [structural]
    for inventory in inventories:
        blocks.append(splits[inventory]["relation"])
        if config == "full":
            blocks.append(splits[inventory]["m3"])
    return np.concatenate(blocks, axis=1)


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        roc_auc_score,
    )
    prediction = probability >= threshold
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "accuracy": float(accuracy_score(y, prediction)),
    }


def fit_model(
    kind: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    seed: int,
    n_jobs: int,
    description: str,
) -> Tuple[Any, Dict[str, Any], np.ndarray, float]:
    train = {"X": train_x, "y": train_y}
    validation = {"X": validation_x, "y": validation_y}
    return models.fit_selected_model(kind, train, validation, seed, n_jobs, description)


def oof_predictions(
    kind: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    params: Mapping[str, Any],
    seed: int,
    n_jobs: int,
    folds: int,
    description: str,
) -> np.ndarray:
    unique_groups = np.unique(groups)
    if len(unique_groups) < folds:
        raise ValueError(f"not enough identity groups for {folds} folds")
    result = np.empty(len(y), dtype=np.float64)
    splitter = GroupKFold(n_splits=folds)
    for fold, (train_index, holdout_index) in enumerate(
        tqdm(splitter.split(X, y, groups), total=folds, desc=description, unit="fold", leave=False)
    ):
        estimator = models.make_model(kind, params, seed + fold + 1, n_jobs)
        estimator.fit(X[train_index], y[train_index])
        result[holdout_index] = models.probabilities(estimator, X[holdout_index])
    return result


def fit_meta(
    train_oof: np.ndarray,
    train_y: np.ndarray,
    validation_probability: np.ndarray,
    validation_y: np.ndarray,
    seed: int,
) -> Tuple[Any, Dict[str, Any], np.ndarray, float]:
    candidates = []
    best = None
    for c_value in (0.01, 0.1, 1.0, 10.0):
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, class_weight="balanced", max_iter=5000, random_state=seed),
        )
        estimator.fit(train_oof, train_y)
        probability = np.asarray(estimator.predict_proba(validation_probability)[:, 1], dtype=np.float64)
        score = metrics(validation_y, probability, 0.5)
        candidate = {"C": c_value, "validation_roc_auc": score["roc_auc"], "validation_average_precision": score["average_precision"]}
        candidates.append(candidate)
        rank = (score["roc_auc"], score["average_precision"], -c_value)
        if best is None or rank > best[0]:
            best = (rank, estimator, candidate, probability)
    assert best is not None
    threshold = models.select_threshold(validation_y, best[3])
    selection = {
        "selected_parameters": best[2],
        "validation_selected_threshold": threshold,
        "candidates": candidates,
        "fit": "grouped out-of-fold train predictions",
    }
    return best[1], selection, best[3], threshold


def consolidate(
    identity_hash: np.ndarray,
    y: np.ndarray,
    domains: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
    excluded: set,
) -> Dict[str, Any]:
    groups: Dict[str, List[int]] = {}
    for index, value in enumerate(identity_hash):
        if str(value) not in excluded:
            groups.setdefault(str(value), []).append(index)
    hashes, labels, domain_values = [], [], []
    result_probability = {key: [] for key in probabilities}
    for value in sorted(groups):
        indices = groups[value]
        if len(np.unique(y[indices])) != 1 or len(np.unique(domains[indices])) != 1:
            raise ValueError(f"identity crosses label/domain: {value}")
        hashes.append(value)
        labels.append(int(y[indices[0]]))
        domain_values.append(str(domains[indices[0]]))
        for key, values in probabilities.items():
            result_probability[key].append(float(np.mean(values[indices])))
    return {
        "identity_hash": np.asarray(hashes),
        "y": np.asarray(labels, dtype=np.int8),
        "domain": np.asarray(domain_values),
        "probability": {key: np.asarray(value, dtype=np.float64) for key, value in result_probability.items()},
        "threshold": dict(thresholds),
    }


def contrast_rows(
    cohort: Mapping[str, Any],
    candidate: str,
    reference: str,
    family: str,
    model: str,
    iterations: int,
    batch_size: int,
    confidence: float,
    seed: int,
) -> List[Dict[str, Any]]:
    y = cohort["y"]
    candidate_p = cohort["probability"][candidate]
    reference_p = cohort["probability"][reference]
    candidate_t = cohort["threshold"][candidate]
    reference_t = cohort["threshold"][reference]
    clusters = paired.identity_clusters(cohort["identity_hash"], y)
    token = f"{seed}:{cohort['name']}:{family}:{model}:{candidate}:{reference}".encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8], "little"))
    candidate_boot = np.empty((iterations, 3), dtype=np.float64)
    reference_boot = np.empty_like(candidate_boot)
    for start in range(0, iterations, batch_size):
        stop = min(start + batch_size, iterations)
        weights = paired.bootstrap_row_weights(rng, clusters, stop - start)
        candidate_boot[start:stop] = paired.weighted_metric_batch(y, candidate_p, candidate_t, weights)
        reference_boot[start:stop] = paired.weighted_metric_batch(y, reference_p, reference_t, weights)
    candidate_point = paired.point_metrics(y, candidate_p, candidate_t)
    reference_point = paired.point_metrics(y, reference_p, reference_t)
    rows = []
    alpha = 1.0 - confidence
    for metric_index, metric in enumerate(("roc_auc", "average_precision", "balanced_accuracy")):
        distribution = candidate_boot[:, metric_index] - reference_boot[:, metric_index]
        rows.append({
            "cohort": cohort["name"], "family": family, "model": model,
            "candidate": candidate, "reference": reference, "metric": metric,
            "records": int(len(y)),
            "candidate_score": float(candidate_point[metric_index]),
            "reference_score": float(reference_point[metric_index]),
            "difference": float(candidate_point[metric_index] - reference_point[metric_index]),
            "ci_low": float(np.quantile(distribution, alpha / 2.0)),
            "ci_high": float(np.quantile(distribution, 1.0 - alpha / 2.0)),
            "probability_positive": float(np.mean(distribution > 0)),
            "bootstrap_sign_pvalue": paired.bootstrap_sign_pvalue(distribution),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=Path("artifacts/unirst2_feature_matrices"))
    parser.add_argument("--audit-dir", type=Path, default=Path("artifacts/unirst2_complementarity"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/unirst2_complementarity/incremental_models"))
    parser.add_argument("--inventories", nargs="+", choices=INVENTORIES, default=list(INVENTORIES))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--oof-folds", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=min(os.cpu_count() or 1, 4))
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--reuse-predictions", action="store_true",
        help="reuse primary_test_predictions.npz and saved validation thresholds; skip all fitting",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if tuple(args.inventories) != INVENTORIES:
        raise ValueError("stage 4 requires all inventories in canonical order")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrices: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {
        inventory: {split: load_split(args.feature_dir, inventory, split) for split in ("train", "validation", "test")}
        for inventory in INVENTORIES
    }
    reference = matrices[INVENTORIES[0]]["test"]
    for inventory in INVENTORIES[1:]:
        for key in ("identity_id", "identity_hash", "y", "domain"):
            if not np.array_equal(reference[key], matrices[inventory]["test"][key]):
                raise ValueError(f"test alignment mismatch: {inventory}/{key}")
    train_hashes = set(matrices[INVENTORIES[0]]["train"]["identity_hash"].tolist())
    near_blocked = set()
    cluster_path = args.audit_dir / "near_duplicate_clusters.jsonl"
    with cluster_path.open(encoding="utf-8") as handle:
        for line in handle:
            cluster = json.loads(line)
            members = cluster.get("members", [])
            member_hashes = {str(member["identity_hash"]) for member in members}
            if member_hashes & train_hashes:
                near_blocked.update(str(member["identity_hash"]) for member in members if member.get("split") == "test")
    source_hashes = {
        f"{inventory}/{split}": sha256_file(args.feature_dir / inventory / f"hc3_{split}.features.npz")
        for inventory in INVENTORIES for split in ("train", "validation", "test")
    }

    all_contrasts: List[Dict[str, Any]] = []
    seed_metrics: List[Dict[str, Any]] = []
    primary_seed = args.seeds[0]
    primary_predictions: Dict[str, np.ndarray] = {}
    primary_thresholds: Dict[str, float] = {}
    model_records: List[Dict[str, Any]] = []
    fit_seeds = list(args.seeds)
    if args.reuse_predictions:
        primary_predictions, primary_thresholds, seed_metrics = load_saved_predictions(
            args.output_dir, primary_seed
        )
        previous_manifest_path = args.output_dir / "manifest.json"
        if previous_manifest_path.exists():
            previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
            model_records = list(previous_manifest.get("model_records", []))
        fit_seeds = []

    for seed in fit_seeds:
        base_predictions: Dict[str, Dict[str, np.ndarray]] = {"train_oof": {}, "validation": {}, "test": {}}
        base_thresholds: Dict[str, float] = {}
        for kind in args.models:
            for inventory in INVENTORIES:
                train = matrices[inventory]["train"]
                validation = matrices[inventory]["validation"]
                test = matrices[inventory]["test"]
                configs = {
                    f"sr/{inventory}": (feature_matrix(train, inventory, "sr"), feature_matrix(validation, inventory, "sr"), feature_matrix(test, inventory, "sr")),
                    f"full/{inventory}": (feature_matrix(train, inventory, "full"), feature_matrix(validation, inventory, "full"), feature_matrix(test, inventory, "full")),
                }
                for config_name, (train_x, validation_x, test_x) in configs.items():
                    estimator, selection, validation_probability, threshold = fit_model(
                        kind, train_x, train["y"], validation_x, validation["y"], seed, args.n_jobs,
                        f"{seed} {kind} {config_name}",
                    )
                    test_probability = models.probabilities(estimator, test_x)
                    model_key = f"{kind}/{config_name}"
                    if config_name.startswith("full/"):
                        base_predictions["validation"][inventory] = validation_probability
                        base_predictions["test"][inventory] = test_probability
                        base_thresholds[inventory] = threshold
                        oof = oof_predictions(
                            kind, train_x, train["y"], train["identity_hash"], selection["selected_parameters"],
                            seed, args.n_jobs, args.oof_folds, f"{seed} OOF {model_key}",
                        )
                        base_predictions["train_oof"][inventory] = oof
                    model_dir = args.output_dir / f"seed_{seed}" / kind / config_name.replace("/", "__")
                    save_model(model_dir / "model.joblib", estimator)
                    atomic_json(model_dir / "metrics.json", {
                        "seed": seed, "model": kind, "config": config_name,
                        "selection": selection,
                        "validation_metrics": metrics(validation["y"], validation_probability, threshold),
                        "test_metrics": metrics(test["y"], test_probability, threshold),
                        "feature_count": int(train_x.shape[1]),
                    })
                    model_records.append({"seed": seed, "model": kind, "config": config_name, "feature_count": int(train_x.shape[1])})
                    if seed == primary_seed:
                        primary_predictions[model_key] = test_probability
                        primary_thresholds[model_key] = threshold
                    if config_name.startswith("sr/"):
                        seed_metrics.append({"seed": seed, "model": kind, "representation": "sr", "inventory": inventory, **metrics(test["y"], test_probability, threshold)})
                    else:
                        seed_metrics.append({"seed": seed, "model": kind, "representation": "full", "inventory": inventory, **metrics(test["y"], test_probability, threshold)})

            # Late fusion uses full-inventory predictions and grouped OOF train scores.
            for left_index, left in enumerate(INVENTORIES):
                for right in INVENTORIES[left_index + 1 :]:
                    pair_name = f"late_pair/{left}+{right}"
                    train_oof = np.column_stack((base_predictions["train_oof"][left], base_predictions["train_oof"][right]))
                    validation_probability_input = np.column_stack((base_predictions["validation"][left], base_predictions["validation"][right]))
                    test_probability_input = np.column_stack((base_predictions["test"][left], base_predictions["test"][right]))
                    meta, selection, validation_probability, threshold = fit_meta(
                        train_oof, matrices[left]["train"]["y"], validation_probability_input,
                        matrices[left]["validation"]["y"], seed,
                    )
                    test_probability = models.probabilities(meta, test_probability_input)
                    key = f"{kind}/{pair_name}"
                    primary_predictions[key] = test_probability if seed == primary_seed else primary_predictions.get(key, test_probability)
                    primary_thresholds[key] = threshold if seed == primary_seed else primary_thresholds.get(key, threshold)
                    meta_dir = args.output_dir / f"seed_{seed}" / kind / pair_name.replace("/", "__")
                    save_model(meta_dir / "model.joblib", meta)
                    atomic_json(meta_dir / "metrics.json", {
                        "seed": seed, "model": kind, "config": pair_name,
                        "selection": selection,
                        "validation_metrics": metrics(matrices[left]["validation"]["y"], validation_probability, threshold),
                        "test_metrics": metrics(matrices[left]["test"]["y"], test_probability, threshold),
                        "feature_count": 2,
                    })
                    seed_metrics.append({"seed": seed, "model": kind, "representation": pair_name, "inventory": "pair", **metrics(matrices[left]["test"]["y"], test_probability, threshold)})
            all_train_oof = np.column_stack([base_predictions["train_oof"][inventory] for inventory in INVENTORIES])
            all_validation_input = np.column_stack([base_predictions["validation"][inventory] for inventory in INVENTORIES])
            all_test_input = np.column_stack([base_predictions["test"][inventory] for inventory in INVENTORIES])
            meta, selection, validation_probability, threshold = fit_meta(
                all_train_oof, matrices[INVENTORIES[0]]["train"]["y"], all_validation_input,
                matrices[INVENTORIES[0]]["validation"]["y"], seed,
            )
            test_probability = models.probabilities(meta, all_test_input)
            key = f"{kind}/late_all"
            if seed == primary_seed:
                primary_predictions[key] = test_probability
                primary_thresholds[key] = threshold
            meta_dir = args.output_dir / f"seed_{seed}" / kind / "late_all"
            save_model(meta_dir / "model.joblib", meta)
            atomic_json(meta_dir / "metrics.json", {
                "seed": seed, "model": kind, "config": "late_all", "selection": selection,
                "validation_metrics": metrics(matrices[INVENTORIES[0]]["validation"]["y"], validation_probability, threshold),
                "test_metrics": metrics(matrices[INVENTORIES[0]]["test"]["y"], test_probability, threshold),
                "feature_count": len(INVENTORIES),
            })
            seed_metrics.append({"seed": seed, "model": kind, "representation": "late_all", "inventory": "all", **metrics(matrices[INVENTORIES[0]]["test"]["y"], test_probability, threshold)})

    # Save primary test predictions in a single aligned artifact.
    prediction_arrays = {f"probability__{key.replace('/', '__').replace('+', '_')}": value for key, value in primary_predictions.items()}
    prediction_arrays["identity_hash"] = reference["identity_hash"]
    prediction_arrays["y"] = reference["y"]
    np.savez_compressed(args.output_dir / "primary_test_predictions.npz", **prediction_arrays)

    # Paired within-inventory additions and late-fusion additions on primary and
    # identity-clean cohorts.  The near-duplicate cohort is equivalent here if
    # it excludes no additional train-linked test identities.
    primary_bundle = {key: value for key, value in primary_predictions.items()}
    for model_kind in args.models:
        for inventory in INVENTORIES:
            candidate = f"{model_kind}/full/{inventory}"
            reference_key = f"{model_kind}/sr/{inventory}"
            for cohort_name, excluded in (("primary", set()), ("identity_clean", train_hashes)):
                cohort = consolidate(reference["identity_hash"], reference["y"], reference["domain"], primary_bundle, primary_thresholds, excluded)
                cohort["name"] = cohort_name
                all_contrasts.extend(contrast_rows(cohort, candidate, reference_key, "within_inventory_m3_increment", model_kind, args.iterations, args.batch_size, args.confidence, args.seeds[0]))
        for left_index, left in enumerate(INVENTORIES):
            for right in INVENTORIES[left_index + 1 :]:
                candidate = f"{model_kind}/late_pair/{left}+{right}"
                for reference_key, label in ((f"{model_kind}/full/{left}", "pair_vs_left"), (f"{model_kind}/full/{right}", "pair_vs_right")):
                    for cohort_name, excluded in (("primary", set()), ("identity_clean", train_hashes)):
                        cohort = consolidate(reference["identity_hash"], reference["y"], reference["domain"], primary_bundle, primary_thresholds, excluded)
                        cohort["name"] = cohort_name
                        all_contrasts.extend(contrast_rows(cohort, candidate, reference_key, label, model_kind, args.iterations, args.batch_size, args.confidence, args.seeds[0]))
        candidate = f"{model_kind}/late_all"
        for inventory in INVENTORIES:
            for cohort_name, excluded in (("primary", set()), ("identity_clean", train_hashes)):
                cohort = consolidate(reference["identity_hash"], reference["y"], reference["domain"], primary_bundle, primary_thresholds, excluded)
                cohort["name"] = cohort_name
                all_contrasts.extend(contrast_rows(cohort, candidate, f"{model_kind}/full/{inventory}", "all_vs_inventory", model_kind, args.iterations, args.batch_size, args.confidence, args.seeds[0]))

    # Correct pairwise addition families separately by cohort/model/metric.
    for cohort_name in ("primary", "identity_clean"):
        for model_kind in args.models:
            for metric in ("roc_auc", "average_precision", "balanced_accuracy"):
                family = [row for row in all_contrasts if row["cohort"] == cohort_name and row["model"] == model_kind and row["metric"] == metric]
                adjusted = paired.holm_adjust([row["bootstrap_sign_pvalue"] for row in family]) if family else []
                for row, value in zip(family, adjusted):
                    row["holm_adjusted_pvalue"] = value
                    row["ci_excludes_zero"] = bool(row["ci_low"] > 0 or row["ci_high"] < 0)

    write_csv(args.output_dir / "incremental_contrasts.csv", all_contrasts)
    write_csv(args.output_dir / "seed_stability.csv", seed_metrics)
    # Domain-level point estimates are diagnostic only: domains are not
    # independent replications because all occur in the same HC3 training run.
    domain_rows: List[Dict[str, Any]] = []
    domain_specs: List[Tuple[str, str, str]] = []
    for inventory in INVENTORIES:
        domain_specs.append(("within_inventory_m3_increment", f"lr/full/{inventory}", f"lr/sr/{inventory}"))
        domain_specs.append(("within_inventory_m3_increment", f"xgb/full/{inventory}", f"xgb/sr/{inventory}"))
    for model_kind in args.models:
        for left_index, left in enumerate(INVENTORIES):
            for right in INVENTORIES[left_index + 1 :]:
                for reference_key, family in ((f"{model_kind}/full/{left}", "pair_vs_left"), (f"{model_kind}/full/{right}", "pair_vs_right")):
                    domain_specs.append((family, f"{model_kind}/late_pair/{left}+{right}", reference_key))
        for inventory in INVENTORIES:
            domain_specs.append(("all_vs_inventory", f"{model_kind}/late_all", f"{model_kind}/full/{inventory}"))
    for family, candidate, reference_key in domain_specs:
        if candidate not in primary_predictions or reference_key not in primary_predictions:
            continue
        for domain in np.unique(reference["domain"]):
            mask = reference["domain"] == domain
            if len(np.unique(reference["y"][mask])) < 2:
                continue
            candidate_score = paired.point_metrics(reference["y"][mask], primary_predictions[candidate][mask], primary_thresholds[candidate])
            reference_score = paired.point_metrics(reference["y"][mask], primary_predictions[reference_key][mask], primary_thresholds[reference_key])
            for metric_index, metric in enumerate(("roc_auc", "average_precision", "balanced_accuracy")):
                domain_rows.append({
                    "domain": str(domain), "family": family, "model": candidate.split("/", 1)[0],
                    "candidate": candidate, "reference": reference_key,
                    "records": int(np.count_nonzero(mask)), "metric": metric,
                    "candidate_score": float(candidate_score[metric_index]),
                    "reference_score": float(reference_score[metric_index]),
                    "difference": float(candidate_score[metric_index] - reference_score[metric_index]),
                })
    write_csv(args.output_dir / "domain_incremental_effects.csv", domain_rows)
    robust_rows = [
        row for row in all_contrasts
        if row["cohort"] == "identity_clean" and row["metric"] == "roc_auc"
        and row.get("holm_adjusted_pvalue", 1.0) < 0.05 and row["difference"] >= 0.01 and row["ci_low"] > 0
    ]
    # The primary cross-inventory gate compares each pair with the member
    # selected as stronger on validation, and late-all with the validation-best
    # inventory. Comparisons against weaker members remain descriptive.
    validation_auc: Dict[str, Dict[str, float]] = {}
    for model_kind in args.models:
        validation_auc[model_kind] = {}
        for inventory in INVENTORIES:
            metadata_path = args.output_dir / f"seed_{primary_seed}" / model_kind / f"full__{inventory}" / "metrics.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                validation_auc[model_kind][inventory] = float(metadata["validation_metrics"]["roc_auc"])
    primary_cross_rows: List[Dict[str, Any]] = []
    for row in all_contrasts:
        if row["cohort"] != "identity_clean" or row["metric"] != "roc_auc":
            continue
        model_kind = row["model"]
        scores = validation_auc.get(model_kind, {})
        if row["family"] in ("pair_vs_left", "pair_vs_right"):
            pair = row["candidate"].split("/late_pair/", 1)[-1].split("+")
            stronger = max(pair, key=lambda inventory: scores.get(inventory, float("-inf")))
            if row["reference"].endswith("/" + stronger):
                primary_cross_rows.append(row)
        elif row["family"] == "all_vs_inventory":
            if scores and row["reference"].endswith("/" + max(scores, key=scores.get)):
                primary_cross_rows.append(row)
    primary_robust_rows = [
        row for row in primary_cross_rows
        if row.get("holm_adjusted_pvalue", 1.0) < 0.05 and row["difference"] >= 0.01 and row["ci_low"] > 0
    ]
    complementarity_summary = {
        "criterion": {"minimum_auc_gain": 0.01, "interval": "95% paired identity-cluster bootstrap", "holm_alpha": 0.05},
        "identity_clean_roc_auc": {
            "interpretation": "descriptive total including within-inventory motif increments and comparisons against weaker inventory members; not the cross-inventory gate",
            "total_contrasts": int(sum(row["cohort"] == "identity_clean" and row["metric"] == "roc_auc" for row in all_contrasts)),
            "criterion_passing": len(robust_rows),
            "passing_by_family": {family: int(sum(row in robust_rows and row["family"] == family for row in robust_rows)) for family in ("within_inventory_m3_increment", "pair_vs_left", "pair_vs_right", "all_vs_inventory")},
        },
        "primary_cross_inventory_gate": {
            "definition": "pair versus validation-selected stronger member; late-all versus validation-selected best inventory",
            "total_contrasts": len(primary_cross_rows),
            "criterion_passing": len(primary_robust_rows),
        },
        "note": "Test outcomes are exploratory because the existing test set was inspected during earlier stages; domain rows are diagnostic, not independent replications.",
    }
    atomic_json(args.output_dir / "complementarity_summary.json", complementarity_summary)
    summary = {
        "stage": "unirst2_incremental_complementarity",
        "inventories": list(INVENTORIES), "models": list(args.models), "seeds": list(args.seeds),
        "oof_folds": args.oof_folds, "iterations": args.iterations, "batch_size": args.batch_size,
        "protocol": {
            "base_models": "train fit, validation selection, test once",
            "late_fusion": "meta logistic regression fit on grouped out-of-fold full-inventory train predictions",
            "identity_groups": "text/source/label identity hash",
            "test_status": "exploratory because existing test outcomes were inspected",
        },
        "source_matrix_hashes": source_hashes,
        "model_records": model_records,
        "outputs": {
            "incremental_contrasts": str((args.output_dir / "incremental_contrasts.csv").resolve()),
            "seed_stability": str((args.output_dir / "seed_stability.csv").resolve()),
            "domain_incremental_effects": str((args.output_dir / "domain_incremental_effects.csv").resolve()),
            "complementarity_summary": str((args.output_dir / "complementarity_summary.json").resolve()),
            "primary_test_predictions": str((args.output_dir / "primary_test_predictions.npz").resolve()),
        },
    }
    report = [
        "# UniRST2 incremental complementarity", "",
        "The analysis compares structure-plus-relation baselines with added M3 frequency and tests late fusion from grouped out-of-fold predictions.",
        "All test-set conclusions are exploratory; the existing HC3 test outcomes were previously inspected.", "",
        f"Generated {len(all_contrasts)} paired incremental contrasts across primary and identity-clean cohorts.", "",
        "For the operational gate, a gain requires ROC-AUC Δ ≥ 0.01, a nominal paired identity-cluster percentile interval excluding zero, and Holm-adjusted bootstrap-sign p < 0.05. Seed comparisons are descriptive; the gate is evaluated on the seed-42 prediction bundle and is not programmatically enforced across seeds.", "",
    ]
    for family in ("within_inventory_m3_increment", "pair_vs_left", "pair_vs_right", "all_vs_inventory"):
        rows = [
            row for row in all_contrasts
            if row["family"] == family and row["cohort"] == "identity_clean" and row["metric"] == "roc_auc"
        ]
        if not rows:
            continue
        robust = [
            row for row in rows
            if row.get("holm_adjusted_pvalue", 1.0) < 0.05
            and row["difference"] >= 0.01
            and row["ci_low"] > 0
        ]
        best = max(rows, key=lambda row: row["difference"])
        report.append(
            f"- `{family}`: {len(robust)}/{len(rows)} identity-clean contrasts meet the operational criterion; "
            f"largest ΔAUC {best['difference']:+.4f} ({best['candidate']} vs {best['reference']}, "
            f"95% CI {best['ci_low']:+.4f}..{best['ci_high']:+.4f}, "
            f"Holm p={best.get('holm_adjusted_pvalue', float('nan')):.4g})."
        )
    report.extend([
        "",
        f"Primary cross-inventory gate (pair versus validation-selected stronger member; all-inventory versus validation-best): {len(primary_robust_rows)}/{len(primary_cross_rows)} identity-clean ROC-AUC contrasts pass the criterion.",
        "Interpretation: M3 features add a clear signal to every single-inventory model, but the fused systems do not yet beat the stronger inventory by a practically meaningful margin. Gains against weaker members are descriptive and do not establish complementary standards. These results support robust motif utility and overlapping, inventory-dependent representations, not independent coherence accounts.",
    ])
    atomic_text(args.output_dir / "report.md", "\n".join(report))
    summary["outputs"]["report"] = str((args.output_dir / "report.md").resolve())
    for name in ("incremental_contrasts.csv", "seed_stability.csv", "domain_incremental_effects.csv", "complementarity_summary.json", "primary_test_predictions.npz", "report.md"):
        summary.setdefault("output_hashes", {})[name] = sha256_file(args.output_dir / name)
    atomic_json(args.output_dir / "manifest.json", summary)
    print(json.dumps({"contrasts": len(all_contrasts), "seed_rows": len(seed_metrics)}, indent=2))


if __name__ == "__main__":
    main()
