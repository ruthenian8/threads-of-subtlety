#!/usr/bin/env python3
"""Fit LR fine/coarse MF+WAD models and compare their gain with MF-only LR."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from tqdm import tqdm
from tos.analysis.granularity import (
    INVENTORIES, METRICS, SPLITS, aggregate_wad, assignment_from_pairs,
    atomic_csv, atomic_json, atomic_npz, atomic_text, coarse_feature_path,
    coarse_lr_prediction_path, coarse_results_path, identity_clean_mask,
    interleave_mf_wad, read_npz, sha256_file, validate_pair,
)


def assignment_from_csv(path: Path, inventory: str, fine_hashes: np.ndarray, coarse_hashes: np.ndarray) -> np.ndarray:
    pairs = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["inventory"] != inventory:
                continue
            pairs.append((row["fine_hash"], row["coarse_hash"]))
    return assignment_from_pairs(fine_hashes, coarse_hashes, pairs, f"{path}/{inventory}")


def assignment_from_results(path: Path, fine_hashes: np.ndarray, coarse_hashes: np.ndarray) -> np.ndarray:
    mapping = json.loads(path.read_text(encoding="utf-8"))["motif_mapping"]
    pairs = [(row["fine_hash"], row["coarse_hash"]) for row in mapping]
    return assignment_from_pairs(fine_hashes, coarse_hashes, pairs, str(path))


def load_splits(
    feature_root: Path,
    granularity_root: Path,
    mapping_file: Path,
    inventory: str,
) -> tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]], Dict[str, str]]:
    fine_splits, coarse_splits, hashes = {}, {}, {}
    for split in SPLITS:
        fine_path = feature_root / inventory / f"hc3_{split}.features.npz"
        projected_path = coarse_feature_path(granularity_root, inventory, split)
        fine, coarse = read_npz(fine_path), read_npz(projected_path)
        validate_pair(fine, coarse, f"{inventory}/{split}")
        if inventory in {"deu.rst.pcc", "nld.rst.nldt"}:
            assignment = assignment_from_csv(
                mapping_file, inventory, fine["m3_hashes"], coarse["m3_hashes"]
            )
        else:
            assignment = assignment_from_results(
                coarse_results_path(granularity_root, inventory),
                fine["m3_hashes"], coarse["m3_hashes"],
            )
        expected_coarse_wad = aggregate_wad(
            fine["X_m3_frequency"].astype(np.float32),
            fine["X_m3_depth"].astype(np.float32),
            assignment,
            coarse["X_m3_frequency"].astype(np.float32),
        )
        if "X_m3_depth" not in coarse:
            coarse["X_m3_depth"] = expected_coarse_wad
        elif not np.allclose(coarse["X_m3_depth"], expected_coarse_wad, atol=2e-6):
            raise ValueError(f"Coarse WAD is not the occurrence-weighted fine WAD: {inventory}/{split}")
        fine_splits[split], coarse_splits[split] = fine, coarse
        hashes[f"fine/{split}"] = sha256_file(fine_path)
        hashes[f"coarse/{split}"] = sha256_file(projected_path)
    if not np.array_equal(fine_splits["train"]["m3_hashes"], fine_splits["test"]["m3_hashes"]):
        raise ValueError(f"Fine motif feature order differs across splits: {inventory}")
    for split in ("validation", "test"):
        if not np.array_equal(fine_splits["train"]["m3_hashes"], fine_splits[split]["m3_hashes"]):
            raise ValueError(f"Fine motif feature order differs across splits: {inventory}/{split}")
    return fine_splits, coarse_splits, hashes


def fit_variant(
    evaluator: Any,
    inventory: str,
    granularity: str,
    splits: Mapping[str, Mapping[str, np.ndarray]],
    output_dir: Path,
    seed: int,
    n_jobs: int,
) -> tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    arrays = {
        split: {
            "X": interleave_mf_wad(value["X_m3_frequency"], value["X_m3_depth"]),
            "y": np.asarray(value["y"], dtype=np.int8),
        }
        for split, value in splits.items()
    }
    model, selection, validation_probability, threshold = evaluator.fit_selected_model(
        "lr", arrays["train"], arrays["validation"], seed, n_jobs,
        f"{inventory} LR MF+WAD/{granularity}",
    )
    test_probability = evaluator.probabilities(model, arrays["test"]["X"])
    run_dir = output_dir / "predictions" / inventory / granularity
    prediction_path = run_dir / "predictions.npz"
    atomic_npz(
        prediction_path,
        test_identity_id=np.asarray(splits["test"]["identity_id"]),
        test_identity_hash=np.asarray(splits["test"]["identity_hash"]),
        test_y=arrays["test"]["y"],
        test_probability=test_probability,
        threshold=np.asarray(threshold),
    )
    result = {
        "inventory": inventory,
        "granularity": granularity,
        "classifier": "lr",
        "representation": "mf_wad",
        "feature_count": int(arrays["train"]["X"].shape[1]),
        "seed": seed,
        "selection": selection,
        "protocol": {
            "fit": "train only",
            "model_selection": "validation ROC-AUC; AP tie-break",
            "threshold_selection": "validation balanced accuracy",
            "test_usage": "single evaluation after model and threshold selection",
        },
        "records": {split: int(len(value["y"])) for split, value in arrays.items()},
        "validation_metrics": evaluator.metrics(arrays["validation"]["y"], validation_probability, threshold),
        "test_metrics": evaluator.metrics(arrays["test"]["y"], test_probability, threshold),
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": sha256_file(prediction_path),
    }
    atomic_json(run_dir / "metrics.json", result)
    return result, {
        "identity_id": np.asarray(splits["test"]["identity_id"]),
        "identity_hash": np.asarray(splits["test"]["identity_hash"]),
        "y": arrays["test"]["y"],
        "probability": np.asarray(test_probability, dtype=np.float64),
        "threshold": float(threshold),
    }


def load_prediction(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as values:
        return {
            "identity_id": np.asarray(values["test_identity_id"]),
            "identity_hash": np.asarray(values["test_identity_hash"]),
            "y": np.asarray(values["test_y"], dtype=np.int8),
            "probability": np.asarray(values["test_probability"], dtype=np.float64),
            "threshold": float(values["threshold"]),
        }


def load_mf_pair(
    inventory: str,
    prediction_root: Path,
    granularity_root: Path,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    fine = load_prediction(prediction_root / inventory / "m3_frequency" / "lr" / "predictions.npz")
    coarse = load_prediction(coarse_lr_prediction_path(granularity_root, inventory))
    for key in ("identity_id", "identity_hash", "y"):
        if not np.array_equal(fine[key], coarse[key]):
            raise ValueError(f"Existing MF prediction rows differ for {inventory}/{key}")
    return fine, coarse


def calculate_comparisons(
    paired: Any,
    inventory: str,
    train_hashes: np.ndarray,
    predictions: Mapping[str, Mapping[str, Any]],
    iterations: int,
    batch_size: int,
    seed: int,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    mf_fine, mf_coarse = predictions["mf_fine"], predictions["mf_coarse"]
    wad_fine, wad_coarse = predictions["wad_fine"], predictions["wad_coarse"]
    for key in ("identity_id", "identity_hash", "y"):
        reference = mf_fine[key]
        for run in (mf_coarse, wad_fine, wad_coarse):
            if not np.array_equal(reference, run[key]):
                raise ValueError(f"Prediction rows differ for {inventory}/{key}")
    masks = {
        "primary": np.ones(len(mf_fine["y"]), dtype=bool),
        "identity_clean": identity_clean_mask(train_hashes, mf_fine["identity_hash"]),
    }
    comparisons, interactions = [], []
    for cohort, mask in masks.items():
        y = mf_fine["y"][mask]
        identity_hash = mf_fine["identity_hash"][mask]
        clusters = paired.identity_clusters(identity_hash, y)
        run_names = ("mf_fine", "mf_coarse", "wad_fine", "wad_coarse")
        point = {
            name: paired.point_metrics(
                y,
                predictions[name]["probability"][mask],
                predictions[name]["threshold"],
            )
            for name in run_names
        }
        boot = np.empty((iterations, len(run_names), len(METRICS)), dtype=np.float64)
        interaction_boot = np.empty((iterations, len(METRICS)), dtype=np.float64)
        rng = np.random.default_rng(seed)
        for start in tqdm(
            range(0, iterations, batch_size),
            desc=f"paired LR MF+WAD {inventory}/{cohort}",
            unit="batch",
            leave=False,
        ):
            stop = min(start + batch_size, iterations)
            weights = paired.bootstrap_row_weights(rng, clusters, stop - start)
            values = {}
            for run_index, name in enumerate(run_names):
                values[name] = paired.weighted_metric_batch(
                    y,
                    predictions[name]["probability"][mask],
                    predictions[name]["threshold"],
                    weights,
                )
                boot[start:stop, run_index] = values[name]
            interaction_boot[start:stop] = (
                values["wad_fine"] - values["wad_coarse"]
                - values["mf_fine"] + values["mf_coarse"]
            )
        for metric_index, metric in enumerate(METRICS):
            samples = boot[:, :, metric_index]
            for representation, fine_name, coarse_name in (
                ("mf", "mf_fine", "mf_coarse"),
                ("mf_wad", "wad_fine", "wad_coarse"),
            ):
                fine_index = run_names.index(fine_name)
                coarse_index = run_names.index(coarse_name)
                differences = samples[:, fine_index] - samples[:, coarse_index]
                point_difference = point[fine_name][metric_index] - point[coarse_name][metric_index]
                comparisons.append({
                    "cohort": cohort,
                    "rows": int(mask.sum()),
                    "inventory": inventory,
                    "classifier": "lr",
                    "representation": representation,
                    "metric": metric,
                    "fine_score": float(point[fine_name][metric_index]),
                    "coarse_score": float(point[coarse_name][metric_index]),
                    "difference": float(point_difference),
                    "ci_low": float(np.quantile(differences, 0.025)),
                    "ci_high": float(np.quantile(differences, 0.975)),
                    "bootstrap_sign_pvalue": paired.bootstrap_sign_pvalue(differences),
                    "holm_adjusted_pvalue": 0.0,
                })
            delta = interaction_boot[:, metric_index]
            interactions.append({
                "cohort": cohort,
                "rows": int(mask.sum()),
                "inventory": inventory,
                "metric": metric,
                "mf_fine_minus_coarse": float(point["mf_fine"][metric_index] - point["mf_coarse"][metric_index]),
                "mf_wad_fine_minus_coarse": float(point["wad_fine"][metric_index] - point["wad_coarse"][metric_index]),
                "delta_gain_mf_wad_minus_mf": float(
                    point["wad_fine"][metric_index] - point["wad_coarse"][metric_index]
                    - point["mf_fine"][metric_index] + point["mf_coarse"][metric_index]
                ),
                "ci_low": float(np.quantile(delta, 0.025)),
                "ci_high": float(np.quantile(delta, 0.975)),
                "bootstrap_sign_pvalue": paired.bootstrap_sign_pvalue(delta),
                "holm_adjusted_pvalue": 0.0,
            })
    return comparisons, interactions


def adjust_holm(paired: Any, rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> None:
    grouped: Dict[tuple[str, str], list[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(str(row[key]) for key in keys), []).append(row)
    for group in grouped.values():
        adjusted = paired.holm_adjust([float(row["bootstrap_sign_pvalue"]) for row in group])
        for row, value in zip(group, adjusted):
            row["holm_adjusted_pvalue"] = value


def report_text(comparisons: Sequence[Mapping[str, Any]], interactions: Sequence[Mapping[str, Any]], iterations: int) -> str:
    labels = {
        "deu.rst.pcc": "PCC", "eng.erst.gum.new": "eRST",
        "eng.rst.rstdt.new": "RST-DT", "nld.rst.nldt": "NLDT",
    }
    primary = {
        (row["inventory"], row["representation"]): row
        for row in comparisons
        if row["cohort"] == "primary" and row["metric"] == "f1"
    }
    delta = {
        row["inventory"]: row
        for row in interactions if row["cohort"] == "primary" and row["metric"] == "f1"
    }
    lines = [
        "# LR fine-to-coarse comparison with MF and MF+WAD",
        "",
        "Fine/coarse LR models use the same train/validation/test splits, standardization, "
        "regularization search, validation ROC-AUC model selection, and validation balanced-accuracy "
        "threshold selection. MF+WAD interleaves the per-motif normalized frequency and weighted "
        "average depth. For merged motifs, coarse WAD is occurrence-weighted.",
        "",
        "| Standard | MF fine F1 | MF coarse F1 | MF gain (95% CI) | MF+WAD fine F1 | MF+WAD coarse F1 | MF+WAD gain (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for inventory in INVENTORIES:
        mf = primary[(inventory, "mf")]
        mf_wad = primary[(inventory, "mf_wad")]
        lines.append(
            f"| {labels[inventory]} | {mf['fine_score']:.3f} | {mf['coarse_score']:.3f} | "
            f"{mf['difference']:+.3f} [{mf['ci_low']:+.3f}, {mf['ci_high']:+.3f}] | "
            f"{mf_wad['fine_score']:.3f} | {mf_wad['coarse_score']:.3f} | "
            f"{mf_wad['difference']:+.3f} [{mf_wad['ci_low']:+.3f}, {mf_wad['ci_high']:+.3f}] |"
        )
    lines.extend([
        "",
        "## Does WAD change the fine-to-coarse gain?",
        "",
        "The paired contrast is (fine − coarse) MF+WAD minus (fine − coarse) MF; negative values "
        "mean the estimated granularity gain is smaller after adding WAD.",
        "",
        "| Standard | Change in fine-to-coarse F1 gain (95% CI) | Holm-adjusted p |",
        "|---|---:|---:|",
    ])
    for inventory in INVENTORIES:
        row = delta[inventory]
        lines.append(
            f"| {labels[inventory]} | {row['delta_gain_mf_wad_minus_mf']:+.3f} "
            f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] | {row['holm_adjusted_pvalue']:.4f} |"
        )
    lines.extend([
        "",
        f"Intervals use {iterations:,} paired, label-stratified identity-cluster bootstrap samples. "
        "The primary cohort retains the shared HC3 test split; the identity-clean sensitivity "
        "removes train-overlap identities and collapses duplicate test identities. Holm correction "
        "covers the four standards separately for each metric and contrast family. The RF MF+WAD "
        "comparison remains in `../rf_mf_wad/`; its fixed 0.5 threshold differs from LR's "
        "validation-selected threshold.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=Path("artifacts/canonical/feature_matrices"))
    parser.add_argument("--granularity-root", type=Path, default=Path("artifacts/canonical/granularity"))
    parser.add_argument("--mapping-file", type=Path, default=Path("artifacts/canonical/granularity/pcc_nldt/motif_mapping.csv"))
    parser.add_argument("--prediction-root", type=Path, default=Path("artifacts/canonical/model_comparison"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/canonical/granularity/lr_mf_wad"))
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-seed", type=int, default=20260909)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from scripts.canonical import compare_paired_performance as paired
    from scripts.canonical import evaluate_feature_models as evaluator

    manifest_path = args.output_dir / "manifest.json"
    required = (
        args.output_dir / "paired_comparisons.csv",
        args.output_dir / "mf_gain_interaction.csv",
        args.output_dir / "report.md",
    )
    input_paths = [args.mapping_file]
    for inventory in INVENTORIES:
        for split in SPLITS:
            input_paths.extend((
                args.feature_root / inventory / f"hc3_{split}.features.npz",
                coarse_feature_path(args.granularity_root, inventory, split),
            ))
    input_paths.extend(
        args.prediction_root / inventory / "m3_frequency" / "lr" / "predictions.npz"
        for inventory in INVENTORIES
    )
    input_paths.extend(coarse_lr_prediction_path(args.granularity_root, inventory) for inventory in INVENTORIES)
    input_paths.extend(
        args.granularity_root / directory / "results.json"
        for directory in ("erst", "rstdt")
    )
    inputs = {str(path): sha256_file(path) for path in input_paths}
    settings = {
        "seed": args.seed,
        "bootstrap_seed": args.bootstrap_seed,
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "n_jobs": args.n_jobs,
        "features": "per-motif interleaved normalized MF and occurrence-weighted WAD",
        "selection": "validation ROC-AUC with AP tie-break; validation balanced-accuracy threshold",
    }
    if manifest_path.is_file() and all(path.is_file() for path in required) and not args.force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("inputs") == inputs and manifest.get("settings") == settings:
            print(f"skipping complete LR MF+WAD granularity run; wrote {args.output_dir}")
            return

    comparison_rows, interaction_rows, metrics_rows = [], [], []
    for inventory in INVENTORIES:
        fine_splits, coarse_splits, split_hashes = load_splits(
            args.feature_root, args.granularity_root, args.mapping_file, inventory
        )
        wad_metrics, wad_predictions = {}, {}
        for granularity, splits in (("fine", fine_splits), ("coarse", coarse_splits)):
            row, predictions = fit_variant(
                evaluator, inventory, granularity, splits,
                args.output_dir, args.seed, args.n_jobs,
            )
            row["feature_matrix_sha256"] = split_hashes
            atomic_json(Path(row["prediction_path"]).with_name("metrics.json"), row)
            metrics_rows.append(row)
            wad_metrics[granularity] = row
            wad_predictions["wad_" + granularity] = predictions
        mf_fine, mf_coarse = load_mf_pair(inventory, args.prediction_root, args.granularity_root)
        wad_predictions["mf_fine"], wad_predictions["mf_coarse"] = mf_fine, mf_coarse
        train_hashes = fine_splits["train"]["identity_hash"]
        comparisons, interactions = calculate_comparisons(
            paired, inventory, train_hashes, wad_predictions,
            args.iterations, args.batch_size, args.bootstrap_seed,
        )
        comparison_rows.extend(comparisons)
        interaction_rows.extend(interactions)
        atomic_csv(args.output_dir / "model_metrics.partial.csv", metrics_rows)
        atomic_csv(args.output_dir / "paired_comparisons.partial.csv", comparison_rows)
        atomic_csv(args.output_dir / "mf_gain_interaction.partial.csv", interaction_rows)

    adjust_holm(paired, comparison_rows, ("cohort", "representation", "metric"))
    adjust_holm(paired, interaction_rows, ("cohort", "metric"))
    atomic_csv(args.output_dir / "model_metrics.csv", metrics_rows)
    atomic_csv(args.output_dir / "paired_comparisons.csv", comparison_rows)
    atomic_csv(args.output_dir / "mf_gain_interaction.csv", interaction_rows)
    for name in ("model_metrics.partial.csv", "paired_comparisons.partial.csv", "mf_gain_interaction.partial.csv"):
        path = args.output_dir / name
        if path.exists():
            path.unlink()
    atomic_text(args.output_dir / "report.md", report_text(comparison_rows, interaction_rows, args.iterations))
    atomic_json(manifest_path, {
        "description": "LR fixed-tree fine/coarse comparison with MF and MF+WAD",
        "inputs": inputs,
        "settings": settings,
        "model_rows": len(metrics_rows),
        "paired_rows": len(comparison_rows),
        "interaction_rows": len(interaction_rows),
        "model_metrics_sha256": sha256_file(args.output_dir / "model_metrics.csv"),
        "paired_comparisons_sha256": sha256_file(args.output_dir / "paired_comparisons.csv"),
        "mf_gain_interaction_sha256": sha256_file(args.output_dir / "mf_gain_interaction.csv"),
        "report_sha256": sha256_file(args.output_dir / "report.md"),
    })
    print(f"wrote {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
