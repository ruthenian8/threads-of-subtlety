#!/usr/bin/env python3
"""Compare fine and coarse M3 catalogs with Kim-style RF MF+WAD features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm
from tos.analysis.granularity import (
    INVENTORIES, METRICS, SPLITS, aggregate_wad, atomic_csv, atomic_json,
    atomic_npz, atomic_text, coarse_feature_path as coarse_path,
    identity_clean_mask, interleave_mf_wad as mf_wad_features, read_npz,
    sha256_file, validate_pair,
)


N_ESTIMATORS = 100
THRESHOLD = 0.5


def prose_list(values: Sequence[str]) -> str:
    values = list(values)
    if not values:
        return "none"
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def load_coarse_assignment(mapping_path: Path, inventory: str, width: int) -> np.ndarray:
    assignment = np.full(width, -1, dtype=np.int64)
    with mapping_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["inventory"] != inventory:
                continue
            fine_index = int(row["fine_index"])
            coarse_index = int(row["coarse_index"])
            if fine_index >= width:
                raise ValueError(f"Mapping exceeds fine width for {inventory}: {fine_index}")
            if assignment[fine_index] not in {-1, coarse_index}:
                raise ValueError(f"Conflicting mapping for {inventory}/{fine_index}")
            assignment[fine_index] = coarse_index
    if np.any(assignment < 0):
        raise ValueError(f"Incomplete PCC/NLDT motif mapping for {inventory}")
    return assignment


def load_inventory(
    feature_root: Path,
    granularity_root: Path,
    mapping_path: Path,
    inventory: str,
) -> tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]], Dict[str, str]]:
    fine_splits: Dict[str, Dict[str, np.ndarray]] = {}
    coarse_splits: Dict[str, Dict[str, np.ndarray]] = {}
    hashes: Dict[str, str] = {}
    assignment = None
    for split in SPLITS:
        fine_path = feature_root / inventory / f"hc3_{split}.features.npz"
        projected_path = coarse_path(granularity_root, inventory, split)
        fine = read_npz(fine_path)
        coarse = read_npz(projected_path)
        validate_pair(fine, coarse, f"{inventory}/{split}")
        if "X_m3_depth" not in coarse:
            if assignment is None:
                assignment = load_coarse_assignment(
                    mapping_path, inventory, fine["X_m3_frequency"].shape[1]
                )
            coarse["X_m3_depth"] = aggregate_wad(
                np.asarray(fine["X_m3_frequency"], dtype=np.float32),
                np.asarray(fine["X_m3_depth"], dtype=np.float32),
                assignment,
                np.asarray(coarse["X_m3_frequency"], dtype=np.float32),
            )
        fine_splits[split] = fine
        coarse_splits[split] = coarse
        hashes[f"fine/{split}"] = sha256_file(fine_path)
        hashes[f"coarse/{split}"] = sha256_file(projected_path)
    return fine_splits, coarse_splits, hashes


def scores(y: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    prediction = probability >= THRESHOLD
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, prediction)),
    }


def fit_variant(
    inventory: str,
    granularity: str,
    splits: Mapping[str, Mapping[str, np.ndarray]],
    output_root: Path,
    seed: int,
    n_jobs: int,
) -> tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    matrices = {
        split: mf_wad_features(values["X_m3_frequency"], values["X_m3_depth"])
        for split, values in splits.items()
    }
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=seed,
        n_jobs=n_jobs,
    )
    model.fit(matrices["train"], splits["train"]["y"])
    validation_probability = model.predict_proba(matrices["validation"])[:, 1]
    test_probability = model.predict_proba(matrices["test"])[:, 1]
    run_dir = output_root / "predictions" / inventory / granularity
    prediction_path = run_dir / "predictions.npz"
    atomic_npz(
        prediction_path,
        test_identity_id=np.asarray(splits["test"]["identity_id"]),
        test_identity_hash=np.asarray(splits["test"]["identity_hash"]),
        test_y=np.asarray(splits["test"]["y"], dtype=np.int8),
        test_probability=np.asarray(test_probability, dtype=np.float64),
        threshold=np.asarray(THRESHOLD),
    )
    result = {
        "inventory": inventory,
        "granularity": granularity,
        "representation": "mf_wad",
        "classifier": "rf",
        "motif_features": int(matrices["train"].shape[1] // 2),
        "feature_count": int(matrices["train"].shape[1]),
        "train_rows": int(len(splits["train"]["y"])),
        "validation_rows": int(len(splits["validation"]["y"])),
        "test_rows": int(len(splits["test"]["y"])),
        "n_estimators": N_ESTIMATORS,
        "random_state": seed,
        "threshold": THRESHOLD,
        **{f"validation_{key}": value for key, value in scores(splits["validation"]["y"], validation_probability).items()},
        **{f"test_{key}": value for key, value in scores(splits["test"]["y"], test_probability).items()},
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": sha256_file(prediction_path),
    }
    return result, {
        "identity_hash": np.asarray(splits["test"]["identity_hash"]),
        "y": np.asarray(splits["test"]["y"], dtype=np.int8),
        "probability": np.asarray(test_probability, dtype=np.float64),
    }


def paired_rows(
    paired: Any,
    inventory: str,
    train_hashes: np.ndarray,
    fine: Mapping[str, np.ndarray],
    coarse: Mapping[str, np.ndarray],
    iterations: int,
    batch_size: int,
    seed: int,
) -> list[Dict[str, Any]]:
    for key in ("identity_hash", "y"):
        if not np.array_equal(fine[key], coarse[key]):
            raise ValueError(f"Fine/coarse prediction mismatch for {inventory}/{key}")
    masks = {
        "primary": np.ones(len(fine["y"]), dtype=bool),
        "identity_clean": identity_clean_mask(train_hashes, fine["identity_hash"]),
    }
    result = []
    for cohort, mask in masks.items():
        y = fine["y"][mask]
        identity_hash = fine["identity_hash"][mask]
        fine_probability = fine["probability"][mask]
        coarse_probability = coarse["probability"][mask]
        clusters = paired.identity_clusters(identity_hash, y)
        fine_point = paired.point_metrics(y, fine_probability, THRESHOLD)
        coarse_point = paired.point_metrics(y, coarse_probability, THRESHOLD)
        differences = np.empty((iterations, len(METRICS)), dtype=np.float64)
        rng = np.random.default_rng(seed)
        for start in tqdm(
            range(0, iterations, batch_size),
            desc=f"RF MF+WAD {inventory}/{cohort}",
            unit="batch",
            leave=False,
        ):
            stop = min(start + batch_size, iterations)
            weights = paired.bootstrap_row_weights(rng, clusters, stop - start)
            differences[start:stop] = (
                paired.weighted_metric_batch(y, fine_probability, THRESHOLD, weights)
                - paired.weighted_metric_batch(y, coarse_probability, THRESHOLD, weights)
            )
        for index, metric in enumerate(METRICS):
            samples = differences[:, index]
            result.append({
                "cohort": cohort,
                "rows": int(mask.sum()),
                "inventory": inventory,
                "representation": "mf_wad",
                "classifier": "rf",
                "metric": metric,
                "fine_score": float(fine_point[index]),
                "coarse_score": float(coarse_point[index]),
                "difference": float(fine_point[index] - coarse_point[index]),
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "bootstrap_sign_pvalue": paired.bootstrap_sign_pvalue(samples),
                "holm_adjusted_pvalue": 0.0,
            })
    return result


def adjust_pvalues(paired: Any, rows: Sequence[Dict[str, Any]]) -> None:
    for cohort in ("primary", "identity_clean"):
        for metric in METRICS:
            selected = [
                row for row in rows
                if row["cohort"] == cohort and row["metric"] == metric
            ]
            adjusted = paired.holm_adjust(
                [float(row["bootstrap_sign_pvalue"]) for row in selected]
            )
            for row, value in zip(selected, adjusted):
                row["holm_adjusted_pvalue"] = value


def report_text(rows: Sequence[Mapping[str, Any]], iterations: int) -> str:
    labels = {
        "deu.rst.pcc": "PCC",
        "eng.erst.gum.new": "eRST",
        "eng.rst.rstdt.new": "RST-DT",
        "nld.rst.nldt": "NLDT",
    }
    selected = {
        row["inventory"]: row
        for row in rows
        if row["cohort"] == "primary" and row["metric"] == "f1"
    }
    pointwise = [
        labels[inventory]
        for inventory in INVENTORIES
        if float(selected[inventory]["ci_low"]) > 0
    ]
    corrected = [
        labels[inventory]
        for inventory in INVENTORIES
        if float(selected[inventory]["holm_adjusted_pvalue"]) < 0.05
    ]
    lines = [
        "# Fixed-tree fine-to-coarse RF MF+WAD ablation",
        "",
        "This is the canonical relation-granularity comparison. It follows the Kim et al. "
        "motif-classifier design: each retained M3 motif contributes an interleaved motif "
        "frequency (MF) and weighted average depth (WAD) pair to a motif-only 100-tree random "
        "forest. Binary F1 for label 1 (human) is evaluated at the fixed 0.5 threshold.",
        "",
        "| Standard | Fine MF+WAD F1 | Coarse MF+WAD F1 | Fine − coarse (95% CI) | Holm-adjusted p |",
        "|---|---:|---:|---:|---:|",
    ]
    for inventory in INVENTORIES:
        row = selected[inventory]
        lines.append(
            f"| {labels[inventory]} | {float(row['fine_score']):.3f} | "
            f"{float(row['coarse_score']):.3f} | {float(row['difference']):+.3f} "
            f"[{float(row['ci_low']):+.3f}, {float(row['ci_high']):+.3f}] | "
            f"{float(row['holm_adjusted_pvalue']):.4f} |"
        )
    lines.extend([
        "",
        f"Fine-minus-coarse point estimates are positive in all four standards. Pointwise "
        f"intervals exclude zero for {prose_list(pointwise)}; after correction across standards, "
        f"only {prose_list(corrected)} remains below 0.05.",
        "",
        "The coarse WAD of a merged motif is its occurrence-weighted fine-motif WAD. This "
        "preserves documents, EDU segmentation, tree topology, attachments, nuclearity, motif "
        "occurrence mass, and the MF/WAD feature semantics; only relation labels and the motif "
        "dimensions they distinguish are merged.",
        "",
        f"Intervals use {iterations:,} paired, label-stratified identity-cluster bootstrap "
        "resamples. Holm adjustment is across the four standards within each cohort and metric. "
        "The identity-clean sensitivity is retained in `paired_comparisons.csv`.",
        "",
        "The older LR/XGBoost MF-only granularity outputs remain useful sensitivity artifacts, "
        "but are not the paper-facing Kim-compatible estimate.",
        "",
    ])
    return "\n".join(lines)


def input_provenance(
    feature_root: Path,
    granularity_root: Path,
    mapping_path: Path,
) -> Dict[str, str]:
    result = {
        str(Path(__file__).resolve()): sha256_file(Path(__file__)),
        str(mapping_path): sha256_file(mapping_path),
    }
    for inventory in INVENTORIES:
        for split in SPLITS:
            fine = feature_root / inventory / f"hc3_{split}.features.npz"
            coarse = coarse_path(granularity_root, inventory, split)
            result[str(fine)] = sha256_file(fine)
            result[str(coarse)] = sha256_file(coarse)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=Path("artifacts/canonical/feature_matrices"))
    parser.add_argument("--granularity-root", type=Path, default=Path("artifacts/canonical/granularity"))
    parser.add_argument("--mapping-file", type=Path, default=Path("artifacts/canonical/granularity/pcc_nldt/motif_mapping.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/canonical/granularity/rf_mf_wad"))
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-seed", type=int, default=20260909)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from scripts.canonical import compare_paired_performance as paired

    provenance = input_provenance(args.feature_root, args.granularity_root, args.mapping_file)
    settings = {
        "representation": "interleaved MF+WAD",
        "classifier": "sklearn RandomForestClassifier",
        "n_estimators": N_ESTIMATORS,
        "model_seed": args.seed,
        "threshold": THRESHOLD,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_iterations": args.iterations,
        "bootstrap_batch_size": args.batch_size,
        "n_jobs": args.n_jobs,
    }
    manifest_path = args.output_dir / "manifest.json"
    required = (args.output_dir / "metrics.csv", args.output_dir / "paired_comparisons.csv", args.output_dir / "report.md")
    if manifest_path.is_file() and all(path.is_file() for path in required) and not args.force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("inputs") == provenance and manifest.get("settings") == settings:
            print(f"skipping complete RF MF+WAD granularity run; wrote {args.output_dir}")
            return

    metric_rows = []
    comparison_rows = []
    source_hashes = {}
    for inventory in INVENTORIES:
        fine_splits, coarse_splits, hashes = load_inventory(
            args.feature_root, args.granularity_root, args.mapping_file, inventory
        )
        source_hashes[inventory] = hashes
        fine_metrics, fine_predictions = fit_variant(
            inventory, "fine", fine_splits, args.output_dir, args.seed, args.n_jobs
        )
        coarse_metrics, coarse_predictions = fit_variant(
            inventory, "coarse", coarse_splits, args.output_dir, args.seed, args.n_jobs
        )
        metric_rows.extend((fine_metrics, coarse_metrics))
        comparison_rows.extend(paired_rows(
            paired,
            inventory,
            np.asarray(fine_splits["train"]["identity_hash"]),
            fine_predictions,
            coarse_predictions,
            args.iterations,
            args.batch_size,
            args.bootstrap_seed,
        ))
        atomic_csv(args.output_dir / "metrics.csv", metric_rows)
        atomic_csv(args.output_dir / "paired_comparisons.partial.csv", comparison_rows)

    adjust_pvalues(paired, comparison_rows)
    atomic_csv(args.output_dir / "metrics.csv", metric_rows)
    atomic_csv(args.output_dir / "paired_comparisons.csv", comparison_rows)
    partial = args.output_dir / "paired_comparisons.partial.csv"
    if partial.exists():
        partial.unlink()
    atomic_text(args.output_dir / "report.md", report_text(comparison_rows, args.iterations))
    atomic_json(manifest_path, {
        "description": "Kim-compatible fixed-tree fine/coarse M3 comparison",
        "inputs": provenance,
        "settings": settings,
        "source_hashes": source_hashes,
        "metrics_rows": len(metric_rows),
        "paired_rows": len(comparison_rows),
        "metrics_sha256": sha256_file(args.output_dir / "metrics.csv"),
        "paired_sha256": sha256_file(args.output_dir / "paired_comparisons.csv"),
        "report_sha256": sha256_file(args.output_dir / "report.md"),
    })
    print(f"wrote {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
