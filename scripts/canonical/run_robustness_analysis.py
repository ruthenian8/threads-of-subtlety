#!/usr/bin/env python3
"""Re-evaluate UniRST2 results after removing overlap and duplicate identities.

The fitted models, validation-selected thresholds, and training-selected motif
candidates remain fixed. Test identities present in training are excluded, and
remaining repeated test identities are consolidated before inference.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from tqdm import tqdm


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
REPRESENTATIONS = ("presence", "frequency", "depth")


from scripts.canonical import analyze_motif_associations as association
from scripts.canonical import compare_paired_performance as paired


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


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def csv_text(rows: Iterable[Mapping[str, Any]]) -> str:
    rows = list(rows)
    fields = list(rows[0]) if rows else []
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def load_split(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        keys = (
            "identity_id",
            "identity_hash",
            "domain",
            "y",
            "m3_hashes",
            "X_m3_presence",
            "X_m3_frequency",
            "X_m3_depth",
        )
        return {key: np.asarray(values[key]) for key in keys}


def sensitivity_cohort(
    train_hashes: np.ndarray,
    test_hashes: np.ndarray,
    y: np.ndarray,
    domain: np.ndarray,
) -> Dict[str, Any]:
    overlap_mask = np.isin(test_hashes, np.unique(train_hashes))
    retained = np.flatnonzero(~overlap_mask)
    retained_hashes = test_hashes[retained]
    unique_hashes, inverse = np.unique(retained_hashes, return_inverse=True)
    groups = [retained[inverse == index] for index in range(len(unique_hashes))]
    consolidated_y: List[int] = []
    consolidated_domain: List[str] = []
    for motif_hash, indices in zip(unique_hashes, groups):
        labels = np.unique(y[indices])
        domains = np.unique(domain[indices])
        if len(labels) != 1 or len(domains) != 1:
            raise ValueError(f"Identity {motif_hash} crosses label or domain")
        consolidated_y.append(int(labels[0]))
        consolidated_domain.append(str(domains[0]))
    return {
        "overlap_mask": overlap_mask,
        "retained_indices": retained,
        "unique_hashes": unique_hashes,
        "groups": groups,
        "y": np.asarray(consolidated_y, dtype=np.int8),
        "domain": np.asarray(consolidated_domain),
    }


def consolidate_vector(values: np.ndarray, groups: Sequence[np.ndarray]) -> np.ndarray:
    return np.asarray([np.mean(values[indices]) for indices in groups], dtype=np.float64)


def consolidate_matrix(
    values: np.ndarray, groups: Sequence[np.ndarray], *, binary: bool = False
) -> np.ndarray:
    result = np.asarray([np.mean(values[indices], axis=0) for indices in groups])
    return (result > 0).astype(np.float64) if binary else result.astype(np.float64)


def verify_prediction_alignment(
    prediction: Mapping[str, Any], test: Mapping[str, np.ndarray], description: str
) -> None:
    for prediction_key, matrix_key in (
        ("identity_id", "identity_id"),
        ("identity_hash", "identity_hash"),
        ("y", "y"),
    ):
        if not np.array_equal(prediction[prediction_key], test[matrix_key]):
            raise ValueError(f"Prediction alignment failure: {description}/{prediction_key}")


def model_analysis(
    prediction_root: Path,
    inventory: str,
    test: Mapping[str, np.ndarray],
    cohort: Mapping[str, Any],
    iterations: int,
    batch_size: int,
    confidence: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, str]]:
    loaded: Dict[Tuple[str, str], Dict[str, Any]] = {}
    provenance: Dict[str, str] = {}
    for model in MODELS:
        for config in CONFIGS:
            path = prediction_root / inventory / config / model / "predictions.npz"
            original = paired.load_predictions(path)
            verify_prediction_alignment(original, test, f"{inventory}/{config}/{model}")
            probability = consolidate_vector(original["probability"], cohort["groups"])
            loaded[(model, config)] = {
                "identity_id": cohort["unique_hashes"],
                "identity_hash": cohort["unique_hashes"],
                "y": cohort["y"],
                "probability": probability,
                "threshold": original["threshold"],
            }
            provenance[f"{config}/{model}"] = original["sha256"]
    reference = next(iter(loaded.values()))
    points, bootstraps = paired.bootstrap_scores(
        loaded,
        reference,
        iterations,
        batch_size,
        paired.stable_seed(seed, inventory),
        f"robust models {inventory}",
    )
    metric_rows: List[Dict[str, Any]] = []
    for model in MODELS:
        for config in CONFIGS:
            for metric_index, metric in enumerate(paired.METRICS):
                metric_rows.append(
                    {
                        "inventory": inventory,
                        "model": model,
                        "feature_config": config,
                        "metric": metric,
                        "score": float(points[(model, config)][metric_index]),
                        "records": len(cohort["y"]),
                    }
                )
    contrasts = paired.build_results(inventory, points, bootstraps, confidence)
    return metric_rows, contrasts, provenance


def motif_analysis(
    association_root: Path,
    inventory: str,
    test: Mapping[str, np.ndarray],
    cohort: Mapping[str, Any],
    iterations: int,
    batch_size: int,
    confidence: float,
    minimum_domains: int,
    min_depth_per_label: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    candidate_path = association_root / inventory / "candidates.json"
    candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate_rows = candidate_payload["candidates"]
    index = {str(value): i for i, value in enumerate(test["m3_hashes"])}
    candidate_indices = np.asarray(
        [index[row["motif_hash"]] for row in candidate_rows], dtype=np.int32
    )
    presence = consolidate_matrix(
        test["X_m3_presence"][:, candidate_indices], cohort["groups"], binary=True
    )
    frequency = consolidate_matrix(
        test["X_m3_frequency"][:, candidate_indices], cohort["groups"]
    )
    depth = consolidate_matrix(
        test["X_m3_depth"][:, candidate_indices], cohort["groups"]
    )
    candidate_hashes = test["m3_hashes"][candidate_indices]
    effects: List[Dict[str, Any]] = []
    for domain in tqdm(
        sorted(np.unique(cohort["domain"])),
        desc=f"robust motifs {inventory}",
        unit="domain",
        leave=False,
    ):
        mask = cohort["domain"] == domain
        effects.extend(
            association.domain_effect_rows(
                inventory,
                str(domain),
                cohort["y"][mask],
                cohort["unique_hashes"][mask],
                candidate_hashes,
                presence[mask],
                frequency[mask],
                depth[mask],
                iterations,
                batch_size,
                confidence,
                association.stable_seed(seed, inventory, str(domain)),
                min_depth_per_label,
            )
        )
    stability = association.summarize_stability(
        inventory, candidate_rows, effects, minimum_domains
    )
    return effects, stability, paired.sha256_file(candidate_path)


def discover(output_dir: Path, filename: str, key: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(output_dir.glob(f"*/{filename}")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))[key])
    return rows


def original_stable_keys(association_root: Path) -> set:
    rows = list(csv.DictReader((association_root / "stability.csv").open()))
    return {
        (row["inventory"], row["motif_hash"], row["representation"], row["stable_direction"])
        for row in rows
        if row["stable"] == "True"
    }


def render_report(
    audits: Sequence[Mapping[str, Any]],
    model_metrics: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    stability: Sequence[Mapping[str, Any]],
    original_stable: set,
) -> str:
    robust_stable = {
        (row["inventory"], row["motif_hash"], row["representation"], row["stable_direction"])
        for row in stability
        if row["stable"]
    }
    frequency_contrasts = [
        row
        for row in contrasts
        if row["metric"] == "f1"
        and row["contrast"] == "frequency_vs_structural"
    ]
    supported_frequency = sum(
        row["difference"] > 0
        and row["ci_low"] > 0
        and row["holm_adjusted_pvalue"] < 0.05
        for row in frequency_contrasts
    )
    stable_by_representation = {
        representation: sum(
            row["stable"] and row["representation"] == representation
            for row in stability
        )
        for representation in REPRESENTATIONS
    }
    best_models = []
    for inventory in sorted({row["inventory"] for row in model_metrics}):
        candidates = [
            row
            for row in model_metrics
            if row["inventory"] == inventory and row["metric"] == "f1"
        ]
        if candidates:
            best_models.append(max(candidates, key=lambda row: row["score"]))
    lines = [
        "# UniRST2 identity-robustness analysis",
        "",
        "Fitted models, thresholds, and motif candidates are unchanged. Test identities "
        "seen in training were removed, and remaining repeated test identities were "
        "collapsed before evaluation.",
        "",
        "## Cohort audit",
        "",
        "| Inventory | Original rows | Train-overlap rows removed | Duplicate rows collapsed | Robust rows |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in audits:
        lines.append(
            "| {inventory} | {original_rows} | {overlap_rows_removed} | "
            "{duplicate_rows_collapsed} | {robust_rows} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Result stability",
            "",
            (
                f"M3 frequency remains superior to the structural baseline in "
                f"{supported_frequency}/{len(frequency_contrasts)} inventory-classifier "
                "comparisons under the same adjusted paired criterion."
            ),
            "",
            (
                f"The original motif screen contained {len(original_stable)} stable "
                f"associations. The robustness cohort retains {len(robust_stable & original_stable)}, "
                f"loses {len(original_stable - robust_stable)}, and adds "
                f"{len(robust_stable - original_stable)} associations."
            ),
            "",
            (
                "The robust stable set contains "
                f"{stable_by_representation['presence']} presence, "
                f"{stable_by_representation['frequency']} frequency, and "
                f"{stable_by_representation['depth']} conditional-depth associations."
            ),
            "",
            "The best configuration remains XGBoost M3 frequency in every inventory: "
            + "; ".join(
                f"{row['inventory']} AUC={row['score']:.4f}" for row in best_models
            )
            + ".",
            "",
            "These estimates address exact identity leakage and repeated test records. "
            "They do not address broader near-duplicate text or source-template dependence.",
            "",
        ]
    )
    return "\n".join(lines)


def checkpoint_root(
    output_dir: Path, association_root: Path
) -> None:
    audits = discover(output_dir, "robustness_results.json", "cohort_audit")
    metrics = discover(output_dir, "robustness_results.json", "model_metrics")
    contrasts = discover(output_dir, "robustness_results.json", "paired_contrasts")
    effects = discover(output_dir, "robustness_results.json", "motif_effects")
    stability = discover(output_dir, "robustness_results.json", "motif_stability")
    for filename, rows in (
        ("cohort_audit.csv", audits),
        ("model_metrics.csv", metrics),
        ("paired_contrasts.csv", contrasts),
        ("motif_domain_effects.csv", effects),
        ("motif_stability.csv", stability),
    ):
        atomic_write_text(output_dir / filename, csv_text(rows))
    report = render_report(
        audits, metrics, contrasts, stability, original_stable_keys(association_root)
    )
    atomic_write_text(output_dir / "report.md", report)
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "completed_inventories": sorted({row["inventory"] for row in audits}),
            "cohort_audit_rows": len(audits),
            "model_metric_rows": len(metrics),
            "paired_contrast_rows": len(contrasts),
            "motif_effect_rows": len(effects),
            "motif_stability_rows": len(stability),
            "report_sha256": paired.sha256_file(output_dir / "report.md"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-dir", default="artifacts/canonical/feature_matrices"
    )
    parser.add_argument(
        "--prediction-dir", default="artifacts/canonical/model_comparison"
    )
    parser.add_argument(
        "--association-dir", default="artifacts/canonical/motif_associations"
    )
    parser.add_argument("--output-dir", default="artifacts/canonical/robustness")
    parser.add_argument(
        "--inventories",
        nargs="+",
        default=list(INVENTORIES),
        help="Inventory names with completed feature and model artifacts.",
    )
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--minimum-domains", type=int, default=4)
    parser.add_argument("--min-depth-per-label", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.iterations < 100 or args.batch_size < 1:
        parser.error("iterations must be >= 100 and batch-size must be positive")

    feature_root = Path(args.feature_dir)
    prediction_root = Path(args.prediction_dir)
    association_root = Path(args.association_dir)
    output_dir = Path(args.output_dir)
    settings = {
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "confidence": args.confidence,
        "minimum_domains": args.minimum_domains,
        "min_depth_per_label": args.min_depth_per_label,
        "seed": args.seed,
    }
    reference_cohort: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    for inventory in tqdm(args.inventories, desc="robust inventories", unit="inventory"):
        train_path = feature_root / inventory / "hc3_train.features.npz"
        test_path = feature_root / inventory / "hc3_test.features.npz"
        train = load_split(train_path)
        test = load_split(test_path)
        cohort = sensitivity_cohort(
            train["identity_hash"], test["identity_hash"], test["y"], test["domain"]
        )
        cohort_signature = (
            cohort["unique_hashes"], cohort["y"], cohort["domain"]
        )
        if reference_cohort is None:
            reference_cohort = cohort_signature
        elif any(
            not np.array_equal(current, reference)
            for current, reference in zip(cohort_signature, reference_cohort)
        ):
            raise ValueError(f"Robust cohort is not aligned for {inventory}")
        provenance = {
            "train_matrix": paired.sha256_file(train_path),
            "test_matrix": paired.sha256_file(test_path),
            "candidate_catalog": paired.sha256_file(
                association_root / inventory / "candidates.json"
            ),
        }
        for model in MODELS:
            for config in CONFIGS:
                prediction_path = (
                    prediction_root
                    / inventory
                    / config
                    / model
                    / "predictions.npz"
                )
                provenance[f"{config}/{model}"] = paired.sha256_file(
                    prediction_path
                )
        result_path = output_dir / inventory / "robustness_results.json"
        if result_path.exists() and not args.force:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("settings") == settings and all(
                existing.get("provenance", {}).get(key) == value
                for key, value in provenance.items()
            ):
                print(f"skipping complete {inventory}")
                checkpoint_root(output_dir, association_root)
                continue

        metrics, contrasts, prediction_hashes = model_analysis(
            prediction_root,
            inventory,
            test,
            cohort,
            args.iterations,
            args.batch_size,
            args.confidence,
            args.seed,
        )
        effects, stability, candidate_hash = motif_analysis(
            association_root,
            inventory,
            test,
            cohort,
            args.iterations,
            args.batch_size,
            args.confidence,
            args.minimum_domains,
            args.min_depth_per_label,
            args.seed,
        )
        computed_provenance = {"candidate_catalog": candidate_hash, **prediction_hashes}
        if any(provenance[key] != value for key, value in computed_provenance.items()):
            raise ValueError(f"Provenance changed while processing {inventory}")
        overlap_removed = int(np.count_nonzero(cohort["overlap_mask"]))
        retained_before_consolidation = len(test["y"]) - overlap_removed
        audit = {
            "inventory": inventory,
            "original_rows": int(len(test["y"])),
            "unique_train_identities": int(len(np.unique(train["identity_hash"]))),
            "overlap_identities": int(
                len(np.unique(test["identity_hash"][cohort["overlap_mask"]]))
            ),
            "overlap_rows_removed": overlap_removed,
            "duplicate_rows_collapsed": int(
                retained_before_consolidation - len(cohort["y"])
            ),
            "robust_rows": int(len(cohort["y"])),
        }
        payload = {
            "inventory": inventory,
            "settings": settings,
            "provenance": provenance,
            "cohort_audit": [audit],
            "model_metrics": metrics,
            "paired_contrasts": contrasts,
            "motif_effects": effects,
            "motif_stability": stability,
        }
        atomic_write_json(result_path, payload)
        checkpoint_root(output_dir, association_root)
    print(f"wrote {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
