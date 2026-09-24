#!/usr/bin/env python3
"""Compare saved UniRST2 predictions between relation inventories.

No models are fitted.  The same identity-clean cohorts and paired bootstrap
resamples are used for both members of every inventory pair.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from tqdm import tqdm


INVENTORIES = (
    "deu.rst.pcc",
    "eng.erst.gum",
    "eng.rst.rstdt",
    "nld.rst.nldt",
)
MODELS = ("lr", "xgb")
METRICS = ("roc_auc", "average_precision", "balanced_accuracy")
COHORTS = ("primary", "identity_clean", "near_duplicate_clean")


def load_paired_module():
    path = Path(__file__).resolve().with_name("13_compare_unirst2_paired_performance.py")
    spec = importlib.util.spec_from_file_location("unirst2_paired", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def write_cases_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> bool:
    """Write the planned machine-readable case table when Arrow is available."""
    try:
        import pandas as pd
    except ImportError:
        return False
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    os.replace(temporary, path)
    return True


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def correlation(left: np.ndarray, right: np.ndarray, spearman: bool = False) -> float:
    if spearman:
        left, right = rankdata(left), rankdata(right)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def load_prediction(path: Path) -> Dict[str, Any]:
    loaded = paired.load_predictions(path)
    return loaded


def load_inputs(
    prediction_dir: Path,
    feature_dir: Path,
    inventories: Sequence[str],
    model: str,
) -> Tuple[Dict[str, Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    predictions: Dict[str, Dict[str, Any]] = {}
    reference_ids = reference_hashes = reference_y = None
    domains = None
    for inventory in inventories:
        prediction_path = prediction_dir / inventory / "m3_frequency" / model / "predictions.npz"
        current = load_prediction(prediction_path)
        matrix_path = feature_dir / inventory / "hc3_test.features.npz"
        with np.load(matrix_path, allow_pickle=False) as matrix:
            for key, values in (
                ("identity_id", matrix["identity_id"]),
                ("identity_hash", matrix["identity_hash"]),
                ("y", matrix["y"]),
            ):
                prediction_key = {"identity_id": "identity_id", "identity_hash": "identity_hash", "y": "y"}[key]
                if not np.array_equal(current[prediction_key], values):
                    raise ValueError(f"prediction/matrix alignment mismatch: {inventory}/{model}/{key}")
            current["domain"] = np.asarray(matrix["domain"])
            current["source"] = np.asarray(matrix["source"])
        if reference_ids is None:
            reference_ids = current["identity_id"]
            reference_hashes = current["identity_hash"]
            reference_y = current["y"]
            domains = current["domain"]
        else:
            for key, expected, actual in (
                ("identity_id", reference_ids, current["identity_id"]),
                ("identity_hash", reference_hashes, current["identity_hash"]),
                ("y", reference_y, current["y"]),
                ("domain", domains, current["domain"]),
            ):
                if not np.array_equal(expected, actual):
                    raise ValueError(f"cross-inventory alignment mismatch: {inventory}/{model}/{key}")
        predictions[inventory] = current
    assert reference_hashes is not None and reference_y is not None and domains is not None
    return predictions, reference_hashes, reference_y, domains


def train_hashes(feature_dir: Path, inventory: str) -> set:
    path = feature_dir / inventory / "hc3_train.features.npz"
    with np.load(path, allow_pickle=False) as matrix:
        return set(matrix["identity_hash"].tolist())


def near_train_test_hashes(cluster_path: Path, train: set) -> set:
    blocked = set()
    if not cluster_path.exists():
        raise FileNotFoundError(cluster_path)
    with cluster_path.open(encoding="utf-8") as handle:
        for line in handle:
            cluster = json.loads(line)
            members = cluster.get("members", [])
            member_hashes = {str(member["identity_hash"]) for member in members}
            if member_hashes & train:
                blocked.update(
                    str(member["identity_hash"])
                    for member in members
                    if member.get("split") == "test"
                )
    return blocked


def consolidated_cohort(
    predictions: Mapping[str, Mapping[str, Any]],
    identity_hashes: np.ndarray,
    y: np.ndarray,
    domains: np.ndarray,
    excluded: set,
) -> Dict[str, Any]:
    kept = [index for index, value in enumerate(identity_hashes) if value not in excluded]
    groups: Dict[str, List[int]] = {}
    for index in kept:
        groups.setdefault(str(identity_hashes[index]), []).append(index)
    hashes, labels, domain_values = [], [], []
    probabilities = {inventory: [] for inventory in predictions}
    thresholds = {}
    for inventory, current in predictions.items():
        thresholds[inventory] = float(current["threshold"])
    for value in sorted(groups):
        indices = groups[value]
        labels_seen = np.unique(y[indices])
        domains_seen = np.unique(domains[indices])
        if len(labels_seen) != 1 or len(domains_seen) != 1:
            raise ValueError(f"identity crosses label/domain: {value}")
        hashes.append(value)
        labels.append(int(labels_seen[0]))
        domain_values.append(str(domains_seen[0]))
        for inventory, current in predictions.items():
            probabilities[inventory].append(float(np.mean(current["probability"][indices])))
    return {
        "identity_hash": np.asarray(hashes),
        "y": np.asarray(labels, dtype=np.int8),
        "domain": np.asarray(domain_values),
        "probability": {key: np.asarray(value, dtype=np.float64) for key, value in probabilities.items()},
        "threshold": thresholds,
        "source_rows": len(kept),
    }


def stable_pair_seed(seed: int, cohort: str, model: str, left: str, right: str) -> int:
    payload = f"{seed}:{cohort}:{model}:{left}:{right}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def pair_results(
    cohort: Mapping[str, Any],
    left: str,
    right: str,
    model: str,
    iterations: int,
    batch_size: int,
    confidence: float,
    seed: int,
) -> List[Dict[str, Any]]:
    y = cohort["y"]
    left_probability = cohort["probability"][left]
    right_probability = cohort["probability"][right]
    left_threshold = cohort["threshold"][left]
    right_threshold = cohort["threshold"][right]
    clusters = paired.identity_clusters(cohort["identity_hash"], y)
    rng = np.random.default_rng(stable_pair_seed(seed, cohort["name"], model, left, right))
    left_boot = np.empty((iterations, len(METRICS)), dtype=np.float64)
    right_boot = np.empty_like(left_boot)
    for start in tqdm(
        range(0, iterations, batch_size),
        desc=f"bootstrap {cohort['name']}/{model}/{left}-{right}",
        unit="batch",
        leave=False,
    ):
        stop = min(start + batch_size, iterations)
        weights = paired.bootstrap_row_weights(rng, clusters, stop - start)
        left_boot[start:stop] = paired.weighted_metric_batch(y, left_probability, left_threshold, weights)
        right_boot[start:stop] = paired.weighted_metric_batch(y, right_probability, right_threshold, weights)
    left_point = paired.point_metrics(y, left_probability, left_threshold)
    right_point = paired.point_metrics(y, right_probability, right_threshold)
    rows = []
    alpha = 1.0 - confidence
    for metric_index, metric in enumerate(METRICS):
        difference = left_boot[:, metric_index] - right_boot[:, metric_index]
        rows.append(
            {
                "cohort": cohort["name"],
                "model": model,
                "left_inventory": left,
                "right_inventory": right,
                "metric": metric,
                "records": int(len(y)),
                "left_score": float(left_point[metric_index]),
                "right_score": float(right_point[metric_index]),
                "difference": float(left_point[metric_index] - right_point[metric_index]),
                "ci_low": float(np.quantile(difference, alpha / 2.0)),
                "ci_high": float(np.quantile(difference, 1.0 - alpha / 2.0)),
                "probability_positive": float(np.mean(difference > 0)),
                "bootstrap_sign_pvalue": paired.bootstrap_sign_pvalue(difference),
            }
        )
    return rows


def domain_rows(cohort: Mapping[str, Any], left: str, right: str, model: str) -> List[Dict[str, Any]]:
    y = cohort["y"]
    lp, rp = cohort["probability"][left], cohort["probability"][right]
    lt, rt = cohort["threshold"][left], cohort["threshold"][right]
    left_pred, right_pred = lp >= lt, rp >= rt
    result = []
    for domain in sorted(np.unique(cohort["domain"])):
        mask = cohort["domain"] == domain
        left_wrong, right_wrong = left_pred[mask] != y[mask], right_pred[mask] != y[mask]
        result.append(
            {
                "cohort": cohort["name"],
                "model": model,
                "left_inventory": left,
                "right_inventory": right,
                "domain": str(domain),
                "records": int(np.sum(mask)),
                "score_pearson": correlation(lp[mask], rp[mask]),
                "score_spearman": correlation(lp[mask], rp[mask], spearman=True),
                "threshold_disagreement": float(np.mean(left_pred[mask] != right_pred[mask])),
                "both_wrong": int(np.sum(left_wrong & right_wrong)),
                "left_only_wrong": int(np.sum(left_wrong & ~right_wrong)),
                "right_only_wrong": int(np.sum(right_wrong & ~left_wrong)),
            }
        )
    return result


def case_rows(cohort: Mapping[str, Any], left: str, right: str, model: str) -> List[Dict[str, Any]]:
    y = cohort["y"]
    lp, rp = cohort["probability"][left], cohort["probability"][right]
    lt, rt = cohort["threshold"][left], cohort["threshold"][right]
    left_pred, right_pred = lp >= lt, rp >= rt
    rows = []
    for index, identity in enumerate(cohort["identity_hash"]):
        rows.append(
            {
                "cohort": cohort["name"],
                "model": model,
                "left_inventory": left,
                "right_inventory": right,
                "identity_hash": str(identity),
                "domain": str(cohort["domain"][index]),
                "y": int(y[index]),
                "left_probability": float(lp[index]),
                "right_probability": float(rp[index]),
                "left_threshold": lt,
                "right_threshold": rt,
                "left_prediction": int(left_pred[index]),
                "right_prediction": int(right_pred[index]),
                "disagreement": int(left_pred[index] != right_pred[index]),
                "left_wrong": int(left_pred[index] != y[index]),
                "right_wrong": int(right_pred[index] != y[index]),
            }
        )
    return rows


def render_report(metrics: Sequence[Mapping[str, Any]], domains: Sequence[Mapping[str, Any]], cohort_sizes: Mapping[str, Any]) -> str:
    auc = [row for row in metrics if row["metric"] == "roc_auc"]
    lines = [
        "# UniRST2 between-inventory prediction comparison",
        "",
        "Saved M3-frequency predictions are compared on identical identity-cluster cohorts; no models are refit.",
        "The primary HC3 test results are exploratory because prior test outcomes were inspected.",
        "",
        "## Cohorts",
        "",
        "| Cohort | Records | Excluded test identities |",
        "|---|---:|---:|",
    ]
    for name, values in cohort_sizes.items():
        lines.append(f"| {name} | {values['records']} | {values['excluded_test_identities']} |")
    lines.extend(["", "## ROC-AUC differences", "", "| Cohort | Model | Pair (left-right) | Difference | 95% CI | Holm p |", "|---|---|---|---:|---:|---:|"])
    for row in auc:
        lines.append(
            f"| {row['cohort']} | {row['model']} | {row['left_inventory']} – {row['right_inventory']} | "
            f"{row['difference']:+.4f} | [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {row.get('holm_adjusted_pvalue', float('nan')):.4g} |"
        )
    robust = sum(row.get("ci_excludes_zero", False) and row.get("holm_adjusted_pvalue", 1.0) < 0.05 for row in auc)
    lines.extend(["", f"{robust}/{len(auc)} ROC-AUC pairwise contrasts have an adjusted interval excluding zero.", "", "## Interpretation", "", "Positive differences favor the left inventory. Correlation and disagreement indicate non-equivalence but do not establish predictive complementarity; that requires the incremental modeling stage.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, default=Path("artifacts/unirst2_model_comparison"))
    parser.add_argument("--feature-dir", type=Path, default=Path("artifacts/unirst2_feature_matrices"))
    parser.add_argument("--audit-dir", type=Path, default=Path("artifacts/unirst2_complementarity"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/unirst2_complementarity/prediction_comparison"))
    parser.add_argument("--inventories", nargs="+", choices=INVENTORIES, default=list(INVENTORIES))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--cohorts", nargs="+", choices=COHORTS, default=list(COHORTS))
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if len(args.inventories) != 4:
        raise ValueError("stage 2 requires all four inventories")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = train_hashes(args.feature_dir, args.inventories[0])
    near_blocked = near_train_test_hashes(args.audit_dir / "near_duplicate_clusters.jsonl", train)
    all_metrics: List[Dict[str, Any]] = []
    all_domains: List[Dict[str, Any]] = []
    all_cases: List[Dict[str, Any]] = []
    cohort_sizes: Dict[str, Any] = {}
    source_hashes: Dict[str, str] = {}
    for model in args.models:
        predictions, identity_hashes, y, domains = load_inputs(
            args.prediction_dir, args.feature_dir, args.inventories, model
        )
        for inventory in args.inventories:
            source_hashes[f"{inventory}/m3_frequency/{model}"] = predictions[inventory]["sha256"]
        excluded_by_name = {
            "primary": set(),
            "identity_clean": train,
            "near_duplicate_clean": near_blocked,
        }
        cohorts = {}
        for name in args.cohorts:
            cohort = consolidated_cohort(
                predictions, identity_hashes, y, domains, excluded_by_name[name]
            )
            cohort["name"] = name
            cohorts[name] = cohort
            cohort_sizes.setdefault(name, {
                "records": int(len(cohort["y"])),
                "source_rows": int(cohort["source_rows"]),
                "excluded_test_identities": int(len(excluded_by_name[name] & set(identity_hashes.tolist()))),
            })
        for cohort_name, cohort in cohorts.items():
            for left_index, left in enumerate(args.inventories):
                for right in args.inventories[left_index + 1 :]:
                    all_metrics.extend(
                        pair_results(
                            cohort, left, right, model, args.iterations, args.batch_size,
                            args.confidence, args.seed,
                        )
                    )
                    all_domains.extend(domain_rows(cohort, left, right, model))
                    if cohort_name in ("identity_clean", "near_duplicate_clean"):
                        all_cases.extend(case_rows(cohort, left, right, model))

    for metric in METRICS:
        for cohort_name in args.cohorts:
            for model in args.models:
                family = [
                    row for row in all_metrics
                    if row["metric"] == metric and row["cohort"] == cohort_name and row["model"] == model
                ]
                adjusted = paired.holm_adjust([row["bootstrap_sign_pvalue"] for row in family]) if family else []
                for row, value in zip(family, adjusted):
                    row["holm_adjusted_pvalue"] = value
                    row["ci_excludes_zero"] = bool(row["ci_low"] > 0 or row["ci_high"] < 0)

    write_csv(args.output_dir / "between_inventory_contrasts.csv", all_metrics)
    write_csv(args.output_dir / "domain_disagreement.csv", all_domains)
    write_csv(args.output_dir / "prediction_cases.csv", all_cases)
    parquet_written = write_cases_parquet(args.output_dir / "prediction_cases.parquet", all_cases)
    summary = {
        "stage": "unirst2_between_inventory_prediction_comparison",
        "inventories": list(args.inventories),
        "models": list(args.models),
        "cohorts": list(args.cohorts),
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "confidence": args.confidence,
        "seed": args.seed,
        "cohort_sizes": cohort_sizes,
        "near_duplicate_blocked_test_identities": len(near_blocked),
        "prediction_hashes": source_hashes,
        "outputs": {
            "between_inventory_contrasts": str((args.output_dir / "between_inventory_contrasts.csv").resolve()),
            "domain_disagreement": str((args.output_dir / "domain_disagreement.csv").resolve()),
            "prediction_cases": str((args.output_dir / "prediction_cases.csv").resolve()),
        },
    }
    summary["output_hashes"] = {
        name: sha256_file(args.output_dir / name)
        for name in ("between_inventory_contrasts.csv", "domain_disagreement.csv", "prediction_cases.csv")
    }
    if parquet_written:
        summary["outputs"]["prediction_cases_parquet"] = str(
            (args.output_dir / "prediction_cases.parquet").resolve()
        )
        summary["output_hashes"]["prediction_cases.parquet"] = sha256_file(
            args.output_dir / "prediction_cases.parquet"
        )
    atomic_json(args.output_dir / "manifest.json", summary)
    atomic_text(args.output_dir / "report.md", render_report(all_metrics, all_domains, cohort_sizes))
    print(json.dumps({"contrasts": len(all_metrics), "domain_rows": len(all_domains), "case_rows": len(all_cases)}, indent=2))


if __name__ == "__main__":
    main()
