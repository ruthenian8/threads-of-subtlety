#!/usr/bin/env python3
"""Estimate paired test-performance differences from saved UniRST2 predictions.

No model is fitted here. Test rows are resampled in label-stratified identity
clusters, so repeated copies of the same text/source/label identity remain in
the same bootstrap unit. The same resamples are used for every feature model
within an inventory, making all reported differences paired.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
METRICS = ("roc_auc", "average_precision", "balanced_accuracy", "f1", "macro_f1")
CONTRASTS = (
    ("relation_vs_structural", "relation_frequency", "structural"),
    ("presence_vs_structural", "m3_presence", "structural"),
    ("frequency_vs_structural", "m3_frequency", "structural"),
    ("depth_vs_structural", "m3_depth", "structural"),
    ("frequency_vs_presence", "m3_frequency", "m3_presence"),
    ("depth_vs_presence", "m3_depth", "m3_presence"),
    ("frequency_vs_depth", "m3_frequency", "m3_depth"),
)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(seed: int, inventory: str) -> int:
    payload = f"{seed}:{inventory}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def load_predictions(path: Path) -> Dict[str, Any]:
    metrics_path = path.with_name("metrics.json")
    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected_hash = metadata["artifacts"]["predictions_sha256"]
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(f"Prediction hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as values:
        result = {
            "identity_id": np.asarray(values["test_identity_id"]),
            "identity_hash": np.asarray(values["test_identity_hash"]),
            "y": np.asarray(values["test_y"], dtype=np.int8),
            "probability": np.asarray(values["test_probability"], dtype=np.float64),
            "threshold": float(values["threshold"]),
        }
    if not np.isfinite(result["probability"]).all():
        raise ValueError(f"Non-finite probabilities: {path}")
    result["sha256"] = actual_hash
    result["reported_test_metrics"] = metadata["test_metrics"]
    return result


def load_inventory(
    prediction_root: Path, inventory: str
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[str, Any]]:
    loaded: Dict[Tuple[str, str], Dict[str, Any]] = {}
    reference: Dict[str, Any] | None = None
    for model in MODELS:
        for config in CONFIGS:
            path = prediction_root / inventory / config / model / "predictions.npz"
            current = load_predictions(path)
            if reference is None:
                reference = current
            for key in ("identity_id", "identity_hash", "y"):
                if not np.array_equal(current[key], reference[key]):
                    raise ValueError(
                        f"Unaligned {key} in {inventory}/{config}/{model}"
                    )
            loaded[(model, config)] = current
    assert reference is not None
    return loaded, reference


def identity_clusters(identity_hash: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    unique, inverse = np.unique(identity_hash, return_inverse=True)
    cluster_y = np.empty(len(unique), dtype=np.int8)
    for cluster in range(len(unique)):
        labels = np.unique(y[inverse == cluster])
        if len(labels) != 1:
            raise ValueError(f"Identity cluster {unique[cluster]} crosses labels")
        cluster_y[cluster] = labels[0]
    by_label = {label: np.flatnonzero(cluster_y == label) for label in (0, 1)}
    if any(len(indices) == 0 for indices in by_label.values()):
        raise ValueError("Both labels are required for stratified bootstrap")
    return {
        "unique": unique,
        "inverse": inverse,
        "cluster_y": cluster_y,
        "by_label": by_label,
    }


def bootstrap_row_weights(
    rng: np.random.Generator,
    clusters: Mapping[str, Any],
    size: int,
) -> np.ndarray:
    cluster_weights = np.zeros((size, len(clusters["unique"])), dtype=np.int16)
    for label in (0, 1):
        indices = clusters["by_label"][label]
        draws = rng.multinomial(
            len(indices), np.full(len(indices), 1.0 / len(indices)), size=size
        )
        cluster_weights[:, indices] = draws
    return cluster_weights[:, clusters["inverse"]]


def weighted_metric_batch(
    y: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    weights: np.ndarray,
) -> np.ndarray:
    """Return weighted ROC-AUC, AP, balanced accuracy, F1 and macro-F1."""
    y_float = y.astype(np.float64)
    positives = weights @ y_float
    negatives = weights @ (1.0 - y_float)

    # Aggregate tied scores before applying the weighted Mann-Whitney formula.
    ascending = np.argsort(probability, kind="stable")
    sorted_probability = probability[ascending]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_probability) != 0) + 1]
    ordered_weights = weights[:, ascending]
    ordered_y = y_float[ascending]
    positive_groups = np.add.reduceat(ordered_weights * ordered_y, starts, axis=1)
    negative_groups = np.add.reduceat(
        ordered_weights * (1.0 - ordered_y), starts, axis=1
    )
    negatives_below = np.cumsum(negative_groups, axis=1) - negative_groups
    auc = np.sum(
        positive_groups * (negatives_below + 0.5 * negative_groups), axis=1
    ) / (positives * negatives)

    # sklearn's non-interpolated AP: precision at each descending score group,
    # weighted by that group's increase in recall.
    positive_desc = positive_groups[:, ::-1]
    total_desc = (positive_groups + negative_groups)[:, ::-1]
    cumulative_positive = np.cumsum(positive_desc, axis=1)
    cumulative_total = np.cumsum(total_desc, axis=1)
    precision = np.divide(
        cumulative_positive,
        cumulative_total,
        out=np.zeros_like(cumulative_positive),
        where=cumulative_total != 0,
    )
    ap = np.sum(
        (positive_desc / positives[:, None])
        * precision,
        axis=1,
    )

    prediction = probability >= threshold
    true_positive = weights @ (y_float * prediction)
    true_negative = weights @ ((1.0 - y_float) * (~prediction))
    balanced_accuracy = 0.5 * (
        true_positive / positives + true_negative / negatives
    )
    false_positive = negatives - true_negative
    false_negative = positives - true_positive
    positive_f1 = 2.0 * true_positive / (
        2.0 * true_positive + false_positive + false_negative
    )
    negative_f1 = 2.0 * true_negative / (
        2.0 * true_negative + false_positive + false_negative
    )
    macro_f1 = 0.5 * (positive_f1 + negative_f1)
    return np.column_stack((auc, ap, balanced_accuracy, positive_f1, macro_f1))


def point_metrics(
    y: np.ndarray, probability: np.ndarray, threshold: float
) -> np.ndarray:
    return weighted_metric_batch(
        y, probability, threshold, np.ones((1, len(y)), dtype=np.int16)
    )[0]


def bootstrap_scores(
    loaded: Mapping[Tuple[str, str], Mapping[str, Any]],
    reference: Mapping[str, Any],
    iterations: int,
    batch_size: int,
    seed: int,
    description: str,
) -> Tuple[Dict[Tuple[str, str], np.ndarray], Dict[Tuple[str, str], np.ndarray]]:
    clusters = identity_clusters(reference["identity_hash"], reference["y"])
    bootstrap = {
        key: np.empty((iterations, len(METRICS)), dtype=np.float64) for key in loaded
    }
    points = {
        key: point_metrics(
            reference["y"], value["probability"], value["threshold"]
        )
        for key, value in loaded.items()
    }
    rng = np.random.default_rng(seed)
    offsets = range(0, iterations, batch_size)
    for start in tqdm(offsets, desc=description, unit="batch", leave=False):
        stop = min(start + batch_size, iterations)
        weights = bootstrap_row_weights(rng, clusters, stop - start)
        for key, value in loaded.items():
            bootstrap[key][start:stop] = weighted_metric_batch(
                reference["y"], value["probability"], value["threshold"], weights
            )
    return points, bootstrap


def bootstrap_sign_pvalue(differences: np.ndarray) -> float:
    """Two-sided smoothed bootstrap sign probability around zero."""
    lower = (np.count_nonzero(differences <= 0) + 1) / (len(differences) + 1)
    upper = (np.count_nonzero(differences >= 0) + 1) / (len(differences) + 1)
    return float(min(1.0, 2.0 * min(lower, upper)))


def holm_adjust(pvalues: Sequence[float]) -> List[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=np.float64)
    running = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def build_results(
    inventory: str,
    points: Mapping[Tuple[str, str], np.ndarray],
    bootstrap: Mapping[Tuple[str, str], np.ndarray],
    confidence: float,
) -> List[Dict[str, Any]]:
    alpha = 1.0 - confidence
    rows: List[Dict[str, Any]] = []
    for model in MODELS:
        for contrast, candidate, reference in CONTRASTS:
            delta = bootstrap[(model, candidate)] - bootstrap[(model, reference)]
            for metric_index, metric in enumerate(METRICS):
                distribution = delta[:, metric_index]
                rows.append(
                    {
                        "inventory": inventory,
                        "model": model,
                        "contrast": contrast,
                        "candidate": candidate,
                        "reference": reference,
                        "metric": metric,
                        "candidate_score": float(points[(model, candidate)][metric_index]),
                        "reference_score": float(points[(model, reference)][metric_index]),
                        "difference": float(
                            points[(model, candidate)][metric_index]
                            - points[(model, reference)][metric_index]
                        ),
                        "ci_low": float(np.quantile(distribution, alpha / 2.0)),
                        "ci_high": float(np.quantile(distribution, 1.0 - alpha / 2.0)),
                        "probability_positive": float(np.mean(distribution > 0)),
                        "bootstrap_sign_pvalue": bootstrap_sign_pvalue(distribution),
                    }
                )
        # Correction family: seven pre-specified feature contrasts for one
        # inventory, model and metric.
        for metric in METRICS:
            family = [
                row for row in rows if row["model"] == model and row["metric"] == metric
            ]
            adjusted = holm_adjust(
                [row["bootstrap_sign_pvalue"] for row in family]
            )
            for row, value in zip(family, adjusted):
                row["holm_adjusted_pvalue"] = value
                row["ci_excludes_zero"] = bool(
                    row["ci_low"] > 0 or row["ci_high"] < 0
                )
    return rows


def csv_text(rows: Iterable[Mapping[str, Any]]) -> str:
    rows = list(rows)
    fields = list(rows[0]) if rows else []
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def discover_inventory_results(output_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(output_dir.glob("*/paired_results.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["results"])
    return rows


def render_report(
    rows: Sequence[Mapping[str, Any]], iterations: int, confidence: float
) -> str:
    primary = [row for row in rows if row["metric"] == "f1"]
    interval_label = f"{100 * confidence:g}%"
    lines = [
        "# UniRST2 paired feature-model comparison",
        "",
        (
            f"Test differences use {iterations:,} label-stratified cluster-bootstrap "
            "resamples. A positive difference favors the first-named feature set. "
            f"Intervals are {interval_label} percentile intervals; Holm adjustment covers the seven "
            "pre-specified contrasts within each inventory and classifier."
        ),
        "",
        "## F1 differences",
        "",
        f"| Inventory | Model | Contrast | Difference | {interval_label} CI | Holm p |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            "| {inventory} | {model} | {contrast} | {difference:+.4f} | "
            "[{ci_low:+.4f}, {ci_high:+.4f}] | {holm_adjusted_pvalue:.4g} |".format(
                **row
            )
        )
    robust = [
        row
        for row in primary
        if row["ci_excludes_zero"] and row["holm_adjusted_pvalue"] < 0.05
    ]
    frequency_structural = [
        row for row in primary if row["contrast"] == "frequency_vs_structural"
    ]
    frequency_presence = [
        row for row in primary if row["contrast"] == "frequency_vs_presence"
    ]
    depth_presence_lr = [
        row
        for row in primary
        if row["contrast"] == "depth_vs_presence" and row["model"] == "lr"
    ]
    depth_presence_xgb = [
        row
        for row in primary
        if row["contrast"] == "depth_vs_presence" and row["model"] == "xgb"
    ]

    def directional_count(values: Sequence[Mapping[str, Any]], positive: bool) -> int:
        return sum(
            bool(row["ci_excludes_zero"])
            and row["holm_adjusted_pvalue"] < 0.05
            and ((row["difference"] > 0) if positive else (row["difference"] < 0))
            for row in values
        )

    lines.extend(
        [
            "",
            "## Compact interpretation",
            "",
            (
                f"{len(robust)} of {len(primary)} F1 contrasts have a {interval_label} interval "
                "excluding zero and Holm-adjusted bootstrap sign probability below 0.05."
            ),
            "",
            (
                "M3 frequency exceeds the structural baseline in "
                f"{directional_count(frequency_structural, True)}/{len(frequency_structural)} "
                "inventory-classifier pairs, and exceeds M3 presence in "
                f"{directional_count(frequency_presence, True)}/{len(frequency_presence)}."
            ),
            "",
            (
                "Depth relative to presence is model-dependent: it is reliably lower in "
                f"{directional_count(depth_presence_lr, False)}/{len(depth_presence_lr)} "
                "logistic-regression comparisons, but reliably higher in "
                f"{directional_count(depth_presence_xgb, True)}/{len(depth_presence_xgb)} "
                "XGBoost comparisons. This is evidence for nonlinear depth effects, not "
                "a stable positive linear depth association."
            ),
            "",
            (
                "The adjusted bootstrap sign values quantify directional stability under "
                "resampling; they should not be interpreted as exact randomization-test "
                "p-values. The released split's train–test identity overlap remains a "
                "separate sensitivity analysis."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-dir", default="artifacts/canonical/model_comparison"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/canonical/paired_comparison"
    )
    parser.add_argument(
        "--inventories",
        nargs="+",
        default=list(INVENTORIES),
        help="Inventory names below the prediction directory.",
    )
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.iterations < 100:
        parser.error("--iterations must be at least 100")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not 0 < args.confidence < 1:
        parser.error("--confidence must be between zero and one")

    prediction_root = Path(args.prediction_dir)
    output_dir = Path(args.output_dir)
    for inventory in tqdm(args.inventories, desc="paired inventories", unit="inventory"):
        loaded, reference = load_inventory(prediction_root, inventory)
        hashes = {
            f"{model}/{config}": value["sha256"]
            for (model, config), value in loaded.items()
        }
        inventory_path = output_dir / inventory / "paired_results.json"
        settings = {
            "iterations": args.iterations,
            "batch_size": args.batch_size,
            "confidence": args.confidence,
            "seed": args.seed,
        }
        if inventory_path.exists() and not args.force:
            existing = json.loads(inventory_path.read_text(encoding="utf-8"))
            if existing.get("settings") == settings and existing.get(
                "prediction_sha256"
            ) == hashes:
                print(f"skipping complete {inventory}")
                continue
        points, bootstrap = bootstrap_scores(
            loaded,
            reference,
            args.iterations,
            args.batch_size,
            stable_seed(args.seed, inventory),
            f"bootstrap {inventory}",
        )
        results = build_results(inventory, points, bootstrap, args.confidence)
        clusters = identity_clusters(reference["identity_hash"], reference["y"])
        payload = {
            "inventory": inventory,
            "settings": settings,
            "protocol": {
                "data": "saved test predictions; no fitting or selection",
                "resampling_unit": "text/source/label identity_hash cluster",
                "stratification": "target label",
                "pairing": "same resamples for all feature configurations and models",
                "interval": "percentile cluster bootstrap",
                "multiplicity": (
                    "Holm across seven contrasts within inventory/model/metric"
                ),
            },
            "rows": int(len(reference["y"])),
            "unique_identity_clusters": int(len(clusters["unique"])),
            "duplicate_rows": int(len(reference["y"]) - len(clusters["unique"])),
            "prediction_sha256": hashes,
            "results": results,
        }
        atomic_write_json(inventory_path, payload)
        atomic_write_text(inventory_path.with_suffix(".csv"), csv_text(results))

    all_rows = discover_inventory_results(output_dir)
    atomic_write_text(output_dir / "summary.csv", csv_text(all_rows))
    atomic_write_text(
        output_dir / "report.md",
        render_report(all_rows, args.iterations, args.confidence),
    )
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "prediction_dir": str(prediction_root.resolve()),
            "settings": {
                "iterations": args.iterations,
                "batch_size": args.batch_size,
                "confidence": args.confidence,
                "seed": args.seed,
            },
            "inventories": sorted({row["inventory"] for row in all_rows}),
            "result_rows": len(all_rows),
            "summary_sha256": sha256_file(output_dir / "summary.csv"),
            "report_sha256": sha256_file(output_dir / "report.md"),
        },
    )
    print(f"wrote {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
