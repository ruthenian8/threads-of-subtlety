#!/usr/bin/env python3
"""Estimate domain-specific UniRST2 fine-versus-coarse OOD effects.

The script reuses the feature matrices and frozen predictions emitted by script
31.  It fixes nested source suffixes (for example ``cnn_human_para`` ->
``cnn``) and uses one Poisson cluster-bootstrap weight per identity across both
OOD collections, preserving their overlap instead of treating them as
independent replications.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from tqdm import tqdm


INVENTORIES = ("deu.rst.pcc", "nld.rst.nldt")
DISPLAY = {"deu.rst.pcc": "PCC", "nld.rst.nldt": "NLDT"}
SETS = ("gpt", "gpt_para")
MODELS = ("lr", "xgb")
METRICS = ("roc_auc", "average_precision", "balanced_accuracy")
DOMAINS = ("cnn", "dialogsum", "imdb", "pubmed")
SCRIPT_VERSION = "unirst2-ood-domains-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
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


def csv_text(rows: Iterable[Mapping[str, Any]]) -> str:
    rows = list(rows)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]) if rows else [])
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def canonical_domain(source: str) -> str:
    """Remove stacked generation/provenance suffixes until stable."""
    suffixes = ("_para", "_human", "_gpt4", "_llm", "_machine")
    value = str(source)
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if value.endswith(suffix):
                value = value[: -len(suffix)]
                changed = True
                break
    return value


def load_bundle(root: Path, inventory: str, ood_set: str, model: str) -> dict[str, np.ndarray | float]:
    matrix_path = root / "feature_matrices" / inventory / f"{ood_set}.npz"
    prediction_path = root / "predictions" / inventory / ood_set / f"{model}.npz"
    with np.load(matrix_path, allow_pickle=False) as values:
        matrix = {key: np.asarray(values[key]) for key in values.files}
    with np.load(prediction_path, allow_pickle=False) as values:
        prediction = {key: np.asarray(values[key]) for key in values.files}
    for key in ("identity_hash", "source", "y"):
        if not np.array_equal(matrix[key], prediction[key]):
            raise ValueError(f"Matrix/prediction mismatch: {inventory}/{ood_set}/{model}/{key}")
    domains = np.asarray([canonical_domain(value) for value in matrix["source"].astype(str)])
    observed = set(domains.tolist())
    if observed != set(DOMAINS):
        raise ValueError(f"Unexpected normalized domains for {inventory}/{ood_set}: {sorted(observed)}")
    return {
        "identity_hash": matrix["identity_hash"].astype(str),
        "source": matrix["source"].astype(str),
        "domain": domains,
        "y": matrix["y"].astype(np.int8),
        "full_probability": prediction["full_probability"].astype(np.float64),
        "coarse_probability": prediction["coarse_probability"].astype(np.float64),
        "full_threshold": float(prediction["full_threshold"]),
        "coarse_threshold": float(prediction["coarse_threshold"]),
        "matrix_path": np.asarray(str(matrix_path.resolve())),
        "prediction_path": np.asarray(str(prediction_path.resolve())),
    }


def weighted_metrics(y: np.ndarray, probability: np.ndarray, threshold: float, weights: np.ndarray) -> np.ndarray:
    """Vectorized weighted AUC, AP, and balanced accuracy."""
    y = np.asarray(y, dtype=np.int8)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(probability, kind="mergesort")
    sorted_probability = probability[order]
    sorted_y = y[order]
    sorted_weights = weights[:, order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_probability)) + 1, len(y)]
    positive_groups = np.add.reduceat(sorted_weights * sorted_y, boundaries[:-1], axis=1)
    negative_groups = np.add.reduceat(sorted_weights * (1 - sorted_y), boundaries[:-1], axis=1)
    positives = positive_groups.sum(axis=1)
    negatives = negative_groups.sum(axis=1)
    negatives_below = np.cumsum(negative_groups, axis=1) - negative_groups
    auc = np.sum(positive_groups * (negatives_below + 0.5 * negative_groups), axis=1) / (positives * negatives)
    pos_desc = positive_groups[:, ::-1]
    total_desc = (positive_groups + negative_groups)[:, ::-1]
    cum_pos = np.cumsum(pos_desc, axis=1)
    cum_total = np.cumsum(total_desc, axis=1)
    precision = np.divide(cum_pos, cum_total, out=np.zeros_like(cum_pos), where=cum_total != 0)
    ap = np.sum((pos_desc / positives[:, None]) * precision, axis=1)
    predicted = probability >= threshold
    tp = weights @ (y * predicted)
    tn = weights @ ((1 - y) * (~predicted))
    ba = 0.5 * (tp / positives + tn / negatives)
    return np.column_stack((auc, ap, ba))


def point_metrics(bundle: Mapping[str, Any], mask: np.ndarray, which: str) -> np.ndarray:
    return weighted_metrics(
        bundle["y"][mask], bundle[f"{which}_probability"][mask],
        float(bundle[f"{which}_threshold"]), np.ones((1, int(mask.sum()))),
    )[0]


def sign_pvalue(values: np.ndarray) -> float:
    lower = (np.count_nonzero(values <= 0) + 1) / (len(values) + 1)
    upper = (np.count_nonzero(values >= 0) + 1) / (len(values) + 1)
    return float(min(1.0, 2 * min(lower, upper)))


def holm(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(values[order] * (len(values) - np.arange(len(values))))
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def analyze_pair(
    bundles: Mapping[str, Mapping[str, Any]], iterations: int, batch_size: int, seed: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    identities = sorted({value for bundle in bundles.values() for value in bundle["identity_hash"]})
    identity_index = {value: index for index, value in enumerate(identities)}
    row_identity = {
        name: np.asarray([identity_index[value] for value in bundle["identity_hash"]], dtype=np.int32)
        for name, bundle in bundles.items()
    }
    distributions: dict[str, np.ndarray] = {
        f"{ood_set}:{domain}": np.empty((iterations, len(METRICS)), dtype=np.float64)
        for ood_set in SETS for domain in DOMAINS
    }
    rng = np.random.default_rng(seed)
    for start in tqdm(range(0, iterations, batch_size), desc="joint OOD bootstrap", unit="batch", leave=False):
        stop = min(start + batch_size, iterations)
        shared_weights = rng.poisson(1.0, size=(stop - start, len(identities))).astype(np.int16)
        for ood_set, bundle in bundles.items():
            for domain in DOMAINS:
                mask = bundle["domain"] == domain
                y = bundle["y"][mask]
                weights = shared_weights[:, row_identity[ood_set][mask]]
                # Degenerate label weights are extraordinarily rare; replace only those rows.
                bad = (weights[:, y == 0].sum(axis=1) == 0) | (weights[:, y == 1].sum(axis=1) == 0)
                weights[bad] = 1
                fine = weighted_metrics(y, bundle["full_probability"][mask], float(bundle["full_threshold"]), weights)
                coarse = weighted_metrics(y, bundle["coarse_probability"][mask], float(bundle["coarse_threshold"]), weights)
                distributions[f"{ood_set}:{domain}"][start:stop] = fine - coarse

    rows: list[dict[str, Any]] = []
    for ood_set, bundle in bundles.items():
        for domain in DOMAINS:
            mask = bundle["domain"] == domain
            fine = point_metrics(bundle, mask, "full")
            coarse = point_metrics(bundle, mask, "coarse")
            samples = distributions[f"{ood_set}:{domain}"]
            for metric_index, metric in enumerate(METRICS):
                values = samples[:, metric_index]
                rows.append({
                    "scope": "collection_domain", "ood_set": ood_set, "domain": domain,
                    "metric": metric, "fine_score": float(fine[metric_index]),
                    "coarse_score": float(coarse[metric_index]), "difference": float(fine[metric_index] - coarse[metric_index]),
                    "ci_low": float(np.quantile(values, .025)), "ci_high": float(np.quantile(values, .975)),
                    "bootstrap_sign_p": sign_pvalue(values), "rows": int(mask.sum()),
                    "generated": int(np.count_nonzero(bundle["y"][mask] == 0)),
                    "human": int(np.count_nonzero(bundle["y"][mask] == 1)),
                })
    # Joint summaries retain dependence because they are functions of the same draws.
    for domain in DOMAINS:
        left = distributions[f"gpt:{domain}"]
        right = distributions[f"gpt_para:{domain}"]
        for scope, samples in (("collection_average", (left + right) / 2), ("para_minus_gpt", right - left)):
            for metric_index, metric in enumerate(METRICS):
                values = samples[:, metric_index]
                point_values = [row["difference"] for row in rows if row["domain"] == domain and row["metric"] == metric]
                point = float(np.mean(point_values)) if scope == "collection_average" else float(point_values[1] - point_values[0])
                rows.append({
                    "scope": scope, "ood_set": "joint", "domain": domain, "metric": metric,
                    "fine_score": "", "coarse_score": "", "difference": point,
                    "ci_low": float(np.quantile(values, .025)), "ci_high": float(np.quantile(values, .975)),
                    "bootstrap_sign_p": sign_pvalue(values), "rows": "", "generated": "", "human": "",
                })
    return rows, distributions


def render_report(rows: Sequence[Mapping[str, Any]], iterations: int) -> str:
    primary = [row for row in rows if row["scope"] == "collection_domain" and row["metric"] == "roc_auc"]
    lines = [
        "# Domain-specific frozen fine-versus-coarse OOD effects", "",
        "Frozen HC3-trained models and script-31 probabilities are reused. Source suffixes are recursively normalized to CNN, DialogSum, IMDb, and PubMed. The two OOD collections are analyzed as related stress tests: exact shared identities receive identical weights in a joint Poisson cluster bootstrap.", "",
        f"Intervals use {iterations:,} shared bootstrap draws. Holm adjustment covers the eight domain-by-collection contrasts within each inventory and classifier. Collection averages are descriptive repeated-condition summaries, not independent-study meta-analysis.", "",
        "| Inventory | Model | Collection | Domain | Fine AUC | Coarse AUC | Fine − coarse (95% CI) | Holm p |", "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in primary:
        mark = "†" if float(row["ci_low"]) > 0 and float(row["holm_p"]) < .05 else ""
        lines.append(f"| {DISPLAY[row['inventory']]} | {str(row['model']).upper()} | {row['ood_set']} | {row['domain']} | {float(row['fine_score']):.3f} | {float(row['coarse_score']):.3f} | {float(row['difference']):+.3f}{mark} [{float(row['ci_low']):+.3f}, {float(row['ci_high']):+.3f}] | {float(row['holm_p']):.4g} |")
    lines.extend(["", "† interval excludes zero and Holm-adjusted bootstrap-sign value is below 0.05.", "", "## Related-collection summaries", "", "| Inventory | Model | Domain | Mean AUC difference (95% CI) | Para − GPT difference (95% CI) |", "|---|---|---|---:|---:|"])
    lookup = {(row["inventory"], row["model"], row["scope"], row["domain"], row["metric"]): row for row in rows}
    for inventory in INVENTORIES:
        for model in MODELS:
            for domain in DOMAINS:
                avg = lookup[(inventory, model, "collection_average", domain, "roc_auc")]
                het = lookup[(inventory, model, "para_minus_gpt", domain, "roc_auc")]
                lines.append(f"| {DISPLAY[inventory]} | {model.upper()} | {domain} | {float(avg['difference']):+.3f} [{float(avg['ci_low']):+.3f}, {float(avg['ci_high']):+.3f}] | {float(het['difference']):+.3f} [{float(het['ci_low']):+.3f}, {float(het['ci_high']):+.3f}] |")
    lines.extend(["", "These are exploratory domain decompositions of previously inspected OOD results. The collections share human documents and are not independent replications; no OOD fitting or threshold selection is performed.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("artifacts/unirst2_ood_coarse_comparison"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/unirst2_minimal_contribution/ood_domains"))
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    all_rows: list[dict[str, Any]] = []
    all_distributions: dict[str, np.ndarray] = {}
    input_hashes: dict[str, str] = {}
    for inventory in INVENTORIES:
        for model in MODELS:
            bundles = {ood_set: load_bundle(args.input_root, inventory, ood_set, model) for ood_set in SETS}
            rows, distributions = analyze_pair(bundles, args.iterations, args.batch_size, args.seed)
            for row in rows:
                row.update({"inventory": inventory, "model": model})
            all_rows.extend(rows)
            for key, value in distributions.items():
                all_distributions[f"{inventory}|{model}|{key}"] = value
        for ood_set in SETS:
            for kind, path in {
                "matrix": args.input_root / "feature_matrices" / inventory / f"{ood_set}.npz",
                **{f"prediction_{model}": args.input_root / "predictions" / inventory / ood_set / f"{model}.npz" for model in MODELS},
            }.items():
                input_hashes[f"{inventory}/{ood_set}/{kind}"] = sha256_file(path)
    # Correction family: 2 collections x 4 domains for one inventory/model/metric.
    for inventory in INVENTORIES:
        for model in MODELS:
            for metric in METRICS:
                family = [row for row in all_rows if row["inventory"] == inventory and row["model"] == model and row["metric"] == metric and row["scope"] == "collection_domain"]
                for row, value in zip(family, holm([float(row["bootstrap_sign_p"]) for row in family])):
                    row["holm_p"] = float(value)
    for row in all_rows:
        row.setdefault("holm_p", "")
    args.output_root.mkdir(parents=True, exist_ok=True)
    results_path = args.output_root / "domain_effects.csv"
    bootstrap_path = args.output_root / "joint_bootstrap.npz"
    report_path = args.output_root / "report.md"
    atomic_text(results_path, csv_text(all_rows))
    atomic_npz(bootstrap_path, **all_distributions)
    atomic_text(report_path, render_report(all_rows, args.iterations))
    settings = dict(vars(args))
    settings.update({"input_root": str(args.input_root), "output_root": str(args.output_root)})
    atomic_json(args.output_root / "manifest.json", {
        "script_version": SCRIPT_VERSION, "settings": settings,
        "canonical_domains": DOMAINS, "related_test_method": "shared identity-level Poisson bootstrap weights",
        "input_hashes": input_hashes, "output_hashes": {path.name: sha256_file(path) for path in (results_path, bootstrap_path, report_path)},
    })
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
