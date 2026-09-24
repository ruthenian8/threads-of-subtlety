#!/usr/bin/env python3
"""Localize fine-versus-coarse M3 gains to selected relation-merge groups.

For each PCC and NLDT merge group selected from training and corroborated on
validation by the domain-interaction analysis, fit two fixed-tree ablations:

* restore: replace one coarse feature by its fine member features;
* collapse: replace those fine member features by their coarse sum.

The first estimates whether a group is sufficient to recover coarse-model
performance; the second estimates whether it is necessary within the complete
fine representation. Models and thresholds are selected on validation using
the same evaluator as the existing feature-model experiments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from tqdm import tqdm


INVENTORIES = ("deu.rst.pcc", "nld.rst.nldt")
DISPLAY = {"deu.rst.pcc": "PCC", "nld.rst.nldt": "NLDT"}
MODELS = ("lr", "xgb")
SPLITS = ("train", "validation", "test")
METRICS = ("roc_auc", "average_precision", "balanced_accuracy")
SCRIPT_VERSION = "localized-coarse-merge-ablation-v1"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


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
    fields = list(rows[0]) if rows else []
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def load_split(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {
            "X": np.asarray(values["X_m3_frequency"], dtype=np.float32),
            "feature_names": np.asarray(values["m3_hashes"]).astype(str),
            "y": np.asarray(values["y"], dtype=np.int8),
            "identity_id": np.asarray(values["identity_id"]),
            "identity_hash": np.asarray(values["identity_hash"]).astype(str),
            "source": np.asarray(values["source"]).astype(str),
            "domain": np.asarray(values["domain"]).astype(str),
        }


def load_prediction(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as values:
        return {
            "identity_id": np.asarray(values["test_identity_id"]),
            "identity_hash": np.asarray(values["test_identity_hash"]).astype(str),
            "y": np.asarray(values["test_y"], dtype=np.int8),
            "probability": np.asarray(values["test_probability"], dtype=np.float64),
            "threshold": float(values["threshold"]),
        }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def discover_groups(
    interaction_path: Path, mapping_path: Path
) -> dict[str, list[dict[str, Any]]]:
    interactions = read_rows(interaction_path)
    mapping = read_rows(mapping_path)
    mapping_by_fine = {(row["inventory"], row["fine_hash"]): row for row in mapping}
    mapping_by_group: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in mapping:
        mapping_by_group[(row["inventory"], row["coarse_hash"])].append(row)
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in interactions:
        inventory = row["inventory"]
        if inventory not in INVENTORIES or row["validation_direction_agrees"] != "True":
            continue
        mapped = mapping_by_fine[(inventory, row["motif_hash"])]
        if int(mapped["coarse_group_size"]) <= 1:
            continue
        key = (inventory, mapped["coarse_hash"])
        if key not in selected:
            members = sorted(mapping_by_group[key], key=lambda value: int(value["fine_index"]))
            selected[key] = {
                "inventory": inventory,
                "coarse_hash": mapped["coarse_hash"],
                "coarse_index": int(mapped["coarse_index"]),
                "coarse_relations": mapped["coarse_relations"],
                "fine_indices": [int(value["fine_index"]) for value in members],
                "fine_hashes": [value["fine_hash"] for value in members],
                "fine_relations": [value["fine_relations"] for value in members],
                "selection_domains": set(),
                "highlighted_domains": set(),
                "highlighted": False,
            }
        selected[key]["selection_domains"].add(row["domain"])
        if row["robust_both_cohorts"] == "True":
            selected[key]["highlighted"] = True
            selected[key]["highlighted_domains"].add(row["domain"])
    result: dict[str, list[dict[str, Any]]] = {inventory: [] for inventory in INVENTORIES}
    for group in selected.values():
        group["selection_domains"] = sorted(group["selection_domains"])
        group["highlighted_domains"] = sorted(group["highlighted_domains"])
        group["group_id"] = f"{group['coarse_relations']}:{group['coarse_hash'][:8]}"
        result[group["inventory"]].append(group)
    for inventory in INVENTORIES:
        result[inventory].sort(key=lambda value: (value["coarse_relations"], value["coarse_hash"]))
    return result


def hybrid_split(
    fine: Mapping[str, np.ndarray],
    coarse: Mapping[str, np.ndarray],
    group: Mapping[str, Any],
    variant: str,
) -> dict[str, np.ndarray]:
    members = np.asarray(group["fine_indices"], dtype=np.int32)
    coarse_index = int(group["coarse_index"])
    if variant == "restore":
        keep = np.arange(coarse["X"].shape[1]) != coarse_index
        X = np.column_stack((coarse["X"][:, keep], fine["X"][:, members]))
        names = np.concatenate((coarse["feature_names"][keep], fine["feature_names"][members]))
    elif variant == "collapse":
        keep = np.ones(fine["X"].shape[1], dtype=bool)
        keep[members] = False
        merged = fine["X"][:, members].sum(axis=1, keepdims=True)
        X = np.column_stack((fine["X"][:, keep], merged))
        names = np.concatenate((fine["feature_names"][keep], np.asarray([group["coarse_hash"]])))
    else:
        raise ValueError(variant)
    if not np.allclose(X.sum(axis=1), fine["X"].sum(axis=1), atol=2e-6):
        raise ValueError(f"Frequency mass changed in {group['group_id']}/{variant}")
    return {
        "X": np.asarray(X, dtype=np.float32),
        "feature_names": names,
        **{key: fine[key] for key in ("y", "identity_id", "identity_hash", "source", "domain")},
    }


def fit_variant(
    evaluator: ModuleType,
    inventory: str,
    group: Mapping[str, Any],
    variant: str,
    model_kind: str,
    splits: Mapping[str, Mapping[str, np.ndarray]],
    output_root: Path,
    seed: int,
    n_jobs: int,
) -> dict[str, Any]:
    description = f"{inventory}/{group['group_id']}/{variant}/{model_kind}"
    model, selection, validation_probability, threshold = evaluator.fit_selected_model(
        model_kind, splits["train"], splits["validation"], seed, n_jobs, description
    )
    test_probability = evaluator.probabilities(model, splits["test"]["X"])
    run_dir = output_root / "models" / inventory / group["coarse_hash"] / variant / model_kind
    prediction_path = run_dir / "predictions.npz"
    model_path = run_dir / "model.joblib"
    atomic_npz(
        prediction_path,
        test_identity_id=splits["test"]["identity_id"],
        test_identity_hash=splits["test"]["identity_hash"],
        test_y=splits["test"]["y"],
        test_domain=splits["test"]["domain"],
        test_probability=test_probability,
        threshold=np.asarray(threshold),
        feature_names=splits["test"]["feature_names"],
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_name(model_path.name + ".tmp")
    joblib.dump(model, temporary)
    os.replace(temporary, model_path)
    result = {
        "inventory": inventory,
        "group_id": group["group_id"],
        "coarse_hash": group["coarse_hash"],
        "variant": variant,
        "model": model_kind,
        "feature_count": int(splits["train"]["X"].shape[1]),
        "selection": selection,
        "validation_metrics": evaluator.metrics(
            splits["validation"]["y"], validation_probability, threshold
        ),
        "test_metrics": evaluator.metrics(splits["test"]["y"], test_probability, threshold),
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": sha256_file(prediction_path),
        "model_path": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
    }
    atomic_json(run_dir / "metrics.json", result)
    return result


def clean_mask(train_hashes: np.ndarray, test_hashes: np.ndarray) -> np.ndarray:
    training = set(train_hashes.tolist())
    seen: set[str] = set()
    keep = np.zeros(len(test_hashes), dtype=bool)
    for index, identity in enumerate(test_hashes.tolist()):
        if identity not in training and identity not in seen:
            keep[index] = True
            seen.add(identity)
    return keep


def sign_pvalue(values: np.ndarray) -> float:
    lower = (np.count_nonzero(values <= 0) + 1) / (len(values) + 1)
    upper = (np.count_nonzero(values >= 0) + 1) / (len(values) + 1)
    return float(min(1.0, 2.0 * min(lower, upper)))


def holm(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    adjusted_ordered = np.maximum.accumulate(values[order] * (len(values) - np.arange(len(values))))
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_ordered, 1.0)
    return adjusted


def add_baseline_fractions(rows: Sequence[dict[str, Any]]) -> None:
    baseline_scores: dict[tuple[str, str, str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (row["inventory"], row["model"], row["cohort"], row["domain"], row["metric"])
        if row["contrast"] == "restoration_gain":
            baseline_scores[key]["coarse"] = float(row["right_score"])
        else:
            baseline_scores[key]["fine"] = float(row["left_score"])
    for row in rows:
        key = (row["inventory"], row["model"], row["cohort"], row["domain"], row["metric"])
        baseline = baseline_scores[key]
        total = baseline["fine"] - baseline["coarse"]
        row["total_fine_minus_coarse"] = total
        row["fraction_of_total_difference"] = (
            float(row["difference"]) / total if abs(total) > 1e-12 else ""
        )


def paired_comparison(
    bootstrap: ModuleType,
    evaluator: ModuleType,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    domain: str,
    cohort_mask: np.ndarray,
    iterations: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    scope_mask = cohort_mask.copy()
    if domain != "all":
        scope_mask &= left["domain"] == domain
    y = left["y"][scope_mask]
    if len(np.unique(y)) != 2:
        raise ValueError(f"Both labels required for {domain}")
    clusters = bootstrap.identity_clusters(left["identity_hash"][scope_mask], y)
    left_point = evaluator.metrics(y, left["probability"][scope_mask], left["threshold"])
    right_point = evaluator.metrics(y, right["probability"][scope_mask], right["threshold"])
    distributions = np.empty((iterations, len(METRICS)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, iterations, batch_size):
        stop = min(iterations, start + batch_size)
        weights = bootstrap.bootstrap_row_weights(rng, clusters, stop - start)
        left_scores = bootstrap.weighted_metric_batch(
            y, left["probability"][scope_mask], left["threshold"], weights
        )
        right_scores = bootstrap.weighted_metric_batch(
            y, right["probability"][scope_mask], right["threshold"], weights
        )
        distributions[start:stop] = left_scores - right_scores
    rows = []
    for metric_index, metric in enumerate(METRICS):
        values = distributions[:, metric_index]
        rows.append(
            {
                "metric": metric,
                "left_score": float(left_point[metric]),
                "right_score": float(right_point[metric]),
                "difference": float(left_point[metric] - right_point[metric]),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "bootstrap_sign_p": sign_pvalue(values),
                "rows": int(scope_mask.sum()),
            }
        )
    return rows


def baseline_prediction(
    path: Path, domain: np.ndarray, expected: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    result = load_prediction(path)
    for key in ("identity_id", "identity_hash", "y"):
        if not np.array_equal(result[key], expected[key]):
            raise ValueError(f"Prediction identity mismatch: {path}:{key}")
    result["domain"] = domain
    return result


def compare_all(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    feature_root: Path,
    full_model_root: Path,
    coarse_model_root: Path,
    output_root: Path,
    evaluator: ModuleType,
    bootstrap: ModuleType,
    iterations: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for inventory in INVENTORIES:
        train = load_split(feature_root / inventory / "hc3_train.features.npz")
        test = load_split(feature_root / inventory / "hc3_test.features.npz")
        masks = {
            "primary": np.ones(len(test["y"]), dtype=bool),
            "identity_clean": clean_mask(train["identity_hash"], test["identity_hash"]),
        }
        for group in tqdm(groups[inventory], desc=f"compare {DISPLAY[inventory]}", unit="group"):
            domains = ["all", *group["selection_domains"]]
            for model_kind in MODELS:
                fine = baseline_prediction(
                    full_model_root / inventory / "m3_frequency" / model_kind / "predictions.npz",
                    test["domain"], test,
                )
                coarse = baseline_prediction(
                    coarse_model_root / "models" / inventory / "coarse_all" / model_kind / "predictions.npz",
                    test["domain"], test,
                )
                restore = baseline_prediction(
                    output_root / "models" / inventory / group["coarse_hash"] / "restore" / model_kind / "predictions.npz",
                    test["domain"], test,
                )
                collapse = baseline_prediction(
                    output_root / "models" / inventory / group["coarse_hash"] / "collapse" / model_kind / "predictions.npz",
                    test["domain"], test,
                )
                comparisons = (
                    ("restoration_gain", restore, coarse),
                    ("collapse_loss", fine, collapse),
                )
                for cohort, cohort_mask in masks.items():
                    for domain in domains:
                        for contrast, left, right in comparisons:
                            estimates = paired_comparison(
                                bootstrap, evaluator, left, right, domain, cohort_mask,
                                iterations, batch_size,
                                int.from_bytes(hashlib.sha256(f"{seed}:{inventory}:{group['coarse_hash']}:{model_kind}:{cohort}:{domain}:{contrast}".encode()).digest()[:8], "little"),
                            )
                            for estimate in estimates:
                                rows.append(
                                    {
                                        "inventory": inventory,
                                        "group_id": group["group_id"],
                                        "coarse_hash": group["coarse_hash"],
                                        "coarse_relations": group["coarse_relations"],
                                        "fine_relations": ";".join(group["fine_relations"]),
                                        "group_size": len(group["fine_indices"]),
                                        "selected_domains": ";".join(group["selection_domains"]),
                                        "highlighted_domains": ";".join(group["highlighted_domains"]),
                                        "highlighted": group["highlighted"],
                                        "model": model_kind,
                                        "cohort": cohort,
                                        "domain": domain,
                                        "contrast": contrast,
                                        **estimate,
                                    }
                                )
        atomic_text(output_root / "paired_comparisons.csv", csv_text(rows))
    families: dict[tuple[str, str, str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = (row["inventory"], row["model"], row["cohort"], row["domain"], row["contrast"])
        families[key].append(index)
    for indices in families.values():
        adjusted = holm([float(rows[index]["bootstrap_sign_p"]) for index in indices])
        for index, value in zip(indices, adjusted):
            rows[index]["holm_p"] = float(value)
    add_baseline_fractions(rows)
    return rows


def render_report(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    rows: Sequence[Mapping[str, Any]],
    iterations: int,
) -> str:
    primary_auc = {
        (row["inventory"], row["coarse_hash"], row["model"], row["contrast"]): row
        for row in rows
        if row["cohort"] == "primary" and row["domain"] == "all" and row["metric"] == "roc_auc"
    }
    lines = [
        "# Localized UniRST2 coarse-merge ablation",
        "",
        "## Design",
        "",
        "Each experiment changes one coarse merge group while retaining the same documents, split membership, EDU segmentation, tree topology, motif catalog coverage, and total normalized motif-frequency mass. `Restoration gain` replaces one coarse feature with its fine members in the coarse representation. `Collapse loss` replaces those members with their sum in the full fine representation.",
        "",
        "Groups were selected by training-domain interactions and a same-direction validation check. Groups highlighted by the preceding identity-robust test analysis are marked below, but test outcomes did not determine the fitted set. Hyperparameters and thresholds are selected on validation using the existing protocol. Intervals use " + f"{iterations:,}" + " paired identity-cluster bootstrap samples. Holm values correct across the fitted groups within inventory, model, cohort, domain, and contrast.",
        "",
    ]
    for inventory in INVENTORIES:
        lines.extend(
            [
                f"## {DISPLAY[inventory]}",
                "",
                "| Model | Group: fine members → coarse | Highlighted domains | Restore − coarse AUC (share of total) | Fine − collapsed AUC (share of total) |",
                "|---|---|---|---:|---:|",
            ]
        )
        for model_kind in MODELS:
            for group in groups[inventory]:
                restore = primary_auc[(inventory, group["coarse_hash"], model_kind, "restoration_gain")]
                collapse = primary_auc[(inventory, group["coarse_hash"], model_kind, "collapse_loss")]
                marker = "★ " if group["highlighted"] else ""
                domain = ", ".join(group["highlighted_domains"]) or "validation only"
                fine = ", ".join(group["fine_relations"])
                restore_supported = "†" if float(restore["ci_low"]) > 0 and float(restore["holm_p"]) < 0.05 else ""
                collapse_supported = "†" if float(collapse["ci_low"]) > 0 and float(collapse["holm_p"]) < 0.05 else ""
                lines.append(
                    f"| {model_kind.upper()} | {marker}{fine} → {group['coarse_relations']} [{group['coarse_hash'][:8]}] | {domain} | "
                    f"{float(restore['difference']):+.4f}{restore_supported} ({float(restore['fraction_of_total_difference']):.0%}) "
                    f"[{float(restore['ci_low']):+.4f}, {float(restore['ci_high']):+.4f}] | "
                    f"{float(collapse['difference']):+.4f}{collapse_supported} ({float(collapse['fraction_of_total_difference']):.0%}) "
                    f"[{float(collapse['ci_low']):+.4f}, {float(collapse['ci_high']):+.4f}] |"
                )
        lines.append("")
        lines.append("★ previously highlighted by the identity-robust domain analysis; † positive interval and Holm-adjusted bootstrap-sign value below 0.05.")
        lines.append("")

    highlighted_auc = [
        row for row in rows
        if row["highlighted"]
        and row["cohort"] == "primary"
        and row["domain"] == "all"
        and row["metric"] == "roc_auc"
    ]
    supported_restore = [
        row for row in highlighted_auc
        if row["contrast"] == "restoration_gain" and float(row["ci_low"]) > 0 and float(row["holm_p"]) < 0.05
    ]
    supported_collapse = [
        row for row in highlighted_auc
        if row["contrast"] == "collapse_loss" and float(row["ci_low"]) > 0 and float(row["holm_p"]) < 0.05
    ]
    lines.extend(
        [
            "## Reading",
            "",
            f"Among highlighted inventory–group–model combinations, {len(supported_restore)} restoration contrasts recover a positive global AUC increment with a positive interval and Holm value below 0.05; {len(supported_collapse)} collapse contrasts show a corresponding necessary contribution inside the full representation.",
            "",
            "| Inventory | Highlighted group | Corrected restoration support | Corrected collapse-loss support | Interpretation |",
            "|---|---|---|---|---|",
        ]
    )
    for inventory in INVENTORIES:
        for group in groups[inventory]:
            if not group["highlighted"]:
                continue
            restore_models = sorted(
                row["model"].upper() for row in supported_restore
                if row["inventory"] == inventory and row["coarse_hash"] == group["coarse_hash"]
            )
            collapse_models = sorted(
                row["model"].upper() for row in supported_collapse
                if row["inventory"] == inventory and row["coarse_hash"] == group["coarse_hash"]
            )
            if restore_models and collapse_models:
                reading = "sufficient from coarse and necessary within fine"
            elif collapse_models:
                reading = "contributes conditionally with other fine groups"
            elif restore_models:
                reading = "recovers coarse performance; no corrected collapse loss"
            else:
                reading = "no corrected global AUC contribution detected"
            lines.append(
                f"| {DISPLAY[inventory]} | {', '.join(group['fine_relations'])} → {group['coarse_relations']} [{group['coarse_hash'][:8]}] | "
                f"{', '.join(restore_models) or 'none'} | {', '.join(collapse_models) or 'none'} | {reading} |"
            )
    domain_supported = [
        row for row in rows
        if row["highlighted"]
        and row["cohort"] == "primary"
        and row["domain"] != "all"
        and row["metric"] == "roc_auc"
        and float(row["holm_p"]) < 0.05
        and (float(row["ci_low"]) > 0 or float(row["ci_high"]) < 0)
    ]
    lines.extend(
        [
            "",
            "Restoration and collapse answer different questions and need not agree. A group can help only when other distinctions remain fine, or it can recover coarse-model signal yet become redundant in the full representation. Domain rows in `paired_comparisons.csv` show whether a small global change hides a larger effect in the domain that originally highlighted the relation group.",
            "",
            "## Corrected domain-localized AUC effects",
            "",
            "| Inventory | Model | Group | Domain | Contrast | AUC difference (95% CI) |",
            "|---|---|---|---|---|---:|",
        ]
    )
    for row in domain_supported:
        lines.append(
            f"| {DISPLAY[row['inventory']]} | {row['model'].upper()} | {row['fine_relations']} → {row['coarse_relations']} [{row['coarse_hash'][:8]}] | "
            f"{row['domain']} | {row['contrast']} | {float(row['difference']):+.4f} "
            f"[{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}] |"
        )
    if not domain_supported:
        lines.append("| — | — | — | — | — | No corrected domain-localized effect |")
    lines.extend(
        [
            "",
            "Opposite-signed domain effects explain why a relation distinction can be descriptively strong yet contribute little to aggregate discrimination. These rows are secondary outcomes and share the project's prior-test-inspection limitation.",
            "",
            "This remains an exploratory localization analysis because the broader project has already inspected HC3 test outcomes. The group-fitting rule itself uses training and validation only; the star records which of those groups was also highlighted by the earlier test analysis.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=Path("artifacts/unirst2_feature_matrices"))
    parser.add_argument("--full-model-root", type=Path, default=Path("artifacts/unirst2_model_comparison"))
    parser.add_argument("--coarse-root", type=Path, default=Path("artifacts/unirst2_granularity_ablation"))
    parser.add_argument("--domain-results", type=Path, default=Path("artifacts/unirst2_domain_distinctions/domain_interactions.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/unirst2_localized_ablation"))
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=min(os.cpu_count() or 1, 4))
    parser.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args()
    if args.iterations < 100 or args.batch_size < 1 or args.n_jobs < 1:
        parser.error("invalid compute settings")

    evaluator = load_module("unirst2_evaluator_local", Path("12_evaluate_unirst2_feature_models.py"))
    bootstrap = load_module("unirst2_bootstrap_local", Path("13_compare_unirst2_paired_performance.py"))
    mapping_path = args.coarse_root / "motif_mapping.csv"
    groups = discover_groups(args.domain_results, mapping_path)
    if [len(groups[inventory]) for inventory in INVENTORIES] != [7, 7]:
        raise ValueError("Expected seven train/validation-selected merge groups per inventory")

    if args.summarize_existing:
        comparison_path = args.output_root / "paired_comparisons.csv"
        comparisons: list[dict[str, Any]] = read_rows(comparison_path)
        for row in comparisons:
            row["highlighted"] = row["highlighted"] == "True"
        add_baseline_fractions(comparisons)
        atomic_text(comparison_path, csv_text(comparisons))
        atomic_text(args.output_root / "report.md", render_report(groups, comparisons, args.iterations))
        manifest_path = args.output_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["paired_comparisons_sha256"] = sha256_file(comparison_path)
        manifest["report_sha256"] = sha256_file(args.output_root / "report.md")
        atomic_json(manifest_path, manifest)
        print(f"updated {args.output_root / 'report.md'} from existing comparisons")
        return

    model_results = []
    for inventory in INVENTORIES:
        fine_splits = {
            split: load_split(args.feature_root / inventory / f"hc3_{split}.features.npz")
            for split in SPLITS
        }
        coarse_splits = {
            split: load_split(args.coarse_root / "feature_matrices" / inventory / f"hc3_{split}.features.npz")
            for split in SPLITS
        }
        for split in SPLITS:
            for key in ("y", "identity_id", "identity_hash", "domain"):
                if not np.array_equal(fine_splits[split][key], coarse_splits[split][key]):
                    raise ValueError(f"Fine/coarse split mismatch: {inventory}/{split}/{key}")
        for group in tqdm(groups[inventory], desc=f"fit {DISPLAY[inventory]}", unit="group"):
            for variant in ("restore", "collapse"):
                splits = {
                    split: hybrid_split(fine_splits[split], coarse_splits[split], group, variant)
                    for split in SPLITS
                }
                for model_kind in MODELS:
                    model_results.append(
                        fit_variant(
                            evaluator, inventory, group, variant, model_kind,
                            splits, args.output_root, args.seed, args.n_jobs,
                        )
                    )
            atomic_json(
                args.output_root / "progress.json",
                {"completed_models": len(model_results), "model_results": model_results},
            )

    comparisons = compare_all(
        groups, args.feature_root, args.full_model_root, args.coarse_root,
        args.output_root, evaluator, bootstrap, args.iterations,
        args.batch_size, args.seed,
    )
    atomic_text(args.output_root / "paired_comparisons.csv", csv_text(comparisons))
    group_rows = []
    for inventory in INVENTORIES:
        for group in groups[inventory]:
            group_rows.append({
                **{key: value for key, value in group.items() if key not in {"selection_domains", "highlighted_domains", "fine_indices", "fine_hashes", "fine_relations"}},
                "selection_domains": ";".join(group["selection_domains"]),
                "highlighted_domains": ";".join(group["highlighted_domains"]),
                "fine_indices": ";".join(map(str, group["fine_indices"])),
                "fine_hashes": ";".join(group["fine_hashes"]),
                "fine_relations": ";".join(group["fine_relations"]),
            })
    atomic_text(args.output_root / "groups.csv", csv_text(group_rows))
    atomic_text(args.output_root / "report.md", render_report(groups, comparisons, args.iterations))
    manifest = {
        "script_version": SCRIPT_VERSION,
        "settings": {"iterations": args.iterations, "batch_size": args.batch_size, "seed": args.seed, "n_jobs": args.n_jobs},
        "selection": "training top interaction and validation direction agreement; highlighted flag from robust test analysis",
        "group_counts": {inventory: len(groups[inventory]) for inventory in INVENTORIES},
        "highlighted_group_counts": {inventory: sum(bool(group["highlighted"]) for group in groups[inventory]) for inventory in INVENTORIES},
        "model_results": model_results,
        "provenance": {
            "domain_results_sha256": sha256_file(args.domain_results),
            "mapping_sha256": sha256_file(mapping_path),
        },
        "groups_sha256": sha256_file(args.output_root / "groups.csv"),
        "paired_comparisons_sha256": sha256_file(args.output_root / "paired_comparisons.csv"),
        "report_sha256": sha256_file(args.output_root / "report.md"),
    }
    atomic_json(args.output_root / "manifest.json", manifest)
    print(json.dumps({"groups": manifest["group_counts"], "highlighted": manifest["highlighted_group_counts"], "models": len(model_results), "comparison_rows": len(comparisons)}, indent=2))
    print(f"wrote {args.output_root / 'report.md'}")


if __name__ == "__main__":
    main()
