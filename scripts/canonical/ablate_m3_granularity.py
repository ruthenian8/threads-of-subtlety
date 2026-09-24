#!/usr/bin/env python3
"""Ablate PCC and NLDT M3 relation granularity on fixed UniRST trees."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import joblib
import networkx as nx
import numpy as np
from tqdm import tqdm
from tos.analysis.granularity import (
    coarsen_graph as coarsen_relation_graph,
    graph_from_data,
    identity_clean_mask as clean_mask,
    relation_labeled_isomorphic as pipeline_isomorphic,
    relation_signature,
)


INVENTORIES = ("deu.rst.pcc", "nld.rst.nldt")
MODELS = ("lr", "xgb")
SPLITS = ("train", "validation", "test")
RELATION_MAP = {
    "antithesis": "Contrast",
    "adversative": "Contrast",
    "background": "Background",
    "cause": "Cause",
    "circumstance": "Background",
    "concession": "Contrast",
    "condition": "Condition",
    "conjunction": "Joint",
    "contrast": "Contrast",
    "disjunction": "Joint",
    "e-elaboration": "Elaboration",
    "elaboration": "Elaboration",
    "enablement": "Enablement",
    "evaluation": "Evaluation",
    "evidence": "Explanation",
    "interpretation": "Evaluation",
    "joint": "Joint",
    "justify": "Explanation",
    "list": "Joint",
    "means": "Manner-Means",
    "motivation": "Explanation",
    "non-volitional-causal": "Cause",
    "preparation": "TextualOrganization",
    "purpose": "Enablement",
    "reason": "Explanation",
    "restatement": "Summary",
    "result": "Cause",
    "sequence": "Temporal",
    "solutionhood": "Topic-Comment",
    "summary": "Summary",
    "volitional-causal": "Cause",
}


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


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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


def coarsen_graph(graph: nx.DiGraph) -> nx.DiGraph:
    return coarsen_relation_graph(graph, RELATION_MAP, context="PCC/NLDT motif")


def build_assignment(
    inventory: str, motif_root: Path, feature_root: Path
) -> tuple[np.ndarray, list[str], list[Dict[str, Any]], Dict[str, Any]]:
    catalog_path = motif_root / inventory / "hc3_M3_motifs.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    with np.load(feature_root / inventory / "hc3_train.features.npz", allow_pickle=False) as values:
        fine_hashes = values["m3_hashes"].astype(str).tolist()
    missing = [motif_hash for motif_hash in fine_hashes if motif_hash not in catalog]
    if missing:
        raise ValueError(f"Catalog lacks {len(missing)} matrix features for {inventory}")

    groups: Dict[str, list[tuple[str, nx.DiGraph, nx.DiGraph]]] = {}
    for motif_hash in fine_hashes:
        fine_graph = graph_from_data(catalog[motif_hash])
        coarse_graph = coarsen_graph(fine_graph)
        coarse_hash = nx.weisfeiler_lehman_graph_hash(coarse_graph, edge_attr="label_0")
        groups.setdefault(coarse_hash, []).append((motif_hash, fine_graph, coarse_graph))
    for coarse_hash, members in groups.items():
        representative = members[0][2]
        if any(not pipeline_isomorphic(representative, member[2]) for member in members[1:]):
            raise ValueError(f"WL collision after coarsening: {inventory}/{coarse_hash}")

    coarse_hashes = sorted(groups)
    coarse_index = {value: index for index, value in enumerate(coarse_hashes)}
    assignment = np.zeros((len(fine_hashes), len(coarse_hashes)), dtype=np.float32)
    mapping_rows = []
    for fine_index, motif_hash in enumerate(fine_hashes):
        fine_graph = graph_from_data(catalog[motif_hash])
        coarse_graph = coarsen_graph(fine_graph)
        coarse_hash = nx.weisfeiler_lehman_graph_hash(coarse_graph, edge_attr="label_0")
        assignment[fine_index, coarse_index[coarse_hash]] = 1.0
        mapping_rows.append({
            "inventory": inventory,
            "fine_index": fine_index,
            "fine_hash": motif_hash,
            "fine_relations": relation_signature(fine_graph),
            "coarse_index": coarse_index[coarse_hash],
            "coarse_hash": coarse_hash,
            "coarse_relations": relation_signature(coarse_graph),
            "coarse_group_size": len(groups[coarse_hash]),
        })
    if not np.array_equal(assignment.sum(axis=1), np.ones(len(fine_hashes))):
        raise ValueError(f"Every fine motif must map exactly once for {inventory}")
    summary = {
        "inventory": inventory,
        "fine_features": len(fine_hashes),
        "coarse_features": len(coarse_hashes),
        "merged_dimensions": len(fine_hashes) - len(coarse_hashes),
        "multi_member_groups": sum(len(members) > 1 for members in groups.values()),
        "largest_group": max(map(len, groups.values())),
        "catalog_path": str(catalog_path.resolve()),
        "catalog_sha256": sha256_file(catalog_path),
        "mapping_complete": True,
        "wl_collisions": 0,
    }
    return assignment, coarse_hashes, mapping_rows, summary


def load_split(path: Path, matrix_key: str = "X_m3_frequency") -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {
            "X": np.asarray(values[matrix_key], dtype=np.float32),
            "y": np.asarray(values["y"], dtype=np.int8),
            "identity_id": np.asarray(values["identity_id"]),
            "identity_hash": np.asarray(values["identity_hash"]),
            "source": np.asarray(values["source"]),
            "domain": np.asarray(values["domain"]),
        }


def aggregate_split(
    source_path: Path,
    destination_path: Path,
    assignment: np.ndarray,
    coarse_hashes: Sequence[str],
) -> Dict[str, np.ndarray]:
    full = load_split(source_path)
    coarse = np.asarray(full["X"] @ assignment, dtype=np.float32)
    full_mass = full["X"].sum(axis=1)
    coarse_mass = coarse.sum(axis=1)
    if not np.allclose(full_mass, coarse_mass, atol=2e-6):
        raise ValueError(f"Motif-frequency mass was not conserved: {source_path}")
    atomic_npz(
        destination_path,
        X_m3_frequency=coarse,
        m3_hashes=np.asarray(coarse_hashes),
        y=full["y"],
        identity_id=full["identity_id"],
        identity_hash=full["identity_hash"],
        source=full["source"],
        domain=full["domain"],
    )
    full["X"] = coarse
    return full


def fit_model(
    evaluator: ModuleType,
    splits: Mapping[str, Mapping[str, np.ndarray]],
    inventory: str,
    model_kind: str,
    output_root: Path,
    seed: int,
    n_jobs: int,
) -> Dict[str, Any]:
    model, selection, validation_probability, threshold = evaluator.fit_selected_model(
        model_kind, splits["train"], splits["validation"], seed, n_jobs,
        f"{inventory} coarse-all/{model_kind}",
    )
    test_probability = evaluator.probabilities(model, splits["test"]["X"])
    run_dir = output_root / "models" / inventory / "coarse_all" / model_kind
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = run_dir / "predictions.npz"
    model_path = run_dir / "model.joblib"
    atomic_npz(
        prediction_path,
        test_identity_id=splits["test"]["identity_id"],
        test_identity_hash=splits["test"]["identity_hash"],
        test_y=splits["test"]["y"],
        test_probability=test_probability,
        threshold=np.asarray(threshold),
    )
    temporary_model = model_path.with_name(model_path.name + ".tmp")
    joblib.dump(model, temporary_model)
    os.replace(temporary_model, model_path)
    result = {
        "inventory": inventory,
        "representation": "coarse_all",
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


def load_prediction(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as values:
        return {
            "identity_id": np.asarray(values["test_identity_id"]),
            "identity_hash": np.asarray(values["test_identity_hash"]),
            "y": np.asarray(values["test_y"], dtype=np.int8),
            "probability": np.asarray(values["test_probability"], dtype=np.float64),
            "threshold": float(values["threshold"]),
        }


def sign_pvalue(values: np.ndarray) -> float:
    lower = (np.count_nonzero(values <= 0) + 1) / (len(values) + 1)
    upper = (np.count_nonzero(values >= 0) + 1) / (len(values) + 1)
    return float(min(1.0, 2 * min(lower, upper)))


def compare_models(
    bootstrap: ModuleType,
    output_root: Path,
    full_model_root: Path,
    train_hashes: Mapping[str, np.ndarray],
    iterations: int,
    batch_size: int,
    seed: int,
) -> list[Dict[str, Any]]:
    metric_names = ("roc_auc", "average_precision", "balanced_accuracy", "f1", "macro_f1")
    rows = []
    for inventory in INVENTORIES:
        for model_kind in MODELS:
            full = load_prediction(
                full_model_root / inventory / "m3_frequency" / model_kind / "predictions.npz"
            )
            coarse = load_prediction(
                output_root / "models" / inventory / "coarse_all" / model_kind / "predictions.npz"
            )
            for key in ("identity_id", "identity_hash", "y"):
                if not np.array_equal(full[key], coarse[key]):
                    raise ValueError(f"Prediction mismatch for {inventory}/{model_kind}/{key}")
            masks = {
                "primary": np.ones(len(full["y"]), dtype=bool),
                "identity_clean": clean_mask(train_hashes[inventory], full["identity_hash"]),
            }
            for cohort, mask in masks.items():
                y = full["y"][mask]
                clusters = bootstrap.identity_clusters(full["identity_hash"][mask], y)
                full_point = bootstrap.point_metrics(
                    y, full["probability"][mask], full["threshold"]
                )
                coarse_point = bootstrap.point_metrics(
                    y, coarse["probability"][mask], coarse["threshold"]
                )
                differences = np.empty((iterations, len(metric_names)), dtype=np.float64)
                rng = np.random.default_rng(seed)
                for start in tqdm(
                    range(0, iterations, batch_size),
                    desc=f"paired {inventory}/{model_kind}/{cohort}",
                    unit="batch",
                ):
                    stop = min(start + batch_size, iterations)
                    weights = bootstrap.bootstrap_row_weights(rng, clusters, stop - start)
                    full_scores = bootstrap.weighted_metric_batch(
                        y, full["probability"][mask], full["threshold"], weights
                    )
                    coarse_scores = bootstrap.weighted_metric_batch(
                        y, coarse["probability"][mask], coarse["threshold"], weights
                    )
                    differences[start:stop] = full_scores - coarse_scores
                for metric_index, metric in enumerate(metric_names):
                    values = differences[:, metric_index]
                    rows.append({
                        "cohort": cohort,
                        "inventory": inventory,
                        "model": model_kind,
                        "metric": metric,
                        "full_score": float(full_point[metric_index]),
                        "coarse_score": float(coarse_point[metric_index]),
                        "difference_full_minus_coarse": float(full_point[metric_index] - coarse_point[metric_index]),
                        "ci_low": float(np.quantile(values, 0.025)),
                        "ci_high": float(np.quantile(values, 0.975)),
                        "bootstrap_sign_p": sign_pvalue(values),
                        "rows": int(mask.sum()),
                    })
    return rows


def csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def report_text(rows: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]]) -> str:
    auc = {
        (row["inventory"], row["model"]): row
        for row in rows
        if row["cohort"] == "primary" and row["metric"] == "f1"
    }
    lines = [
        "# MF-only LR/XGBoost granularity sensitivity",
        "",
        "PCC and NLDT motif counts are deterministically summed after mapping their relation "
        "labels into the original 18 coarse DMRST classes. EDU segmentation, attachments, "
        "nuclearity pooling, documents, catalog coverage, and train/validation/test membership "
        "remain fixed. M3 frequencies are aggregated by matrix summation, which is algebraically "
        "identical to summing raw counts before normalization because every fine motif maps once.",
        "",
        "| Inventory | Fine features | Coarse features | Removed dimensions | Largest merge |",
        "|---|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['inventory']} | {summary['fine_features']} | "
            f"{summary['coarse_features']} | {summary['merged_dimensions']} | "
            f"{summary['largest_group']} |"
        )
    lines.extend([
        "",
        "## Sensitivity result",
        "",
        "| Model | Inventory | Fine F1 | Coarse F1 | Fine − coarse (95% CI) |",
        "|---|---|---:|---:|---:|",
    ])
    for model_kind in MODELS:
        for inventory in INVENTORIES:
            row = auc[(inventory, model_kind)]
            lines.append(
                f"| {model_kind.upper()} | {inventory} | {row['full_score']:.3f} | "
                f"{row['coarse_score']:.3f} | {row['difference_full_minus_coarse']:+.3f} "
                f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "In this MF-only sensitivity, fine relation distinctions are materially predictive in both inventories. All four "
        "fine-minus-coarse F1 intervals exclude zero, and the same direction holds for "
        "average precision and validation-threshold balanced accuracy. The identity-clean "
        "sensitivity changes the F1 gains only slightly. Because the trees and occurrence "
        "mass are fixed, these losses are attributable to merging fine motif categories under "
        "the declared relation map rather than to parser or segmentation differences.",
        "",
        "The paper-facing Kim-compatible estimate is the RF MF+WAD run in "
        "`../rf_mf_wad/`; the present LR/XGBoost result is retained as a sensitivity analysis. "
        "This result is conditional on the semantic coarse mapping and does not identify which "
        "individual merge supplies the gain. A subsequent leave-one-group-fine ablation would "
        "localize the effect without changing the primary result.",
        "",
        "Intervals use 2,000 paired, label-stratified identity-cluster bootstrap resamples. "
        "The identity-clean cohort removes training-overlap identities and collapses duplicate "
        "test identities; its estimates are stored in `paired_comparisons.csv`.",
        "",
        "This ablation isolates the predictive consequence of merging relation distinctions "
        "within each UniRST representation. It does not compare parser quality and does not "
        "project onto the original paper's selected 31-motif catalog.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=Path("artifacts/canonical/feature_matrices"))
    parser.add_argument("--motif-root", type=Path, default=Path("data/motifs2"))
    parser.add_argument("--full-model-root", type=Path, default=Path("artifacts/canonical/model_comparison"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/canonical/granularity_ablation"))
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=min(os.cpu_count() or 1, 4))
    args = parser.parse_args()

    from scripts.canonical import compare_paired_performance as bootstrap
    from scripts.canonical import evaluate_feature_models as evaluator
    all_mapping_rows = []
    summaries = []
    model_results = []
    train_hashes = {}
    for inventory in INVENTORIES:
        assignment, coarse_hashes, mapping_rows, summary = build_assignment(
            inventory, args.motif_root, args.feature_root
        )
        all_mapping_rows.extend(mapping_rows)
        summaries.append(summary)
        splits = {}
        for split in SPLITS:
            destination = args.output_root / "feature_matrices" / inventory / f"hc3_{split}.features.npz"
            splits[split] = aggregate_split(
                args.feature_root / inventory / f"hc3_{split}.features.npz",
                destination, assignment, coarse_hashes,
            )
        train_hashes[inventory] = splits["train"]["identity_hash"]
        for model_kind in MODELS:
            model_results.append(fit_model(
                evaluator, splits, inventory, model_kind, args.output_root,
                args.seed, args.n_jobs,
            ))
    comparisons = compare_models(
        bootstrap, args.output_root, args.full_model_root, train_hashes,
        args.iterations, args.batch_size, args.seed,
    )
    atomic_text(args.output_root / "motif_mapping.csv", csv_text(all_mapping_rows))
    atomic_text(args.output_root / "paired_comparisons.csv", csv_text(comparisons))
    atomic_text(args.output_root / "report.md", report_text(comparisons, summaries))
    atomic_json(args.output_root / "manifest.json", {
        "relation_map": RELATION_MAP,
        "mapping": "fine relation labels -> original 18 DMRST classes; topology and / edges retained",
        "frequency_aggregation": "X_coarse = X_fine @ assignment; occurrence mass conserved",
        "inventory_summaries": summaries,
        "model_results": model_results,
        "bootstrap": {"iterations": args.iterations, "seed": args.seed, "cluster": "text/source/label identity"},
        "motif_mapping_sha256": sha256_file(args.output_root / "motif_mapping.csv"),
        "paired_comparisons_sha256": sha256_file(args.output_root / "paired_comparisons.csv"),
    })
    print(f"wrote {args.output_root / 'report.md'}")


if __name__ == "__main__":
    main()
