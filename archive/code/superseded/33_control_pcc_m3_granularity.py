#!/usr/bin/env python3
"""Test PCC fine M3 granularity after stringent document and parse controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import joblib
import networkx as nx
import numpy as np
from tqdm import tqdm


INVENTORY = "deu.rst.pcc"
SPLITS = ("train", "validation", "test")
MODELS = ("lr", "xgb")
REPRESENTATIONS = ("controls", "controls_coarse", "controls_fine")
METRICS = ("roc_auc", "average_precision", "balanced_accuracy")
SCRIPT_VERSION = "pcc-controlled-granularity-v1"


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


def structural_assignment(catalog_path: Path, fine_hashes: Sequence[str]) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    representatives: list[nx.DiGraph] = []
    structural_hashes: list[str] = []
    fine_to_structural: list[int] = []
    rows: list[dict[str, Any]] = []
    edge_match = lambda left, right: left.get("label_0") == right.get("label_0")
    for fine_index, motif_hash in enumerate(fine_hashes):
        graph = nx.node_link_graph(catalog[motif_hash])
        original_relations = sorted(str(data.get("label_0")) for _, _, data in graph.edges(data=True) if data.get("label_0") != "/")
        for _, _, data in graph.edges(data=True):
            if data.get("label_0") != "/":
                data["label_0"] = "REL"
        target = None
        for index, representative in enumerate(representatives):
            if nx.is_isomorphic(graph, representative, edge_match=edge_match):
                target = index
                break
        if target is None:
            target = len(representatives)
            representatives.append(graph)
            structural_hashes.append(nx.weisfeiler_lehman_graph_hash(graph, edge_attr="label_0"))
        fine_to_structural.append(target)
        rows.append({"fine_index": fine_index, "fine_hash": motif_hash, "fine_relations": "|".join(original_relations) or "(structural only)", "structural_index": target, "structural_hash": structural_hashes[target]})
    assignment = np.zeros((len(fine_hashes), len(representatives)), dtype=np.float32)
    assignment[np.arange(len(fine_hashes)), fine_to_structural] = 1
    if not np.array_equal(assignment.sum(axis=1), np.ones(len(fine_hashes))):
        raise ValueError("Each fine motif must map to exactly one structural motif")
    return assignment, structural_hashes, rows


def load_split(path: Path, structural_assignment_matrix: np.ndarray, coarse_assignment: np.ndarray) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as values:
        structural_names = values["structural_names"].astype(str).tolist()
        text_index = structural_names.index("text_chars")
        edu_index = structural_names.index("edu_count")
        scalar = np.log1p(np.asarray(values["X_structural"][:, [text_index, edu_index]], dtype=np.float32))
        relation = np.asarray(values["X_relation_frequency"], dtype=np.float32)
        fine = np.asarray(values["X_m3_frequency"], dtype=np.float32)
        structure_m3 = np.asarray(fine @ structural_assignment_matrix, dtype=np.float32)
        coarse = np.asarray(fine @ coarse_assignment, dtype=np.float32)
        controls = np.column_stack((scalar, relation, structure_m3)).astype(np.float32)
        common = {
            "y": np.asarray(values["y"], dtype=np.int8), "identity_id": np.asarray(values["identity_id"]),
            "identity_hash": np.asarray(values["identity_hash"]), "domain": np.asarray(values["domain"]).astype(str),
        }
        return {
            **common, "controls": controls,
            "controls_coarse": np.column_stack((controls, coarse)).astype(np.float32),
            "controls_fine": np.column_stack((controls, fine)).astype(np.float32),
            "control_names": ["log1p_text_chars", "log1p_edu_count", *[f"relation:{x}" for x in values["relation_names"].astype(str)], *[f"structural_m3:{i}" for i in range(structure_m3.shape[1])]],
        }


def fit_all(evaluator: ModuleType, splits: Mapping[str, Mapping[str, Any]], output_root: Path, seed: int, n_jobs: int) -> dict[tuple[str, str], dict[str, Any]]:
    results: dict[tuple[str, str], dict[str, Any]] = {}
    for model_kind in MODELS:
        for representation in REPRESENTATIONS:
            prepared = {split: {"X": values[representation], "y": values["y"]} for split, values in splits.items()}
            model, selection, validation_probability, threshold = evaluator.fit_selected_model(
                model_kind, prepared["train"], prepared["validation"], seed, n_jobs, f"PCC {representation}/{model_kind}"
            )
            test_probability = evaluator.probabilities(model, prepared["test"]["X"])
            run_dir = output_root / "models" / representation / model_kind
            run_dir.mkdir(parents=True, exist_ok=True)
            model_path = run_dir / "model.joblib"
            temporary = model_path.with_name(model_path.name + ".tmp")
            joblib.dump(model, temporary)
            os.replace(temporary, model_path)
            prediction_path = run_dir / "predictions.npz"
            atomic_npz(prediction_path, identity_id=splits["test"]["identity_id"], identity_hash=splits["test"]["identity_hash"], domain=splits["test"]["domain"], y=splits["test"]["y"], probability=test_probability, threshold=np.asarray(threshold))
            metadata = {
                "representation": representation, "model": model_kind, "feature_count": int(prepared["train"]["X"].shape[1]),
                "selection": selection, "validation_metrics": evaluator.metrics(prepared["validation"]["y"], validation_probability, threshold),
                "test_metrics": evaluator.metrics(prepared["test"]["y"], test_probability, threshold),
                "model_sha256": sha256_file(model_path), "prediction_sha256": sha256_file(prediction_path),
            }
            atomic_json(run_dir / "metrics.json", metadata)
            results[(model_kind, representation)] = {**metadata, "probability": test_probability, "threshold": threshold}
    return results


def stratified_cluster_weights(rng: np.random.Generator, identity: np.ndarray, domain: np.ndarray, y: np.ndarray, size: int) -> np.ndarray:
    unique, inverse = np.unique(identity, return_inverse=True)
    cluster_domain = np.empty(len(unique), dtype=object)
    cluster_y = np.empty(len(unique), dtype=np.int8)
    for index in range(len(unique)):
        rows = inverse == index
        domains = np.unique(domain[rows])
        labels = np.unique(y[rows])
        if len(domains) != 1 or len(labels) != 1:
            raise ValueError("Identity cluster crosses domain or label")
        cluster_domain[index], cluster_y[index] = domains[0], labels[0]
    cluster_weights = np.zeros((size, len(unique)), dtype=np.int16)
    for current_domain in sorted(set(domain.tolist())):
        for label in (0, 1):
            indices = np.flatnonzero((cluster_domain == current_domain) & (cluster_y == label))
            if not len(indices):
                continue
            cluster_weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=size)
    return cluster_weights[:, inverse]


def compare(results: Mapping[tuple[str, str], Mapping[str, Any]], splits: Mapping[str, Mapping[str, Any]], bootstrap: ModuleType, iterations: int, batch_size: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    test = splits["test"]
    training = set(splits["train"]["identity_hash"].tolist())
    seen: set[str] = set()
    clean = np.zeros(len(test["y"]), dtype=bool)
    for index, value in enumerate(test["identity_hash"].tolist()):
        if value not in training and value not in seen:
            clean[index] = True
            seen.add(value)
    scopes = {"primary": np.ones(len(clean), dtype=bool), "identity_clean": clean}
    scopes.update({f"domain:{domain}": test["domain"] == domain for domain in sorted(set(test["domain"].tolist()))})
    contrasts = (("fine_minus_coarse", "controls_fine", "controls_coarse"), ("fine_minus_controls", "controls_fine", "controls"))
    rows: list[dict[str, Any]] = []
    saved: dict[str, np.ndarray] = {}
    for model in MODELS:
        for scope, mask in scopes.items():
            y = test["y"][mask]
            if len(np.unique(y)) != 2:
                continue
            distributions = {name: np.empty((iterations, len(METRICS)), dtype=np.float64) for name, _, _ in contrasts}
            rng = np.random.default_rng(seed)
            for start in tqdm(range(0, iterations, batch_size), desc=f"bootstrap {model}/{scope}", unit="batch", leave=False):
                stop = min(start + batch_size, iterations)
                weights = stratified_cluster_weights(rng, test["identity_hash"][mask], test["domain"][mask], y, stop - start)
                scores = {}
                for representation in REPRESENTATIONS:
                    item = results[(model, representation)]
                    scores[representation] = bootstrap.weighted_metric_batch(y, item["probability"][mask], float(item["threshold"]), weights)
                for name, candidate, reference in contrasts:
                    distributions[name][start:stop] = scores[candidate] - scores[reference]
            for name, candidate, reference in contrasts:
                saved[f"{model}|{scope}|{name}"] = distributions[name]
                candidate_item, reference_item = results[(model, candidate)], results[(model, reference)]
                candidate_point = bootstrap.point_metrics(y, candidate_item["probability"][mask], float(candidate_item["threshold"]))
                reference_point = bootstrap.point_metrics(y, reference_item["probability"][mask], float(reference_item["threshold"]))
                for metric_index, metric in enumerate(METRICS):
                    values = distributions[name][:, metric_index]
                    rows.append({
                        "model": model, "scope": scope, "contrast": name, "metric": metric,
                        "candidate_score": float(candidate_point[metric_index]), "reference_score": float(reference_point[metric_index]),
                        "difference": float(candidate_point[metric_index] - reference_point[metric_index]),
                        "ci_low": float(np.quantile(values, .025)), "ci_high": float(np.quantile(values, .975)),
                        "bootstrap_sign_p": bootstrap.bootstrap_sign_pvalue(values), "rows": int(mask.sum()),
                    })
    # Two global contrasts form the correction family; domain rows are descriptive.
    for model in MODELS:
        for scope in ("primary", "identity_clean"):
            for metric in METRICS:
                family = [row for row in rows if row["model"] == model and row["scope"] == scope and row["metric"] == metric]
                adjusted = bootstrap.holm_adjust([row["bootstrap_sign_p"] for row in family])
                for row, value in zip(family, adjusted):
                    row["holm_p"] = value
    for row in rows:
        row.setdefault("holm_p", "")
    return rows, saved


def render_report(rows: Sequence[Mapping[str, Any]], structural_features: int, iterations: int) -> str:
    primary = [row for row in rows if row["scope"] in ("primary", "identity_clean") and row["metric"] == "roc_auc"]
    lines = [
        "# PCC fine-granularity test with stringent controls", "",
        "All candidates use the same released HC3 splits. The common control block contains log document length, log EDU count, full PCC relation-frequency marginals, and relation-erased structural M3 frequencies. Structural M3 retains directed topology and `/` scaffolding but replaces every discourse relation with `REL`; node roles are ignored, matching the edge-labelled isomorphism semantics of the source motif counts.", "",
        "The primary contrast adds either fixed-tree coarse or full fine PCC M3 to that identical control block. Thus it asks whether fine relational configurations add predictive information beyond size, EDU count, relation marginals, and relation-free motif structure.", "",
        f"There are {structural_features} relation-erased structural M3 categories. Intervals use {iterations:,} paired domain-and-label-stratified identity-cluster bootstrap samples.", "",
        "| Cohort | Model | Contrast | Candidate AUC | Reference AUC | Difference (95% CI) | Holm p |", "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(f"| {row['scope']} | {str(row['model']).upper()} | {row['contrast']} | {float(row['candidate_score']):.3f} | {float(row['reference_score']):.3f} | {float(row['difference']):+.3f} [{float(row['ci_low']):+.3f}, {float(row['ci_high']):+.3f}] | {float(row['holm_p']):.4g} |")
    lr = next(row for row in primary if row["scope"] == "primary" and row["model"] == "lr" and row["contrast"] == "fine_minus_coarse")
    survives = float(lr["ci_low"]) > 0
    lines.extend(["", "## Decision", "", ("The prespecified LR fine-versus-coarse PCC difference remains positive with an interval excluding zero; the PCC granularity result survives the declared controls." if survives else "The prespecified LR fine-versus-coarse PCC interval includes zero or is negative; the current experiment does not establish that the PCC granularity result survives the declared controls."), "", "XGBoost is a robustness analysis. Domain-specific rows are saved in `paired_comparisons.csv` and are descriptive rather than a new confirmatory family.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=Path("artifacts/unirst2_feature_matrices"))
    parser.add_argument("--motif-root", type=Path, default=Path("data/motifs2"))
    parser.add_argument("--coarse-mapping", type=Path, default=Path("artifacts/unirst2_granularity_ablation/motif_mapping.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/unirst2_minimal_contribution/pcc_controls"))
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=min(os.cpu_count() or 1, 4))
    args = parser.parse_args()
    evaluator = load_module("unirst2_evaluator_controls", Path("12_evaluate_unirst2_feature_models.py"))
    bootstrap = load_module("unirst2_bootstrap_controls", Path("13_compare_unirst2_paired_performance.py"))
    train_path = args.feature_root / INVENTORY / "hc3_train.features.npz"
    with np.load(train_path, allow_pickle=False) as values:
        fine_hashes = values["m3_hashes"].astype(str).tolist()
    catalog_path = args.motif_root / INVENTORY / "hc3_M3_motifs.json"
    structural_map, structural_hashes, structural_rows = structural_assignment(catalog_path, fine_hashes)
    with args.coarse_mapping.open(newline="", encoding="utf-8") as handle:
        mapping_rows = sorted((row for row in csv.DictReader(handle) if row["inventory"] == INVENTORY), key=lambda row: int(row["fine_index"]))
    if [row["fine_hash"] for row in mapping_rows] != fine_hashes:
        raise ValueError("Coarse mapping does not match PCC feature order")
    coarse_count = max(int(row["coarse_index"]) for row in mapping_rows) + 1
    coarse_map = np.zeros((len(fine_hashes), coarse_count), dtype=np.float32)
    for row in mapping_rows:
        coarse_map[int(row["fine_index"]), int(row["coarse_index"])] = 1
    splits = {split: load_split(args.feature_root / INVENTORY / f"hc3_{split}.features.npz", structural_map, coarse_map) for split in SPLITS}
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_text(args.output_root / "structural_motif_mapping.csv", csv_text(structural_rows))
    results = fit_all(evaluator, splits, args.output_root, args.seed, args.n_jobs)
    rows, distributions = compare(results, splits, bootstrap, args.iterations, args.batch_size, args.seed)
    comparisons_path = args.output_root / "paired_comparisons.csv"
    bootstrap_path = args.output_root / "bootstrap_distributions.npz"
    report_path = args.output_root / "report.md"
    atomic_text(comparisons_path, csv_text(rows))
    atomic_npz(bootstrap_path, **distributions)
    atomic_text(report_path, render_report(rows, len(structural_hashes), args.iterations))
    settings = dict(vars(args)); settings.update({key: str(value) for key, value in vars(args).items() if isinstance(value, Path)})
    inputs = [*(args.feature_root / INVENTORY / f"hc3_{split}.features.npz" for split in SPLITS), catalog_path, args.coarse_mapping]
    atomic_json(args.output_root / "manifest.json", {
        "script_version": SCRIPT_VERSION, "settings": settings,
        "control_definition": {"scalar": ["log1p_text_chars", "log1p_edu_count"], "relation_frequency": "full PCC marginals", "structural_m3": "non-/ edge labels replaced by REL; direction and scaffolding retained; node attributes ignored consistently with source motif matching"},
        "input_hashes": {str(path): sha256_file(path) for path in inputs},
        "output_hashes": {path.name: sha256_file(path) for path in (comparisons_path, bootstrap_path, report_path, args.output_root / "structural_motif_mapping.csv")},
    })
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
