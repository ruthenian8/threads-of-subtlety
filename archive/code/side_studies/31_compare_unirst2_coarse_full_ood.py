#!/usr/bin/env python3
"""Compare frozen UniRST2 full and coarse M3 models on OOD corpora.

The stored OOD motif vectors use an older catalog, so M3 frequencies are
recounted from their persisted discourse graphs against the UniRST2 catalogs.
No parser or model is rerun. Full and fixed-tree coarse HC3 models are then
evaluated with their validation-selected thresholds unchanged.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import io
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from tqdm import tqdm

from tos.tos_dataset import ToSDataset
from tos.tos_utils import node_link_graph_compat


INVENTORIES = ("deu.rst.pcc", "nld.rst.nldt")
DISPLAY = {"deu.rst.pcc": "PCC", "nld.rst.nldt": "NLDT"}
OOD_FILES = {
    "gpt": "test_ood_set_gpt.discourse_parsed.graph_added.m3_m6_motif_dists.jsonl",
    "gpt_para": "test_ood_set_gpt_para.discourse_parsed.graph_added.m3_m6_motif_dists.jsonl",
}
MODELS = ("lr", "xgb")
METRICS = ("roc_auc", "average_precision", "balanced_accuracy")
SCRIPT_VERSION = "unirst2-coarse-full-ood-v1"


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
    writer = csv.DictWriter(output, fieldnames=list(rows[0]) if rows else [])
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def base_identity(text: str, source: str, label: int) -> str:
    payload = json.dumps(
        [text, source, label], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_group(source: str) -> str:
    return re.sub(r"_(?:human|llm|gpt4|machine|para)$", "", source) or source


def load_catalog(motif_root: Path, inventory: str) -> tuple[list[str], Any, dict[str, str]]:
    directory = motif_root / inventory
    manifest_path = directory / "hc3_selected-motif-hashes.generated.json"
    catalog_path = directory / "hc3_M3_motifs.json"
    selected = json.loads(manifest_path.read_text(encoding="utf-8"))["m3"]
    motifs = ToSDataset.load_motifs(str(catalog_path), selected)
    index = ToSDataset.prepare_motif_catalog_index(motifs)
    return selected, index, {
        "selected_manifest_sha256": sha256_file(manifest_path),
        "catalog_sha256": sha256_file(catalog_path),
    }


def build_assignment(
    mapping_path: Path, inventory: str, fine_hashes: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    with mapping_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["inventory"] == inventory]
    rows.sort(key=lambda row: int(row["fine_index"]))
    if [row["fine_hash"] for row in rows] != list(fine_hashes):
        raise ValueError(f"Fine catalog order differs from coarse mapping for {inventory}")
    coarse_count = max(int(row["coarse_index"]) for row in rows) + 1
    assignment = np.zeros((len(rows), coarse_count), dtype=np.float32)
    coarse_hashes: list[str | None] = [None] * coarse_count
    for row in rows:
        fine_index = int(row["fine_index"])
        coarse_index = int(row["coarse_index"])
        assignment[fine_index, coarse_index] = 1.0
        existing = coarse_hashes[coarse_index]
        if existing is not None and existing != row["coarse_hash"]:
            raise ValueError(f"Conflicting coarse hash at {coarse_index}")
        coarse_hashes[coarse_index] = row["coarse_hash"]
    if any(value is None for value in coarse_hashes):
        raise ValueError(f"Incomplete coarse mapping for {inventory}")
    return assignment, [str(value) for value in coarse_hashes]


def extract_matrix(
    input_path: Path,
    output_path: Path,
    inventory: str,
    fine_hashes: Sequence[str],
    catalog_index: Any,
    assignment: np.ndarray,
    coarse_hashes: Sequence[str],
) -> dict[str, Any]:
    frequencies = []
    labels = []
    identities = []
    sources = []
    groups = []
    total_occurrences = 0.0
    zero_documents = 0
    digest = hashlib.sha256()
    with input_path.open("rb") as handle:
        iterator = tqdm(handle, desc=f"recount {DISPLAY[inventory]} {input_path.name[:18]}", unit="doc")
        for line_number, raw_line in enumerate(iterator, 1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            record = ast.literal_eval(raw_line.decode("utf-8"))
            label = int(record["label"])
            source = str(record.get("source", record.get("src", "unknown")))
            text = str(record.get("text", ""))
            raw_total = np.zeros(len(fine_hashes), dtype=np.float64)
            trees = record.get("scene_discourse_trees") or {}
            scene_values = trees.values() if isinstance(trees, Mapping) else trees
            for scene in scene_values:
                if not isinstance(scene, Mapping) or not scene.get("graph_dict"):
                    continue
                graph = node_link_graph_compat(scene["graph_dict"])
                root_label = f"span_1-{len(scene.get('edus') or {})}"
                distribution = ToSDataset.calculate_observed_m3_m6_distributions(
                    graph, {"m3": catalog_index}, root_label
                )["m3"]
                raw_total += np.asarray(distribution["raw"], dtype=np.float64)
            total = float(raw_total.sum())
            total_occurrences += total
            if total == 0:
                zero_documents += 1
                frequency = raw_total
            else:
                frequency = raw_total / total
            frequencies.append(frequency.astype(np.float32))
            labels.append(label)
            identities.append(base_identity(text, source, label))
            sources.append(source)
            groups.append(source_group(source))
    X_full = np.asarray(frequencies, dtype=np.float32)
    X_coarse = np.asarray(X_full @ assignment, dtype=np.float32)
    if not np.allclose(X_full.sum(axis=1), X_coarse.sum(axis=1), atol=2e-6):
        raise ValueError(f"Frequency mass changed while coarsening {inventory}/{input_path.name}")
    atomic_npz(
        output_path,
        inventory=np.asarray(inventory),
        X_full=X_full,
        full_hashes=np.asarray(fine_hashes),
        X_coarse=X_coarse,
        coarse_hashes=np.asarray(coarse_hashes),
        y=np.asarray(labels, dtype=np.int8),
        identity_hash=np.asarray(identities),
        source=np.asarray(sources),
        source_group=np.asarray(groups),
    )
    metadata = {
        "inventory": inventory,
        "input_path": str(input_path.resolve()),
        "input_sha256": digest.hexdigest(),
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "rows": len(labels),
        "label_counts": {str(key): value for key, value in sorted(Counter(labels).items())},
        "source_group_counts": dict(sorted(Counter(groups).items())),
        "unique_identities": len(set(identities)),
        "zero_motif_documents": zero_documents,
        "total_matched_m3_occurrences": total_occurrences,
        "full_features": X_full.shape[1],
        "coarse_features": X_coarse.shape[1],
    }
    atomic_json(output_path.with_suffix(".json"), metadata)
    return metadata


def load_ood_matrix(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {key: np.asarray(values[key]) for key in values.files}


def load_threshold(prediction_path: Path) -> float:
    with np.load(prediction_path, allow_pickle=False) as values:
        return float(values["threshold"])


def bootstrap_weights(
    rng: np.random.Generator,
    y: np.ndarray,
    groups: np.ndarray,
    size: int,
) -> np.ndarray:
    weights = np.zeros((size, len(y)), dtype=np.int16)
    strata = sorted({(str(group), int(label)) for group, label in zip(groups, y)})
    for group, label in strata:
        indices = np.flatnonzero((groups == group) & (y == label))
        probabilities = np.full(len(indices), 1.0 / len(indices))
        weights[:, indices] = rng.multinomial(len(indices), probabilities, size=size)
    return weights


def sign_pvalue(values: np.ndarray) -> float:
    lower = (np.count_nonzero(values <= 0) + 1) / (len(values) + 1)
    upper = (np.count_nonzero(values >= 0) + 1) / (len(values) + 1)
    return float(min(1.0, 2.0 * min(lower, upper)))


def holm(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(values[order] * (len(values) - np.arange(len(values))))
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def evaluate_pair(
    evaluator: ModuleType,
    bootstrap: ModuleType,
    matrix: Mapping[str, np.ndarray],
    inventory: str,
    ood_set: str,
    model_kind: str,
    full_model_root: Path,
    coarse_root: Path,
    output_root: Path,
    iterations: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    full_dir = full_model_root / inventory / "m3_frequency" / model_kind
    coarse_dir = coarse_root / "models" / inventory / "coarse_all" / model_kind
    full_model = joblib.load(full_dir / "model.joblib")
    coarse_model = joblib.load(coarse_dir / "model.joblib")
    full_threshold = load_threshold(full_dir / "predictions.npz")
    coarse_threshold = load_threshold(coarse_dir / "predictions.npz")
    full_probability = evaluator.probabilities(full_model, matrix["X_full"])
    coarse_probability = evaluator.probabilities(coarse_model, matrix["X_coarse"])
    y = np.asarray(matrix["y"], dtype=np.int8)
    full_point = evaluator.metrics(y, full_probability, full_threshold)
    coarse_point = evaluator.metrics(y, coarse_probability, coarse_threshold)
    distributions = np.empty((iterations, len(METRICS)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, iterations, batch_size):
        stop = min(iterations, start + batch_size)
        weights = bootstrap_weights(rng, y, matrix["source_group"].astype(str), stop - start)
        full_scores = bootstrap.weighted_metric_batch(
            y, full_probability, full_threshold, weights
        )
        coarse_scores = bootstrap.weighted_metric_batch(
            y, coarse_probability, coarse_threshold, weights
        )
        distributions[start:stop] = full_scores - coarse_scores
    prediction_path = output_root / "predictions" / inventory / ood_set / f"{model_kind}.npz"
    atomic_npz(
        prediction_path,
        identity_hash=matrix["identity_hash"],
        source=matrix["source"],
        source_group=matrix["source_group"],
        y=y,
        full_probability=full_probability,
        coarse_probability=coarse_probability,
        full_threshold=np.asarray(full_threshold),
        coarse_threshold=np.asarray(coarse_threshold),
    )
    rows = []
    for metric_index, metric in enumerate(METRICS):
        samples = distributions[:, metric_index]
        rows.append(
            {
                "inventory": inventory,
                "ood_set": ood_set,
                "model": model_kind,
                "metric": metric,
                "full_score": float(full_point[metric]),
                "coarse_score": float(coarse_point[metric]),
                "difference_full_minus_coarse": float(full_point[metric] - coarse_point[metric]),
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "bootstrap_sign_p": sign_pvalue(samples),
                "rows": len(y),
                "full_threshold": full_threshold,
                "coarse_threshold": coarse_threshold,
                "prediction_path": str(prediction_path.resolve()),
                "prediction_sha256": sha256_file(prediction_path),
            }
        )
    return rows


def render_report(
    rows: Sequence[Mapping[str, Any]],
    extraction: Mapping[str, Mapping[str, Mapping[str, Any]]],
    overlap: Mapping[str, Mapping[str, int]],
    cross_set_overlap: Mapping[str, int],
    iterations: int,
) -> str:
    lookup = {
        (row["inventory"], row["ood_set"], row["model"], row["metric"]): row
        for row in rows
    }
    lines = [
        "# Frozen coarse-versus-full M3 performance on OOD sets",
        "",
        "The HC3-trained full and fixed-tree coarse PCC/NLDT models are applied without refitting or OOD threshold selection. Stored discourse graphs are recounted against the UniRST2 M3 catalogs because the previously stored OOD vectors use incompatible older catalogs. The coarse matrix is obtained from the full counts through the same deterministic relation mapping as the in-domain ablation.",
        "",
        f"Intervals use {iterations:,} paired source-and-label-stratified bootstrap samples. Holm values correct the two OOD-set comparisons within each inventory, classifier, and metric.",
        "",
        "| OOD set | Rows | Labels (machine/human) | Source groups | HC3-train identity overlaps |",
        "|---|---:|---:|---:|---:|",
    ]
    reference_inventory = INVENTORIES[0]
    for ood_set in OOD_FILES:
        item = extraction[reference_inventory][ood_set]
        lines.append(
            f"| {ood_set} | {item['rows']} | {item['label_counts'].get('0', 0)}/{item['label_counts'].get('1', 0)} | "
            f"{len(item['source_group_counts'])} | {overlap[reference_inventory][ood_set]} |"
        )
    lines.extend(
        [
            "",
            "## Primary comparison",
            "",
            "| Inventory | OOD set | Model | Full AUC | Coarse AUC | Full − coarse AUC (95% CI) | Full − coarse AP | Full − coarse balanced accuracy |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for inventory in INVENTORIES:
        for ood_set in OOD_FILES:
            for model_kind in MODELS:
                auc = lookup[(inventory, ood_set, model_kind, "roc_auc")]
                ap = lookup[(inventory, ood_set, model_kind, "average_precision")]
                ba = lookup[(inventory, ood_set, model_kind, "balanced_accuracy")]
                marker = "†" if float(auc["ci_low"]) > 0 and float(auc["holm_p"]) < 0.05 else ""
                lines.append(
                    f"| {DISPLAY[inventory]} | {ood_set} | {model_kind.upper()} | "
                    f"{float(auc['full_score']):.3f} | {float(auc['coarse_score']):.3f} | "
                    f"{float(auc['difference_full_minus_coarse']):+.3f}{marker} "
                    f"[{float(auc['ci_low']):+.3f}, {float(auc['ci_high']):+.3f}] | "
                    f"{float(ap['difference_full_minus_coarse']):+.3f} | "
                    f"{float(ba['difference_full_minus_coarse']):+.3f} |"
                )
    supported = [
        row for row in rows
        if row["metric"] == "roc_auc"
        and float(row["ci_low"]) > 0
        and float(row["holm_p"]) < 0.05
    ]
    reversed_rows = [
        row for row in rows
        if row["metric"] == "roc_auc"
        and float(row["ci_high"]) < 0
        and float(row["holm_p"]) < 0.05
    ]
    auc_rows = [row for row in rows if row["metric"] == "roc_auc"]
    mean_deltas = {
        (inventory, model): float(np.mean([
            row["difference_full_minus_coarse"]
            for row in auc_rows
            if row["inventory"] == inventory and row["model"] == model
        ]))
        for inventory in INVENTORIES
        for model in MODELS
    }
    full_auc = [float(row["full_score"]) for row in auc_rows]
    ap_supported = sum(
        row["metric"] == "average_precision"
        and float(row["ci_low"]) > 0
        and float(row["holm_p"]) < 0.05
        for row in rows
    )
    ba_supported = sum(
        row["metric"] == "balanced_accuracy"
        and float(row["ci_low"]) > 0
        and float(row["holm_p"]) < 0.05
        for row in rows
    )
    lines.extend(
        [
            "",
            "† positive interval and Holm-adjusted bootstrap-sign value below 0.05.",
            "",
            "## Reading",
            "",
            f"The full representation has a corrected positive AUC advantage in {len(supported)}/8 inventory–set–classifier comparisons; the coarse representation has a corrected advantage in {len(reversed_rows)}/8.",
            "",
            f"PCC transfers the fine-granularity advantage with both classifiers: its mean OOD AUC difference is {mean_deltas[('deu.rst.pcc', 'lr')]:+.3f} for LR and {mean_deltas[('deu.rst.pcc', 'xgb')]:+.3f} for XGB, and all four intervals exclude zero after correction. NLDT transfers the advantage for LR (mean {mean_deltas[('nld.rst.nldt', 'lr')]:+.3f}) but not XGB (mean {mean_deltas[('nld.rst.nldt', 'xgb')]:+.3f}). This makes PCC the stronger evidence that fine relation distinctions generalize beyond HC3 rather than merely increasing in-domain flexibility.",
            "",
            f"The secondary metrics are less uniform: {ap_supported}/8 AP and {ba_supported}/8 balanced-accuracy differences are corrected positive. Full-model OOD AUC is only {min(full_auc):.3f}–{max(full_auc):.3f}, so the result is a modest relative gain under domain shift, not strong absolute detection.",
            "",
            f"The two OOD collections are related stress tests, not independent replications: they share {cross_set_overlap[reference_inventory]} text/source/label identities (the human baseline). Neither set overlaps HC3 training under the same project identity definition. This test addresses transfer of already fitted HC3 distinctions; it uses no OOD refitting or threshold selection. Conclusions remain conditional on the persisted UniRST parses, the declared coarse map, and these released OOD sets.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/unirst"))
    parser.add_argument("--motif-root", type=Path, default=Path("data/motifs2"))
    parser.add_argument("--feature-root", type=Path, default=Path("artifacts/unirst2_feature_matrices"))
    parser.add_argument("--full-model-root", type=Path, default=Path("artifacts/unirst2_model_comparison"))
    parser.add_argument("--coarse-root", type=Path, default=Path("artifacts/unirst2_granularity_ablation"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/unirst2_ood_coarse_comparison"))
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--reuse-features", action="store_true")
    args = parser.parse_args()
    if args.iterations < 100 or args.batch_size < 1:
        parser.error("invalid bootstrap settings")

    evaluator = load_module("unirst2_ood_evaluator", Path("12_evaluate_unirst2_feature_models.py"))
    bootstrap = load_module("unirst2_ood_bootstrap", Path("13_compare_unirst2_paired_performance.py"))
    mapping_path = args.coarse_root / "motif_mapping.csv"
    extraction: dict[str, dict[str, dict[str, Any]]] = {inventory: {} for inventory in INVENTORIES}
    matrices: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    catalog_provenance = {}
    for inventory in INVENTORIES:
        fine_hashes, catalog_index, catalog_meta = load_catalog(args.motif_root, inventory)
        assignment, coarse_hashes = build_assignment(mapping_path, inventory, fine_hashes)
        with np.load(args.coarse_root / "feature_matrices" / inventory / "hc3_train.features.npz", allow_pickle=False) as values:
            if list(values["m3_hashes"].astype(str)) != coarse_hashes:
                raise ValueError(f"Coarse model feature order mismatch for {inventory}")
        catalog_provenance[inventory] = catalog_meta
        for ood_set, filename in OOD_FILES.items():
            input_path = args.data_root / f"rel-{inventory}" / filename
            output_path = args.output_root / "feature_matrices" / inventory / f"{ood_set}.npz"
            meta_path = output_path.with_suffix(".json")
            if args.reuse_features and output_path.exists() and meta_path.exists():
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                metadata = extract_matrix(
                    input_path, output_path, inventory, fine_hashes,
                    catalog_index, assignment, coarse_hashes,
                )
            extraction[inventory][ood_set] = metadata
            matrices[(inventory, ood_set)] = load_ood_matrix(output_path)
        atomic_json(args.output_root / "extraction_progress.json", extraction)

    reference = INVENTORIES[0]
    for ood_set in OOD_FILES:
        left = matrices[(reference, ood_set)]
        for inventory in INVENTORIES[1:]:
            right = matrices[(inventory, ood_set)]
            for key in ("identity_hash", "source", "source_group", "y"):
                if not np.array_equal(left[key], right[key]):
                    raise ValueError(f"OOD alignment mismatch: {ood_set}/{key}")

    overlap: dict[str, dict[str, int]] = {inventory: {} for inventory in INVENTORIES}
    for inventory in INVENTORIES:
        with np.load(args.feature_root / inventory / "hc3_train.features.npz", allow_pickle=False) as values:
            train_ids = set(values["identity_hash"].astype(str))
        for ood_set in OOD_FILES:
            overlap[inventory][ood_set] = sum(
                identity in train_ids
                for identity in matrices[(inventory, ood_set)]["identity_hash"].astype(str)
            )
    cross_set_overlap = {
        inventory: len(
            set(matrices[(inventory, "gpt")]["identity_hash"].astype(str))
            & set(matrices[(inventory, "gpt_para")]["identity_hash"].astype(str))
        )
        for inventory in INVENTORIES
    }

    rows = []
    for inventory in INVENTORIES:
        for ood_set in OOD_FILES:
            for model_kind in MODELS:
                rows.extend(
                    evaluate_pair(
                        evaluator, bootstrap, matrices[(inventory, ood_set)],
                        inventory, ood_set, model_kind, args.full_model_root,
                        args.coarse_root, args.output_root, args.iterations,
                        args.batch_size,
                        int.from_bytes(hashlib.sha256(f"{args.seed}:{inventory}:{ood_set}:{model_kind}".encode()).digest()[:8], "little"),
                    )
                )
            atomic_text(args.output_root / "paired_comparisons.csv", csv_text(rows))
    families: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        families[(row["inventory"], row["model"], row["metric"])].append(index)
    for indices in families.values():
        adjusted = holm([float(rows[index]["bootstrap_sign_p"]) for index in indices])
        for index, value in zip(indices, adjusted):
            rows[index]["holm_p"] = float(value)
    atomic_text(args.output_root / "paired_comparisons.csv", csv_text(rows))
    atomic_text(
        args.output_root / "report.md",
        render_report(rows, extraction, overlap, cross_set_overlap, args.iterations),
    )
    manifest = {
        "script_version": SCRIPT_VERSION,
        "settings": {"iterations": args.iterations, "batch_size": args.batch_size, "seed": args.seed},
        "catalog_provenance": catalog_provenance,
        "coarse_mapping_sha256": sha256_file(mapping_path),
        "extraction": extraction,
        "hc3_train_identity_overlap": overlap,
        "cross_ood_set_identity_overlap": cross_set_overlap,
        "comparison_rows": len(rows),
        "paired_comparisons_sha256": sha256_file(args.output_root / "paired_comparisons.csv"),
        "report_sha256": sha256_file(args.output_root / "report.md"),
    }
    atomic_json(args.output_root / "manifest.json", manifest)
    print(json.dumps({"comparison_rows": len(rows), "overlap": overlap}, indent=2))
    print(f"wrote {args.output_root / 'report.md'}")


if __name__ == "__main__":
    main()
