#!/usr/bin/env python3
"""Evaluate frozen UniRST2 models on the paired Human Detectors holdout.

No model, threshold, catalog, or coarse mapping is estimated from this set.
The 150 source identifiers each link one human and one machine document, so
uncertainty is estimated by resampling those matched pairs.
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
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from tqdm.auto import tqdm

from tos.tos_dataset import ToSDataset
from tos.tos_utils import node_link_graph_compat


INVENTORIES = (
    "deu.rst.pcc",
    "eng.erst.gum",
    "eng.rst.rstdt",
    "nld.rst.nldt",
)
DISPLAY = {
    "deu.rst.pcc": "PCC",
    "eng.erst.gum": "eRST",
    "eng.rst.rstdt": "RST-DT",
    "nld.rst.nldt": "NLDT",
}
CONFIGS = (
    "structural",
    "relation_frequency",
    "m3_presence",
    "m3_frequency",
    "m3_depth",
)
MODELS = ("lr", "xgb")
METRICS = ("roc_auc", "average_precision", "balanced_accuracy")
COARSE_INVENTORIES = ("deu.rst.pcc", "nld.rst.nldt")
SCRIPT_VERSION = "human-detectors-frozen-holdout-v1"


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


def source_parts(source: str) -> tuple[str, str, str]:
    parts = source.split("::")
    if len(parts) != 4 or parts[0] != "human_detectors":
        raise ValueError(f"Unexpected Human Detectors source identifier: {source!r}")
    return parts[1], parts[2], parts[3]


def load_threshold(path: Path) -> float:
    with np.load(path, allow_pickle=False) as values:
        return float(values["threshold"])


def load_catalog(motif_root: Path, inventory: str) -> tuple[list[str], Any, dict[str, str]]:
    directory = motif_root / inventory
    manifest_path = directory / "hc3_selected-motif-hashes.generated.json"
    catalog_path = directory / "hc3_M3_motifs.json"
    selected = json.loads(manifest_path.read_text(encoding="utf-8"))["m3"]
    motifs = ToSDataset.load_motifs(str(catalog_path), selected)
    return selected, ToSDataset.prepare_motif_catalog_index(motifs), {
        "catalog": sha256_file(catalog_path),
        "selected_manifest": sha256_file(manifest_path),
    }


def count_lines(paths: Sequence[Path]) -> int:
    return sum(
        sum(bool(line.strip()) for line in path.open("rb"))
        for path in paths
    )


def extract_features(
    inventory: str,
    paths: Sequence[Path],
    fine_hashes: Sequence[str],
    catalog_index: Any,
    training_matrix: Path,
    builder: ModuleType,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(training_matrix, allow_pickle=False) as train:
        if train["m3_hashes"].astype(str).tolist() != list(fine_hashes):
            raise ValueError(f"Training feature/catalog order mismatch for {inventory}")
        relation_names_all = train["relation_names"].astype(str).tolist()
        structural_names = train["structural_names"].astype(str)
    relation_vocabulary = [name for name in relation_names_all if name != builder.OTHER_RELATION]

    labels: list[int] = []
    sources: list[str] = []
    generators: list[str] = []
    prompts: list[str] = []
    publishers: list[str] = []
    presence: list[np.ndarray] = []
    frequency: list[np.ndarray] = []
    depth: list[np.ndarray] = []
    relation_counts: list[dict[str, int]] = []
    structural: list[np.ndarray] = []
    scene_counts: list[int] = []
    zero_motif_flags: list[bool] = []
    digest = hashlib.sha256()
    graph_scenes = 0
    zero_motif_documents = 0

    total = count_lines(paths)
    for path in paths:
        with path.open("rb") as handle:
            iterator = tqdm(handle, total=sum(bool(x.strip()) for x in path.open("rb")),
                            desc=f"features {DISPLAY[inventory]}/{path.name.split('.')[0]}", unit="doc")
            for line_number, raw in enumerate(iterator, 1):
                digest.update(raw)
                if not raw.strip():
                    continue
                record = ast.literal_eval(raw.decode("utf-8"))
                trees = record.get("scene_discourse_trees") or {}
                scene_values = trees.values() if isinstance(trees, Mapping) else trees
                for scene in scene_values:
                    if not isinstance(scene, Mapping) or not scene.get("graph_dict"):
                        raise ValueError(f"{path}:{line_number}: scene has no graph")
                    graph = node_link_graph_compat(scene["graph_dict"])
                    root_label = f"span_1-{len(scene.get('edus') or {})}"
                    distribution = ToSDataset.calculate_observed_m3_m6_distributions(
                        graph, {"m3": catalog_index}, root_label
                    )["m3"]
                    scene["motif_dists"] = {"m3": distribution}
                    graph_scenes += 1
                values = builder.aggregate_document(record, len(fine_hashes))
                label = int(record["label"])
                source = str(record["source"])
                generator, prompt, publisher = source_parts(source)
                labels.append(label)
                sources.append(source)
                generators.append(generator)
                prompts.append(prompt)
                publishers.append(publisher)
                presence.append(values[0])
                frequency.append(values[1])
                depth.append(values[2])
                relation_counts.append(values[3])
                structural.append(values[4])
                scene_counts.append(len(trees))
                is_zero_motif = not np.asarray(values[0]).any()
                zero_motif_flags.append(is_zero_motif)
                if is_zero_motif:
                    zero_motif_documents += 1

    input_rows = len(labels)
    source_counts = Counter(sources)
    labels_by_source: dict[str, set[int]] = {}
    for source, label in zip(sources, labels):
        labels_by_source.setdefault(source, set()).add(label)
    complete_sources = {
        source for source, count in source_counts.items()
        if count == 2 and labels_by_source[source] == {0, 1}
    }
    dropped_sources = sorted(set(sources) - complete_sources)
    keep = np.asarray([source in complete_sources for source in sources], dtype=bool)
    labels = [value for value, selected in zip(labels, keep) if selected]
    sources = [value for value, selected in zip(sources, keep) if selected]
    generators = [value for value, selected in zip(generators, keep) if selected]
    prompts = [value for value, selected in zip(prompts, keep) if selected]
    publishers = [value for value, selected in zip(publishers, keep) if selected]
    presence = [value for value, selected in zip(presence, keep) if selected]
    frequency = [value for value, selected in zip(frequency, keep) if selected]
    depth = [value for value, selected in zip(depth, keep) if selected]
    relation_counts = [value for value, selected in zip(relation_counts, keep) if selected]
    structural = [value for value, selected in zip(structural, keep) if selected]
    scene_counts = [value for value, selected in zip(scene_counts, keep) if selected]
    zero_motif_flags = [value for value, selected in zip(zero_motif_flags, keep) if selected]

    rows = {"records": len(labels), "relation_counts": relation_counts}
    relations, relation_totals = builder.relation_matrix(rows, relation_vocabulary)
    result = {
        "inventory": np.asarray(inventory),
        "y": np.asarray(labels, dtype=np.int8),
        "source": np.asarray(sources),
        "pair_id": np.asarray(sources),
        "generator": np.asarray(generators),
        "prompt": np.asarray(prompts),
        "publisher": np.asarray(publishers),
        "X_structural": np.asarray(structural, dtype=np.float32),
        "structural_names": structural_names,
        "X_relation_frequency": relations,
        "relation_names": np.asarray(relation_names_all),
        "relation_total": relation_totals,
        "X_m3_presence": np.asarray(presence, dtype=np.uint8),
        "X_m3_frequency": np.asarray(frequency, dtype=np.float32),
        "X_m3_depth": np.asarray(depth, dtype=np.float32),
        "m3_hashes": np.asarray(fine_hashes),
    }
    if input_rows != total:
        raise ValueError(f"Expected {total} graph rows but extracted {input_rows}")
    retained_source_counts = Counter(sources)
    if set(retained_source_counts.values()) != {2}:
        raise ValueError("Complete-case filtering did not produce matched pairs")
    for source in retained_source_counts:
        pair_labels = [label for label, value in zip(labels, sources) if value == source]
        if sorted(pair_labels) != [0, 1]:
            raise ValueError(f"Pair {source!r} does not contain both labels")
    metadata = {
        "graph_rows_before_pair_filter": input_rows,
        "rows": len(labels),
        "label_counts": dict(sorted(Counter(labels).items())),
        "matched_pairs": len(retained_source_counts),
        "excluded_incomplete_pairs": dropped_sources,
        "generator_pair_counts": {
            key: value // 2 for key, value in sorted(Counter(generators).items())
        },
        "publisher_pair_counts": {
            key: value // 2 for key, value in sorted(Counter(publishers).items())
        },
        "graph_scenes_before_pair_filter": graph_scenes,
        "retained_graph_scenes": sum(scene_counts),
        "zero_motif_documents_before_pair_filter": zero_motif_documents,
        "zero_motif_documents": sum(zero_motif_flags),
        "input_combined_sha256": digest.hexdigest(),
    }
    return result, metadata


def paired_weights(rng: np.random.Generator, pair_ids: np.ndarray, size: int) -> np.ndarray:
    unique, inverse = np.unique(pair_ids.astype(str), return_inverse=True)
    counts = rng.multinomial(len(unique), np.full(len(unique), 1.0 / len(unique)), size=size)
    return counts[:, inverse]


def bootstrap_scores(
    bootstrap: ModuleType,
    y: np.ndarray,
    pair_ids: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    iterations: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    output = np.empty((iterations, len(METRICS)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, iterations, batch_size):
        stop = min(iterations, start + batch_size)
        weights = paired_weights(rng, pair_ids, stop - start)
        output[start:stop] = bootstrap.weighted_metric_batch(
            y, probability, threshold, weights
        )
    return output


def point_and_ci(
    evaluator: ModuleType,
    bootstrap: ModuleType,
    y: np.ndarray,
    pair_ids: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    iterations: int,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    return (
        evaluator.metrics(y, probability, threshold),
        bootstrap_scores(
            bootstrap, y, pair_ids, probability, threshold,
            iterations, batch_size, seed,
        ),
    )


def assignment_from_mapping(
    mapping_path: Path, inventory: str, fine_hashes: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    with mapping_path.open(newline="", encoding="utf-8") as handle:
        rows = sorted(
            (row for row in csv.DictReader(handle) if row["inventory"] == inventory),
            key=lambda row: int(row["fine_index"]),
        )
    if [row["fine_hash"] for row in rows] != list(fine_hashes):
        raise ValueError(f"Coarse mapping order mismatch for {inventory}")
    count = max(int(row["coarse_index"]) for row in rows) + 1
    assignment = np.zeros((len(rows), count), dtype=np.float32)
    hashes: list[str | None] = [None] * count
    for row in rows:
        i, j = int(row["fine_index"]), int(row["coarse_index"])
        assignment[i, j] = 1
        hashes[j] = row["coarse_hash"]
    if not np.array_equal(assignment.sum(axis=1), np.ones(len(rows))) or any(x is None for x in hashes):
        raise ValueError(f"Incomplete coarse assignment for {inventory}")
    return assignment, [str(x) for x in hashes]


def evaluate_frozen(
    evaluator: ModuleType,
    bootstrap: ModuleType,
    features: Mapping[str, np.ndarray],
    X: np.ndarray,
    model_path: Path,
    threshold_path: Path,
    inventory: str,
    representation: str,
    model_kind: str,
    output_root: Path,
    iterations: int,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model = joblib.load(model_path)
    if int(getattr(model, "n_features_in_", X.shape[1])) != X.shape[1]:
        raise ValueError(f"Feature width mismatch for {inventory}/{representation}/{model_kind}")
    threshold = load_threshold(threshold_path)
    probability = evaluator.probabilities(model, X)
    y, pairs = features["y"], features["pair_id"]
    point, samples = point_and_ci(
        evaluator, bootstrap, y, pairs, probability, threshold,
        iterations, batch_size, seed,
    )
    prediction_path = output_root / "predictions" / inventory / representation / f"{model_kind}.npz"
    atomic_npz(
        prediction_path,
        y=y, pair_id=pairs, source=features["source"],
        generator=features["generator"], publisher=features["publisher"],
        probability=probability, threshold=np.asarray(threshold),
    )
    row: dict[str, Any] = {
        "inventory": inventory,
        "representation": representation,
        "model": model_kind,
        "rows": len(y),
        "pairs": len(np.unique(pairs)),
        "feature_count": X.shape[1],
        "threshold": threshold,
        "model_sha256": sha256_file(model_path),
        "predictions_sha256": sha256_file(prediction_path),
    }
    for index, metric in enumerate(METRICS):
        row[metric] = float(point[metric])
        row[f"{metric}_ci_low"] = float(np.quantile(samples[:, index], 0.025))
        row[f"{metric}_ci_high"] = float(np.quantile(samples[:, index], 0.975))
    return row, probability, samples


def contrast_rows(
    bootstrap: ModuleType,
    inventory: str,
    contrast: str,
    model_kind: str,
    candidate_row: Mapping[str, Any],
    reference_row: Mapping[str, Any],
    candidate_samples: np.ndarray,
    reference_samples: np.ndarray,
) -> list[dict[str, Any]]:
    difference = candidate_samples - reference_samples
    rows = []
    for index, metric in enumerate(METRICS):
        values = difference[:, index]
        rows.append({
            "inventory": inventory,
            "contrast": contrast,
            "model": model_kind,
            "metric": metric,
            "candidate_score": candidate_row[metric],
            "reference_score": reference_row[metric],
            "difference": float(candidate_row[metric] - reference_row[metric]),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "bootstrap_sign_p": bootstrap.bootstrap_sign_pvalue(values),
            "pairs": candidate_row["pairs"],
        })
    return rows


def render_report(
    performance: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    extraction: Mapping[str, Mapping[str, Any]],
    iterations: int,
) -> str:
    auc = [row for row in performance if row["representation"] in CONFIGS]
    contrast_auc = [row for row in contrasts if row["metric"] == "roc_auc"]
    lines = [
        "# Frozen UniRST2 evaluation on Human Detectors",
        "",
        "Human Detectors is used once as a paired held-out set. All parsers were already run with shared segmentation. HC3-derived motif catalogs, fitted models, validation-selected thresholds, and fixed-tree coarse mappings are applied unchanged. No Human Detectors outcome was used for fitting or selection.",
        "",
        f"The complete-case cohort contains {extraction[INVENTORIES[0]]['matched_pairs']} matched human–machine source/prompt pairs ({extraction[INVENTORIES[0]]['rows']} documents). Intervals use {iterations:,} matched-pair bootstrap samples.",
        "",
        "## Absolute frozen-model performance",
        "",
        "| Inventory | Representation | Model | AUC (95% CI) | AP | Balanced accuracy |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in auc:
        lines.append(
            f"| {DISPLAY[str(row['inventory'])]} | {row['representation']} | {str(row['model']).upper()} | "
            f"{float(row['roc_auc']):.3f} [{float(row['roc_auc_ci_low']):.3f}, {float(row['roc_auc_ci_high']):.3f}] | "
            f"{float(row['average_precision']):.3f} | {float(row['balanced_accuracy']):.3f} |"
        )
    lines.extend([
        "",
        "## Frozen fine-versus-coarse tests",
        "",
        "| Inventory | Contrast | Model | Candidate AUC | Reference AUC | Difference (95% CI) |",
        "|---|---|---|---:|---:|---:|",
    ])
    for row in contrast_auc:
        lines.append(
            f"| {DISPLAY[str(row['inventory'])]} | {row['contrast']} | {str(row['model']).upper()} | "
            f"{float(row['candidate_score']):.3f} | {float(row['reference_score']):.3f} | "
            f"{float(row['difference']):+.3f} [{float(row['ci_low']):+.3f}, {float(row['ci_high']):+.3f}] |"
        )
    primary = next(
        row for row in contrast_auc
        if row["inventory"] == "deu.rst.pcc"
        and row["contrast"] == "controlled_fine_minus_coarse"
        and row["model"] == "lr"
    )
    if float(primary["ci_low"]) > 0:
        decision = "The prespecified controlled PCC LR fine-over-coarse advantage transfers to Human Detectors: its matched-pair interval excludes zero."
    elif float(primary["ci_high"]) < 0:
        decision = "The prespecified controlled PCC LR contrast reverses on Human Detectors: the coarse representation is better and its matched-pair interval excludes zero."
    else:
        decision = "The prespecified controlled PCC LR interval includes zero. Human Detectors does not independently establish transfer of the fine-over-coarse advantage."
    lines.extend([
        "",
        "## Prespecified decision",
        "",
        decision,
        "",
        "Absolute accuracy and generator/publisher breakdowns are descriptive indicators of temporal and domain shift. The assigned generator in a human row identifies its matched comparison condition; it does not mean that the human article was generated by that model.",
        "",
        "## Coverage",
        "",
        "| Inventory | Parsed documents | Matched pairs | Graph scenes | Zero-M3 documents |",
        "|---|---:|---:|---:|---:|",
    ])
    for inventory in INVENTORIES:
        item = extraction[inventory]
        lines.append(
            f"| {DISPLAY[inventory]} | {item['rows']} | {item['matched_pairs']} | "
            f"{item['retained_graph_scenes']} | {item['zero_motif_documents']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/unirst2"))
    parser.add_argument("--motif-root", type=Path, default=Path("data/motifs2"))
    parser.add_argument("--feature-root", type=Path, default=Path("artifacts/unirst2_feature_matrices"))
    parser.add_argument("--model-root", type=Path, default=Path("artifacts/unirst2_model_comparison"))
    parser.add_argument("--coarse-root", type=Path, default=Path("artifacts/unirst2_granularity_ablation"))
    parser.add_argument("--control-root", type=Path, default=Path("artifacts/unirst2_minimal_contribution/pcc_controls"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/unirst2_human_detectors"))
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--reuse-features", action="store_true")
    args = parser.parse_args()
    if args.iterations < 100 or args.batch_size < 1:
        parser.error("invalid bootstrap settings")

    evaluator = load_module("hd_evaluator", Path("12_evaluate_unirst2_feature_models.py"))
    bootstrap = load_module("hd_bootstrap", Path("13_compare_unirst2_paired_performance.py"))
    builder = load_module("hd_builder", Path("11_build_unirst2_feature_matrices.py"))
    controls = load_module("hd_controls", Path("33_control_pcc_m3_granularity.py"))
    args.output_root.mkdir(parents=True, exist_ok=True)

    features: dict[str, dict[str, np.ndarray]] = {}
    extraction: dict[str, dict[str, Any]] = {}
    catalog_provenance: dict[str, dict[str, str]] = {}
    for inventory in INVENTORIES:
        output_path = args.output_root / "feature_matrices" / inventory / "human_detectors.features.npz"
        metadata_path = output_path.with_suffix(".json")
        fine_hashes, catalog_index, catalog_meta = load_catalog(args.motif_root, inventory)
        catalog_provenance[inventory] = catalog_meta
        if args.reuse_features and output_path.exists() and metadata_path.exists():
            with np.load(output_path, allow_pickle=False) as values:
                matrix = {key: np.asarray(values[key]) for key in values.files}
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            directory = args.data_root / f"rel-{inventory}"
            paths = [
                directory / "human_detectors_test_human.discourse_parsed.graph_added.jsonl",
                directory / "human_detectors_test_machine.discourse_parsed.graph_added.jsonl",
            ]
            missing = [str(path) for path in paths if not path.exists()]
            if missing:
                raise FileNotFoundError(f"Missing graph-added Human Detectors files: {missing}")
            matrix, metadata = extract_features(
                inventory, paths, fine_hashes, catalog_index,
                args.feature_root / inventory / "hc3_train.features.npz", builder,
            )
            atomic_npz(output_path, **matrix)
            metadata.update({"artifact_sha256": sha256_file(output_path)})
            atomic_json(metadata_path, metadata)
        features[inventory] = matrix
        extraction[inventory] = metadata
        atomic_json(args.output_root / "extraction_progress.json", extraction)

    reference = features[INVENTORIES[0]]
    for inventory in INVENTORIES[1:]:
        for key in ("y", "source", "pair_id", "generator", "prompt", "publisher"):
            if not np.array_equal(reference[key], features[inventory][key]):
                raise ValueError(f"Cross-inventory alignment mismatch: {inventory}/{key}")

    performance: list[dict[str, Any]] = []
    probabilities: dict[tuple[str, str, str], np.ndarray] = {}
    samples: dict[tuple[str, str, str], np.ndarray] = {}
    key_map = {
        "structural": "X_structural",
        "relation_frequency": "X_relation_frequency",
        "m3_presence": "X_m3_presence",
        "m3_frequency": "X_m3_frequency",
        "m3_depth": "X_m3_depth",
    }
    for inventory in INVENTORIES:
        matrix = features[inventory]
        for config in CONFIGS:
            for model_kind in MODELS:
                model_dir = args.model_root / inventory / config / model_kind
                row, probability, distribution = evaluate_frozen(
                    evaluator, bootstrap, matrix,
                    np.asarray(matrix[key_map[config]], dtype=np.float32),
                    model_dir / "model.joblib", model_dir / "predictions.npz",
                    inventory, config, model_kind, args.output_root,
                    args.iterations, args.batch_size,
                    int.from_bytes(hashlib.sha256(f"{args.seed}:{inventory}:{config}:{model_kind}".encode()).digest()[:8], "little"),
                )
                performance.append(row)
                probabilities[(inventory, config, model_kind)] = probability
                samples[(inventory, config, model_kind)] = distribution
            atomic_text(args.output_root / "performance.csv", csv_text(performance))

    comparisons: list[dict[str, Any]] = []
    mapping_path = args.coarse_root / "motif_mapping.csv"
    for inventory in COARSE_INVENTORIES:
        matrix = features[inventory]
        assignment, coarse_hashes = assignment_from_mapping(
            mapping_path, inventory, matrix["m3_hashes"].astype(str).tolist()
        )
        X_coarse = np.asarray(matrix["X_m3_frequency"] @ assignment, dtype=np.float32)
        with np.load(args.coarse_root / "feature_matrices" / inventory / "hc3_train.features.npz", allow_pickle=False) as train:
            if train["m3_hashes"].astype(str).tolist() != coarse_hashes:
                raise ValueError(f"Coarse model order mismatch for {inventory}")
        for model_kind in MODELS:
            coarse_dir = args.coarse_root / "models" / inventory / "coarse_all" / model_kind
            coarse_row, _, coarse_samples = evaluate_frozen(
                evaluator, bootstrap, matrix, X_coarse,
                coarse_dir / "model.joblib", coarse_dir / "predictions.npz",
                inventory, "coarse_m3_frequency", model_kind, args.output_root,
                args.iterations, args.batch_size,
                int.from_bytes(hashlib.sha256(f"{args.seed}:{inventory}:fine-coarse:{model_kind}".encode()).digest()[:8], "little"),
            )
            performance.append(coarse_row)
            fine_row = next(row for row in performance if row["inventory"] == inventory and row["representation"] == "m3_frequency" and row["model"] == model_kind)
            # Recompute both candidates with a common resampling stream for an exactly paired contrast.
            common_seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{inventory}:paired:{model_kind}".encode()).digest()[:8], "little")
            fine_common = bootstrap_scores(bootstrap, matrix["y"], matrix["pair_id"], probabilities[(inventory, "m3_frequency", model_kind)], float(fine_row["threshold"]), args.iterations, args.batch_size, common_seed)
            coarse_model = joblib.load(coarse_dir / "model.joblib")
            coarse_probability = evaluator.probabilities(coarse_model, X_coarse)
            coarse_common = bootstrap_scores(bootstrap, matrix["y"], matrix["pair_id"], coarse_probability, float(coarse_row["threshold"]), args.iterations, args.batch_size, common_seed)
            comparisons.extend(contrast_rows(bootstrap, inventory, "fine_minus_coarse", model_kind, fine_row, coarse_row, fine_common, coarse_common))

    inventory = "deu.rst.pcc"
    matrix = features[inventory]
    fine_hashes = matrix["m3_hashes"].astype(str).tolist()
    structural_map, _, _ = controls.structural_assignment(args.motif_root / inventory / "hc3_M3_motifs.json", fine_hashes)
    coarse_map, _ = assignment_from_mapping(mapping_path, inventory, fine_hashes)
    names = matrix["structural_names"].astype(str).tolist()
    scalar = np.log1p(matrix["X_structural"][:, [names.index("text_chars"), names.index("edu_count")]]).astype(np.float32)
    common_controls = np.column_stack((scalar, matrix["X_relation_frequency"], matrix["X_m3_frequency"] @ structural_map)).astype(np.float32)
    controlled = {
        "controls": common_controls,
        "controls_coarse": np.column_stack((common_controls, matrix["X_m3_frequency"] @ coarse_map)).astype(np.float32),
        "controls_fine": np.column_stack((common_controls, matrix["X_m3_frequency"])).astype(np.float32),
    }
    controlled_results: dict[tuple[str, str], tuple[dict[str, Any], np.ndarray]] = {}
    for representation, X in controlled.items():
        for model_kind in MODELS:
            model_dir = args.control_root / "models" / representation / model_kind
            row, probability, _ = evaluate_frozen(
                evaluator, bootstrap, matrix, X,
                model_dir / "model.joblib", model_dir / "predictions.npz",
                inventory, representation, model_kind, args.output_root,
                args.iterations, args.batch_size,
                int.from_bytes(hashlib.sha256(f"{args.seed}:controlled:{representation}:{model_kind}".encode()).digest()[:8], "little"),
            )
            performance.append(row)
            controlled_results[(representation, model_kind)] = (row, probability)
    for model_kind in MODELS:
        candidate_row, candidate_probability = controlled_results[("controls_fine", model_kind)]
        reference_row, reference_probability = controlled_results[("controls_coarse", model_kind)]
        common_seed = int.from_bytes(hashlib.sha256(f"{args.seed}:controlled-paired:{model_kind}".encode()).digest()[:8], "little")
        candidate_samples = bootstrap_scores(bootstrap, matrix["y"], matrix["pair_id"], candidate_probability, float(candidate_row["threshold"]), args.iterations, args.batch_size, common_seed)
        reference_samples = bootstrap_scores(bootstrap, matrix["y"], matrix["pair_id"], reference_probability, float(reference_row["threshold"]), args.iterations, args.batch_size, common_seed)
        comparisons.extend(contrast_rows(bootstrap, inventory, "controlled_fine_minus_coarse", model_kind, candidate_row, reference_row, candidate_samples, reference_samples))

    performance_path = args.output_root / "performance.csv"
    comparisons_path = args.output_root / "paired_comparisons.csv"
    report_path = args.output_root / "report.md"
    atomic_text(performance_path, csv_text(performance))
    atomic_text(comparisons_path, csv_text(comparisons))
    atomic_text(report_path, render_report(performance, comparisons, extraction, args.iterations))
    atomic_json(args.output_root / "manifest.json", {
        "script_version": SCRIPT_VERSION,
        "settings": {"iterations": args.iterations, "batch_size": args.batch_size, "seed": args.seed},
        "protocol": "Frozen HC3 models/catalogs/thresholds; matched source-pair bootstrap; no HD fitting or selection.",
        "catalog_provenance": catalog_provenance,
        "coarse_mapping_sha256": sha256_file(mapping_path),
        "extraction": extraction,
        "outputs": {path.name: sha256_file(path) for path in (performance_path, comparisons_path, report_path)},
    })
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
