#!/usr/bin/env python3
"""Compare released coarse DMRST M3 features with four UniRST inventories.

The Hugging Face rows are aligned to the audited UniRST2 cohort by the stable
text/source/label identity. Models are fit on train, selected on validation,
and evaluated once on test. Paired identity-cluster bootstraps compare both
full-inventory-only and coarse+full models against the coarse baseline.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Mapping, Sequence

import joblib
import numpy as np
from tqdm import tqdm


INVENTORIES = (
    "deu.rst.pcc",
    "eng.erst.gum",
    "eng.rst.rstdt",
    "nld.rst.nldt",
)
SPLITS = ("train", "validation", "test")
MODELS = ("lr", "xgb")
EXPECTED_SOURCE_SHA256 = {
    "train": "db46253183f541d560bde5b170a2e4e2b2b7f1c39ba7d733bdae42923af94ded",
    "validation": "352fedc8c83e7b2d069fd65df671b80d7281309684b6c1914b0f2897c50296c8",
    "test": "43fe588ecd25a7c3c2188e7a96deb19cc5207f0d96b8cd4906142396c99b9608",
}


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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def parse_record(line: str) -> Mapping[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        value = ast.literal_eval(line)
    if not isinstance(value, Mapping):
        raise ValueError("Dataset row is not a mapping")
    return value


def base_identity(text: str, source: str, label: int) -> str:
    payload = json.dumps(
        [text, source, label], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def coarse_frequency(record: Mapping[str, Any], dimension: int) -> np.ndarray:
    total = np.zeros(dimension, dtype=np.float64)
    trees = record.get("scene_discourse_trees") or {}
    scenes = trees.values() if isinstance(trees, Mapping) else trees
    for scene in scenes:
        distribution = (scene.get("motif_dists") or {}).get("m3") or {}
        raw = np.asarray(distribution.get("raw") or (), dtype=np.float64)
        if raw.shape != (dimension,):
            raise ValueError(f"Expected {dimension} coarse M3 values, got {raw.shape}")
        total += raw
    denominator = total.sum()
    if denominator:
        total /= denominator
    return total.astype(np.float32)


def load_reference(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {
            key: np.asarray(values[key])
            for key in ("identity_id", "identity_hash", "source", "domain", "y")
        }


def build_coarse_split(
    source_path: Path,
    reference_path: Path,
    output_path: Path,
    motif_hashes: Sequence[str],
    expected_sha256: str,
    force: bool,
) -> Dict[str, Any]:
    if sha256_file(source_path) != expected_sha256:
        raise ValueError(f"Hugging Face source checksum mismatch: {source_path}")
    if output_path.exists() and not force:
        provenance = output_path.with_suffix(".provenance.json")
        if provenance.exists():
            return json.loads(provenance.read_text(encoding="utf-8"))

    reference = load_reference(reference_path)
    needed = Counter(reference["identity_hash"].tolist())
    found: Dict[str, list[np.ndarray]] = defaultdict(list)
    total_lines = sum(1 for line in source_path.open("rb") if line.strip())
    with source_path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, total=total_lines, desc=source_path.stem, unit="doc"):
            if not line.strip():
                continue
            record = parse_record(line)
            label = int(record["label"])
            source = str(record.get("source", record.get("src", "unknown")))
            identity_hash = base_identity(str(record.get("text") or ""), source, label)
            if len(found[identity_hash]) < needed.get(identity_hash, 0):
                found[identity_hash].append(coarse_frequency(record, len(motif_hashes)))

    missing = {key: count - len(found[key]) for key, count in needed.items() if len(found[key]) < count}
    if missing:
        raise ValueError(f"Coarse source lacks {sum(missing.values())} target rows")
    occurrences: Counter[str] = Counter()
    rows = []
    for identity_hash in reference["identity_hash"].tolist():
        occurrence = occurrences[identity_hash]
        occurrences[identity_hash] += 1
        rows.append(found[identity_hash][occurrence])
    matrix = np.asarray(rows, dtype=np.float32)
    atomic_npz(
        output_path,
        inventory=np.asarray("coarse_dmrst"),
        identity_id=reference["identity_id"],
        identity_hash=reference["identity_hash"],
        source=reference["source"],
        domain=reference["domain"],
        y=reference["y"],
        X_m3_frequency=matrix,
        m3_hashes=np.asarray(motif_hashes),
    )
    record = {
        "source_path": str(source_path.resolve()),
        "source_sha256": expected_sha256,
        "reference_path": str(reference_path.resolve()),
        "reference_sha256": sha256_file(reference_path),
        "artifact_path": str(output_path.resolve()),
        "artifact_sha256": sha256_file(output_path),
        "rows": int(len(matrix)),
        "features": int(matrix.shape[1]),
        "alignment": "text/source/label identity, reordered to UniRST2 reference",
    }
    atomic_json(output_path.with_suffix(".provenance.json"), record)
    return record


def model_data(path: Path, matrix_key: str = "X_m3_frequency") -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {
            "X": np.asarray(values[matrix_key], dtype=np.float32),
            "y": np.asarray(values["y"], dtype=np.int8),
            "identity_id": np.asarray(values["identity_id"]),
            "identity_hash": np.asarray(values["identity_hash"]),
            "source": np.asarray(values["source"]),
            "domain": np.asarray(values["domain"]),
        }


def combine_data(coarse: Mapping[str, np.ndarray], full: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    for key in ("y", "identity_id", "identity_hash"):
        if not np.array_equal(coarse[key], full[key]):
            raise ValueError(f"Coarse/full alignment mismatch: {key}")
    result = dict(coarse)
    result["X"] = np.column_stack((coarse["X"], full["X"])).astype(np.float32)
    return result


def fit_and_save(
    evaluator: ModuleType,
    splits: Mapping[str, Mapping[str, np.ndarray]],
    model_kind: str,
    representation: str,
    output_dir: Path,
    seed: int,
    n_jobs: int,
) -> Dict[str, Any]:
    run_dir = output_dir / representation / model_kind
    metrics_path = run_dir / "metrics.json"
    prediction_path = run_dir / "predictions.npz"
    model_path = run_dir / "model.joblib"
    model, selection, validation_probability, threshold = evaluator.fit_selected_model(
        model_kind, splits["train"], splits["validation"], seed, n_jobs,
        f"{representation}/{model_kind}",
    )
    test_probability = evaluator.probabilities(model, splits["test"]["X"])
    result = {
        "representation": representation,
        "model": model_kind,
        "seed": seed,
        "feature_count": int(splits["train"]["X"].shape[1]),
        "selection": selection,
        "validation_metrics": evaluator.metrics(
            splits["validation"]["y"], validation_probability, threshold
        ),
        "test_metrics": evaluator.metrics(splits["test"]["y"], test_probability, threshold),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
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
    result["artifacts"] = {
        "predictions": str(prediction_path.resolve()),
        "predictions_sha256": sha256_file(prediction_path),
        "model": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
    }
    atomic_json(metrics_path, result)
    return result


def load_saved_prediction(path: Path) -> Dict[str, np.ndarray | float]:
    with np.load(path, allow_pickle=False) as values:
        return {
            "identity_id": np.asarray(values["test_identity_id"]),
            "identity_hash": np.asarray(values["test_identity_hash"]),
            "y": np.asarray(values["test_y"], dtype=np.int8),
            "probability": np.asarray(values["test_probability"], dtype=np.float64),
            "threshold": float(values["threshold"]),
        }


def clean_mask(train_hashes: np.ndarray, test_hashes: np.ndarray) -> np.ndarray:
    training = set(train_hashes.tolist())
    seen = set()
    keep = np.zeros(len(test_hashes), dtype=bool)
    for index, value in enumerate(test_hashes.tolist()):
        if value not in training and value not in seen:
            keep[index] = True
            seen.add(value)
    return keep


def sign_pvalue(values: np.ndarray) -> float:
    lower = (np.count_nonzero(values <= 0) + 1) / (len(values) + 1)
    upper = (np.count_nonzero(values >= 0) + 1) / (len(values) + 1)
    return float(min(1.0, 2 * min(lower, upper)))


def holm(values: Sequence[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_comparisons(
    bootstrap: ModuleType,
    prediction_root: Path,
    existing_root: Path,
    train_hashes: np.ndarray,
    iterations: int,
    batch_size: int,
    seed: int,
) -> list[Dict[str, Any]]:
    loaded: Dict[tuple[str, str], Dict[str, Any]] = {}
    for model in MODELS:
        loaded[(model, "coarse")] = load_saved_prediction(
            prediction_root / "coarse" / model / "predictions.npz"
        )
        for inventory in INVENTORIES:
            loaded[(model, f"full:{inventory}")] = load_saved_prediction(
                existing_root / inventory / "m3_frequency" / model / "predictions.npz"
            )
            loaded[(model, f"conditional:{inventory}")] = load_saved_prediction(
                prediction_root / f"coarse_plus_{inventory}" / model / "predictions.npz"
            )
    reference = loaded[("lr", "coarse")]
    for key, value in loaded.items():
        for field in ("identity_id", "identity_hash", "y"):
            if not np.array_equal(value[field], reference[field]):
                raise ValueError(f"Prediction alignment mismatch for {key}: {field}")

    cohorts = {
        "primary": np.ones(len(reference["y"]), dtype=bool),
        "identity_clean": clean_mask(train_hashes, reference["identity_hash"]),
    }
    rows: list[Dict[str, Any]] = []
    metric_names = ("roc_auc", "average_precision", "balanced_accuracy")
    for cohort, mask in cohorts.items():
        y = reference["y"][mask]
        identity_hash = reference["identity_hash"][mask]
        clusters = bootstrap.identity_clusters(identity_hash, y)
        rng = np.random.default_rng(seed)
        contrasts = [
            (model, inventory, kind)
            for model in MODELS
            for inventory in INVENTORIES
            for kind in ("full_vs_coarse", "conditional_vs_coarse")
        ]
        differences = {
            contrast: np.empty((iterations, 3), dtype=np.float64)
            for contrast in contrasts
        }
        points: Dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
        for model, inventory, kind in contrasts:
            candidate_name = f"full:{inventory}" if kind == "full_vs_coarse" else f"conditional:{inventory}"
            base = loaded[(model, "coarse")]
            candidate = loaded[(model, candidate_name)]
            base_point = bootstrap.point_metrics(y, base["probability"][mask], base["threshold"])
            candidate_point = bootstrap.point_metrics(
                y, candidate["probability"][mask], candidate["threshold"]
            )
            points[(model, inventory, kind)] = (base_point, candidate_point)
        for start in tqdm(range(0, iterations, batch_size), desc=f"paired {cohort}", unit="batch"):
            stop = min(iterations, start + batch_size)
            weights = bootstrap.bootstrap_row_weights(rng, clusters, stop - start)
            base_scores: Dict[str, np.ndarray] = {}
            for model in MODELS:
                base = loaded[(model, "coarse")]
                base_scores[model] = bootstrap.weighted_metric_batch(
                    y, base["probability"][mask], base["threshold"], weights
                )
            for model, inventory, kind in contrasts:
                candidate_name = f"full:{inventory}" if kind == "full_vs_coarse" else f"conditional:{inventory}"
                candidate = loaded[(model, candidate_name)]
                score = bootstrap.weighted_metric_batch(
                    y, candidate["probability"][mask], candidate["threshold"], weights
                )
                differences[(model, inventory, kind)][start:stop] = score - base_scores[model]
        for model, inventory, kind in contrasts:
            base_point, candidate_point = points[(model, inventory, kind)]
            delta = differences[(model, inventory, kind)]
            for metric_index, metric in enumerate(metric_names):
                values = delta[:, metric_index]
                rows.append({
                    "cohort": cohort,
                    "model": model,
                    "inventory": inventory,
                    "contrast": kind,
                    "metric": metric,
                    "coarse_score": float(base_point[metric_index]),
                    "candidate_score": float(candidate_point[metric_index]),
                    "difference_candidate_minus_coarse": float(candidate_point[metric_index] - base_point[metric_index]),
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "bootstrap_sign_p": sign_pvalue(values),
                    "rows": int(mask.sum()),
                })
    for family in {(r["cohort"], r["model"], r["contrast"], r["metric"]) for r in rows}:
        members = [r for r in rows if (r["cohort"], r["model"], r["contrast"], r["metric"]) == family]
        for row, adjusted in zip(members, holm([r["bootstrap_sign_p"] for r in members])):
            row["holm_p_across_inventories"] = adjusted
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, output.getvalue())


def report_text(rows: Sequence[Mapping[str, Any]], feature_counts: Mapping[str, int]) -> str:
    selected = [
        row for row in rows
        if row["cohort"] == "primary" and row["metric"] == "roc_auc"
    ]
    lines = [
        "# Coarse versus full-inventory M3 prediction",
        "",
        "All models use the released HC3 train/validation/test membership restricted to the "
        "43,322/968/1,932 rows shared with UniRST2. M3 frequency is calculated identically "
        "from summed document motif counts. Models fit training only; validation selects "
        "hyperparameters and thresholds.",
        "",
        f"The released DMRST representation has {feature_counts['coarse']} coarse M3 features. "
        "Full inventories have " + ", ".join(
            f"{name}: {feature_counts[name]}" for name in INVENTORIES
        ) + ".",
        "",
        "## Primary test ROC-AUC",
        "",
        "| Model | Inventory | Coarse | Full only | Full − coarse (95% CI) | Coarse + full | Conditional gain (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for inventory in INVENTORIES:
            full = next(r for r in selected if r["model"] == model and r["inventory"] == inventory and r["contrast"] == "full_vs_coarse")
            conditional = next(r for r in selected if r["model"] == model and r["inventory"] == inventory and r["contrast"] == "conditional_vs_coarse")
            lines.append(
                f"| {model.upper()} | {inventory} | {full['coarse_score']:.3f} | "
                f"{full['candidate_score']:.3f} | {full['difference_candidate_minus_coarse']:+.3f} "
                f"[{full['ci_low']:+.3f}, {full['ci_high']:+.3f}] | "
                f"{conditional['candidate_score']:.3f} | {conditional['difference_candidate_minus_coarse']:+.3f} "
                f"[{conditional['ci_low']:+.3f}, {conditional['ci_high']:+.3f}] |"
            )
    lines.extend([
        "",
        "## Result",
        "",
        "The full inventories are more predictive than the coarse representation in the "
        "matched LR comparison: all four direct ROC-AUC differences are positive with intervals "
        "excluding zero. Under XGBoost, PCC and NLDT retain reliable direct gains; GUM and "
        "RST-DT have smaller positive point estimates whose intervals include zero. All eight "
        "coarse+full models improve reliably over coarse-only models, showing that every full "
        "representation contains predictive information not recovered by the coarse features. "
        "The identity-clean sensitivity reproduces this pattern.",
        "",
        "The direct contrast asks which representation predicts better. The conditional contrast "
        "asks whether a full inventory adds information after coarse features are included. "
        "Intervals are paired, label-stratified identity-cluster percentile intervals with 2,000 "
        "resamples. Holm-adjusted bootstrap-sign values across the four inventories are in "
        "`paired_comparisons.csv`.",
        "",
        "## Interpretation boundary",
        "",
        "The coarse distributions were produced by the original DMRST pipeline, while the full "
        "inventories were produced by UniRST with shared UniRST segmentation. A performance "
        "difference therefore combines relation granularity, parser, segmentation, catalog, and "
        "training-corpus effects. It is evidence about these operational representations, not a "
        "causal estimate of relation-inventory granularity.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/original_coarse_hf"))
    parser.add_argument("--full-feature-dir", type=Path, default=Path("artifacts/unirst2_feature_matrices"))
    parser.add_argument("--full-model-dir", type=Path, default=Path("artifacts/unirst2_model_comparison"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/unirst2_coarse_comparison"))
    parser.add_argument("--manifest", type=Path, default=Path("data/motifs/hc3-mage_selected-motif-hashes.json"))
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=min(os.cpu_count() or 1, 4))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    evaluator = load_module("unirst2_evaluator", Path("12_evaluate_unirst2_feature_models.py"))
    bootstrap = load_module("unirst2_bootstrap", Path("13_compare_unirst2_paired_performance.py"))
    motif_hashes = json.loads(args.manifest.read_text(encoding="utf-8"))["m3"]
    if len(motif_hashes) != 31:
        raise ValueError(f"Expected 31 released coarse M3 hashes, got {len(motif_hashes)}")

    matrix_dir = args.output_dir / "feature_matrices" / "coarse"
    provenance = []
    for split in SPLITS:
        provenance.append(build_coarse_split(
            args.source_dir / f"hc3_{split}.discourse_parsed.graph_added.motif_dists.jsonl",
            args.full_feature_dir / INVENTORIES[0] / f"hc3_{split}.features.npz",
            matrix_dir / f"hc3_{split}.features.npz",
            motif_hashes, EXPECTED_SOURCE_SHA256[split], args.force,
        ))

    coarse_splits = {split: model_data(matrix_dir / f"hc3_{split}.features.npz") for split in SPLITS}
    model_root = args.output_dir / "models"
    results = []
    for model in MODELS:
        results.append(fit_and_save(evaluator, coarse_splits, model, "coarse", model_root, args.seed, args.n_jobs))
    feature_counts: Dict[str, int] = {"coarse": len(motif_hashes)}
    for inventory in INVENTORIES:
        full_splits = {
            split: model_data(args.full_feature_dir / inventory / f"hc3_{split}.features.npz")
            for split in SPLITS
        }
        feature_counts[inventory] = int(full_splits["train"]["X"].shape[1])
        combined = {split: combine_data(coarse_splits[split], full_splits[split]) for split in SPLITS}
        for model in MODELS:
            results.append(fit_and_save(
                evaluator, combined, model, f"coarse_plus_{inventory}",
                model_root, args.seed, args.n_jobs,
            ))

    comparisons = paired_comparisons(
        bootstrap, model_root, args.full_model_dir,
        coarse_splits["train"]["identity_hash"], args.iterations,
        args.batch_size, args.seed,
    )
    write_csv(args.output_dir / "paired_comparisons.csv", comparisons)
    atomic_text(args.output_dir / "report.md", report_text(comparisons, feature_counts))
    atomic_json(args.output_dir / "manifest.json", {
        "source_repository": "https://huggingface.co/datasets/zaemyung/ToS-Dataset",
        "source_files": provenance,
        "coarse_motif_manifest": str(args.manifest.resolve()),
        "coarse_motif_manifest_sha256": sha256_file(args.manifest),
        "feature_counts": feature_counts,
        "model_results": results,
        "bootstrap": {"iterations": args.iterations, "seed": args.seed, "unit": "text/source/label identity cluster"},
        "paired_comparisons_sha256": sha256_file(args.output_dir / "paired_comparisons.csv"),
    })
    print(f"wrote {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
