#!/usr/bin/env python3
"""Measure motif/label dependency across the UniRST HC3 and OOD sets.

The validation split is the only split used for model selection and fitting.
The HC3 test and OOD files are read only for the final, no-refit evaluation.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm


INVENTORIES = ("deu.rst.pcc", "eng.erst.gum", "eng.rst.rstdt", "nld.rst.nldt")
SPLITS = ("validation", "test", "ood_gpt", "ood_para")
FILE_NAMES = {
    "validation": "hc3_validation_pcc_aligned.discourse_parsed.graph_added.m3_m6_motif_dists.jsonl",
    "test": "hc3_test_pcc_aligned.discourse_parsed.graph_added.m3_m6_motif_dists.jsonl",
    "ood_gpt": "test_ood_set_gpt.discourse_parsed.graph_added.m3_m6_motif_dists.jsonl",
    "ood_para": "test_ood_set_gpt_para.discourse_parsed.graph_added.m3_m6_motif_dists.jsonl",
}


@dataclass
class Corpus:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    sources: np.ndarray
    feature_names: List[str]
    records: int
    structural: np.ndarray


def _read_record(line: str) -> Mapping[str, Any]:
    """Read both JSONL and the Python-literal JSONL emitted by project scripts."""
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        value = ast.literal_eval(line)
    if not isinstance(value, Mapping):
        raise ValueError("Each dataset line must contain a mapping")
    return value


def _source_group(source: Any) -> str:
    source = str(source or "unknown")
    # Source names are domain + provenance suffixes (for example
    # medicine_human and medicine_llm).  Group both labels by domain.
    group = re.sub(r"_(?:human|llm|gpt4|machine|para)$", "", source)
    return group or source


def _scene_values(scene: Mapping[str, Any], size: str) -> Tuple[np.ndarray, np.ndarray]:
    distributions = scene.get("motif_dists") or {}
    dist = distributions.get(size)
    if not isinstance(dist, Mapping):
        raise ValueError(f"Scene is missing {size} motif distributions")
    raw = np.asarray(dist.get("raw", []), dtype=float)
    wad = np.asarray(dist.get("wad", []), dtype=float)
    if raw.ndim != 1 or wad.shape != raw.shape:
        raise ValueError(f"Malformed {size} raw/wad arrays")
    return raw, wad


def _aggregate_record(record: Mapping[str, Any], dimensions: Mapping[str, int]) -> Tuple[np.ndarray, Dict[str, float]]:
    trees = record.get("scene_discourse_trees") or {}
    if isinstance(trees, Mapping):
        scenes = [scene for scene in trees.values() if isinstance(scene, Mapping)]
    elif isinstance(trees, Sequence):
        scenes = [scene for scene in trees if isinstance(scene, Mapping)]
    else:
        scenes = []

    feature_groups: List[np.ndarray] = []
    total_edus = 0
    total_nodes = 0
    total_edges = 0
    for size in ("m3", "m6"):
        expected = dimensions[size]
        raw_total = np.zeros(expected, dtype=float)
        depth_total = np.zeros(expected, dtype=float)
        for scene in scenes:
            raw, wad = _scene_values(scene, size)
            if len(raw) != expected:
                raise ValueError(f"{size} dimension {len(raw)} does not match manifest ({expected})")
            present = (raw > 0) & np.isfinite(wad) & (wad >= 0)
            raw_total += raw
            depth_total[present] += raw[present] * wad[present]
            total_edus += len(scene.get("edus") or {}) if size == "m3" else 0
            graph = scene.get("graph_dict") or {}
            total_nodes += len(graph.get("nodes") or []) if size == "m3" else 0
            total_edges += len(graph.get("links") or []) if size == "m3" else 0
        total = raw_total.sum()
        mf = raw_total / total if total > 0 else np.zeros_like(raw_total)
        wad = np.full(expected, -1.0, dtype=float)
        present = raw_total > 0
        wad[present] = depth_total[present] / raw_total[present]
        interleaved = np.empty(expected * 2, dtype=float)
        interleaved[0::2] = mf
        interleaved[1::2] = wad
        feature_groups.append(interleaved)

    text = str(record.get("text") or "")
    structural = {
        "text_chars": float(len(text)),
        "scene_count": float(len(scenes)),
        "edu_count": float(total_edus),
        "node_count": float(total_nodes),
        "edge_count": float(total_edges),
    }
    return np.concatenate(feature_groups), structural


def _feature_names(manifest: Mapping[str, Sequence[str]]) -> List[str]:
    names: List[str] = []
    for size in ("m3", "m6"):
        for i, motif_hash in enumerate(manifest[size]):
            names.extend((f"{size}:{i}:{motif_hash}:mf", f"{size}:{i}:{motif_hash}:wad"))
    return names


def load_corpus(
    path: Path,
    manifest: Mapping[str, Sequence[str]],
    *,
    show_progress: bool = True,
    progress_desc: Optional[str] = None,
) -> Corpus:
    dimensions = {size: len(manifest[size]) for size in ("m3", "m6")}
    motif_rows: List[np.ndarray] = []
    structural_rows: List[List[float]] = []
    labels: List[int] = []
    groups: List[str] = []
    sources: List[str] = []
    with path.open(encoding="utf-8") as handle:
        iterator: Iterable[str] = handle
        if show_progress:
            iterator = tqdm(
                iterator,
                desc=progress_desc or path.name,
                unit="doc",
                leave=False,
            )
        for line_number, line in enumerate(iterator, 1):
            if not line.strip():
                continue
            record = _read_record(line)
            if "label" not in record:
                raise ValueError(f"{path}:{line_number} has no label")
            label = int(record["label"])
            if label not in (0, 1):
                raise ValueError(f"{path}:{line_number} has non-binary label {label}")
            motif, structural = _aggregate_record(record, dimensions)
            motif_rows.append(motif)
            structural_rows.append([structural[k] for k in ("text_chars", "scene_count", "edu_count", "node_count", "edge_count")])
            labels.append(label)
            source = str(record.get("source", record.get("src", "unknown")))
            sources.append(source)
            groups.append(_source_group(source))
    if not motif_rows:
        raise ValueError(f"No records found in {path}")
    names = _feature_names(manifest)
    X = np.asarray(motif_rows, dtype=float)
    X[~np.isfinite(X)] = 0.0
    return Corpus(X, np.asarray(labels, dtype=int), np.asarray(groups), np.asarray(sources), names, len(labels), np.asarray(structural_rows, dtype=float))


def _metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> Dict[str, Any]:
    from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score, roc_auc_score, recall_score)
    prediction = (probability >= threshold).astype(int)
    result: Dict[str, Any] = {
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else float("nan"),
        "average_precision": float(average_precision_score(y, probability)) if len(np.unique(y)) == 2 else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, prediction)),
        "recall_human": float(recall_score(y, prediction, pos_label=1, zero_division=0)),
        "recall_generated": float(recall_score(y, prediction, pos_label=0, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }
    return result


def _threshold(y: np.ndarray, probability: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score
    candidates = np.unique(np.r_[0.0, probability, 1.0])
    scores = np.asarray([balanced_accuracy_score(y, probability >= value) for value in candidates])
    best = np.flatnonzero(scores == scores.max())
    return float(candidates[best[len(best) // 2]])


def _model(kind: str, params: Mapping[str, Any], seed: int, n_jobs: int):
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(C=float(params["C"]), class_weight="balanced", max_iter=5000, random_state=seed))
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError("XGBoost is required; install requirements.txt") from exc
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        max_depth=int(params["max_depth"]), min_child_weight=int(params["min_child_weight"]),
        n_estimators=int(params["n_estimators"]), scale_pos_weight=float(params["scale_pos_weight"]),
        random_state=seed, n_jobs=n_jobs,
    )


def _probability(model, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def select_and_fit(
    corpus: Corpus,
    kind: str,
    seed: int,
    n_jobs: int,
    *,
    progress_desc: Optional[str] = None,
    show_progress: bool = True,
) -> Tuple[Any, Dict[str, Any], float, Dict[str, float]]:
    from sklearn.model_selection import LeaveOneGroupOut
    if kind == "lr":
        grid = [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)]
    else:
        positives = max(1, int(corpus.y.sum()))
        negatives = max(1, len(corpus.y) - positives)
        ratio = negatives / positives
        grid = [{"max_depth": d, "min_child_weight": w, "n_estimators": e, "scale_pos_weight": ratio}
                for d in (2, 3) for w in (1, 5) for e in (200, 400)]
    splitter = LeaveOneGroupOut()
    groups = corpus.groups if len(np.unique(corpus.groups)) >= 2 else np.arange(len(corpus.y))
    folds = list(splitter.split(corpus.X, corpus.y, groups))
    best_score = -math.inf
    best_params: Optional[Dict[str, Any]] = None
    best_oof: Optional[np.ndarray] = None
    progress = tqdm(
        total=len(grid) * len(folds) + 1,
        desc=progress_desc or f"tuning {kind}",
        unit="fit",
        leave=False,
        disable=not show_progress,
    )
    try:
        for candidate_index, params in enumerate(grid, 1):
            oof = np.full(len(corpus.y), np.nan)
            for fold_index, (train_idx, valid_idx) in enumerate(folds, 1):
                progress.set_postfix_str(
                    f"candidate {candidate_index}/{len(grid)}, "
                    f"fold {fold_index}/{len(folds)}",
                    refresh=False,
                )
                if len(np.unique(corpus.y[train_idx])) < 2 or len(np.unique(corpus.y[valid_idx])) < 2:
                    progress.update(1)
                    continue
                model = _model(kind, params, seed, n_jobs)
                model.fit(corpus.X[train_idx], corpus.y[train_idx])
                oof[valid_idx] = _probability(model, corpus.X[valid_idx])
                progress.update(1)
            valid = np.isfinite(oof)
            if valid.sum() == 0 or len(np.unique(corpus.y[valid])) < 2:
                continue
            from sklearn.metrics import roc_auc_score
            score = float(roc_auc_score(corpus.y[valid], oof[valid]))
            if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and (best_params is None or str(params) < str(best_params))):
                best_score, best_params, best_oof = score, dict(params), oof
            progress.set_postfix_str(
                f"candidate {candidate_index}/{len(grid)}, best AUC={best_score:.4f}",
                refresh=True,
            )
        if best_params is None or best_oof is None:
            raise ValueError("Unable to produce grouped validation folds with both labels")
        threshold = _threshold(corpus.y[np.isfinite(best_oof)], best_oof[np.isfinite(best_oof)])
        progress.set_postfix_str("refitting selected model", refresh=True)
        fitted = _model(kind, best_params, seed, n_jobs)
        fitted.fit(corpus.X, corpus.y)
        progress.update(1)
    finally:
        progress.close()
    return fitted, best_params, threshold, {"roc_auc": best_score, "threshold": threshold}


def _bootstrap_ci(y: np.ndarray, probability: np.ndarray, threshold: float, seed: int, n: int = 300) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    indices = [np.flatnonzero(y == label) for label in (0, 1)]
    samples: Dict[str, List[float]] = {k: [] for k in ("roc_auc", "average_precision", "balanced_accuracy", "macro_f1")}
    for _ in range(n):
        chosen = np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in indices])
        result = _metrics(y[chosen], probability[chosen], threshold)
        for key in samples:
            if np.isfinite(result[key]):
                samples[key].append(result[key])
    return {key: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) if values else (float("nan"), float("nan")) for key, values in samples.items()}


def _config_matrix(corpus: Corpus, config: str) -> Tuple[np.ndarray, List[str]]:
    motif_dim = len(corpus.feature_names)
    if config == "m3":
        idx = [i for i, name in enumerate(corpus.feature_names) if name.startswith("m3:")]
        return corpus.X[:, idx], [corpus.feature_names[i] for i in idx]
    if config == "m6":
        idx = [i for i, name in enumerate(corpus.feature_names) if name.startswith("m6:")]
        return corpus.X[:, idx], [corpus.feature_names[i] for i in idx]
    structural_names = ["struct:text_chars", "struct:scene_count", "struct:edu_count", "struct:node_count", "struct:edge_count"]
    if config == "motifs":
        return corpus.X, corpus.feature_names
    if config == "structural":
        return corpus.structural, structural_names
    if config == "combined":
        return np.concatenate([corpus.X, corpus.structural], axis=1), corpus.feature_names + structural_names
    raise ValueError(config)


def _format(value: Any) -> str:
    if isinstance(value, float):
        return "nan" if not np.isfinite(value) else f"{value:.4f}"
    return str(value)


def _importance(model, names: Sequence[str], kind: str, limit: int = 10) -> List[Tuple[str, float]]:
    if kind == "lr":
        estimator = model[-1]
        signed = np.asarray(estimator.coef_[0])
        values = np.abs(signed)
    else:
        values = np.asarray(model.feature_importances_)
        signed = values
    order = np.argsort(values)[::-1][:limit]
    return [(names[i], float(signed[i])) for i in order if values[i] > 0]


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _render_report(lines: Sequence[str]) -> str:
    return "\n".join(
        list(lines)
        + [
            "Interpretation: predictive association is not evidence of causation; source, length, and parser effects may confound motif distributions."
        ]
    ) + "\n"


def analyze(args: argparse.Namespace) -> str:
    from sklearn.inspection import permutation_importance
    root = Path(args.data_dir)
    report: List[str] = ["Motif–label dependency analysis", "Validation-only fitting; evaluation splits are never used for refitting.", ""]
    output_path = Path(args.output)
    partial_path = output_path.with_name(output_path.name + ".partial")
    artifact_arg = getattr(args, "artifact_dir", None)
    artifact_dir = (
        Path(artifact_arg)
        if artifact_arg
        else output_path.with_name(f"{output_path.stem}_artifacts")
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    configs = tuple(args.configs)
    overall = tqdm(
        total=len(args.inventories) * len(configs) * len(args.models),
        desc="analysis",
        unit="model",
        disable=args.quiet,
    )
    try:
        for inventory in args.inventories:
            inventory_start = len(report)
            data_dir = root / "unirst" / f"rel-{inventory}"
            motif_dir = root / "motifs" / inventory
            manifest_path = motif_dir / "hc3_selected-motif-hashes.generated.json"
            if not manifest_path.exists():
                manifest_path = motif_dir / "hc3_selected-motif-hashes.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"No motif manifest for {inventory}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            corpora: Dict[str, Corpus] = {}
            report.append(f"[{inventory}]")
            for split in SPLITS:
                path = data_dir / FILE_NAMES[split]
                if not path.exists():
                    raise FileNotFoundError(path)
                overall.set_postfix_str(f"loading {inventory}/{split}", refresh=True)
                corpus = load_corpus(
                    path,
                    manifest,
                    show_progress=not args.quiet,
                    progress_desc=f"load {inventory}/{split}",
                )
                corpora[split] = corpus
                counts = np.bincount(corpus.y, minlength=2)
                report.append(f"{split}: n={len(corpus.y)}, labels={{0:{counts[0]}, 1:{counts[1]}}}, motif_dim={corpus.X.shape[1]}")
            for config in configs:
                train_X, names = _config_matrix(corpora["validation"], config)
                eval_corpora = {}
                for split, corpus in corpora.items():
                    eval_corpora[split], eval_names = _config_matrix(corpus, config)
                    if eval_names != names:
                        raise ValueError(f"Feature mapping differs between validation and {split} for {config}")
                train = Corpus(train_X, corpora["validation"].y, corpora["validation"].groups, corpora["validation"].sources, names, len(train_X), corpora["validation"].structural)
                for kind in args.models:
                    stage = f"{inventory} {config}/{kind}"
                    overall.set_postfix_str(stage, refresh=True)
                    model, params, threshold, cv = select_and_fit(
                        train,
                        kind,
                        args.seed,
                        args.n_jobs,
                        progress_desc=f"tune {stage}",
                        show_progress=not args.quiet,
                    )
                    report.append(f"{config}/{kind}: validation_cv_roc_auc={_format(cv['roc_auc'])}, threshold={_format(threshold)}, params={params}")
                    for split in ("test", "ood_gpt", "ood_para"):
                        overall.set_postfix_str(f"evaluate {stage}/{split}", refresh=True)
                        probabilities = _probability(model, eval_corpora[split])
                        metrics = _metrics(corpora[split].y, probabilities, threshold)
                        ci = _bootstrap_ci(corpora[split].y, probabilities, threshold, args.seed, args.bootstrap)
                        report.append(f"  {split}: " + ", ".join(f"{key}={_format(value)}" for key, value in metrics.items() if key != "confusion_matrix") + f", confusion={metrics['confusion_matrix']}")
                        report.append("    CI95: " + ", ".join(f"{key}=({_format(value[0])},{_format(value[1])})" for key, value in ci.items()))
                    report.append("  top_features: " + ", ".join(f"{name}={value:.5g}" for name, value in _importance(model, names, kind)))
                    if args.permutation and config in ("motifs", "combined"):
                        overall.set_postfix_str(f"permutation {stage}", refresh=True)
                        perm = permutation_importance(model, eval_corpora["test"], corpora["test"].y, scoring="roc_auc", n_repeats=3, random_state=args.seed, n_jobs=args.n_jobs)
                        order = np.argsort(perm.importances_mean)[::-1][:10]
                        report.append("  test_permutation: " + ", ".join(f"{names[i]}={perm.importances_mean[i]:.5g}" for i in order if perm.importances_mean[i] > 0))
                    overall.update(1)
            report.append("")
            inventory_report = _render_report(
                [
                    "Motif–label dependency analysis",
                    "Validation-only fitting; evaluation splits are never used for refitting.",
                    "",
                ]
                + report[inventory_start:]
            )
            artifact_path = artifact_dir / f"{inventory}.txt"
            _atomic_write_text(artifact_path, inventory_report)
            _atomic_write_text(partial_path, _render_report(report))
            overall.set_postfix_str(f"saved {artifact_path.name}", refresh=True)
            if not args.quiet:
                tqdm.write(f"saved {artifact_path} and checkpointed {partial_path}")
    finally:
        overall.close()
    return _render_report(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="motif_label_dependency_report.txt")
    parser.add_argument(
        "--artifact-dir",
        help="per-inventory artifact directory (default: <output stem>_artifacts)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--permutation", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--inventories", nargs="+", choices=INVENTORIES, default=list(INVENTORIES))
    parser.add_argument("--configs", nargs="+", choices=("m3", "m6", "motifs", "structural", "combined"), default=["m3", "m6", "motifs", "structural", "combined"])
    parser.add_argument("--models", nargs="+", choices=("lr", "xgb"), default=["lr", "xgb"])
    args = parser.parse_args()
    # Avoid oversubscribing constrained workstations when ``-1`` is supplied.
    # XGBoost and BLAS can otherwise each create one worker per visible CPU.
    if args.n_jobs < 0:
        args.n_jobs = min(os.cpu_count() or 1, 4)
    output = Path(args.output)
    report = analyze(args)
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        os.replace(partial, output)
    else:
        _atomic_write_text(output, report)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
