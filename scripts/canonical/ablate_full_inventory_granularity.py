#!/usr/bin/env python3
"""Compare full and parent-coarsened relation features on fixed UniRST trees."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
from tqdm import tqdm
from tos.analysis.granularity import (
    coarsen_graph as coarsen_relation_graph,
    graph_from_data,
    identity_clean_mask,
    relation_labeled_isomorphic,
)


INVENTORY = "eng.erst.gum.new"
SPLITS = ("train", "validation", "test")
REPRESENTATIONS = {
    "relation_frequency": ("X_relation_frequency", "relation_names"),
    "m3_presence": ("X_m3_presence", "m3_hashes"),
    "m3_frequency": ("X_m3_frequency", "m3_hashes"),
    "m3_depth": ("X_m3_depth", "m3_hashes"),
}
MODELS = ("lr", "xgb")
FINE_TO_PARENT = {
    "adversative-antithesis": "adversative",
    "adversative-concession": "adversative",
    "adversative-contrast": "adversative",
    "attribution-negative": "attribution",
    "attribution-positive": "attribution",
    "causal-cause": "causal",
    "causal-result": "causal",
    "context-background": "context",
    "context-circumstance": "context",
    "contingency-condition": "condition",
    "elaboration-additional": "elaboration",
    "elaboration-attribute": "elaboration",
    "evaluation-comment": "evaluation",
    "explanation-evidence": "explanation",
    "explanation-justify": "explanation",
    "explanation-motivation": "explanation",
    "joint-disjunction": "joint",
    "joint-list": "joint",
    "joint-other": "joint",
    "joint-sequence": "joint",
    "mode-manner": "mode",
    "mode-means": "mode",
    "organization-heading": "organization",
    "organization-phatic": "organization",
    "organization-preparation": "organization",
    "purpose-attribute": "purpose",
    "purpose-goal": "purpose",
    "restatement-partial": "restatement",
    "restatement-repetition": "restatement",
    "same-unit": "same-unit",
    "topic-question": "topic",
    "topic-solutionhood": "topic",
    "__OTHER__": "__OTHER__",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def coarsen_motif(graph: nx.DiGraph) -> nx.DiGraph:
    return coarsen_relation_graph(graph, FINE_TO_PARENT, context=INVENTORY)


isomorphic = relation_labeled_isomorphic


def build_assignment(feature_root: Path, motif_root: Path):
    train_path = feature_root / INVENTORY / "hc3_train.features.npz"
    with np.load(train_path, allow_pickle=False) as matrix:
        fine_relations = matrix["relation_names"].astype(str).tolist()
        fine_hashes = matrix["m3_hashes"].astype(str).tolist()

    missing_relations = sorted(set(fine_relations) - set(FINE_TO_PARENT))
    if missing_relations:
        raise ValueError(f"Unmapped relation features: {missing_relations}")
    coarse_relations = sorted(set(FINE_TO_PARENT[r] for r in fine_relations))
    relation_assignment = np.zeros(
        (len(fine_relations), len(coarse_relations)), dtype=np.float32
    )
    relation_index = {name: index for index, name in enumerate(coarse_relations)}
    for fine_index, name in enumerate(fine_relations):
        relation_assignment[fine_index, relation_index[FINE_TO_PARENT[name]]] = 1

    catalog_path = motif_root / INVENTORY / "hc3_M3_motifs.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    groups = {}
    fine_graphs = {}
    for motif_hash in fine_hashes:
        fine_graphs[motif_hash] = graph_from_data(catalog[motif_hash])
        coarse_graph = coarsen_motif(fine_graphs[motif_hash])
        coarse_hash = nx.weisfeiler_lehman_graph_hash(
            coarse_graph, edge_attr="label_0"
        )
        members = groups.setdefault(coarse_hash, [])
        if members and not isomorphic(members[0][1], coarse_graph):
            raise ValueError(f"WL collision for coarse motif {coarse_hash}")
        members.append((motif_hash, coarse_graph))
    coarse_hashes = sorted(groups)
    motif_index = {value: index for index, value in enumerate(coarse_hashes)}
    motif_assignment = np.zeros(
        (len(fine_hashes), len(coarse_hashes)), dtype=np.float32
    )
    mapping = []
    for fine_index, motif_hash in enumerate(fine_hashes):
        coarse_graph = coarsen_motif(fine_graphs[motif_hash])
        coarse_hash = nx.weisfeiler_lehman_graph_hash(
            coarse_graph, edge_attr="label_0"
        )
        motif_assignment[fine_index, motif_index[coarse_hash]] = 1
        mapping.append({
            "fine_hash": motif_hash,
            "coarse_hash": coarse_hash,
            "coarse_index": motif_index[coarse_hash],
            "coarse_group_size": len(groups[coarse_hash]),
        })
    return {
        "fine_relations": fine_relations,
        "coarse_relations": coarse_relations,
        "relation_assignment": relation_assignment,
        "fine_hashes": fine_hashes,
        "coarse_hashes": coarse_hashes,
        "motif_assignment": motif_assignment,
        "motif_mapping": mapping,
        "catalog_sha256": sha256_file(catalog_path),
    }


def build_coarse_splits(feature_root: Path, output_root: Path, assignments):
    result = {}
    for split in SPLITS:
        source = feature_root / INVENTORY / f"hc3_{split}.features.npz"
        with np.load(source, allow_pickle=False) as values:
            arrays = {key: np.asarray(values[key]) for key in values.files}
        relation = arrays["X_relation_frequency"] @ assignments["relation_assignment"]
        frequency = arrays["X_m3_frequency"] @ assignments["motif_assignment"]
        presence = (
            arrays["X_m3_presence"] @ assignments["motif_assignment"] > 0
        ).astype(np.uint8)
        depth_numerator = (
            arrays["X_m3_frequency"] * arrays["X_m3_depth"]
        ) @ assignments["motif_assignment"]
        depth = np.divide(
            depth_numerator, frequency,
            out=np.zeros_like(depth_numerator, dtype=np.float32),
            where=frequency > 0,
        )
        if not np.allclose(
            arrays["X_relation_frequency"].sum(1), relation.sum(1), atol=2e-6
        ):
            raise ValueError(f"Relation mass changed in {split}")
        if not np.allclose(
            arrays["X_m3_frequency"].sum(1), frequency.sum(1), atol=2e-6
        ):
            raise ValueError(f"Motif mass changed in {split}")
        destination = output_root / "features" / f"hc3_{split}.coarse.features.npz"
        atomic_npz(
            destination,
            identity_id=arrays["identity_id"], identity_hash=arrays["identity_hash"],
            source=arrays["source"], domain=arrays["domain"], y=arrays["y"],
            X_relation_frequency=relation.astype(np.float32),
            relation_names=np.asarray(assignments["coarse_relations"]),
            X_m3_presence=presence, X_m3_frequency=frequency.astype(np.float32),
            X_m3_depth=depth, m3_hashes=np.asarray(assignments["coarse_hashes"]),
        )
        result[split] = {"fine": arrays, "coarse_path": destination}
    return result


def model_data(arrays, representation, coarse=False):
    key, names_key = REPRESENTATIONS[representation]
    return {
        "X": np.asarray(arrays[key], dtype=np.float32),
        "y": np.asarray(arrays["y"], dtype=np.int8),
        "identity_id": np.asarray(arrays["identity_id"]),
        "identity_hash": np.asarray(arrays["identity_hash"]),
        "feature_names": np.asarray(arrays[names_key]),
    }


def read_npz(path: Path):
    with np.load(path, allow_pickle=False) as values:
        return {key: np.asarray(values[key]) for key in values.files}


def fit_coarse(evaluator, split_arrays, output_root: Path, n_jobs: int, seed: int):
    results = []
    for representation in REPRESENTATIONS:
        splits = {
            split: model_data(read_npz(split_arrays[split]["coarse_path"]), representation)
            for split in SPLITS
        }
        for kind in MODELS:
            model, selection, validation_probability, threshold = evaluator.fit_selected_model(
                kind, splits["train"], splits["validation"], seed, n_jobs,
                f"coarse {INVENTORY} {representation}/{kind}",
            )
            test_probability = evaluator.probabilities(model, splits["test"]["X"])
            run_dir = output_root / "models" / representation / kind
            prediction_path = run_dir / "predictions.npz"
            model_path = run_dir / "model.joblib"
            atomic_npz(
                prediction_path,
                test_identity_id=splits["test"]["identity_id"],
                test_identity_hash=splits["test"]["identity_hash"],
                test_y=splits["test"]["y"],
                test_probability=test_probability, threshold=np.asarray(threshold),
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            temporary = model_path.with_name(model_path.name + ".tmp")
            joblib.dump(model, temporary)
            os.replace(temporary, model_path)
            row = {
                "representation": representation, "model": kind,
                "coarse_features": int(splits["train"]["X"].shape[1]),
                "selection": selection,
                "validation_metrics": evaluator.metrics(
                    splits["validation"]["y"], validation_probability, threshold
                ),
                "test_metrics": evaluator.metrics(
                    splits["test"]["y"], test_probability, threshold
                ),
                "prediction_path": str(prediction_path.resolve()),
            }
            atomic_json(run_dir / "metrics.json", row)
            results.append(row)
    return results


def holm_adjust(rows):
    for cohort in ("primary", "identity_clean"):
        for metric in ("roc_auc", "average_precision", "balanced_accuracy", "f1", "macro_f1"):
            group = [r for r in rows if r["cohort"] == cohort and r["metric"] == metric]
            ordered = sorted(group, key=lambda r: r["bootstrap_sign_pvalue"])
            running = 0.0
            total = len(ordered)
            for rank, row in enumerate(ordered):
                running = max(running, min(1.0, (total - rank) * row["bootstrap_sign_pvalue"]))
                row["holm_adjusted_pvalue"] = running


def paired_comparisons(paired, feature_root: Path, fine_root: Path, coarse_results, iterations, batch_size, seed):
    rows = []
    with np.load(feature_root / INVENTORY / "hc3_train.features.npz", allow_pickle=False) as values:
        train_hashes = np.asarray(values["identity_hash"])
    for coarse in coarse_results:
        representation, kind = coarse["representation"], coarse["model"]
        fine = paired.load_predictions(
            fine_root / INVENTORY / representation / kind / "predictions.npz"
        )
        with np.load(coarse["prediction_path"], allow_pickle=False) as values:
            coarse_probability = np.asarray(values["test_probability"], dtype=np.float64)
            coarse_threshold = float(values["threshold"])
            for key, coarse_key in (
                ("identity_id", "test_identity_id"),
                ("identity_hash", "test_identity_hash"), ("y", "test_y")
            ):
                if not np.array_equal(fine[key], values[coarse_key]):
                    raise ValueError(f"Fine/coarse row mismatch: {representation}/{kind}")
        masks = {
            "primary": np.ones(len(fine["y"]), dtype=bool),
            "identity_clean": identity_clean_mask(train_hashes, fine["identity_hash"]),
        }
        for cohort, mask in masks.items():
            y = fine["y"][mask]
            fine_probability = fine["probability"][mask]
            coarse_probability_masked = coarse_probability[mask]
            clusters = paired.identity_clusters(fine["identity_hash"][mask], y)
            fine_point = paired.point_metrics(y, fine_probability, fine["threshold"])
            coarse_point = paired.point_metrics(y, coarse_probability_masked, coarse_threshold)
            metric_names = ("roc_auc", "average_precision", "balanced_accuracy", "f1", "macro_f1")
            boot = np.empty((iterations, len(metric_names)), dtype=np.float64)
            rng = np.random.default_rng(seed)
            for start in tqdm(
                range(0, iterations, batch_size),
                desc=f"paired {cohort} {representation}/{kind}",
            ):
                size = min(batch_size, iterations - start)
                weights = paired.bootstrap_row_weights(rng, clusters, size)
                fine_scores = paired.weighted_metric_batch(
                    y, fine_probability, fine["threshold"], weights
                )
                coarse_scores = paired.weighted_metric_batch(
                    y, coarse_probability_masked, coarse_threshold, weights
                )
                boot[start:start + size] = fine_scores - coarse_scores
            for index, metric in enumerate(metric_names):
                differences = boot[:, index]
                rows.append({
                    "cohort": cohort, "rows": int(mask.sum()),
                    "representation": representation, "model": kind, "metric": metric,
                    "fine_score": float(fine_point[index]),
                    "coarse_score": float(coarse_point[index]),
                    "difference": float(fine_point[index] - coarse_point[index]),
                    "ci_low": float(np.quantile(differences, 0.025)),
                    "ci_high": float(np.quantile(differences, 0.975)),
                    "bootstrap_sign_pvalue": float(min(1, 2 * min(
                        (np.count_nonzero(differences <= 0) + 1) / (iterations + 1),
                        (np.count_nonzero(differences >= 0) + 1) / (iterations + 1),
                    ))),
                })
    holm_adjust(rows)
    return rows


def write_csv(path: Path, rows) -> None:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(output.getvalue())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    global INVENTORY, FINE_TO_PARENT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default=INVENTORY)
    parser.add_argument(
        "--mapping-json",
        type=Path,
        help=(
            "JSON object mapping every fine relation feature to its coarse parent. "
            "Omit only for the built-in full-eRST mapping."
        ),
    )
    parser.add_argument("--feature-root", type=Path, default=Path("artifacts/canonical/feature_matrices"))
    parser.add_argument("--motif-root", type=Path, default=Path("data/motifs2"))
    parser.add_argument("--fine-model-root", type=Path, default=Path("artifacts/canonical/model_comparison"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Model seed; defaults to the seed used by the saved fine models.",
    )
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    INVENTORY = args.inventory
    if args.mapping_json is not None:
        loaded_mapping = json.loads(args.mapping_json.read_text(encoding="utf-8"))
        if not isinstance(loaded_mapping, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in loaded_mapping.items()
        ):
            raise ValueError("--mapping-json must contain a string-to-string JSON object")
        FINE_TO_PARENT = loaded_mapping
    elif INVENTORY != "eng.erst.gum.new":
        parser.error("--mapping-json is required unless --inventory is eng.erst.gum.new")
    if args.output_dir is None:
        if INVENTORY == "eng.erst.gum.new":
            args.output_dir = Path("artifacts/canonical/granularity/erst")
        else:
            safe_inventory = INVENTORY.replace("/", "_")
            args.output_dir = Path("artifacts") / f"full_{safe_inventory}_granularity_ablation"
    from scripts.canonical import compare_paired_performance as paired
    from scripts.canonical import evaluate_feature_models as evaluator
    assignments = build_assignment(args.feature_root, args.motif_root)
    split_arrays = build_coarse_splits(args.feature_root, args.output_dir, assignments)
    results = fit_coarse(evaluator, split_arrays, args.output_dir, args.n_jobs, args.seed)
    comparisons = paired_comparisons(
        paired, args.feature_root, args.fine_model_root, results, args.iterations,
        args.batch_size, args.seed,
    )
    write_csv(args.output_dir / "fine_vs_coarse.csv", comparisons)
    audit = {
        "inventory": INVENTORY,
        "settings": {
            "model_seed": args.seed,
            "bootstrap_seed": args.seed,
            "bootstrap_iterations": args.iterations,
            "bootstrap_batch_size": args.batch_size,
            "n_jobs": args.n_jobs,
            "confidence": 0.95,
            "multiplicity": (
                "Holm across eight representation/classifier comparisons "
                "within cohort and metric"
            ),
        },
        "mapping": FINE_TO_PARENT,
        "fine_relation_features": len(assignments["fine_relations"]),
        "coarse_relation_features": len(assignments["coarse_relations"]),
        "fine_m3_features": len(assignments["fine_hashes"]),
        "coarse_m3_features": len(assignments["coarse_hashes"]),
        "motif_mapping_complete": bool(np.all(assignments["motif_assignment"].sum(1) == 1)),
        "relation_mapping_complete": bool(np.all(assignments["relation_assignment"].sum(1) == 1)),
        "catalog_sha256": assignments["catalog_sha256"],
        "motif_mapping": assignments["motif_mapping"],
        "results": results,
        "comparisons": comparisons,
    }
    atomic_json(args.output_dir / "results.json", audit)
    print(f"wrote {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
