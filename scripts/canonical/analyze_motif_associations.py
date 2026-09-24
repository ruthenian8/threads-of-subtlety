#!/usr/bin/env python3
"""Test signed M3 motif associations across held-out HC3 domains.

Candidates are selected from existing train-fitted logistic-regression models.
All association estimates use the untouched test split. Presence is expressed
as a human-minus-generated prevalence difference; frequency as a marginal
Hedges' g; and depth as Hedges' g conditional on the motif being present.
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

import joblib
import numpy as np
from tqdm import tqdm


INVENTORIES = (
    "deu.rst.pcc",
    "eng.erst.gum.new",
    "eng.rst.rstdt.new",
    "nld.rst.nldt",
)
REPRESENTATIONS = ("presence", "frequency", "depth")


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


def stable_seed(seed: int, inventory: str, domain: str) -> int:
    payload = f"{seed}:{inventory}:{domain}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def csv_text(rows: Iterable[Mapping[str, Any]]) -> str:
    rows = list(rows)
    fields = list(rows[0]) if rows else []
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def load_test_matrix(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        result = {
            "identity_hash": np.asarray(values["identity_hash"]),
            "domain": np.asarray(values["domain"]),
            "y": np.asarray(values["y"], dtype=np.int8),
            "hashes": np.asarray(values["m3_hashes"]),
            "presence": np.asarray(values["X_m3_presence"], dtype=np.float64),
            "frequency": np.asarray(values["X_m3_frequency"], dtype=np.float64),
            "depth": np.asarray(values["X_m3_depth"], dtype=np.float64),
        }
    dimensions = {result[key].shape for key in REPRESENTATIONS}
    expected = (len(result["y"]), len(result["hashes"]))
    if dimensions != {expected}:
        raise ValueError(f"M3 matrix shape mismatch in {path}: {dimensions}")
    if any(not np.isfinite(result[key]).all() for key in REPRESENTATIONS):
        raise ValueError(f"Non-finite M3 values in {path}")
    return result


def load_fitted_importances(
    model_root: Path, inventory: str, hashes: np.ndarray
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, str]]:
    coefficients: Dict[str, np.ndarray] = {}
    xgb_importances: Dict[str, np.ndarray] = {}
    provenance: Dict[str, str] = {}
    for representation in REPRESENTATIONS:
        config = f"m3_{representation}"
        for model_kind in ("lr", "xgb"):
            directory = model_root / inventory / config / model_kind
            model_path = directory / "model.joblib"
            metrics_path = directory / "metrics.json"
            metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
            actual_hash = sha256_file(model_path)
            if actual_hash != metadata["artifacts"]["model_sha256"]:
                raise ValueError(f"Model hash mismatch: {model_path}")
            provenance[f"{config}/{model_kind}"] = actual_hash
            model = joblib.load(model_path)
            if model_kind == "lr":
                values = np.asarray(
                    model.named_steps["logisticregression"].coef_[0],
                    dtype=np.float64,
                )
                coefficients[representation] = values
            else:
                xgb_importances[representation] = np.asarray(
                    model.feature_importances_, dtype=np.float64
                )
        if coefficients[representation].shape != hashes.shape:
            raise ValueError(f"LR feature mismatch for {inventory}/{representation}")
        if xgb_importances[representation].shape != hashes.shape:
            raise ValueError(f"XGB feature mismatch for {inventory}/{representation}")
    return coefficients, xgb_importances, provenance


def select_candidates(
    hashes: np.ndarray,
    coefficients: Mapping[str, np.ndarray],
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rank motifs by their largest within-representation relative LR magnitude."""
    scores = np.zeros(len(hashes), dtype=np.float64)
    for representation in REPRESENTATIONS:
        magnitude = np.abs(coefficients[representation])
        maximum = float(magnitude.max(initial=0.0))
        if maximum > 0:
            scores = np.maximum(scores, magnitude / maximum)
    order = sorted(range(len(hashes)), key=lambda i: (-scores[i], str(hashes[i])))
    selected = np.asarray(order[: min(top_k, len(order))], dtype=np.int32)
    return selected, scores[selected]


def motif_description(graph: Mapping[str, Any]) -> str:
    node_labels = sorted(str(node.get("label_0", "?")) for node in graph.get("nodes", ()))
    relation_labels = sorted(
        str(edge.get("label_0", "?"))
        for edge in graph.get("links", graph.get("edges", ()))
        if edge.get("label_0") != "/"
    )
    relations = "|".join(relation_labels) if relation_labels else "(structural only)"
    return "nodes=" + "|".join(node_labels) + "; relations=" + relations


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
        raise ValueError("Both labels are required within every domain")
    return {"unique": unique, "inverse": inverse, "by_label": by_label}


def bootstrap_row_weights(
    rng: np.random.Generator, clusters: Mapping[str, Any], size: int
) -> np.ndarray:
    cluster_weights = np.zeros((size, len(clusters["unique"])), dtype=np.int16)
    for label in (0, 1):
        indices = clusters["by_label"][label]
        cluster_weights[:, indices] = rng.multinomial(
            len(indices), np.full(len(indices), 1.0 / len(indices)), size=size
        )
    return cluster_weights[:, clusters["inverse"]]


def prevalence_difference(
    weights: np.ndarray, y: np.ndarray, presence: np.ndarray
) -> np.ndarray:
    human_weights = weights * y
    generated_weights = weights * (1 - y)
    human = (human_weights @ presence) / human_weights.sum(axis=1)[:, None]
    generated = (generated_weights @ presence) / generated_weights.sum(axis=1)[:, None]
    return human - generated


def hedges_g(
    weights: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    include: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if include is None:
        include = np.ones_like(values, dtype=np.float64)
    else:
        include = include.astype(np.float64)
    human_weights = weights[:, :, None] * y[None, :, None] * include[None, :, :]
    generated_weights = (
        weights[:, :, None] * (1 - y)[None, :, None] * include[None, :, :]
    )
    human_n = human_weights.sum(axis=1)
    generated_n = generated_weights.sum(axis=1)
    human_mean = np.divide(
        np.sum(human_weights * values[None, :, :], axis=1),
        human_n,
        out=np.full_like(human_n, np.nan),
        where=human_n > 0,
    )
    generated_mean = np.divide(
        np.sum(generated_weights * values[None, :, :], axis=1),
        generated_n,
        out=np.full_like(generated_n, np.nan),
        where=generated_n > 0,
    )
    human_ss = np.sum(
        human_weights * (values[None, :, :] - human_mean[:, None, :]) ** 2,
        axis=1,
    )
    generated_ss = np.sum(
        generated_weights
        * (values[None, :, :] - generated_mean[:, None, :]) ** 2,
        axis=1,
    )
    degrees = human_n + generated_n - 2
    pooled_sd = np.sqrt(
        np.divide(
            human_ss + generated_ss,
            degrees,
            out=np.full_like(degrees, np.nan),
            where=degrees > 0,
        )
    )
    correction = 1.0 - np.divide(
        3.0,
        4.0 * (human_n + generated_n) - 9.0,
        out=np.zeros_like(human_n),
        where=(4.0 * (human_n + generated_n) - 9.0) != 0,
    )
    effect = np.divide(
        (human_mean - generated_mean) * correction,
        pooled_sd,
        out=np.full_like(pooled_sd, np.nan),
        where=pooled_sd > 0,
    )
    return effect, human_n, generated_n


def bootstrap_sign_pvalue(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    lower = (np.count_nonzero(values <= 0) + 1) / (len(values) + 1)
    upper = (np.count_nonzero(values >= 0) + 1) / (len(values) + 1)
    return float(min(1.0, 2.0 * min(lower, upper)))


def holm_adjust(pvalues: Sequence[float]) -> List[float]:
    result = np.full(len(pvalues), np.nan, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(pvalues))
    order = valid[np.argsort(np.asarray(pvalues)[valid])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * pvalues[index])
        result[index] = min(1.0, running)
    return result.tolist()


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def domain_effect_rows(
    inventory: str,
    domain: str,
    y: np.ndarray,
    identity_hash: np.ndarray,
    hashes: np.ndarray,
    presence: np.ndarray,
    frequency: np.ndarray,
    depth: np.ndarray,
    iterations: int,
    batch_size: int,
    confidence: float,
    seed: int,
    min_depth_per_label: int,
) -> List[Dict[str, Any]]:
    clusters = identity_clusters(identity_hash, y)
    point_weights = np.ones((1, len(y)), dtype=np.int16)
    point = {
        "presence": prevalence_difference(point_weights, y, presence)[0],
        "frequency": hedges_g(point_weights, y, frequency)[0][0],
        "depth": hedges_g(point_weights, y, depth, presence > 0)[0][0],
    }
    depth_counts = hedges_g(point_weights, y, depth, presence > 0)[1:]
    distributions = {
        representation: np.empty((iterations, len(hashes)), dtype=np.float64)
        for representation in REPRESENTATIONS
    }
    rng = np.random.default_rng(seed)
    for start in range(0, iterations, batch_size):
        stop = min(start + batch_size, iterations)
        weights = bootstrap_row_weights(rng, clusters, stop - start)
        distributions["presence"][start:stop] = prevalence_difference(
            weights, y, presence
        )
        distributions["frequency"][start:stop] = hedges_g(
            weights, y, frequency
        )[0]
        distributions["depth"][start:stop] = hedges_g(
            weights, y, depth, presence > 0
        )[0]

    alpha = 1.0 - confidence
    rows: List[Dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        for index, motif_hash in enumerate(hashes):
            valid = True
            human_present = int(depth_counts[0][0, index])
            generated_present = int(depth_counts[1][0, index])
            if representation == "depth" and min(
                human_present, generated_present
            ) < min_depth_per_label:
                valid = False
            samples = distributions[representation][:, index]
            samples = samples[np.isfinite(samples)]
            if not valid or len(samples) < max(100, iterations // 2):
                effect = ci_low = ci_high = pvalue = float("nan")
            else:
                effect = point[representation][index]
                ci_low, ci_high = np.quantile(
                    samples, [alpha / 2.0, 1.0 - alpha / 2.0]
                )
                pvalue = bootstrap_sign_pvalue(samples)
            rows.append(
                {
                    "inventory": inventory,
                    "domain": domain,
                    "representation": representation,
                    "motif_hash": str(motif_hash),
                    "effect_definition": (
                        "human_minus_generated_prevalence"
                        if representation == "presence"
                        else (
                            "human_minus_generated_hedges_g_all_documents"
                            if representation == "frequency"
                            else "human_minus_generated_hedges_g_given_presence"
                        )
                    ),
                    "effect": finite_or_none(effect),
                    "ci_low": finite_or_none(ci_low),
                    "ci_high": finite_or_none(ci_high),
                    "bootstrap_sign_pvalue": finite_or_none(pvalue),
                    "human_documents": int(np.count_nonzero(y == 1)),
                    "generated_documents": int(np.count_nonzero(y == 0)),
                    "human_present": human_present,
                    "generated_present": generated_present,
                    "sufficient_depth_support": bool(
                        min(human_present, generated_present) >= min_depth_per_label
                    ),
                }
            )

    # Correct the candidate family separately within domain and representation.
    for representation in REPRESENTATIONS:
        family = [row for row in rows if row["representation"] == representation]
        adjusted = holm_adjust(
            [
                float(row["bootstrap_sign_pvalue"])
                if row["bootstrap_sign_pvalue"] is not None
                else float("nan")
                for row in family
            ]
        )
        for row, value in zip(family, adjusted):
            row["holm_adjusted_pvalue"] = finite_or_none(value)
            row["robust_direction"] = (
                "positive"
                if row["ci_low"] is not None
                and row["ci_low"] > 0
                and value < 0.05
                else (
                    "negative"
                    if row["ci_high"] is not None
                    and row["ci_high"] < 0
                    and value < 0.05
                    else "uncertain"
                )
            )
    return rows


def summarize_stability(
    inventory: str,
    candidate_rows: Sequence[Mapping[str, Any]],
    effect_rows: Sequence[Mapping[str, Any]],
    minimum_domains: int,
) -> List[Dict[str, Any]]:
    by_hash = {row["motif_hash"]: row for row in candidate_rows}
    results: List[Dict[str, Any]] = []
    for motif_hash, candidate in by_hash.items():
        for representation in REPRESENTATIONS:
            rows = [
                row
                for row in effect_rows
                if row["motif_hash"] == motif_hash
                and row["representation"] == representation
            ]
            positive = sum(row["robust_direction"] == "positive" for row in rows)
            negative = sum(row["robust_direction"] == "negative" for row in rows)
            valid = sum(row["effect"] is not None for row in rows)
            stable_direction = "none"
            if positive >= minimum_domains and negative == 0:
                stable_direction = "human"
            elif negative >= minimum_domains and positive == 0:
                stable_direction = "generated"
            effects = [float(row["effect"]) for row in rows if row["effect"] is not None]
            results.append(
                {
                    "inventory": inventory,
                    "motif_hash": motif_hash,
                    "description": candidate["description"],
                    "representation": representation,
                    "selection_rank": candidate["selection_rank"],
                    "global_lr_coefficient": candidate[f"lr_{representation}_coefficient"],
                    "global_xgb_importance": candidate[f"xgb_{representation}_importance"],
                    "valid_domains": valid,
                    "robust_human_domains": positive,
                    "robust_generated_domains": negative,
                    "uncertain_domains": len(rows) - positive - negative,
                    "median_domain_effect": finite_or_none(float(np.median(effects)))
                    if effects
                    else None,
                    "stable_direction": stable_direction,
                    "stable": stable_direction != "none",
                }
            )
    return results


def discover(output_dir: Path, filename: str, key: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(output_dir.glob(f"*/{filename}")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))[key])
    return rows


def render_report(
    stability: Sequence[Mapping[str, Any]],
    iterations: int,
    confidence: float,
    minimum_domains: int,
) -> str:
    stable = [row for row in stability if row["stable"]]
    stable_by_representation = {
        representation: sum(
            row["stable"] and row["representation"] == representation
            for row in stability
        )
        for representation in REPRESENTATIONS
    }
    coefficient_agreement = sum(
        (row["global_lr_coefficient"] > 0)
        == (row["stable_direction"] == "human")
        for row in stable
    )
    lines = [
        "# UniRST2 held-out motif association stability",
        "",
        (
            "Twelve motifs per relation inventory were selected from existing "
            "train-fitted logistic-regression coefficients. Associations were then "
            f"estimated independently in five held-out HC3 domains using {iterations:,} "
            "identity-cluster bootstrap resamples."
        ),
        "",
        "Presence is a human-minus-generated prevalence difference. Frequency is a "
        "marginal Hedges' g. Depth is Hedges' g among documents containing the motif, "
        "which prevents motif absence from being interpreted as shallow depth.",
        "",
        (
            f"A direction is called stable when at least {minimum_domains}/5 domains "
            f"have a same-signed {100 * confidence:g}% interval excluding zero and "
            "Holm-adjusted bootstrap sign value below 0.05, with no robust domain in "
            "the opposite direction."
        ),
        "",
        "## Stable associations",
        "",
        "| Inventory | Representation | Direction | Motif | Domains | Median effect | Description |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in stable:
        support = max(row["robust_human_domains"], row["robust_generated_domains"])
        lines.append(
            "| {inventory} | {representation} | {stable_direction} | `{motif_hash}` | "
            "{support}/5 | {median_domain_effect:+.3f} | {description} |".format(
                support=support, **row
            )
        )
    if not stable:
        lines.append("| — | — | — | — | 0 | — | No motif passed the stability rule |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                f"{len(stable)} of {len(stability)} candidate-representation pairs "
                "meet the confirmatory stability rule. These are held-out associations, "
                "not causal effects."
            ),
            "",
            (
                f"The stable set contains {stable_by_representation['presence']} presence, "
                f"{stable_by_representation['frequency']} frequency, and "
                f"{stable_by_representation['depth']} conditional-depth associations. "
                f"Its held-out direction agrees with the global training-model coefficient "
                f"for {coefficient_agreement}/{len(stable)} associations."
            ),
            "",
            (
                "Catalogs are inventory-specific. A hash repeated across inventories can "
                "only be treated as the same motif after its labeled graph is verified; "
                "that graph-level cross-standard synthesis is a later step."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def checkpoint_root(
    output_dir: Path, iterations: int, confidence: float, minimum_domains: int
) -> None:
    effects = discover(output_dir, "association_results.json", "effects")
    stability = discover(output_dir, "stability_results.json", "stability")
    atomic_write_text(output_dir / "domain_effects.csv", csv_text(effects))
    atomic_write_text(output_dir / "stability.csv", csv_text(stability))
    atomic_write_text(
        output_dir / "report.md",
        render_report(stability, iterations, confidence, minimum_domains),
    )
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "completed_inventories": sorted(
                {row["inventory"] for row in stability}
            ),
            "domain_effect_rows": len(effects),
            "stability_rows": len(stability),
            "domain_effects_sha256": sha256_file(output_dir / "domain_effects.csv"),
            "stability_sha256": sha256_file(output_dir / "stability.csv"),
            "report_sha256": sha256_file(output_dir / "report.md"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-dir", default="artifacts/canonical/feature_matrices"
    )
    parser.add_argument(
        "--model-dir", default="artifacts/canonical/model_comparison"
    )
    parser.add_argument("--motif-dir", default="data/motifs2")
    parser.add_argument(
        "--output-dir", default="artifacts/canonical/motif_associations"
    )
    parser.add_argument(
        "--inventories",
        nargs="+",
        default=list(INVENTORIES),
        help="Inventory names with feature, model, and motif artifacts.",
    )
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--minimum-domains", type=int, default=4)
    parser.add_argument("--min-depth-per-label", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.top_k < 1 or args.iterations < 100 or args.batch_size < 1:
        parser.error("top-k and batch-size must be positive; iterations must be >= 100")
    if not 0 < args.confidence < 1 or not 1 <= args.minimum_domains <= 5:
        parser.error("invalid confidence or minimum-domains")

    feature_root = Path(args.feature_dir)
    model_root = Path(args.model_dir)
    motif_root = Path(args.motif_dir)
    output_dir = Path(args.output_dir)
    settings = {
        "top_k": args.top_k,
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "confidence": args.confidence,
        "minimum_domains": args.minimum_domains,
        "min_depth_per_label": args.min_depth_per_label,
        "seed": args.seed,
    }
    for inventory in tqdm(args.inventories, desc="motif inventories", unit="inventory"):
        matrix_path = feature_root / inventory / "hc3_test.features.npz"
        catalog_path = motif_root / inventory / "hc3_M3_motifs.json"
        selected_path = (
            motif_root / inventory / "hc3_selected-motif-hashes.generated.json"
        )
        matrix = load_test_matrix(matrix_path)
        coefficients, xgb_importances, model_hashes = load_fitted_importances(
            model_root, inventory, matrix["hashes"]
        )
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        selected_manifest = json.loads(selected_path.read_text(encoding="utf-8"))
        if list(matrix["hashes"]) != selected_manifest["m3"]:
            raise ValueError(f"Selected motif manifest mismatch for {inventory}")
        missing = [motif_hash for motif_hash in matrix["hashes"] if motif_hash not in catalog]
        if missing:
            raise ValueError(f"Catalog misses {len(missing)} selected motifs for {inventory}")
        provenance = {
            "test_matrix": sha256_file(matrix_path),
            "motif_catalog": sha256_file(catalog_path),
            "selected_manifest": sha256_file(selected_path),
            **model_hashes,
        }
        inventory_dir = output_dir / inventory
        result_path = inventory_dir / "association_results.json"
        stability_path = inventory_dir / "stability_results.json"
        required_paths = (
            result_path,
            stability_path,
            inventory_dir / "candidates.json",
            inventory_dir / "domain_effects.csv",
            inventory_dir / "stability.csv",
        )
        if all(path.exists() for path in required_paths) and not args.force:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            existing_stability = json.loads(stability_path.read_text(encoding="utf-8"))
            if (
                existing.get("settings") == settings
                and existing.get("provenance") == provenance
                and existing_stability.get("settings") == settings
                and existing_stability.get("provenance") == provenance
            ):
                print(f"skipping complete {inventory}")
                checkpoint_root(
                    output_dir, args.iterations, args.confidence, args.minimum_domains
                )
                continue

        candidate_indices, selection_scores = select_candidates(
            matrix["hashes"], coefficients, args.top_k
        )
        candidate_hashes = matrix["hashes"][candidate_indices]
        candidate_rows: List[Dict[str, Any]] = []
        for rank, (index, score) in enumerate(
            zip(candidate_indices, selection_scores), 1
        ):
            motif_hash = str(matrix["hashes"][index])
            row: Dict[str, Any] = {
                "inventory": inventory,
                "selection_rank": rank,
                "motif_hash": motif_hash,
                "selection_score": float(score),
                "description": motif_description(catalog[motif_hash]),
                "graph": catalog[motif_hash],
            }
            for representation in REPRESENTATIONS:
                row[f"lr_{representation}_coefficient"] = float(
                    coefficients[representation][index]
                )
                row[f"xgb_{representation}_importance"] = float(
                    xgb_importances[representation][index]
                )
            candidate_rows.append(row)
        atomic_write_json(
            inventory_dir / "candidates.json",
            {"inventory": inventory, "provenance": provenance, "candidates": candidate_rows},
        )

        effects: List[Dict[str, Any]] = []
        domains = sorted(np.unique(matrix["domain"]))
        for domain in tqdm(
            domains, desc=f"domains {inventory}", unit="domain", leave=False
        ):
            mask = matrix["domain"] == domain
            effects.extend(
                domain_effect_rows(
                    inventory,
                    str(domain),
                    matrix["y"][mask],
                    matrix["identity_hash"][mask],
                    candidate_hashes,
                    matrix["presence"][mask][:, candidate_indices],
                    matrix["frequency"][mask][:, candidate_indices],
                    matrix["depth"][mask][:, candidate_indices],
                    args.iterations,
                    args.batch_size,
                    args.confidence,
                    stable_seed(args.seed, inventory, str(domain)),
                    args.min_depth_per_label,
                )
            )
        stability = summarize_stability(
            inventory, candidate_rows, effects, args.minimum_domains
        )
        atomic_write_json(
            result_path,
            {
                "inventory": inventory,
                "settings": settings,
                "provenance": provenance,
                "domains": domains,
                "effects": effects,
            },
        )
        atomic_write_text(inventory_dir / "domain_effects.csv", csv_text(effects))
        atomic_write_json(
            stability_path,
            {
                "inventory": inventory,
                "settings": settings,
                "provenance": provenance,
                "stability": stability,
            },
        )
        atomic_write_text(inventory_dir / "stability.csv", csv_text(stability))
        checkpoint_root(output_dir, args.iterations, args.confidence, args.minimum_domains)
    print(f"wrote {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
