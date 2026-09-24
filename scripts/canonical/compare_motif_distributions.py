#!/usr/bin/env python3
"""Compare human and machine M3 motif distributions for four UniRST inventories.

The analysis follows the signed comparison in Kim et al. (2024): every
difference is machine minus human.  Motif frequencies are document-normalized,
IDF is learned once from the released training split, and all display choices
are frozen from training before test effects are read.  The released paper
split is primary; a test cohort with training overlaps removed and duplicate
identities consolidated is reported as a sensitivity analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import spearmanr
from tqdm import tqdm


INVENTORIES = (
    "deu.rst.pcc",
    "eng.erst.gum.new",
    "eng.rst.rstdt.new",
    "nld.rst.nldt",
)
DOMAINS = ("finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai")
SPLITS = ("train", "validation", "test")
METRICS = (
    "presence",
    "frequency",
    "mfidf",
    "conditional_depth",
    "depth_contribution",
)
PAPER_URL = "https://aclanthology.org/2024.acl-long.298.pdf"
SCRIPT_VERSION = "paper-style-motif-distributions-v1"


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


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def csv_text(rows: Iterable[Mapping[str, Any]], fields: Sequence[str] | None = None) -> str:
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else []
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def stable_seed(seed: int, *parts: str) -> int:
    payload = ":".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def load_matrix(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        result = {
            "identity_hash": np.asarray(values["identity_hash"]).astype(str),
            "domain": np.asarray(values["domain"]).astype(str),
            "y": np.asarray(values["y"], dtype=np.int8),
            "hashes": np.asarray(values["m3_hashes"]).astype(str),
            "presence": np.asarray(values["X_m3_presence"], dtype=np.float32),
            "frequency": np.asarray(values["X_m3_frequency"], dtype=np.float32),
            "depth": np.asarray(values["X_m3_depth"], dtype=np.float32),
        }
    expected = (len(result["y"]), len(result["hashes"]))
    for name in ("presence", "frequency", "depth"):
        if result[name].shape != expected:
            raise ValueError(f"{path}: {name} shape {result[name].shape}, expected {expected}")
        if not np.isfinite(result[name]).all():
            raise ValueError(f"{path}: non-finite values in {name}")
    if not set(np.unique(result["y"])).issubset({0, 1}):
        raise ValueError(f"{path}: labels must be 0=machine and 1=human")
    return result


def metric_matrices(data: Mapping[str, np.ndarray], idf: np.ndarray) -> dict[str, np.ndarray]:
    frequency = np.asarray(data["frequency"], dtype=np.float32)
    presence = np.asarray(data["presence"], dtype=np.float32)
    depth = np.asarray(data["depth"], dtype=np.float32)
    return {
        "presence": presence,
        "frequency": frequency,
        "mfidf": frequency * idf[None, :],
        "conditional_depth": depth,
        "depth_contribution": frequency * depth,
    }


def collapse_clean_test(data: Mapping[str, np.ndarray], train_identities: set[str]) -> dict[str, np.ndarray]:
    keep = np.asarray([identity not in train_identities for identity in data["identity_hash"]])
    identities = data["identity_hash"][keep]
    unique, inverse = np.unique(identities, return_inverse=True)
    result: dict[str, Any] = {"identity_hash": unique, "hashes": data["hashes"]}
    labels = np.empty(len(unique), dtype=np.int8)
    domains = np.empty(len(unique), dtype=object)
    for cluster in range(len(unique)):
        rows = np.flatnonzero(inverse == cluster)
        observed_labels = np.unique(data["y"][keep][rows])
        observed_domains = np.unique(data["domain"][keep][rows])
        if len(observed_labels) != 1 or len(observed_domains) != 1:
            raise ValueError(f"identity {unique[cluster]} crosses label or domain")
        labels[cluster] = observed_labels[0]
        domains[cluster] = observed_domains[0]
    result["y"] = labels
    result["domain"] = domains.astype(str)
    for name in ("presence", "frequency", "depth"):
        values = np.asarray(data[name][keep], dtype=np.float64)
        sums = np.zeros((len(unique), values.shape[1]), dtype=np.float64)
        counts = np.zeros(len(unique), dtype=np.int32)
        np.add.at(sums, inverse, values)
        np.add.at(counts, inverse, 1)
        averaged = sums / counts[:, None]
        if name == "presence":
            averaged = (averaged > 0).astype(np.float32)
        result[name] = averaged.astype(np.float32)
    result["removed_train_overlap_rows"] = int(np.count_nonzero(~keep))
    result["consolidated_duplicate_rows"] = int(len(identities) - len(unique))
    return result


def class_point_estimates(
    matrices: Mapping[str, np.ndarray], y: np.ndarray
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    by_class: dict[str, dict[str, np.ndarray]] = {}
    present_counts: dict[str, np.ndarray] = {}
    for class_name, label in (("machine", 0), ("human", 1)):
        mask = y == label
        if not np.any(mask):
            raise ValueError(f"missing {class_name} rows")
        presence = matrices["presence"][mask]
        present = presence.sum(axis=0)
        present_counts[class_name] = present
        estimates = {
            "presence": presence.mean(axis=0),
            "frequency": matrices["frequency"][mask].mean(axis=0),
            "mfidf": matrices["mfidf"][mask].mean(axis=0),
            "depth_contribution": matrices["depth_contribution"][mask].mean(axis=0),
        }
        depth_sum = matrices["conditional_depth"][mask].sum(axis=0)
        estimates["conditional_depth"] = np.divide(
            depth_sum,
            present,
            out=np.full(depth_sum.shape, np.nan, dtype=np.float64),
            where=present > 0,
        )
        by_class[class_name] = estimates
    return by_class, present_counts


def clustered_bootstrap_intervals(
    matrices: Mapping[str, np.ndarray],
    y: np.ndarray,
    identities: np.ndarray,
    *,
    iterations: int,
    confidence: float,
    batch_size: int,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Identity-clustered, label-stratified intervals for machine−human means."""
    motif_count = matrices["presence"].shape[1]
    rng = np.random.default_rng(seed)
    distributions = {
        metric: np.empty((iterations, motif_count), dtype=np.float32)
        for metric in METRICS
    }
    class_distributions: dict[int, dict[str, np.ndarray]] = {}

    for label in (0, 1):
        label_rows = np.flatnonzero(y == label)
        label_ids = identities[label_rows]
        unique, inverse = np.unique(label_ids, return_inverse=True)
        cluster_sizes = np.zeros(len(unique), dtype=np.float32)
        np.add.at(cluster_sizes, inverse, 1.0)

        combined = np.concatenate(
            [
                matrices["presence"][label_rows],
                matrices["frequency"][label_rows],
                matrices["mfidf"][label_rows],
                matrices["depth_contribution"][label_rows],
                matrices["conditional_depth"][label_rows],
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        cluster_sums = np.zeros((len(unique), combined.shape[1]), dtype=np.float32)
        np.add.at(cluster_sums, inverse, combined)
        class_result = {
            metric: np.empty((iterations, motif_count), dtype=np.float32)
            for metric in METRICS
        }
        probabilities = np.full(len(unique), 1.0 / len(unique))
        for start in range(0, iterations, batch_size):
            stop = min(start + batch_size, iterations)
            weights = rng.multinomial(len(unique), probabilities, size=stop - start).astype(np.float32)
            sums = weights @ cluster_sums
            row_denominator = weights @ cluster_sizes
            blocks = sums.reshape(stop - start, 5, motif_count)
            for block_index, metric in enumerate(("presence", "frequency", "mfidf", "depth_contribution")):
                class_result[metric][start:stop] = blocks[:, block_index, :] / row_denominator[:, None]
            class_result["conditional_depth"][start:stop] = np.divide(
                blocks[:, 4, :],
                blocks[:, 0, :],
                out=np.full((stop - start, motif_count), np.nan, dtype=np.float32),
                where=blocks[:, 0, :] > 0,
            )
        class_distributions[label] = class_result

    alpha = 1.0 - confidence
    intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for metric in METRICS:
        distributions[metric][:] = class_distributions[0][metric] - class_distributions[1][metric]
        samples = distributions[metric]
        valid_columns = np.isfinite(samples).any(axis=0)
        low = np.full(samples.shape[1], np.nan, dtype=np.float64)
        high = np.full(samples.shape[1], np.nan, dtype=np.float64)
        if valid_columns.any():
            valid_low, valid_high = np.nanquantile(
                samples[:, valid_columns],
                [alpha / 2.0, 1.0 - alpha / 2.0],
                axis=0,
            )
            low[valid_columns] = valid_low
            high[valid_columns] = valid_high
        intervals[metric] = (low, high)
    return intervals


def graph_description(graph: Mapping[str, Any]) -> tuple[str, str]:
    nodes = sorted(str(node.get("label_0", "?")) for node in graph.get("nodes", ()))
    relations = sorted(
        str(edge.get("label_0", "?"))
        for edge in graph.get("links", graph.get("edges", ()))
        if edge.get("label_0") != "/"
    )
    return "|".join(relations) if relations else "(structural only)", "|".join(nodes)


def load_motif_metadata(
    catalog_path: Path,
    hashes: np.ndarray,
    mapping: Mapping[tuple[str, str], Mapping[str, str]],
    inventory: str,
) -> list[dict[str, Any]]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    metadata = []
    for index, motif_hash in enumerate(hashes):
        if motif_hash not in catalog:
            raise ValueError(f"{inventory}: matrix hash absent from catalog: {motif_hash}")
        relation, nodes = graph_description(catalog[motif_hash])
        mapped = mapping.get((inventory, str(motif_hash)), {})
        metadata.append(
            {
                "motif_index": index,
                "motif_hash": str(motif_hash),
                "relation_signature": relation,
                "node_signature": nodes,
                "pipeline_family": mapped.get("pipeline_family", ""),
                "exact_family": mapped.get("exact_family", ""),
                "structural_family": mapped.get("structural_family", ""),
            }
        )
    return metadata


def select_training_motifs(
    inventory: str,
    train: Mapping[str, np.ndarray],
    idf: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    per_direction: int,
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    matrices = metric_matrices(train, idf)
    for domain in DOMAINS:
        mask = train["domain"] == domain
        estimates, _ = class_point_estimates({key: value[mask] for key, value in matrices.items()}, train["y"][mask])
        delta = estimates["machine"]["mfidf"] - estimates["human"]["mfidf"]
        positives = [index for index in np.argsort(-delta) if delta[index] > 0][:per_direction]
        negatives = [index for index in np.argsort(delta) if delta[index] < 0][:per_direction]
        for direction, indices in (("machine", positives), ("human", negatives)):
            for rank, index in enumerate(indices, 1):
                motif = metadata[index]
                selection = {
                    "inventory": inventory,
                    "domain": domain,
                    "direction_from_training": direction,
                    "direction_rank": rank,
                    "train_mfidf_difference": float(delta[index]),
                    **motif,
                }
                selected[(domain, motif["motif_hash"])] = selection
                rows.append(selection)
    return selected, rows


def summarize_cohort(
    inventory: str,
    cohort: str,
    data: Mapping[str, np.ndarray],
    idf: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    selected: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    bootstrap: bool,
    iterations: int,
    confidence: float,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    matrices = metric_matrices(data, idf)
    rows: list[dict[str, Any]] = []
    domains = (*DOMAINS, "all")
    for domain in tqdm(domains, desc=f"{inventory} {cohort}", unit="domain", leave=False):
        mask = np.ones(len(data["y"]), dtype=bool) if domain == "all" else data["domain"] == domain
        subset = {name: values[mask] for name, values in matrices.items()}
        subset_y = data["y"][mask]
        estimates, present_counts = class_point_estimates(subset, subset_y)
        intervals = None
        if bootstrap and domain != "all":
            intervals = clustered_bootstrap_intervals(
                subset,
                subset_y,
                data["identity_hash"][mask],
                iterations=iterations,
                confidence=confidence,
                batch_size=batch_size,
                seed=stable_seed(seed, cohort, domain),
            )
        machine_n = int(np.count_nonzero(subset_y == 0))
        human_n = int(np.count_nonzero(subset_y == 1))
        for metric in METRICS:
            difference = estimates["machine"][metric] - estimates["human"][metric]
            if intervals is None:
                lows = np.full(len(metadata), np.nan)
                highs = np.full(len(metadata), np.nan)
            else:
                lows, highs = intervals[metric]
            for index, motif in enumerate(metadata):
                selection = selected.get((domain, motif["motif_hash"]))
                rows.append(
                    {
                        "inventory": inventory,
                        "cohort": cohort,
                        "domain": domain,
                        **motif,
                        "metric": metric,
                        "machine_mean": finite(estimates["machine"][metric][index]),
                        "human_mean": finite(estimates["human"][metric][index]),
                        "difference_machine_minus_human": finite(difference[index]),
                        "ci_low": finite(lows[index]),
                        "ci_high": finite(highs[index]),
                        "machine_documents": machine_n,
                        "human_documents": human_n,
                        "machine_present": int(present_counts["machine"][index]),
                        "human_present": int(present_counts["human"][index]),
                        "selected_from_training": bool(selection),
                        "training_direction": selection["direction_from_training"] if selection else "",
                        "training_direction_rank": selection["direction_rank"] if selection else "",
                    }
                )
    return rows


def save_figure_atomic(figure: Any, path: Path, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.png")
    try:
        figure.savefig(temporary, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def plot_top_motifs(
    inventory: str,
    rows: Sequence[Mapping[str, Any]],
    selections: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    lookup = {
        (row["domain"], row["motif_hash"]): row
        for row in rows
        if row["cohort"] == "test" and row["metric"] == "mfidf"
    }
    figure, axes = plt.subplots(len(DOMAINS), 1, figsize=(15, 16), sharex=False)
    for axis, domain in zip(axes, DOMAINS):
        domain_selection = sorted(
            (row for row in selections if row["domain"] == domain),
            key=lambda row: (0 if row["direction_from_training"] == "machine" else 1, row["direction_rank"]),
        )
        values = [lookup[(domain, row["motif_hash"])] for row in domain_selection]
        differences = np.asarray([float(row["difference_machine_minus_human"]) for row in values])
        low = np.asarray([float(row["ci_low"]) for row in values])
        high = np.asarray([float(row["ci_high"]) for row in values])
        positions = np.arange(len(values))
        colours = np.where(differences >= 0, "#ef766b", "#2e8b57")
        axis.bar(positions, differences * 1000.0, color=colours, alpha=0.94)
        axis.errorbar(
            positions,
            differences * 1000.0,
            yerr=np.vstack(((differences - low) * 1000.0, (high - differences) * 1000.0)),
            fmt="none",
            ecolor="#333333",
            elinewidth=0.8,
            capsize=2,
        )
        labels = [f"{row['motif_index']}:{row['relation_signature']}" for row in domain_selection]
        axis.set_xticks(positions, labels, rotation=50, ha="right", fontsize=8)
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.set_title(domain)
        axis.set_ylabel("Δ MF-IDF × 10³")
    figure.suptitle(
        f"{inventory}: test MF-IDF differences for training-selected M3 motifs\n"
        "positive = more machine; negative = more human",
        fontsize=14,
        y=1.005,
    )
    figure.tight_layout()
    save_figure_atomic(figure, output)


def plot_full_catalog(
    inventory: str,
    rows: Sequence[Mapping[str, Any]],
    motif_count: int,
    output: Path,
) -> None:
    lookup = {
        (row["domain"], int(row["motif_index"])): float(row["difference_machine_minus_human"])
        for row in rows
        if row["cohort"] == "test" and row["metric"] == "mfidf" and row["domain"] != "all"
    }
    figure, axes = plt.subplots(len(DOMAINS), 1, figsize=(18, 13), sharex=True)
    positions = np.arange(motif_count)
    for axis, domain in zip(axes, DOMAINS):
        values = np.asarray([lookup[(domain, index)] for index in range(motif_count)]) * 1000.0
        colours = np.where(values >= 0, "#ef766b", "#2e8b57")
        axis.bar(positions, values, width=0.9, color=colours)
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.set_title(domain)
        axis.set_ylabel("Δ × 10³")
    axes[-1].set_xlabel("inventory-specific M3 catalog index")
    axes[-1].set_xticks(positions[:: max(1, motif_count // 24)])
    figure.suptitle(
        f"{inventory}: full test M3 MF-IDF difference distribution\n"
        "positive = more machine; negative = more human",
        fontsize=14,
        y=1.005,
    )
    figure.tight_layout()
    save_figure_atomic(figure, output)


def load_mapping(path: Path) -> tuple[dict[tuple[str, str], dict[str, str]], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["inventory"], row["motif_hash"]): row for row in rows}, rows


def cross_standard_outputs(
    result_rows: Sequence[Mapping[str, Any]],
    mapping_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    family_by_member = {
        (row["inventory"], row["motif_hash"]): row["pipeline_family"]
        for row in mapping_rows
    }
    relation_by_member = {
        (row["inventory"], row["motif_hash"]): row["description"]
        for row in mapping_rows
    }
    test = [
        row for row in result_rows
        if row["cohort"] == "test" and row["metric"] in {"frequency", "mfidf"}
    ]
    values: dict[tuple[str, str, str, str], float] = {}
    for row in test:
        values[(row["inventory"], row["domain"], row["motif_hash"], row["metric"])] = float(
            row["difference_machine_minus_human"]
        )
    family_rows: list[dict[str, Any]] = []
    members_by_family: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for member, family in family_by_member.items():
        members_by_family[family].append(member)
    for family, members in sorted(members_by_family.items()):
        if len({inventory for inventory, _ in members}) < 2:
            continue
        for domain in (*DOMAINS, "all"):
            for inventory, motif_hash in sorted(members):
                frequency = values.get((inventory, domain, motif_hash, "frequency"))
                mfidf = values.get((inventory, domain, motif_hash, "mfidf"))
                if frequency is None or mfidf is None:
                    continue
                family_rows.append(
                    {
                        "pipeline_family": family,
                        "domain": domain,
                        "inventory": inventory,
                        "motif_hash": motif_hash,
                        "frequency_difference_machine_minus_human": frequency,
                        "mfidf_difference_machine_minus_human": mfidf,
                        "description": relation_by_member[(inventory, motif_hash)],
                    }
                )

    lookup: dict[tuple[str, str, str], Mapping[str, Any]] = {
        (row["pipeline_family"], row["domain"], row["inventory"]): row
        for row in family_rows
    }
    pair_rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(INVENTORIES):
        for right in INVENTORIES[left_index + 1 :]:
            for domain in (*DOMAINS, "all"):
                families = sorted(
                    family for family in members_by_family
                    if (family, domain, left) in lookup and (family, domain, right) in lookup
                )
                left_values = np.asarray([
                    lookup[(family, domain, left)]["frequency_difference_machine_minus_human"]
                    for family in families
                ], dtype=float)
                right_values = np.asarray([
                    lookup[(family, domain, right)]["frequency_difference_machine_minus_human"]
                    for family in families
                ], dtype=float)
                correlation = spearmanr(left_values, right_values).statistic if len(families) >= 3 else np.nan
                nonzero = (left_values != 0) | (right_values != 0)
                sign_agreement = np.mean(np.sign(left_values[nonzero]) == np.sign(right_values[nonzero])) if np.any(nonzero) else np.nan
                pair_rows.append(
                    {
                        "left_inventory": left,
                        "right_inventory": right,
                        "domain": domain,
                        "shared_pipeline_families": len(families),
                        "spearman_frequency_difference": finite(correlation),
                        "sign_agreement": finite(sign_agreement),
                    }
                )
    return family_rows, pair_rows


def plot_cross_standard_families(family_rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    overall = [row for row in family_rows if row["domain"] == "all"]
    by_family: dict[str, dict[str, float]] = defaultdict(dict)
    descriptions: dict[str, str] = {}
    for row in overall:
        by_family[row["pipeline_family"]][row["inventory"]] = float(
            row["frequency_difference_machine_minus_human"]
        )
        descriptions[row["pipeline_family"]] = str(row["description"]).split("; relations=")[-1]
    ordered = sorted(
        by_family,
        key=lambda family: -max(abs(value) for value in by_family[family].values()),
    )[:24]
    matrix = np.full((len(ordered), len(INVENTORIES)), np.nan)
    for row_index, family in enumerate(ordered):
        for column, inventory in enumerate(INVENTORIES):
            if inventory in by_family[family]:
                matrix[row_index, column] = by_family[family][inventory]
    limit = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
    figure, axis = plt.subplots(figsize=(10, max(8, len(ordered) * 0.42)))
    image = axis.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=-limit, vmax=limit)
    axis.set_xticks(range(len(INVENTORIES)), INVENTORIES, rotation=25, ha="right")
    axis.set_yticks(
        range(len(ordered)),
        [f"{family} · {descriptions[family][:48]}" for family in ordered],
        fontsize=8,
    )
    axis.set_title("Shared M3 families: overall test frequency difference\npositive = more machine; negative = more human")
    figure.colorbar(image, ax=axis, label="machine − human normalized frequency")
    figure.tight_layout()
    save_figure_atomic(figure, output)


def sensitivity_rows(result_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["inventory"], row["domain"], row["motif_hash"], row["metric"], row["cohort"]): row
        for row in result_rows
        if row["cohort"] in {"test", "test_clean"} and row["domain"] != "all"
    }
    summaries = []
    for inventory in INVENTORIES:
        for metric in METRICS:
            primary, clean = [], []
            for domain in DOMAINS:
                hashes = sorted({
                    key[2] for key in lookup
                    if key[0] == inventory and key[1] == domain and key[3] == metric
                })
                for motif_hash in hashes:
                    left = lookup[(inventory, domain, motif_hash, metric, "test")]["difference_machine_minus_human"]
                    right = lookup[(inventory, domain, motif_hash, metric, "test_clean")]["difference_machine_minus_human"]
                    if left is not None and right is not None:
                        primary.append(float(left))
                        clean.append(float(right))
            primary_values = np.asarray(primary)
            clean_values = np.asarray(clean)
            correlation = spearmanr(primary_values, clean_values).statistic if len(primary_values) >= 3 else np.nan
            summaries.append(
                {
                    "inventory": inventory,
                    "metric": metric,
                    "comparisons": len(primary_values),
                    "spearman_primary_vs_clean": finite(correlation),
                    "maximum_absolute_change": finite(np.max(np.abs(primary_values - clean_values))) if len(primary_values) else None,
                    "sign_agreement": finite(np.mean(np.sign(primary_values) == np.sign(clean_values))) if len(primary_values) else None,
                }
            )
    return summaries


def render_report(
    results: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    sensitivity: Sequence[Mapping[str, Any]],
    clean_counts: Mapping[str, Mapping[str, int]],
    motif_counts: Mapping[str, int],
    iterations: int,
    confidence: float,
) -> str:
    test_mfidf = [
        row for row in results
        if row["cohort"] == "test"
        and row["metric"] == "mfidf"
        and row["domain"] in DOMAINS
        and row["selected_from_training"]
    ]
    lines = [
        "# UniRST2 paper-style motif-distribution comparison",
        "",
        "## Design",
        "",
        (
            "This exploratory analysis follows the signed comparison in *Threads of Subtlety*: "
            "all differences are mean machine minus mean human. Positive values therefore indicate "
            "greater motif prevalence in machine-generated documents. IDF is fitted separately for "
            "each inventory using the released HC3 training split only."
        ),
        "",
        f"The primary cohort is the released paper split. Test uncertainty uses {iterations:,} "
        f"label-stratified identity-cluster bootstrap samples and nominal {confidence * 100:g}% percentile intervals. "
        "The clean sensitivity removes identities seen in training and consolidates duplicate test identities.",
        "",
        "| Inventory | M3 motifs | Clean test identities | Removed overlap rows | Consolidated duplicate rows |",
        "|---|---:|---:|---:|---:|",
    ]
    for inventory in INVENTORIES:
        counts = clean_counts[inventory]
        lines.append(
            f"| {inventory} | {motif_counts[inventory]} | {counts['rows']} | "
            f"{counts['removed_train_overlap_rows']} | {counts['consolidated_duplicate_rows']} |"
        )
    lines.extend(
        [
            "",
            "## Strongest test MF-IDF differences among training-selected motifs",
            "",
            "The candidate set and direction-specific ranks were fixed on training before test values were read.",
            "",
        ]
    )
    for inventory in INVENTORIES:
        lines.extend([f"### {inventory}", "", "| Domain | More machine-associated | More human-associated |", "|---|---|---|"])
        for domain in DOMAINS:
            candidates = [row for row in test_mfidf if row["inventory"] == inventory and row["domain"] == domain]
            positive = max(candidates, key=lambda row: float(row["difference_machine_minus_human"]))
            negative = min(candidates, key=lambda row: float(row["difference_machine_minus_human"]))
            lines.append(
                f"| {domain} | {positive['motif_index']} `{positive['relation_signature']}` "
                f"({float(positive['difference_machine_minus_human']) * 1000:+.3f}×10⁻³) | "
                f"{negative['motif_index']} `{negative['relation_signature']}` "
                f"({float(negative['difference_machine_minus_human']) * 1000:+.3f}×10⁻³) |"
            )
        lines.extend(
            [
                "",
                f"![{inventory} training-selected MF-IDF differences]({inventory}/paper_style_mfidf_top.png)",
                "",
                f"[Full-catalog plot]({inventory}/full_catalog_mfidf.png)",
                "",
            ]
        )
    lines.extend(["## Cross-standard comparison", ""])
    for left_index, left in enumerate(INVENTORIES):
        for right in INVENTORIES[left_index + 1 :]:
            rows = [
                row for row in pair_rows
                if row["left_inventory"] == left and row["right_inventory"] == right and row["domain"] in DOMAINS
            ]
            sign_values = [float(row["sign_agreement"]) for row in rows if row["sign_agreement"] is not None]
            rho_values = [float(row["spearman_frequency_difference"]) for row in rows if row["spearman_frequency_difference"] is not None]
            mean_sign = float(np.mean(sign_values)) if sign_values else float("nan")
            mean_rho = float(np.mean(rho_values)) if rho_values else float("nan")
            sign_text = f"{mean_sign:.3f}" if np.isfinite(mean_sign) else "not estimable"
            rho_text = f"{mean_rho:.3f}" if np.isfinite(mean_rho) else "not estimable"
            lines.append(f"- {left} / {right}: mean sign agreement {sign_text}; mean Spearman ρ {rho_text}.")
    lines.extend(
        [
            "",
            "![Cross-standard shared-family effects](cross_standard_family_effects.png)",
            "",
            "Raw hashes and inventory-specific catalog indices are not treated as interchangeable. The cross-standard "
            "summary is restricted to the existing verified pipeline-equivalent graph families; differing IDF scales "
            "are why the shared-family synthesis emphasizes frequency differences, signs, and ranks.",
            "",
            "## Sensitivity to split overlap and duplicates",
            "",
            "| Inventory | Metric | Rank correlation | Sign agreement | Maximum absolute change |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in sensitivity:
        lines.append(
            f"| {row['inventory']} | {row['metric']} | {row['spearman_primary_vs_clean']:.3f} | "
            f"{row['sign_agreement']:.3f} | {row['maximum_absolute_change']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "The HC3 domains differ from the five DeepfakeTextDetect domains shown in the original paper. This is a "
            "methodological analogue, not a numerical reproduction. Motif frequencies are compositional, parser outputs "
            "are not gold annotations, and the current test outcomes were inspected in earlier analyses. The results "
            "should therefore be used to identify cross-inventory patterns and review candidates, not as causal evidence "
            "about human writing or as a new confirmatory benchmark.",
            "",
            f"Original paper: {PAPER_URL}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=Path("artifacts/canonical/feature_matrices"))
    parser.add_argument("--motif-root", type=Path, default=Path("data/motifs2"))
    parser.add_argument("--mapping-members", type=Path, default=Path("artifacts/canonical/cross_standard_mapping/members.csv"))
    parser.add_argument("--cohort-manifest", type=Path, default=Path("artifacts/canonical/feature_matrices/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/canonical/motif_distributions"))
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-batch-size", type=int, default=200)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--top-per-direction", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20240910)
    args = parser.parse_args()
    if args.bootstrap_iterations < 20:
        raise ValueError("--bootstrap-iterations must be at least 20")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping, mapping_rows = load_mapping(args.mapping_members)
    cohort_manifest = json.loads(args.cohort_manifest.read_text(encoding="utf-8"))
    if tuple(cohort_manifest.get("inventories", ())) != INVENTORIES:
        raise ValueError("cohort manifest inventories do not match the requested inventories")
    catalog_audit = {
        row["inventory"]: row for row in cohort_manifest.get("catalogs", ())
    }
    matrix_audit = {
        (row["inventory"], row["split"]): row
        for row in cohort_manifest.get("matrices", ())
    }
    alignment_reference: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    all_rows: list[dict[str, Any]] = []
    all_selections: list[dict[str, Any]] = []
    motif_counts: dict[str, int] = {}
    clean_counts: dict[str, dict[str, int]] = {}
    input_hashes: dict[str, str] = {
        str(args.mapping_members): sha256_file(args.mapping_members),
        str(args.cohort_manifest): sha256_file(args.cohort_manifest),
    }

    for inventory in tqdm(INVENTORIES, desc="inventories", unit="inventory"):
        matrices = {
            split: load_matrix(args.feature_root / inventory / f"hc3_{split}.features.npz")
            for split in SPLITS
        }
        for split in SPLITS:
            audit = matrix_audit.get((inventory, split))
            if not audit or not all(
                audit.get(field) is True
                for field in ("alignment_verified", "artifact_hash_verified", "source_hash_verified")
            ):
                raise ValueError(f"{inventory} {split}: cohort audit is missing or unverified")
            if int(audit.get("records", -1)) != len(matrices[split]["y"]):
                raise ValueError(f"{inventory} {split}: record count differs from cohort audit")
            aligned = (
                matrices[split]["identity_hash"],
                matrices[split]["domain"],
                matrices[split]["y"],
            )
            if split not in alignment_reference:
                alignment_reference[split] = tuple(values.copy() for values in aligned)
            elif any(
                not np.array_equal(reference, observed)
                for reference, observed in zip(alignment_reference[split], aligned)
            ):
                raise ValueError(f"{inventory} {split}: identities, domains, or labels are not aligned")
        hashes = matrices["train"]["hashes"]
        if any(not np.array_equal(hashes, matrices[split]["hashes"]) for split in SPLITS):
            raise ValueError(f"{inventory}: motif hash order differs across splits")
        motif_counts[inventory] = len(hashes)
        for split in SPLITS:
            path = args.feature_root / inventory / f"hc3_{split}.features.npz"
            input_hashes[str(path)] = sha256_file(path)
        catalog_path = args.motif_root / inventory / "hc3_M3_motifs.json"
        provenance_path = args.motif_root / inventory / "hc3_M3_provenance.json"
        catalog_hash = sha256_file(catalog_path)
        provenance_hash = sha256_file(provenance_path)
        input_hashes[str(catalog_path)] = catalog_hash
        input_hashes[str(provenance_path)] = provenance_hash
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        expected_catalog = catalog_audit.get(inventory)
        if not expected_catalog or (
            expected_catalog.get("catalog_sha256") != catalog_hash
            or expected_catalog.get("motif_provenance_sha256") != provenance_hash
            or int(expected_catalog.get("motif_count", -1)) != len(hashes)
            or expected_catalog.get("source_files_verified") != 1
        ):
            raise ValueError(f"{inventory}: catalog differs from the verified cohort audit")
        if (
            provenance.get("dataset_name") != "hc3"
            or provenance.get("split") != "train"
            or provenance.get("relinventory") != inventory
            or int(provenance.get("document_count", -1)) < len(matrices["train"]["y"])
            or int(provenance.get("motif_count", -1)) != len(hashes)
            or provenance.get("catalog", {}).get("sha256") != catalog_hash
        ):
            raise ValueError(f"{inventory}: motif catalog provenance is inconsistent")

        metadata = load_motif_metadata(catalog_path, hashes, mapping, inventory)
        train_presence = matrices["train"]["presence"]
        idf = np.log((1.0 + len(train_presence)) / (1.0 + train_presence.sum(axis=0))).astype(np.float32)
        selected, selection_rows = select_training_motifs(
            inventory,
            matrices["train"],
            idf,
            metadata,
            args.top_per_direction,
        )
        all_selections.extend(selection_rows)
        train_ids = set(matrices["train"]["identity_hash"].tolist())
        clean = collapse_clean_test(matrices["test"], train_ids)
        clean_counts[inventory] = {
            "rows": len(clean["y"]),
            "removed_train_overlap_rows": int(clean["removed_train_overlap_rows"]),
            "consolidated_duplicate_rows": int(clean["consolidated_duplicate_rows"]),
        }

        inventory_rows: list[dict[str, Any]] = []
        for split in ("train", "validation"):
            inventory_rows.extend(
                summarize_cohort(
                    inventory, split, matrices[split], idf, metadata, selected,
                    bootstrap=False,
                    iterations=args.bootstrap_iterations,
                    confidence=args.confidence,
                    batch_size=args.bootstrap_batch_size,
                    seed=args.seed,
                )
            )
        inventory_rows.extend(
            summarize_cohort(
                inventory, "test", matrices["test"], idf, metadata, selected,
                bootstrap=True,
                iterations=args.bootstrap_iterations,
                confidence=args.confidence,
                batch_size=args.bootstrap_batch_size,
                seed=args.seed,
            )
        )
        inventory_rows.extend(
            summarize_cohort(
                inventory, "test_clean", clean, idf, metadata, selected,
                bootstrap=True,
                iterations=args.bootstrap_iterations,
                confidence=args.confidence,
                batch_size=args.bootstrap_batch_size,
                seed=args.seed,
            )
        )
        inventory_dir = args.output_dir / inventory
        atomic_text(inventory_dir / "domain_differences.csv", csv_text(inventory_rows))
        atomic_text(inventory_dir / "training_selected_motifs.csv", csv_text(selection_rows))
        atomic_json(
            inventory_dir / "idf.json",
            {
                "definition": "log((1 + N_train) / (1 + train_document_frequency))",
                "training_documents": len(train_presence),
                "motifs": [
                    {**metadata[index], "document_frequency": int(train_presence[:, index].sum()), "idf": float(idf[index])}
                    for index in range(len(metadata))
                ],
            },
        )
        plot_top_motifs(inventory, inventory_rows, selection_rows, inventory_dir / "paper_style_mfidf_top.png")
        plot_full_catalog(inventory, inventory_rows, len(metadata), inventory_dir / "full_catalog_mfidf.png")
        all_rows.extend(inventory_rows)

    family_rows, pair_rows = cross_standard_outputs(all_rows, mapping_rows)
    sensitivity = sensitivity_rows(all_rows)
    atomic_text(args.output_dir / "all_domain_differences.csv", csv_text(all_rows))
    atomic_text(args.output_dir / "training_selected_motifs.csv", csv_text(all_selections))
    atomic_text(args.output_dir / "cross_standard_families.csv", csv_text(family_rows))
    atomic_text(args.output_dir / "cross_standard_pairwise.csv", csv_text(pair_rows))
    atomic_text(args.output_dir / "clean_cohort_sensitivity.csv", csv_text(sensitivity))
    plot_cross_standard_families(family_rows, args.output_dir / "cross_standard_family_effects.png")
    atomic_text(
        args.output_dir / "report.md",
        render_report(
            all_rows,
            pair_rows,
            sensitivity,
            clean_counts,
            motif_counts,
            args.bootstrap_iterations,
            args.confidence,
        ),
    )

    output_paths = sorted(path for path in args.output_dir.rglob("*") if path.is_file() and path.name != "manifest.json")
    manifest = {
        "stage": "unirst2_paper_style_motif_distribution_comparison",
        "version": SCRIPT_VERSION,
        "paper": PAPER_URL,
        "inventories": list(INVENTORIES),
        "domains": list(DOMAINS),
        "motif_counts": motif_counts,
        "cohorts": {
            "primary": "released HC3 paper split after common parser-complete filtering",
            "sensitivity": "test identities absent from train; duplicate test identities consolidated",
            "clean_counts": clean_counts,
        },
        "parameters": {
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_batch_size": args.bootstrap_batch_size,
            "confidence": args.confidence,
            "top_per_direction": args.top_per_direction,
            "seed": args.seed,
        },
        "definitions": {
            "label_mapping": {"0": "machine", "1": "human"},
            "difference": "mean(machine) - mean(human)",
            "frequency": "document M3 count / all document M3 counts",
            "idf": "log((1 + N_train) / (1 + train document frequency))",
            "mfidf": "frequency * training-fitted IDF",
            "conditional_depth": "mean normalized motif depth among documents where motif is present",
            "depth_contribution": "document frequency * conditional document depth",
            "display_selection": "within each inventory/domain, five largest positive and five largest negative training MF-IDF differences",
            "uncertainty": "label-stratified identity-cluster bootstrap; nominal percentile intervals",
        },
        "input_hashes": input_hashes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "output_hashes": {
            str(path.relative_to(args.output_dir)): sha256_file(path)
            for path in output_paths
        },
        "interpretation": "exploratory; HC3 domains differ from the paper's DeepfakeTextDetect domains and test outcomes were previously inspected",
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "result_rows": len(all_rows),
        "selected_rows": len(all_selections),
        "shared_family_rows": len(family_rows),
        "pairwise_rows": len(pair_rows),
        "clean_counts": clean_counts,
    }, indent=2))


if __name__ == "__main__":
    main()
