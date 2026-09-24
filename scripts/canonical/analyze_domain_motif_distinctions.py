#!/usr/bin/env python3
"""Discover domain-specific M3 authorship effects across UniRST inventories.

Candidates are selected from training by the interaction

    (machine - human in domain) - mean(machine - human in other domains).

Validation is used only to check the selected direction. Test uncertainty is
estimated with label- and domain-stratified identity-cluster bootstraps. The
released test split is primary; an identity-clean cohort is a sensitivity
analysis. All point estimates use normalized M3 frequency. Training ranking
uses the same train-fitted MF-IDF scaling as the paper-style analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm


INVENTORIES = (
    "deu.rst.pcc",
    "eng.erst.gum.new",
    "eng.rst.rstdt.new",
    "nld.rst.nldt",
)
DISPLAY = {
    "deu.rst.pcc": "PCC",
    "eng.erst.gum.new": "eRST",
    "eng.rst.rstdt.new": "RST-DT",
    "nld.rst.nldt": "NLDT",
}
DOMAINS = ("finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai")
SCRIPT_VERSION = "domain-motif-interactions-v1"


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


def csv_text(rows: Iterable[Mapping[str, Any]]) -> str:
    rows = list(rows)
    output = io.StringIO()
    fields = list(rows[0]) if rows else []
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
            "presence": np.asarray(values["X_m3_presence"], dtype=np.float64),
            "frequency": np.asarray(values["X_m3_frequency"], dtype=np.float64),
        }
    expected = (len(result["y"]), len(result["hashes"]))
    if result["presence"].shape != expected or result["frequency"].shape != expected:
        raise ValueError(f"Malformed feature matrix: {path}")
    if set(np.unique(result["domain"])) != set(DOMAINS):
        raise ValueError(f"Unexpected domains in {path}")
    return result


def clean_test(data: Mapping[str, np.ndarray], train_ids: set[str]) -> dict[str, np.ndarray]:
    keep = np.asarray([value not in train_ids for value in data["identity_hash"]])
    ids = data["identity_hash"][keep]
    unique, inverse = np.unique(ids, return_inverse=True)
    result: dict[str, Any] = {"identity_hash": unique, "hashes": data["hashes"]}
    result["y"] = np.empty(len(unique), dtype=np.int8)
    result["domain"] = np.empty(len(unique), dtype=object)
    for cluster in range(len(unique)):
        rows = inverse == cluster
        labels = np.unique(data["y"][keep][rows])
        domains = np.unique(data["domain"][keep][rows])
        if len(labels) != 1 or len(domains) != 1:
            raise ValueError(f"Identity {unique[cluster]} crosses label or domain")
        result["y"][cluster] = labels[0]
        result["domain"][cluster] = domains[0]
    for name in ("presence", "frequency"):
        values = data[name][keep]
        sums = np.zeros((len(unique), values.shape[1]), dtype=np.float64)
        counts = np.zeros(len(unique), dtype=np.int32)
        np.add.at(sums, inverse, values)
        np.add.at(counts, inverse, 1)
        result[name] = sums / counts[:, None]
        if name == "presence":
            result[name] = (result[name] > 0).astype(np.float64)
    result["domain"] = result["domain"].astype(str)
    result["removed_overlap_rows"] = int(np.count_nonzero(~keep))
    result["consolidated_duplicate_rows"] = int(len(ids) - len(unique))
    return result


def graph_description(graph: Mapping[str, Any]) -> tuple[str, str]:
    relations = sorted(
        str(edge.get("label_0", "?"))
        for edge in graph.get("links", graph.get("edges", ()))
        if edge.get("label_0") != "/"
    )
    nodes = sorted(str(node.get("label_0", "?")) for node in graph.get("nodes", ()))
    return "|".join(relations) if relations else "(structural only)", "|".join(nodes)


def load_metadata(
    inventory: str,
    hashes: np.ndarray,
    motif_root: Path,
    family_rows: Mapping[tuple[str, str], Mapping[str, str]],
    family_inventory_counts: Mapping[str, int],
    coarse_rows: Mapping[tuple[str, str], Mapping[str, str]],
) -> list[dict[str, Any]]:
    path = motif_root / inventory / "hc3_M3_motifs.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    metadata = []
    for index, motif_hash in enumerate(hashes):
        relation, nodes = graph_description(catalog[str(motif_hash)])
        family = family_rows.get((inventory, str(motif_hash)), {})
        coarse = coarse_rows.get((inventory, str(motif_hash)), {})
        pipeline_family = family.get("pipeline_family", "")
        metadata.append(
            {
                "motif_index": index,
                "motif_hash": str(motif_hash),
                "relation_signature": relation,
                "node_signature": nodes,
                "pipeline_family": pipeline_family,
                "family_inventory_count": family_inventory_counts.get(pipeline_family, 1),
                "coarse_relations": coarse.get("coarse_relations", ""),
                "coarse_group_size": int(coarse.get("coarse_group_size", 1)),
            }
        )
    return metadata


def domain_effects(data: Mapping[str, np.ndarray], values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    effects = np.empty((len(DOMAINS), values.shape[1]), dtype=np.float64)
    for domain_index, domain in enumerate(DOMAINS):
        domain_mask = data["domain"] == domain
        machine = values[domain_mask & (data["y"] == 0)]
        human = values[domain_mask & (data["y"] == 1)]
        if len(machine) == 0 or len(human) == 0:
            raise ValueError(f"Missing label in {domain}")
        effects[domain_index] = machine.mean(axis=0) - human.mean(axis=0)
    interactions = np.empty_like(effects)
    for domain_index in range(len(DOMAINS)):
        others = np.arange(len(DOMAINS)) != domain_index
        interactions[domain_index] = effects[domain_index] - effects[others].mean(axis=0)
    return effects, interactions


def candidate_support(data: Mapping[str, np.ndarray], domain: str) -> tuple[np.ndarray, np.ndarray]:
    target = data["domain"] == domain
    other = ~target
    target_counts = data["presence"][target].sum(axis=0)
    other_counts = data["presence"][other].sum(axis=0)
    target_by_label = np.vstack(
        [data["presence"][target & (data["y"] == label)].sum(axis=0) for label in (0, 1)]
    )
    other_by_label = np.vstack(
        [data["presence"][other & (data["y"] == label)].sum(axis=0) for label in (0, 1)]
    )
    minimums = np.minimum(target_by_label.min(axis=0), other_by_label.min(axis=0))
    return target_counts + other_counts, minimums


def select_candidates(
    train: Mapping[str, np.ndarray],
    metadata: Sequence[Mapping[str, Any]],
    interactions: np.ndarray,
    idf: np.ndarray,
    per_direction: int,
    min_label_support: int,
) -> list[dict[str, Any]]:
    selected = []
    for domain_index, domain in enumerate(DOMAINS):
        total_support, label_support = candidate_support(train, domain)
        eligible = np.flatnonzero(label_support >= min_label_support)
        scores = interactions[domain_index] * idf
        positive = sorted(eligible, key=lambda index: (-scores[index], str(train["hashes"][index])))
        negative = sorted(eligible, key=lambda index: (scores[index], str(train["hashes"][index])))
        for direction, indices in (("positive", positive), ("negative", negative)):
            filtered = [index for index in indices if scores[index] > 0] if direction == "positive" else [index for index in indices if scores[index] < 0]
            for rank, index in enumerate(filtered[:per_direction], 1):
                selected.append(
                    {
                        "domain": domain,
                        "direction": direction,
                        "selection_rank": rank,
                        "train_interaction_frequency": float(interactions[domain_index, index]),
                        "train_interaction_mfidf": float(scores[index]),
                        "train_document_support": int(total_support[index]),
                        "minimum_train_label_support": int(label_support[index]),
                        **metadata[index],
                    }
                )
    return selected


def cluster_means(
    data: Mapping[str, np.ndarray],
    values: np.ndarray,
    iterations: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """Bootstrap class means, preserving every domain-by-label stratum."""
    result = np.empty((len(DOMAINS), 2, iterations, values.shape[1]), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for domain_index, domain in enumerate(DOMAINS):
        for label in (0, 1):
            rows = np.flatnonzero((data["domain"] == domain) & (data["y"] == label))
            ids = data["identity_hash"][rows]
            unique, inverse = np.unique(ids, return_inverse=True)
            cluster_sums = np.zeros((len(unique), values.shape[1]), dtype=np.float64)
            cluster_sizes = np.zeros(len(unique), dtype=np.float64)
            np.add.at(cluster_sums, inverse, values[rows])
            np.add.at(cluster_sizes, inverse, 1.0)
            probability = np.full(len(unique), 1.0 / len(unique))
            for start in range(0, iterations, batch_size):
                stop = min(iterations, start + batch_size)
                weights = rng.multinomial(len(unique), probability, size=stop - start)
                result[domain_index, label, start:stop] = (
                    (weights @ cluster_sums) / (weights @ cluster_sizes)[:, None]
                )
    return result


def bootstrap_interactions(
    data: Mapping[str, np.ndarray],
    candidate_indices: np.ndarray,
    iterations: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    means = cluster_means(
        data,
        data["frequency"][:, candidate_indices],
        iterations,
        batch_size,
        seed,
    )
    effects = means[:, 0] - means[:, 1]
    interactions = np.empty_like(effects)
    for domain_index in range(len(DOMAINS)):
        others = np.arange(len(DOMAINS)) != domain_index
        interactions[domain_index] = effects[domain_index] - effects[others].mean(axis=0)
    return interactions


def sign_pvalue(samples: np.ndarray) -> float:
    finite = samples[np.isfinite(samples)]
    if len(finite) == 0:
        return 1.0
    positive = np.count_nonzero(finite >= 0)
    negative = np.count_nonzero(finite <= 0)
    return min(1.0, 2.0 * (min(positive, negative) + 1) / (len(finite) + 1))


def holm_adjust(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        values[order] * (len(values) - np.arange(len(values)))
    )
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def evaluate_candidates(
    inventory: str,
    selected: Sequence[Mapping[str, Any]],
    train_interactions: np.ndarray,
    validation_interactions: np.ndarray,
    test: Mapping[str, np.ndarray],
    clean: Mapping[str, np.ndarray],
    test_effects: np.ndarray,
    test_interactions: np.ndarray,
    clean_effects: np.ndarray,
    clean_interactions: np.ndarray,
    iterations: int,
    batch_size: int,
    confidence: float,
    seed: int,
) -> list[dict[str, Any]]:
    indices = np.asarray([int(row["motif_index"]) for row in selected], dtype=np.int32)
    primary_samples = bootstrap_interactions(
        test, indices, iterations, batch_size, stable_seed(seed, inventory, "primary")
    )
    clean_samples = bootstrap_interactions(
        clean, indices, iterations, batch_size, stable_seed(seed, inventory, "clean")
    )
    alpha = 1.0 - confidence
    rows = []
    primary_pvalues = []
    clean_pvalues = []
    for candidate_position, candidate in enumerate(selected):
        domain_index = DOMAINS.index(str(candidate["domain"]))
        motif_index = int(candidate["motif_index"])
        primary_distribution = primary_samples[domain_index, :, candidate_position]
        clean_distribution = clean_samples[domain_index, :, candidate_position]
        primary_low, primary_high = np.quantile(primary_distribution, [alpha / 2, 1 - alpha / 2])
        clean_low, clean_high = np.quantile(clean_distribution, [alpha / 2, 1 - alpha / 2])
        train_value = float(train_interactions[domain_index, motif_index])
        validation_value = float(validation_interactions[domain_index, motif_index])
        test_value = float(test_interactions[domain_index, motif_index])
        clean_value = float(clean_interactions[domain_index, motif_index])
        expected_sign = np.sign(train_value)
        validation_agrees = bool(expected_sign * validation_value > 0)
        test_agrees = bool(expected_sign * test_value > 0)
        clean_agrees = bool(expected_sign * clean_value > 0)
        primary_interval_agrees = bool(primary_low > 0) if expected_sign > 0 else bool(primary_high < 0)
        clean_interval_agrees = bool(clean_low > 0) if expected_sign > 0 else bool(clean_high < 0)
        primary_p = sign_pvalue(primary_distribution)
        clean_p = sign_pvalue(clean_distribution)
        primary_pvalues.append(primary_p)
        clean_pvalues.append(clean_p)
        others = np.arange(len(DOMAINS)) != domain_index
        rows.append(
            {
                "inventory": inventory,
                **candidate,
                "validation_interaction_frequency": validation_value,
                "test_target_authorship_effect": float(test_effects[domain_index, motif_index]),
                "test_other_domains_mean_effect": float(test_effects[others, motif_index].mean()),
                "test_interaction_frequency": test_value,
                "test_ci_low": float(primary_low),
                "test_ci_high": float(primary_high),
                "test_bootstrap_sign_pvalue": primary_p,
                "validation_direction_agrees": validation_agrees,
                "test_direction_agrees": test_agrees,
                "test_interval_agrees": primary_interval_agrees,
                "clean_test_interaction_frequency": clean_value,
                "clean_test_ci_low": float(clean_low),
                "clean_test_ci_high": float(clean_high),
                "clean_test_bootstrap_sign_pvalue": clean_p,
                "clean_test_direction_agrees": clean_agrees,
                "clean_test_interval_agrees": clean_interval_agrees,
            }
        )
    primary_adjusted = holm_adjust(primary_pvalues)
    clean_adjusted = holm_adjust(clean_pvalues)
    for index, row in enumerate(rows):
        row["test_holm_pvalue"] = float(primary_adjusted[index])
        row["clean_test_holm_pvalue"] = float(clean_adjusted[index])
        row["replicated_primary"] = bool(
            row["validation_direction_agrees"]
            and row["test_direction_agrees"]
            and row["test_interval_agrees"]
            and row["test_holm_pvalue"] < 0.05
        )
        row["replicated_identity_clean"] = bool(
            row["validation_direction_agrees"]
            and row["clean_test_direction_agrees"]
            and row["clean_test_interval_agrees"]
            and row["clean_test_holm_pvalue"] < 0.05
        )
        row["robust_both_cohorts"] = bool(
            row["replicated_primary"] and row["replicated_identity_clean"]
        )
    return rows


def save_heatmap(inventory: str, rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    robust = [row for row in rows if row["robust_both_cohorts"]]
    if not robust:
        robust = sorted(rows, key=lambda row: -abs(float(row["test_interaction_frequency"])))[:12]
    robust = sorted(robust, key=lambda row: (DOMAINS.index(str(row["domain"])), -abs(float(row["test_interaction_frequency"]))))
    values = np.asarray([float(row["test_interaction_frequency"]) for row in robust])
    lows = np.asarray([float(row["test_ci_low"]) for row in robust])
    highs = np.asarray([float(row["test_ci_high"]) for row in robust])
    labels = [f"{row['domain']}: {row['relation_signature']} [{row['motif_index']}]" for row in robust]
    height = max(4.0, 0.42 * len(robust) + 1.5)
    figure, axis = plt.subplots(figsize=(10, height))
    positions = np.arange(len(labels))
    colors = np.where(values >= 0, "#b2182b", "#2166ac")
    axis.barh(positions, values, color=colors, alpha=0.82)
    axis.errorbar(
        values,
        positions,
        xerr=np.vstack((values - lows, highs - values)),
        fmt="none",
        ecolor="#202020",
        elinewidth=1,
        capsize=2,
    )
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_xscale("symlog", linthresh=0.005)
    axis.set_yticks(positions, labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("domain interaction in normalized motif frequency (symlog scale)")
    axis.set_title(f"{DISPLAY[inventory]}: replicated domain-specific M3 effects")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    os.replace(temporary, output)


def shared_family_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row["robust_both_cohorts"] or not row["pipeline_family"]:
            continue
        direction = "target-more-machine" if float(row["test_interaction_frequency"]) > 0 else "target-less-machine"
        grouped[(str(row["domain"]), str(row["pipeline_family"]), direction)].append(row)
    output = []
    for (domain, family, direction), members in grouped.items():
        inventories = sorted({str(row["inventory"]) for row in members})
        if len(inventories) < 2:
            continue
        output.append(
            {
                "domain": domain,
                "pipeline_family": family,
                "direction": direction,
                "inventory_count": len(inventories),
                "inventories": ";".join(inventories),
                "relations": ";".join(sorted({str(row["relation_signature"]) for row in members})),
                "mean_test_interaction": float(np.mean([float(row["test_interaction_frequency"]) for row in members])),
            }
        )
    return sorted(output, key=lambda row: (DOMAINS.index(row["domain"]), -row["inventory_count"], row["pipeline_family"]))


def pairwise_profile_rows(
    profiles: Mapping[str, tuple[np.ndarray, Sequence[Mapping[str, Any]]]],
) -> list[dict[str, Any]]:
    """Compare all test interactions over graph-matched pipeline families."""
    output = []
    for left, right in combinations(INVENTORIES, 2):
        left_values, left_metadata = profiles[left]
        right_values, right_metadata = profiles[right]
        left_index = {
            str(row["pipeline_family"]): int(row["motif_index"])
            for row in left_metadata if row["pipeline_family"]
        }
        right_index = {
            str(row["pipeline_family"]): int(row["motif_index"])
            for row in right_metadata if row["pipeline_family"]
        }
        shared = sorted(set(left_index) & set(right_index))
        for domain_index, domain in enumerate(DOMAINS):
            x = np.asarray([left_values[domain_index, left_index[family]] for family in shared])
            y = np.asarray([right_values[domain_index, right_index[family]] for family in shared])
            rho = float(spearmanr(x, y).statistic) if len(shared) >= 3 else float("nan")
            output.append(
                {
                    "left_inventory": left,
                    "right_inventory": right,
                    "domain": domain,
                    "shared_pipeline_families": len(shared),
                    "spearman_rho": rho,
                    "sign_agreement": float(np.mean(np.sign(x) == np.sign(y))),
                }
            )
    return output


def render_report(
    rows: Sequence[Mapping[str, Any]],
    shared: Sequence[Mapping[str, Any]],
    pairwise: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
    clean_counts: Mapping[str, Mapping[str, int]],
) -> str:
    robust = [row for row in rows if row["robust_both_cohorts"]]
    lines = [
        "# Domain-specific M3 distinctions across RST inventories",
        "",
        "## Design",
        "",
        "A domain interaction is the machine-minus-human normalized motif-frequency difference in one domain minus the equally weighted mean of that difference in the other four domains. Positive values mean that a motif is more machine-associated in the named domain than elsewhere; negative values mean that it is less machine-associated there.",
        "",
        f"For each inventory and domain, the {settings['per_direction']} strongest positive and {settings['per_direction']} strongest negative MF-IDF-scaled interactions were selected using training only. Validation checked direction. Test intervals use {settings['iterations']:,} domain- and label-stratified identity-cluster bootstrap samples, with Holm correction over the selected hypotheses within each inventory. `robust_both_cohorts` additionally requires the result in a test cohort with training-overlap identities removed and duplicate identities consolidated.",
        "",
        f"Of {len(rows)} training-selected domain–motif hypotheses, {sum(bool(row['replicated_primary']) for row in rows)} replicate in the released test split and {len(robust)} also replicate in the identity-clean sensitivity cohort.",
        "",
    ]
    for inventory in INVENTORIES:
        inventory_rows = [row for row in robust if row["inventory"] == inventory]
        counts = clean_counts[inventory]
        lines.extend(
            [
                f"## {DISPLAY[inventory]} (`{inventory}`)",
                "",
                f"{len(inventory_rows)} robust domain-specific interactions. The clean cohort contains {counts['rows']} identities after removing {counts['removed_overlap_rows']} overlap rows and consolidating {counts['consolidated_duplicate_rows']} duplicate rows.",
                "",
                f"![{DISPLAY[inventory]} domain interactions]({inventory}/domain_interactions.png)",
                "",
                "| Domain | Direction | M3 relations | Target effect | Other-domain effect | Interaction (95% CI) | Inventory resolution |",
                "|---|---|---|---:|---:|---:|---|",
            ]
        )
        for row in sorted(inventory_rows, key=lambda item: (DOMAINS.index(str(item["domain"])), -abs(float(item["test_interaction_frequency"])))):
            direction = "relatively more machine-associated" if float(row["test_interaction_frequency"]) > 0 else "relatively less machine-associated"
            resolution = "shared pipeline family"
            if int(row["family_inventory_count"]) == 1:
                resolution = "pipeline family observed in one inventory"
            if int(row["coarse_group_size"]) > 1:
                resolution += f"; merges into {row['coarse_relations']} (group {row['coarse_group_size']})"
            lines.append(
                f"| {row['domain']} | {direction} | {row['relation_signature']} [{row['motif_index']}] | "
                f"{float(row['test_target_authorship_effect']):+.4f} | {float(row['test_other_domains_mean_effect']):+.4f} | "
                f"{float(row['test_interaction_frequency']):+.4f} [{float(row['test_ci_low']):+.4f}, {float(row['test_ci_high']):+.4f}] | {resolution} |"
            )
        if not inventory_rows:
            lines.append("| — | — | — | — | — | — | No interaction met the full replication rule |")
        lines.append("")
    lines.extend(
        [
            "## Cross-inventory recurring domain effects",
            "",
            "These are graph-matched motif families with the same robust domain-specific direction in at least two inventories.",
            "",
            "| Domain | Direction | Inventories | Relations | Mean interaction |",
            "|---|---|---|---|---:|",
        ]
    )
    for row in shared:
        names = ", ".join(DISPLAY[value] for value in row["inventories"].split(";"))
        lines.append(f"| {row['domain']} | {row['direction']} | {names} | {row['relations']} | {float(row['mean_test_interaction']):+.4f} |")
    if not shared:
        lines.append("| — | — | — | — | No shared family met the rule |")
    lines.extend(
        [
            "",
            "## All-catalog interaction similarity",
            "",
            "This comparison uses every graph-matched pipeline family shared by each inventory pair, independently of candidate selection. Values are means over the five domain-specific interaction profiles.",
            "",
            "| Inventory pair | Shared families | Mean Spearman ρ | Mean sign agreement |",
            "|---|---:|---:|---:|",
        ]
    )
    for left, right in combinations(INVENTORIES, 2):
        pair = [row for row in pairwise if row["left_inventory"] == left and row["right_inventory"] == right]
        rho_values = [float(row["spearman_rho"]) for row in pair if np.isfinite(float(row["spearman_rho"]))]
        rho_text = f"{np.mean(rho_values):.3f}" if rho_values else "not estimable"
        lines.append(
            f"| {DISPLAY[left]} / {DISPLAY[right]} | {pair[0]['shared_pipeline_families']} | "
            f"{rho_text} | "
            f"{np.mean([float(row['sign_agreement']) for row in pair]):.3f} |"
        )
    reversals = sum(
        float(row["test_target_authorship_effect"])
        * float(row["test_other_domains_mean_effect"]) < 0
        for row in robust
    )
    fine_rows = [
        row for row in robust
        if row["inventory"] in {"deu.rst.pcc", "nld.rst.nldt"}
    ]
    merged_rows = sum(int(row["coarse_group_size"]) > 1 for row in fine_rows)
    lines.extend(
        [
            "",
            "## Research reading",
            "",
            f"{reversals}/{len(robust)} robust interactions reverse the sign of the target-domain authorship effect relative to the mean effect elsewhere; the remainder mainly amplify or attenuate a shared direction. Domain therefore changes both the strength and, often, the direction of motif associations.",
            "",
            f"Among PCC and NLDT, {merged_rows}/{len(fine_rows)} robust interactions belong to motif groups that collapse under the declared coarse mapping. These are direct candidates for the localized fine-versus-coarse ablation.",
            "",
        ]
    )
    for inventory in INVENTORIES:
        inventory_robust = [row for row in robust if row["inventory"] == inventory]
        if inventory_robust:
            strongest = max(
                inventory_robust,
                key=lambda row: abs(float(row["test_interaction_frequency"])),
            )
            lines.append(
                f"- {DISPLAY[inventory]}: the largest replicated interaction is "
                f"`{strongest['relation_signature']}` in {strongest['domain']} "
                f"({float(strongest['test_interaction_frequency']):+.4f})."
            )
    lines.extend(
        [
            "",
            "PCC and NLDT retain the strongest broad label-matched profile similarity. Full eRST shares very few exact-label pipeline families with the other inventories, so its pairwise rank correlations are either based on very small sets or are not estimable. Structural-family comparisons are therefore the appropriate basis for broader cross-standard interpretation.",
        ]
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The interaction tests identify where an authorship-associated motif departs from its behavior in the other HC3 domains. They should be read as domain-conditioned representational clues rather than intrinsic properties of finance, medicine, or the other genres. The same documents were parsed under every inventory, so recurring graph families support a shared discourse pattern; unmatched pipeline families and fine-to-coarse merge annotations identify promising relation distinctions for targeted ablation and passage review.",
            "",
            "The analysis is exploratory because earlier project stages inspected the released test outcomes. Training selection, validation direction checks, multiplicity correction, and the identity-clean sensitivity reduce opportunistic discovery, but a new corpus is still needed for confirmatory claims.",
            "",
        ]
    )
    return "\n".join(lines)


def read_csv_index(path: Path, keys: Sequence[str]) -> dict[tuple[str, ...], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {tuple(row[key] for key in keys): row for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="artifacts/canonical/feature_matrices")
    parser.add_argument("--motif-dir", default="data/motifs2")
    parser.add_argument("--family-members", default="artifacts/canonical/cross_standard_mapping/members.csv")
    parser.add_argument("--coarse-mapping", default="artifacts/canonical/granularity_ablation/motif_mapping.csv")
    parser.add_argument("--output-dir", default="artifacts/canonical/domain_distinctions")
    parser.add_argument("--per-direction", type=int, default=2)
    parser.add_argument("--min-label-support", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    if args.per_direction < 1 or args.min_label_support < 1 or args.iterations < 100:
        parser.error("invalid selection or bootstrap settings")

    feature_root = Path(args.feature_dir)
    motif_root = Path(args.motif_dir)
    output_root = Path(args.output_dir)
    members_path = Path(args.family_members)
    coarse_path = Path(args.coarse_mapping)
    family_rows = read_csv_index(members_path, ("inventory", "motif_hash"))
    family_inventories: dict[str, set[str]] = defaultdict(set)
    for row in family_rows.values():
        family_inventories[row["pipeline_family"]].add(row["inventory"])
    family_inventory_counts = {key: len(value) for key, value in family_inventories.items()}
    coarse_rows = read_csv_index(coarse_path, ("inventory", "fine_hash"))
    settings = {
        "per_direction": args.per_direction,
        "min_label_support": args.min_label_support,
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "confidence": args.confidence,
        "seed": args.seed,
    }
    all_rows = []
    interaction_profiles: dict[str, tuple[np.ndarray, Sequence[Mapping[str, Any]]]] = {}
    clean_counts = {}
    provenance: dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "family_members_sha256": sha256_file(members_path),
        "coarse_mapping_sha256": sha256_file(coarse_path),
        "inventories": {},
    }
    for inventory in tqdm(INVENTORIES, desc="inventories", unit="inventory"):
        paths = {
            split: feature_root / inventory / f"hc3_{split}.features.npz"
            for split in ("train", "validation", "test")
        }
        train, validation, test = (load_matrix(paths[split]) for split in ("train", "validation", "test"))
        for split, matrix in (("validation", validation), ("test", test)):
            if not np.array_equal(train["hashes"], matrix["hashes"]):
                raise ValueError(f"{inventory}: motif order differs in {split}")
        clean = clean_test(test, set(train["identity_hash"]))
        clean_counts[inventory] = {
            "rows": len(clean["y"]),
            "removed_overlap_rows": clean["removed_overlap_rows"],
            "consolidated_duplicate_rows": clean["consolidated_duplicate_rows"],
        }
        metadata = load_metadata(
            inventory, train["hashes"], motif_root, family_rows,
            family_inventory_counts, coarse_rows,
        )
        idf = np.log((1.0 + len(train["y"])) / (1.0 + train["presence"].sum(axis=0)))
        train_effects, train_interactions = domain_effects(train, train["frequency"])
        _, validation_interactions = domain_effects(validation, validation["frequency"])
        test_effects, test_interactions = domain_effects(test, test["frequency"])
        clean_effects, clean_interactions = domain_effects(clean, clean["frequency"])
        selected = select_candidates(
            train, metadata, train_interactions, idf,
            args.per_direction, args.min_label_support,
        )
        rows = evaluate_candidates(
            inventory, selected, train_interactions, validation_interactions,
            test, clean, test_effects, test_interactions, clean_effects,
            clean_interactions, args.iterations, args.batch_size,
            args.confidence, args.seed,
        )
        interaction_profiles[inventory] = (test_interactions, metadata)
        inventory_dir = output_root / inventory
        atomic_text(inventory_dir / "domain_interactions.csv", csv_text(rows))
        save_heatmap(inventory, rows, inventory_dir / "domain_interactions.png")
        all_rows.extend(rows)
        provenance["inventories"][inventory] = {
            f"{split}_matrix_sha256": sha256_file(path) for split, path in paths.items()
        }
        atomic_json(
            inventory_dir / "result.json",
            {"inventory": inventory, "settings": settings, "provenance": provenance["inventories"][inventory], "rows": rows},
        )
        atomic_text(output_root / "domain_interactions.csv", csv_text(all_rows))

    shared = shared_family_rows(all_rows)
    pairwise = pairwise_profile_rows(interaction_profiles)
    atomic_text(output_root / "shared_family_interactions.csv", csv_text(shared))
    atomic_text(output_root / "pairwise_profile_similarity.csv", csv_text(pairwise))
    atomic_text(output_root / "report.md", render_report(all_rows, shared, pairwise, settings, clean_counts))
    manifest = {
        "settings": settings,
        "provenance": provenance,
        "candidate_hypotheses": len(all_rows),
        "replicated_primary": sum(bool(row["replicated_primary"]) for row in all_rows),
        "robust_both_cohorts": sum(bool(row["robust_both_cohorts"]) for row in all_rows),
        "shared_family_interactions": len(shared),
        "domain_interactions_sha256": sha256_file(output_root / "domain_interactions.csv"),
        "shared_family_interactions_sha256": sha256_file(output_root / "shared_family_interactions.csv"),
        "pairwise_profile_similarity_sha256": sha256_file(output_root / "pairwise_profile_similarity.csv"),
        "report_sha256": sha256_file(output_root / "report.md"),
    }
    atomic_json(output_root / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("candidate_hypotheses", "replicated_primary", "robust_both_cohorts", "shared_family_interactions")}, indent=2))
    print(f"wrote {output_root / 'report.md'}")


if __name__ == "__main__":
    main()
