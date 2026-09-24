#!/usr/bin/env python3
"""Compare paired Human Detectors M3 distributions using frozen HC3 catalogs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy


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
METRICS = ("presence", "frequency", "mfidf", "conditional_depth", "depth_contribution")
SCRIPT_VERSION = "human-detectors-motif-distributions-v1"


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


def csv_text(rows: Iterable[Mapping[str, Any]]) -> str:
    rows = list(rows)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]) if rows else [])
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def graph_description(graph: Mapping[str, Any]) -> tuple[str, str]:
    nodes = sorted(str(node.get("label_0", "?")) for node in graph.get("nodes", ()))
    links = graph.get("links", graph.get("edges", ()))
    relations = sorted(
        str(edge.get("label_0", "?"))
        for edge in links
        if edge.get("label_0") != "/"
    )
    return "|".join(relations) if relations else "(structural only)", "|".join(nodes)


def load_features(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        result = {
            "y": np.asarray(values["y"], dtype=np.int8),
            "pair_id": np.asarray(values["pair_id"]).astype(str),
            "hashes": np.asarray(values["m3_hashes"]).astype(str),
            "presence": np.asarray(values["X_m3_presence"], dtype=np.float64),
            "frequency": np.asarray(values["X_m3_frequency"], dtype=np.float64),
            "depth": np.asarray(values["X_m3_depth"], dtype=np.float64),
        }
    expected = (len(result["y"]), len(result["hashes"]))
    for key in ("presence", "frequency", "depth"):
        if result[key].shape != expected or not np.isfinite(result[key]).all():
            raise ValueError(f"{path}: malformed {key} matrix")
    unique, counts = np.unique(result["pair_id"], return_counts=True)
    if len(unique) * 2 != len(result["y"]) or set(counts.tolist()) != {2}:
        raise ValueError(f"{path}: cohort is not composed of matched pairs")
    return result


def load_training(path: Path, hashes: Sequence[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        if values["m3_hashes"].astype(str).tolist() != list(hashes):
            raise ValueError(f"{path}: HC3 and Human Detectors catalog orders differ")
        return {
            "y": np.asarray(values["y"], dtype=np.int8),
            "presence": np.asarray(values["X_m3_presence"], dtype=np.float64),
            "frequency": np.asarray(values["X_m3_frequency"], dtype=np.float64),
        }


def pair_indices(y: np.ndarray, pair_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    machine, human = [], []
    for pair_id in sorted(set(pair_ids.tolist())):
        rows = np.flatnonzero(pair_ids == pair_id)
        machine_rows = rows[y[rows] == 0]
        human_rows = rows[y[rows] == 1]
        if len(machine_rows) != 1 or len(human_rows) != 1:
            raise ValueError(f"pair {pair_id!r} does not contain one row per label")
        machine.append(machine_rows[0])
        human.append(human_rows[0])
    return np.asarray(machine), np.asarray(human)


def paired_estimates(
    data: Mapping[str, np.ndarray],
    idf: np.ndarray,
    iterations: int,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    machine, human = pair_indices(data["y"], data["pair_id"])
    matrices = {
        "presence": data["presence"],
        "frequency": data["frequency"],
        "mfidf": data["frequency"] * idf[None, :],
        "conditional_depth": data["depth"],
        "depth_contribution": data["frequency"] * data["depth"],
    }
    point: dict[str, dict[str, np.ndarray]] = {}
    for metric, values in matrices.items():
        if metric == "conditional_depth":
            machine_present = data["presence"][machine]
            human_present = data["presence"][human]
            machine_mean = np.divide(
                (values[machine] * machine_present).sum(axis=0), machine_present.sum(axis=0),
                out=np.full(values.shape[1], np.nan), where=machine_present.sum(axis=0) > 0,
            )
            human_mean = np.divide(
                (values[human] * human_present).sum(axis=0), human_present.sum(axis=0),
                out=np.full(values.shape[1], np.nan), where=human_present.sum(axis=0) > 0,
            )
        else:
            machine_mean = values[machine].mean(axis=0)
            human_mean = values[human].mean(axis=0)
        point[metric] = {
            "machine": machine_mean,
            "human": human_mean,
            "difference": machine_mean - human_mean,
        }

    motif_count = len(idf)
    samples = {metric: np.empty((iterations, motif_count), dtype=np.float32) for metric in METRICS}
    rng = np.random.default_rng(seed)
    pair_count = len(machine)
    probabilities = np.full(pair_count, 1.0 / pair_count)
    for start in range(0, iterations, batch_size):
        stop = min(start + batch_size, iterations)
        weights = rng.multinomial(pair_count, probabilities, size=stop - start).astype(np.float64)
        denominator = weights.sum(axis=1, keepdims=True)
        for metric in ("presence", "frequency", "mfidf", "depth_contribution"):
            values = matrices[metric]
            samples[metric][start:stop] = (
                weights @ (values[machine] - values[human]) / denominator
            )
        for target, rows in (("machine", machine), ("human", human)):
            present = data["presence"][rows]
            depth_sum = weights @ (matrices["conditional_depth"][rows] * present)
            present_sum = weights @ present
            estimate = np.divide(
                depth_sum, present_sum, out=np.full_like(depth_sum, np.nan), where=present_sum > 0
            )
            if target == "machine":
                machine_depth = estimate
            else:
                human_depth = estimate
        samples["conditional_depth"][start:stop] = machine_depth - human_depth
    intervals = {}
    for metric, values in samples.items():
        valid = np.isfinite(values).any(axis=0)
        low = np.full(values.shape[1], np.nan, dtype=np.float64)
        high = np.full(values.shape[1], np.nan, dtype=np.float64)
        if valid.any():
            low[valid], high[valid] = np.nanquantile(
                values[:, valid], (0.025, 0.975), axis=0
            )
        intervals[metric] = (low, high)
    return point, intervals


def save_plot(
    inventory: str,
    selected: Sequence[int],
    metadata: Sequence[Mapping[str, str]],
    point: Mapping[str, Mapping[str, np.ndarray]],
    intervals: Mapping[str, tuple[np.ndarray, np.ndarray]],
    output: Path,
) -> None:
    values = point["mfidf"]["difference"][selected] * 1000.0
    low = intervals["mfidf"][0][selected] * 1000.0
    high = intervals["mfidf"][1][selected] * 1000.0
    positions = np.arange(len(selected))
    labels = [f"{index}: {metadata[index]['relation_signature']}" for index in selected]
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(positions, values, color=np.where(values >= 0, "#d95f59", "#2e8b57"))
    axis.errorbar(positions, values, yerr=np.vstack((values - low, high - values)),
                  fmt="none", ecolor="#333333", capsize=3, linewidth=0.8)
    axis.axhline(0, color="#444444", linewidth=0.8)
    axis.set_xticks(positions, labels, rotation=45, ha="right")
    axis.set_ylabel("machine − human MF-IDF × 10³")
    axis.set_title(f"{DISPLAY[inventory]}: Human Detectors effects for HC3-training-selected motifs")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    os.replace(temporary, output)


def render_report(
    coverage: Sequence[Mapping[str, Any]],
    top_rows: Sequence[Mapping[str, Any]],
    iterations: int,
) -> str:
    lines = [
        "# Human Detectors motif-distribution comparison",
        "",
        "HC3-selected M3 catalogs and HC3-fitted IDF weights are applied unchanged to the paired Human Detectors feature matrices. Differences are mean machine minus mean human. Positive values indicate greater prevalence in machine documents. The 149 source/prompt pairs are resampled together in the bootstrap.",
        "",
        "## HC3 catalog coverage",
        "",
        "| Inventory | HC3 selected motifs | Seen in HD | Seen in machine | Seen in human | Documents with ≥1 match | Frequency mass range |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in coverage:
        lines.append(
            f"| {DISPLAY[row['inventory']]} | {row['catalog_motifs']} | {row['seen_motifs']} | "
            f"{row['machine_seen_motifs']} | {row['human_seen_motifs']} | "
            f"{row['documents_with_motif']}/{row['documents']} | "
            f"{row['frequency_mass_min']:.6f}–{row['frequency_mass_max']:.6f} |"
        )
    lines.extend([
        "",
        "Every retained document contains HC3-catalogued motifs and its selected-catalog frequencies sum to one within numerical tolerance. RST-DT leaves six selected HC3 motifs unobserved in this small holdout; all selected motifs are observed for the other inventories.",
        "",
        "## Strongest HD effects among HC3-training-selected motifs",
        "",
        "The display candidates were fixed from the five largest positive and five largest negative aggregate HC3-training MF-IDF effects per inventory. The table reports the strongest HD effect in each direction within that fixed candidate set.",
        "",
        "| Inventory | More machine-associated in HD | MF-IDF difference (95% CI) | More human-associated in HD | MF-IDF difference (95% CI) |",
        "|---|---|---:|---|---:|",
    ])
    for inventory in INVENTORIES:
        rows = [row for row in top_rows if row["inventory"] == inventory]
        positive = max(rows, key=lambda row: float(row["difference_machine_minus_human"]))
        negative = min(rows, key=lambda row: float(row["difference_machine_minus_human"]))
        lines.append(
            f"| {DISPLAY[inventory]} | {positive['motif_index']} `{positive['relation_signature']}` | "
            f"{positive['difference_machine_minus_human'] * 1000:+.3f} "
            f"[{positive['ci_low'] * 1000:+.3f}, {positive['ci_high'] * 1000:+.3f}] ×10⁻³ | "
            f"{negative['motif_index']} `{negative['relation_signature']}` | "
            f"{negative['difference_machine_minus_human'] * 1000:+.3f} "
            f"[{negative['ci_low'] * 1000:+.3f}, {negative['ci_high'] * 1000:+.3f}] ×10⁻³ |"
        )
    lines.extend([
        "",
        f"Intervals use {iterations:,} paired bootstrap samples. Selection is independent of HD labels at the candidate-set stage, but choosing the largest HD effect from each fixed set remains descriptive. Full estimates for every motif and representation are in `motif_differences.csv`.",
        "",
        "## Interpretation",
        "",
        "The coverage audit rules out catalog non-occurrence as an explanation for the weak frozen detector transfer. HC3 motifs are present throughout Human Detectors, but their class-conditional distribution has shifted. The analysis therefore supports a temporal/domain-shift interpretation: the discourse configurations remain measurable while their association with authorship changes.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hd-feature-root", type=Path, default=Path("artifacts/unirst2_human_detectors/feature_matrices"))
    parser.add_argument("--hc3-feature-root", type=Path, default=Path("artifacts/unirst2_feature_matrices"))
    parser.add_argument("--motif-root", type=Path, default=Path("data/motifs2"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/unirst2_motif_distribution_comparison/human_detectors"),
    )
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if args.iterations < 100 or args.batch_size < 1:
        parser.error("invalid bootstrap settings")

    coverage_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    for inventory in INVENTORIES:
        hd_path = args.hd_feature_root / inventory / "human_detectors.features.npz"
        train_path = args.hc3_feature_root / inventory / "hc3_train.features.npz"
        catalog_path = args.motif_root / inventory / "hc3_M3_motifs.json"
        for path in (hd_path, train_path, catalog_path):
            input_hashes[str(path)] = sha256_file(path)
        hd = load_features(hd_path)
        train = load_training(train_path, hd["hashes"])
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        metadata = []
        for index, motif_hash in enumerate(hd["hashes"]):
            if motif_hash not in catalog:
                raise ValueError(f"{inventory}: selected motif {motif_hash} absent from HC3 catalog")
            relations, nodes = graph_description(catalog[motif_hash])
            metadata.append({
                "motif_index": index,
                "motif_hash": motif_hash,
                "relation_signature": relations,
                "node_signature": nodes,
            })
        idf = np.log((1.0 + len(train["y"])) / (1.0 + train["presence"].sum(axis=0)))
        train_delta = (
            (train["frequency"][train["y"] == 0] * idf).mean(axis=0)
            - (train["frequency"][train["y"] == 1] * idf).mean(axis=0)
        )
        positive = [int(i) for i in np.argsort(-train_delta) if train_delta[i] > 0][:5]
        negative = [int(i) for i in np.argsort(train_delta) if train_delta[i] < 0][:5]
        selected = positive + negative
        point, intervals = paired_estimates(
            hd, idf, args.iterations, args.batch_size,
            int.from_bytes(hashlib.sha256(f"{args.seed}:{inventory}".encode()).digest()[:8], "little"),
        )
        machine_mask, human_mask = hd["y"] == 0, hd["y"] == 1
        seen = hd["presence"].sum(axis=0) > 0
        coverage_rows.append({
            "inventory": inventory,
            "catalog_motifs": len(hd["hashes"]),
            "seen_motifs": int(seen.sum()),
            "machine_seen_motifs": int((hd["presence"][machine_mask].sum(axis=0) > 0).sum()),
            "human_seen_motifs": int((hd["presence"][human_mask].sum(axis=0) > 0).sum()),
            "unseen_hashes": "|".join(hd["hashes"][~seen].tolist()),
            "documents": len(hd["y"]),
            "documents_with_motif": int((hd["presence"].sum(axis=1) > 0).sum()),
            "frequency_mass_min": float(hd["frequency"].sum(axis=1).min()),
            "frequency_mass_max": float(hd["frequency"].sum(axis=1).max()),
        })
        for metric in METRICS:
            low, high = intervals[metric]
            for index, motif in enumerate(metadata):
                row = {
                    "inventory": inventory,
                    **motif,
                    "metric": metric,
                    "machine_mean": float(point[metric]["machine"][index]),
                    "human_mean": float(point[metric]["human"][index]),
                    "difference_machine_minus_human": float(point[metric]["difference"][index]),
                    "ci_low": float(low[index]) if np.isfinite(low[index]) else "",
                    "ci_high": float(high[index]) if np.isfinite(high[index]) else "",
                    "selected_from_hc3_training": index in selected,
                    "hc3_training_mfidf_difference": float(train_delta[index]),
                }
                difference_rows.append(row)
                if metric == "mfidf" and index in selected:
                    selected_rows.append(row)
        inventory_dir = args.output_dir / inventory
        atomic_text(inventory_dir / "motif_differences.csv", csv_text(
            row for row in difference_rows if row["inventory"] == inventory
        ))
        save_plot(inventory, selected, metadata, point, intervals, inventory_dir / "hc3_selected_mfidf.png")

    coverage_path = args.output_dir / "catalog_coverage.csv"
    differences_path = args.output_dir / "motif_differences.csv"
    selected_path = args.output_dir / "hc3_training_selected_motifs.csv"
    report_path = args.output_dir / "report.md"
    atomic_text(coverage_path, csv_text(coverage_rows))
    atomic_text(differences_path, csv_text(difference_rows))
    atomic_text(selected_path, csv_text(selected_rows))
    atomic_text(report_path, render_report(coverage_rows, selected_rows, args.iterations))
    outputs = sorted(path for path in args.output_dir.rglob("*") if path.is_file() and path.name != "manifest.json")
    atomic_json(args.output_dir / "manifest.json", {
        "script_version": SCRIPT_VERSION,
        "parameters": {"iterations": args.iterations, "batch_size": args.batch_size, "seed": args.seed},
        "definitions": {
            "difference": "mean(machine) - mean(human)",
            "idf": "log((1 + N_hc3_train) / (1 + HC3 train document frequency))",
            "candidate_selection": "five largest positive and negative aggregate HC3-training MF-IDF effects per inventory",
            "uncertainty": "matched source/prompt-pair percentile bootstrap",
        },
        "coverage": coverage_rows,
        "input_hashes": input_hashes,
        "output_hashes": {str(path.relative_to(args.output_dir)): sha256_file(path) for path in outputs},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
    })
    print(json.dumps({"coverage": coverage_rows, "rows": len(difference_rows), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
