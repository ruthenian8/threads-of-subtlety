#!/usr/bin/env python3
"""Create publication-oriented figures from existing UniRST2 artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


INVENTORIES = ("deu.rst.pcc", "eng.erst.gum.new", "eng.rst.rstdt.new", "nld.rst.nldt")
INVENTORY_LABELS = {"deu.rst.pcc": "PCC", "eng.erst.gum.new": "eRST", "eng.rst.rstdt.new": "RST-DT", "nld.rst.nldt": "NLDT"}
MODELS = ("lr", "xgb")
MODEL_LABELS = {"lr": "LR", "xgb": "XGB"}
CONFIGS = ("structural", "relation_frequency", "m3_presence", "m3_frequency", "m3_depth")
CONFIG_LABELS = {"structural": "Structural", "relation_frequency": "Relation\nfrequency", "m3_presence": "M3\npresence", "m3_frequency": "M3\nfrequency", "m3_depth": "M3\ndepth"}
DOMAINS = ("finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai")
DOMAIN_LABELS = ("Finance", "Medicine", "Open QA", "Reddit ELI5", "Wiki CS/AI")
FIGURES = ("model_f1_comparison.png", "paired_frequency_f1_gain.png", "robustness_f1_shift.png", "stable_motif_domain_effects.png", "stable_presence_odds_ratios.png", "relaxed_presence_odds_ratios_3of5.png", "cross_standard_family_overlap.png")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def atomic_savefig(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        fig.savefig(temporary, format="png", dpi=dpi, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        plt.close(fig)
        if temporary.exists():
            temporary.unlink()


def configure_style() -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.facecolor": "white", "axes.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False})


def f1_matrix(rows: Sequence[Mapping[str, str]], robust: bool) -> np.ndarray:
    lookup = {}
    for row in rows:
        if robust and row["metric"] != "f1":
            continue
        key = (row["inventory"], row["model"], row["feature_config"])
        lookup[key] = float(row["score"] if robust else row["test_f1"])
    matrix = np.empty((len(INVENTORIES) * len(MODELS), len(CONFIGS)))
    pairs = [(inventory, model) for inventory in INVENTORIES for model in MODELS]
    for row_index, (inventory, model) in enumerate(pairs):
        for column, config in enumerate(CONFIGS):
            matrix[row_index, column] = lookup[(inventory, model, config)]
    return matrix


def heatmap(ax, values, row_labels, column_labels, *, cmap, vmin, vmax, value_format, colorbar_label) -> None:
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(column_labels)), column_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    midpoint = (vmin + vmax) / 2
    half_range = (vmax - vmin) / 2
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                color = "white" if abs(value - midpoint) > 0.55 * half_range else "black"
                ax.text(column, row, format(value, value_format), ha="center", va="center", color=color, fontsize=7.5)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label(colorbar_label)


def row_labels() -> List[str]:
    return [f"{INVENTORY_LABELS[i]} · {MODEL_LABELS[m]}" for i in INVENTORIES for m in MODELS]


def plot_model_f1(rows, path, dpi) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    heatmap(ax, f1_matrix(rows, False), row_labels(), [CONFIG_LABELS[x] for x in CONFIGS], cmap="viridis", vmin=0.60, vmax=0.90, value_format=".3f", colorbar_label="Test F1 (human class)")
    ax.set_title("Held-out discrimination by relation inventory and feature set")
    atomic_savefig(fig, path, dpi)


def plot_frequency_gain(primary_rows, robust_rows, path, dpi) -> None:
    def select(rows):
        return {(row["inventory"], row["model"]): row for row in rows if row["metric"] == "f1" and row["contrast"] == "frequency_vs_structural"}
    primary, robust = select(primary_rows), select(robust_rows)
    keys = [(i, m) for i in INVENTORIES for m in MODELS]
    y = np.arange(len(keys))
    estimate = np.asarray([float(robust[k]["difference"]) for k in keys])
    low = np.asarray([float(robust[k]["ci_low"]) for k in keys])
    high = np.asarray([float(robust[k]["ci_high"]) for k in keys])
    original = np.asarray([float(primary[k]["difference"]) for k in keys])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.axvline(0, color="#555555", linewidth=1, linestyle="--")
    ax.errorbar(estimate, y, xerr=np.vstack((estimate - low, high - estimate)), fmt="o", color="#176d8c", ecolor="#176d8c", capsize=3, label="Identity-robust estimate (95% CI)")
    ax.scatter(original, y, marker="x", color="#d95f02", s=38, label="Primary estimate")
    ax.set_yticks(y, [f"{INVENTORY_LABELS[i]} · {MODEL_LABELS[m]}" for i, m in keys])
    ax.invert_yaxis()
    ax.set_xlabel("F1 difference: M3 frequency − structural")
    ax.set_title("M3-frequency gain persists after identity-overlap removal")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax.grid(axis="x", alpha=0.25)
    atomic_savefig(fig, path, dpi)


def plot_f1_shift(primary_rows, robust_rows, path, dpi) -> None:
    shift = f1_matrix(robust_rows, True) - f1_matrix(primary_rows, False)
    bound = max(0.002, float(np.max(np.abs(shift))))
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    heatmap(ax, shift, row_labels(), [CONFIG_LABELS[x] for x in CONFIGS], cmap="RdBu_r", vmin=-bound, vmax=bound, value_format="+.3f", colorbar_label="Robust − primary F1")
    ax.set_title("Performance sensitivity to exact identity overlap")
    atomic_savefig(fig, path, dpi)


def relation_label(description: str) -> str:
    return description.split("relations=", 1)[-1].replace("|", "×")


def plot_stable_domain_effects(stability_rows, effect_rows, member_rows, path, dpi) -> None:
    stable = [row for row in stability_rows if row["stable"] == "True" and row["representation"] in ("presence", "frequency")]
    stable.sort(key=lambda row: ((0 if row["representation"] == "presence" else 1), INVENTORIES.index(row["inventory"]), row["motif_hash"]))
    effects = {(row["inventory"], row["motif_hash"], row["representation"], row["domain"]): row for row in effect_rows}
    members = {(row["inventory"], row["motif_hash"]): row for row in member_rows}
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 10.5))
    for ax, representation in zip(axes, ("presence", "frequency")):
        rows = [row for row in stable if row["representation"] == representation]
        values = np.full((len(rows), len(DOMAINS)), np.nan)
        for ri, row in enumerate(rows):
            for ci, domain in enumerate(DOMAINS):
                value = effects.get((row["inventory"], row["motif_hash"], representation, domain))
                if value is not None and value["effect"]:
                    values[ri, ci] = float(value["effect"])
        bound = max(0.1, float(np.nanmax(np.abs(values))))
        image = ax.imshow(values, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
        ax.set_xticks(range(len(DOMAINS)), DOMAIN_LABELS)
        labels = [f"{INVENTORY_LABELS[row['inventory']]} · {relation_label(members[(row['inventory'], row['motif_hash'])]['description'])} · {row['motif_hash'][:6]}" for row in rows]
        ax.set_yticks(range(len(rows)), labels)
        ax.set_title("Presence: prevalence difference" if representation == "presence" else "Frequency: Hedges’ g")
        for ri in range(values.shape[0]):
            for ci in range(values.shape[1]):
                if np.isfinite(values[ri, ci]):
                    color = "white" if abs(values[ri, ci]) > 0.55 * bound else "#222222"
                    ax.text(ci, ri, f"{values[ri, ci]:+.2f}", ha="center", va="center", color=color, fontsize=7)
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle("Identity-robust motif effects across held-out HC3 domains", y=0.995)
    fig.text(0.5, 0.005, "Negative effects indicate greater presence or frequency in generated text.", ha="center", fontsize=9)
    atomic_savefig(fig, path, dpi)


def presence_odds_ratio(row: Mapping[str, str]) -> tuple[float, bool]:
    human_present = int(row["human_present"])
    generated_present = int(row["generated_present"])
    human_absent = int(row["human_documents"]) - human_present
    generated_absent = int(row["generated_documents"]) - generated_present
    counts = np.asarray(
        [human_present, human_absent, generated_present, generated_absent],
        dtype=np.float64,
    )
    if np.any(counts < 0):
        raise ValueError("Motif presence count exceeds its document count")
    corrected = bool(np.any(counts == 0))
    if corrected:
        counts += 0.5
    return float((counts[0] / counts[1]) / (counts[2] / counts[3])), corrected


def plot_stable_presence_odds_ratios(stability_rows, effect_rows, member_rows, path, dpi, minimum_domains=4) -> None:
    stable = [
        row for row in stability_rows
        if row["representation"] == "presence"
        and (
            (int(row["robust_human_domains"]) >= minimum_domains and int(row["robust_generated_domains"]) == 0)
            or (int(row["robust_generated_domains"]) >= minimum_domains and int(row["robust_human_domains"]) == 0)
        )
    ]
    stable.sort(key=lambda row: (INVENTORIES.index(row["inventory"]), row["motif_hash"]))
    if not stable:
        raise ValueError(f"No identity-robust {minimum_domains}/5 presence associations to plot")
    effects = {
        (row["inventory"], row["motif_hash"], row["domain"]): row
        for row in effect_rows if row["representation"] == "presence"
    }
    members = {(row["inventory"], row["motif_hash"]): row for row in member_rows}
    odds_ratios = np.empty((len(stable), len(DOMAINS)), dtype=np.float64)
    corrected = np.zeros_like(odds_ratios, dtype=bool)
    for row_index, row in enumerate(stable):
        key = row["inventory"], row["motif_hash"]
        for domain_index, domain in enumerate(DOMAINS):
            effect = effects.get((*key, domain))
            if effect is None:
                raise ValueError(f"Missing presence counts for {key}/{domain}")
            odds_ratios[row_index, domain_index], corrected[row_index, domain_index] = presence_odds_ratio(effect)

    log_odds = np.log(odds_ratios)
    bound = max(float(np.log(2)), float(np.max(np.abs(log_odds))))
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.55 * len(stable) + 1.8)))
    image = ax.imshow(log_odds, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(range(len(DOMAINS)), DOMAIN_LABELS)
    ax.set_yticks(
        range(len(stable)),
        [
            f"{INVENTORY_LABELS[row['inventory']]} · "
            f"{relation_label(members[(row['inventory'], row['motif_hash'])]['description'])} · "
            f"{row['motif_hash'][:6]}"
            for row in stable
        ],
    )
    title = "Motif presence odds ratios across held-out HC3 domains"
    if minimum_domains != 4:
        title += f" (relaxed stability: {minimum_domains}/5)"
    ax.set_title(title)
    for row_index in range(len(stable)):
        for domain_index in range(len(DOMAINS)):
            value = odds_ratios[row_index, domain_index]
            label = f"{value:.2f}" if value < 10 else f"{value:.1f}"
            if corrected[row_index, domain_index]:
                label += "†"
            color = "white" if abs(log_odds[row_index, domain_index]) > 0.55 * bound else "#222222"
            ax.text(domain_index, row_index, label, ha="center", va="center", color=color, fontsize=8)
    tick_values = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0)
    ticks = [np.log(value) for value in tick_values if abs(np.log(value)) <= bound]
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, ticks=ticks)
    colorbar.ax.set_yticklabels([f"{np.exp(value):g}" for value in ticks])
    colorbar.set_label("Presence odds ratio: human / generated")
    fig.text(
        0.5, 0.005,
        "OR < 1: generated-associated; OR > 1: human-associated. "
        f"Rows have robust same-direction effects in ≥{minimum_domains}/5 domains and none opposite; "
        "† marks a 0.5 zero-cell correction.",
        ha="center", fontsize=8,
    )
    atomic_savefig(fig, path, dpi)


def plot_family_overlap(family_rows, member_rows, path, dpi) -> None:
    families = sorted([row for row in family_rows if int(row["stable_inventory_count"]) >= 2], key=lambda row: row["family_id"])
    family_index = {row["family_id"]: index for index, row in enumerate(families)}
    values = np.zeros((len(families), len(INVENTORIES)), dtype=np.int8)
    for member in member_rows:
        family = member["pipeline_family"]
        if family not in family_index or member["robust_stable"] != "True":
            continue
        representations = {value.split(":", 1)[0] for value in member["stable_associations"].split(";") if value}
        values[family_index[family], INVENTORIES.index(member["inventory"])] = (1 if "presence" in representations else 0) + (2 if "frequency" in representations else 0)
    cmap = ListedColormap(["#f4f4f4", "#1b9e77", "#d95f02", "#7570b3"])
    fig, ax = plt.subplots(figsize=(7.4, max(3.2, 0.58 * len(families))))
    ax.imshow(values, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
    ax.set_xticks(range(len(INVENTORIES)), [INVENTORY_LABELS[x] for x in INVENTORIES])
    ax.set_yticks(range(len(families)), [f"{row['family_id']} · {row['relation_signatures'].replace('|', '×').replace(';', ' / ')}" for row in families])
    tokens = {0: "", 1: "P", 2: "F", 3: "P+F"}
    for ri in range(values.shape[0]):
        for ci in range(values.shape[1]):
            ax.text(ci, ri, tokens[int(values[ri, ci])], ha="center", va="center", color="white" if values[ri, ci] else "#777777", fontsize=8)
    ax.set_title("Robust stable motif families shared across relation inventories")
    ax.set_xlabel("P = presence; F = frequency")
    ax.tick_params(length=0)
    atomic_savefig(fig, path, dpi)


def checkpoint_manifest(output_dir: Path, provenance: Mapping[str, str], completed: Sequence[str], dpi: int) -> None:
    atomic_write_json(output_dir / "manifest.json", {"provenance": dict(provenance), "settings": {"dpi": dpi}, "completed_figures": list(completed), "figure_sha256": {name: sha256_file(output_dir / name) for name in completed}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="artifacts/canonical/model_comparison")
    parser.add_argument("--paired-dir", default="artifacts/canonical/paired_comparison")
    parser.add_argument("--robustness-dir", default="artifacts/canonical/robustness")
    parser.add_argument("--mapping-dir", default="artifacts/canonical/cross_standard_mapping")
    parser.add_argument("--output-dir", default="artifacts/canonical/figures")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    inputs = {
        "primary_models": Path(args.model_dir) / "summary.csv",
        "primary_paired": Path(args.paired_dir) / "summary.csv",
        "robust_models": Path(args.robustness_dir) / "model_metrics.csv",
        "robust_paired": Path(args.robustness_dir) / "paired_contrasts.csv",
        "robust_stability": Path(args.robustness_dir) / "motif_stability.csv",
        "robust_effects": Path(args.robustness_dir) / "motif_domain_effects.csv",
        "mapping_members": Path(args.mapping_dir) / "members.csv",
        "pipeline_families": Path(args.mapping_dir) / "pipeline_families.csv",
    }
    provenance = {name: sha256_file(path) for name, path in inputs.items()}
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "manifest.json"
    required_outputs = FIGURES + ("README.md",)
    if manifest_path.exists() and all((output_dir / name).exists() for name in required_outputs) and not args.force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("provenance") == provenance and manifest.get("settings") == {"dpi": args.dpi}:
            print(f"skipping complete figures; wrote {output_dir}")
            return
    configure_style()
    tables = {name: read_csv(path) for name, path in inputs.items()}
    plotters = (
        (FIGURES[0], lambda path: plot_model_f1(tables["primary_models"], path, args.dpi)),
        (FIGURES[1], lambda path: plot_frequency_gain(tables["primary_paired"], tables["robust_paired"], path, args.dpi)),
        (FIGURES[2], lambda path: plot_f1_shift(tables["primary_models"], tables["robust_models"], path, args.dpi)),
        (FIGURES[3], lambda path: plot_stable_domain_effects(tables["robust_stability"], tables["robust_effects"], tables["mapping_members"], path, args.dpi)),
        (FIGURES[4], lambda path: plot_stable_presence_odds_ratios(tables["robust_stability"], tables["robust_effects"], tables["mapping_members"], path, args.dpi)),
        (FIGURES[5], lambda path: plot_stable_presence_odds_ratios(tables["robust_stability"], tables["robust_effects"], tables["mapping_members"], path, args.dpi, minimum_domains=3)),
        (FIGURES[6], lambda path: plot_family_overlap(tables["pipeline_families"], tables["mapping_members"], path, args.dpi)),
    )
    completed = []
    for filename, plotter in plotters:
        plotter(output_dir / filename)
        completed.append(filename)
        checkpoint_manifest(output_dir, provenance, completed, args.dpi)
        print(f"wrote {output_dir / filename}")
    atomic_write_text(output_dir / "README.md", """# UniRST2 result figures

- `model_f1_comparison.png`: primary held-out binary F1 for the human class for every inventory, classifier, and feature set.
- `paired_frequency_f1_gain.png`: paired F1 gain of M3-frequency over structural features, with identity-robust intervals and primary estimates.
- `robustness_f1_shift.png`: F1 change after removing train-overlap identities and consolidating test duplicates.
- `stable_motif_domain_effects.png`: domain effects for motifs passing the identity-robust stability rule.
- `stable_presence_odds_ratios.png`: human-versus-generated presence odds ratios for identity-robust stable motifs, displayed on a log-centered color scale. A 0.5 correction is applied only to zero-cell tables.
- `relaxed_presence_odds_ratios_3of5.png`: sensitivity version of the odds-ratio plot, requiring robust same-direction presence effects in at least three of five domains and none in the opposite direction.
- `cross_standard_family_overlap.png`: presence/frequency associations for pipeline-equivalent families stable in multiple inventories.

All figures are derived from existing tabular artifacts; no model fitting or motif extraction occurs here.
""")
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
