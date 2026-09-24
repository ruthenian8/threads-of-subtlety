#!/usr/bin/env python3
"""Plot the top M3 motifs from the v2 M3-only XGBoost models.

The script refits each inventory's model on the PCC-aligned HC3 validation
split with the selected hyperparameters recorded in the v2 text report.  MF
and WAD gain shares are summed to rank motifs and shown separately in each
stacked bar.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from tqdm import tqdm


INVENTORIES = (
    "deu.rst.pcc",
    "eng.erst.gum",
    "eng.rst.rstdt",
    "nld.rst.nldt",
)
DISPLAY_NAMES = {
    "deu.rst.pcc": "PCC (deu.rst.pcc)",
    "eng.erst.gum": "GUM (eng.erst.gum)",
    "eng.rst.rstdt": "RST-DT (eng.rst.rstdt)",
    "nld.rst.nldt": "NLDT (nld.rst.nldt)",
}
VALIDATION_FILE = (
    "hc3_validation_pcc_aligned.discourse_parsed.graph_added."
    "m3_m6_motif_dists.jsonl"
)


def _load_analysis_module(path: Path):
    spec = importlib.util.spec_from_file_location("motif_dependency_analysis", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import analysis helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    # Python 3.8 dataclasses resolve postponed annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _selected_params(report_path: Path) -> Dict[str, Dict[str, Any]]:
    """Extract the selected M3-only XGBoost parameters from a report."""
    current_inventory = None
    selected: Dict[str, Dict[str, Any]] = {}
    pattern = re.compile(r"^m3/xgb:.*\bparams=(\{.*\})$")
    for raw_line in report_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            current_inventory = line[1:-1]
            continue
        match = pattern.match(line)
        if match and current_inventory in INVENTORIES:
            value = ast.literal_eval(match.group(1))
            if not isinstance(value, dict):
                raise ValueError(f"Malformed parameters for {current_inventory}")
            selected[current_inventory] = value
    missing = set(INVENTORIES) - set(selected)
    if missing:
        raise ValueError(
            f"The v2 report has no M3-only XGBoost parameters for: "
            f"{', '.join(sorted(missing))}"
        )
    return selected


def _manifest_path(data_dir: Path, inventory: str) -> Path:
    motif_dir = data_dir / "motifs" / inventory
    generated = motif_dir / "hc3_selected-motif-hashes.generated.json"
    return generated if generated.exists() else motif_dir / "hc3_selected-motif-hashes.json"


def _aggregate_importance(
    names: Sequence[str], values: np.ndarray, limit: int
) -> list[Dict[str, Any]]:
    motifs: Dict[tuple[int, str], Dict[str, Any]] = {}
    for name, value in zip(names, values):
        size, raw_index, motif_hash, statistic = name.split(":")
        if size != "m3" or statistic not in ("mf", "wad"):
            raise ValueError(f"Unexpected M3 feature name: {name}")
        key = (int(raw_index), motif_hash)
        row = motifs.setdefault(
            key,
            {
                "motif_index": int(raw_index),
                "motif_hash": motif_hash,
                "mf_importance": 0.0,
                "wad_importance": 0.0,
            },
        )
        row[f"{statistic}_importance"] = float(value)
    for row in motifs.values():
        row["total_importance"] = (
            row["mf_importance"] + row["wad_importance"]
        )
    ordered = sorted(
        motifs.values(),
        key=lambda row: (-row["total_importance"], row["motif_index"]),
    )[:limit]
    for rank, row in enumerate(ordered, 1):
        row["rank"] = rank
    return ordered


def _save_plot(
    rows: Sequence[Mapping[str, Any]],
    inventory: str,
    params: Mapping[str, Any],
    output_stem: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:
        raise RuntimeError(
            "Plot generation requires matplotlib; install requirements.txt first"
        ) from exc

    labels = [
        f"#{row['motif_index']}  {str(row['motif_hash'])[:12]}…"
        for row in rows
    ]
    mf = np.asarray([row["mf_importance"] for row in rows])
    wad = np.asarray([row["wad_importance"] for row in rows])
    totals = mf + wad
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    ax.barh(y, mf, color="#31688e", label="Motif frequency (MF)")
    ax.barh(
        y,
        wad,
        left=mf,
        color="#35b779",
        label="Weighted average depth (WAD)",
    )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.set_xlabel("Share of total M3-only XGBoost gain")
    ax.set_title(
        f"Top {len(rows)} M3 motifs by importance — {DISPLAY_NAMES[inventory]}",
        fontsize=15,
        fontweight="bold",
        pad=16,
    )
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", frameon=False)

    max_total = float(totals.max()) if len(totals) else 1.0
    ax.set_xlim(0, max_total * 1.18)
    for position, total in zip(y, totals):
        ax.text(
            total + max_total * 0.015,
            position,
            f"{total:.1%}",
            va="center",
            fontsize=9,
            color="#333333",
        )

    parameter_text = ", ".join(f"{key}={value}" for key, value in params.items())
    fig.text(
        0.5,
        0.012,
        "Fit: PCC-aligned HC3 validation only; "
        f"XGBoost parameters from v2 ({parameter_text}). "
        "Each motif combines its MF and WAD gain shares.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension, file_format in (("png", "png"), ("svg", "svg")):
        # Inventory names contain dots (for example ``deu.rst.pcc``), so
        # Path.with_suffix() would incorrectly discard the final component.
        destination = Path(f"{output_stem}.{extension}")
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            fig.savefig(
                temporary,
                format=file_format,
                dpi=200,
                bbox_inches="tight",
                facecolor="white",
            )
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = (
        "inventory",
        "rank",
        "motif_index",
        "motif_hash",
        "total_importance",
        "mf_importance",
        "wad_importance",
    )
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows({key: row[key] for key in columns} for row in rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--report", type=Path, default=Path("motif_label_dependency_report.v2.txt")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")

    analysis = _load_analysis_module(
        Path(__file__).resolve().with_name("10_analyze_motif_label_dependency.py")
    )
    selected = _selected_params(args.report)
    all_rows: list[Dict[str, Any]] = []
    progress = tqdm(INVENTORIES, desc="M3 importance plots", unit="inventory")
    for inventory in progress:
        progress.set_postfix_str(inventory, refresh=True)
        manifest_path = _manifest_path(args.data_dir, inventory)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validation_path = (
            args.data_dir / "unirst" / f"rel-{inventory}" / VALIDATION_FILE
        )
        corpus = analysis.load_corpus(
            validation_path,
            manifest,
            progress_desc=f"load {inventory}",
        )
        X, names = analysis._config_matrix(corpus, "m3")
        model = analysis._model(
            "xgb", selected[inventory], args.seed, args.n_jobs
        )
        model.fit(X, corpus.y)
        rows = _aggregate_importance(
            names, np.asarray(model.feature_importances_), args.top
        )
        for row in rows:
            row["inventory"] = inventory
        all_rows.extend(rows)
        output_stem = args.output_dir / f"m3_only_xgb_top{args.top}_{inventory}"
        _save_plot(rows, inventory, selected[inventory], output_stem)
        tqdm.write(f"saved {output_stem}.png and {output_stem}.svg")

    csv_path = args.output_dir / f"m3_only_xgb_top{args.top}_importance.csv"
    _write_csv(csv_path, all_rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
