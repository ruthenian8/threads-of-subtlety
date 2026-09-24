#!/usr/bin/env python3
"""Generate the canonical four-standard research report and paper tables."""

from __future__ import annotations

import csv
import argparse
import json
import os
from pathlib import Path


DEFAULT_ROOT = Path("artifacts/canonical")
INVENTORIES = (
    ("deu.rst.pcc", "PCC"),
    ("eng.erst.gum.new", "eRST"),
    ("eng.rst.rstdt.new", "RST-DT"),
    ("nld.rst.nldt", "NLDT"),
)
CONFIGS = ("structural", "relation_frequency", "m3_presence", "m3_frequency", "m3_depth")
CONFIG_LABELS = {
    "structural": "Structural",
    "relation_frequency": "Relation frequency",
    "m3_presence": "M3 presence",
    "m3_frequency": "M3 frequency",
    "m3_depth": "M3 depth",
}


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prose_list(values):
    values = list(values)
    if not values:
        return "none"
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def model_table(model_rows):
    lines = [
        "| Standard | Model | Structural | Relation frequency | M3 presence | M3 frequency | M3 depth |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for inventory, label in INVENTORIES:
        for model in ("lr", "xgb"):
            values = {
                row["feature_config"]: float(row["test_f1"])
                for row in model_rows
                if row["inventory"] == inventory and row["model"] == model
            }
            cells = [f"{values[config]:.3f}" for config in CONFIGS]
            lines.append(f"| {label} | {model.upper()} | " + " | ".join(cells) + " |")
    return lines


def paired_table(paired_rows):
    lines = [
        "| Standard | Model | M3 frequency − structure F1 | 95% CI | Holm-adjusted value |",
        "|---|---|---:|---:|---:|",
    ]
    labels = dict(INVENTORIES)
    selected = [
        row for row in paired_rows
        if row["contrast"] == "frequency_vs_structural" and row["metric"] == "f1"
    ]
    for row in selected:
        lines.append(
            f"| {labels[row['inventory']]} | {row['model'].upper()} | "
            f"{float(row['difference']):+.3f} | "
            f"[{float(row['ci_low']):+.3f}, {float(row['ci_high']):+.3f}] | "
            f"{float(row['holm_adjusted_pvalue']):.4f} |"
        )
    return lines


def granularity_rows(root: Path):
    result = [
        {
            "inventory": row["inventory"],
            "model": row["classifier"],
            "representation": row["representation"],
            "fine": float(row["fine_score"]),
            "coarse": float(row["coarse_score"]),
            "difference": float(row["difference"]),
            "low": float(row["ci_low"]),
            "high": float(row["ci_high"]),
            "adjusted": float(row["holm_adjusted_pvalue"]),
        }
        for row in rows(root / "granularity" / "rf_mf_wad" / "paired_comparisons.csv")
        if row["cohort"] == "primary"
        and row["metric"] == "f1"
        and row["representation"] == "mf_wad"
        and row["classifier"] == "rf"
    ]
    order = {inventory: index for index, (inventory, _) in enumerate(INVENTORIES)}
    return sorted(result, key=lambda row: order[row["inventory"]])


def lr_wad_granularity_rows(root: Path):
    result = [
        {
            "inventory": row["inventory"],
            "representation": row["representation"],
            "fine": float(row["fine_score"]),
            "coarse": float(row["coarse_score"]),
            "difference": float(row["difference"]),
            "low": float(row["ci_low"]),
            "high": float(row["ci_high"]),
            "adjusted": float(row["holm_adjusted_pvalue"]),
        }
        for row in rows(root / "granularity" / "lr_mf_wad" / "paired_comparisons.csv")
        if row["cohort"] == "primary" and row["metric"] == "f1"
    ]
    interactions = [
        row for row in rows(root / "granularity" / "lr_mf_wad" / "mf_gain_interaction.csv")
        if row["cohort"] == "primary" and row["metric"] == "f1"
    ]
    order = {inventory: index for index, (inventory, _) in enumerate(INVENTORIES)}
    return (
        sorted(result, key=lambda row: (order[row["inventory"]], row["representation"])),
        sorted(interactions, key=lambda row: order[row["inventory"]]),
    )


def granularity_table(values):
    labels = dict(INVENTORIES)
    lines = [
        "| Standard | Fine MF+WAD F1 | Coarse MF+WAD F1 | Fine − coarse (95% CI) | Holm-adjusted p |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in values:
        lines.append(
            f"| {labels[row['inventory']]} | {row['fine']:.3f} | "
            f"{row['coarse']:.3f} | {row['difference']:+.3f} "
            f"[{row['low']:+.3f}, {row['high']:+.3f}] | {row['adjusted']:.4f} |"
        )
    return lines


def lr_granularity_table(values, interactions):
    labels = dict(INVENTORIES)
    lines = [
        "| Standard | Representation | Fine F1 | Coarse F1 | Fine − coarse (95% CI) | Holm-adjusted p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in values:
        lines.append(
            f"| {labels[row['inventory']]} | {row['representation'].upper()} | "
            f"{row['fine']:.3f} | {row['coarse']:.3f} | {row['difference']:+.3f} "
            f"[{row['low']:+.3f}, {row['high']:+.3f}] | {row['adjusted']:.4f} |"
        )
    lines.extend([
        "",
        "Paired change in the fine-minus-coarse gain after adding WAD:",
        "",
        "| Standard | (MF+WAD gain) − (MF gain), F1 (95% CI) | Holm-adjusted p |",
        "|---|---:|---:|",
    ])
    for row in interactions:
        lines.append(
            f"| {labels[row['inventory']]} | {float(row['delta_gain_mf_wad_minus_mf']):+.3f} "
            f"[{float(row['ci_low']):+.3f}, {float(row['ci_high']):+.3f}] | "
            f"{float(row['holm_adjusted_pvalue']):.4f} |"
        )
    return lines


def stable_table(stability):
    lines = [
        "| Standard | Presence | Frequency | Conditional depth |",
        "|---|---:|---:|---:|",
    ]
    for inventory, label in INVENTORIES:
        counts = {
            representation: sum(
                row["inventory"] == inventory
                and row["representation"] == representation
                and row["stable"] == "True"
                for row in stability
            )
            for representation in ("presence", "frequency", "depth")
        }
        lines.append(f"| {label} | {counts['presence']} | {counts['frequency']} | {counts['depth']} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=Path("UNIRST2_RESEARCH_REPORT.md"))
    parser.add_argument("--tables", type=Path, default=Path("PAPER_RELEVANT_RESULTS_TABLES.md"))
    args = parser.parse_args()
    root = args.artifact_root
    model_rows = rows(root / "model_comparison" / "summary.csv")
    paired_rows = rows(root / "paired_comparison" / "summary.csv")
    stability = rows(root / "motif_associations" / "stability.csv")
    granular = granularity_rows(root)
    lr_granular, lr_interactions = lr_wad_granularity_rows(root)
    cohort = json.loads((root / "feature_matrices" / "manifest.json").read_text(encoding="utf-8"))
    mapping = json.loads((root / "cross_standard_mapping" / "manifest.json").read_text(encoding="utf-8"))
    domain = json.loads((root / "domain_distinctions" / "manifest.json").read_text(encoding="utf-8"))
    provenance_path = root / "shared_segmentation_provenance.json"
    shared_segmentation = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else None
    dimensions = {}
    for inventory, _ in INVENTORIES:
        schema = next(
            row for row in model_rows
            if row["inventory"] == inventory and row["model"] == "lr" and row["feature_config"] == "m3_frequency"
        )
        relation = next(
            row for row in model_rows
            if row["inventory"] == inventory and row["model"] == "lr" and row["feature_config"] == "relation_frequency"
        )
        dimensions[inventory] = (int(relation["feature_count"]), int(schema["feature_count"]))

    split_counts = {
        split: int(cohort["splits"][split]["common_rows"])
        for split in ("train", "validation", "test")
    }
    xgb_m3_f1 = [
        float(row["test_f1"])
        for row in model_rows
        if row["model"] == "xgb" and row["feature_config"] == "m3_frequency"
    ]
    xgb_frequency_gains = [
        float(row["difference"])
        for row in paired_rows
        if row["model"] == "xgb"
        and row["contrast"] == "frequency_vs_structural"
        and row["metric"] == "f1"
    ]
    granularity_gains = [row["difference"] for row in granular]
    lr_mf_wad_gains = [row["difference"] for row in lr_granular if row["representation"] == "mf_wad"]
    granularity_clear = [
        dict(INVENTORIES)[row["inventory"]] for row in granular if row["low"] > 0
    ]
    granularity_adjusted = [
        dict(INVENTORIES)[row["inventory"]]
        for row in granular if row["adjusted"] < 0.05
    ]
    granularity_uncertain = [
        dict(INVENTORIES)[row["inventory"]] for row in granular if row["low"] <= 0 <= row["high"]
    ]
    lr_wad_adjusted = [
        dict(INVENTORIES)[row["inventory"]]
        for row in lr_granular
        if row["representation"] == "mf_wad" and row["adjusted"] < 0.05
    ]
    granularity_conclusion = (
        "For Kim-style RF, MF+WAD fine-minus-coarse gains are positive in all standards; "
        f"pointwise intervals exclude zero for {prose_list(granularity_clear)}, while only "
        f"{prose_list(granularity_adjusted)} survives Holm correction. For LR with MF+WAD, "
        f"Holm-adjusted tests support positive gains in {prose_list(lr_wad_adjusted)}."
    )
    stable_counts = {
        representation: sum(
            row["representation"] == representation and row["stable"] == "True"
            for row in stability
        )
        for representation in ("presence", "frequency", "depth")
    }
    stable_total = sum(stable_counts.values())
    segmentation_statement = (
        "All four inventories use the same EDU segmentation. PCC and NLDT define the shared "
        "segmentation target; the eRST and RST-DT inputs were independently verified against it "
        f"with {shared_segmentation['verification']['recoverable_exact_scenes']:,} recoverable scenes "
        "matching exactly and no boundary mismatches."
        if shared_segmentation else
        "All four inventories use the same document identities; segmentation provenance is recorded in the canonical artifacts."
    )

    table_lines = [
        "# Paper-relevant results tables: canonical four-standard analysis", "",
        "These tables use the exact same HC3 identities and shared EDU segmentation for PCC, full eRST, full RST-DT, and NLDT. "
        f"Models are trained on {split_counts['train']:,} documents, selected on {split_counts['validation']:,} validation documents, and evaluated on {split_counts['test']:,} test documents. "
        "Alternative-segmentation runs are retained only as sensitivity artifacts and are not counted as additional standards.", "",
        "## Research questions", "",
        "1. Do M3 motifs add authorship-related information beyond document structure and marginal relation frequency?",
        "2. Does fine relation resolution retain predictive information that is lost under fixed-tree coarse projection?",
        "3. Are motif associations stable in sign across HC3 domains, and are presence and depth empirically distinct?",
        "4. Which effects overlap across standards and which depend on inventory-specific distinctions?", "",
        "## Table 1. Canonical cohort and feature dimensions", "",
        "| Standard | Train | Validation | Test | Relation features | M3 features |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for inventory, label in INVENTORIES:
        relation_count, motif_count = dimensions[inventory]
        table_lines.append(
            f"| {label} | {split_counts['train']:,} | {split_counts['validation']:,} | "
            f"{split_counts['test']:,} | {relation_count} | {motif_count} |"
        )
    table_lines += ["", "## Table 2. HC3 test F1", "", *model_table(model_rows), "", "## Table 3. Paired increment of M3 frequency over structure (F1)", "", *paired_table(paired_rows), "", "## Table 4. Fixed-tree fine-to-coarse RF MF+WAD ablation (F1)", "", *granularity_table(granular), "", "## Table 5. Fixed-tree fine-to-coarse LR ablation with MF and MF+WAD (F1)", "", *lr_granularity_table(lr_granular, lr_interactions), "", "The paired interaction tests whether adding WAD changes LR's fine-minus-coarse gain, using the same identity-cluster bootstrap resamples for MF and MF+WAD. Coarse WAD is occurrence-weighted when fine motifs merge. RF retains its Kim-style fixed 0.5 threshold; LR uses the established validation-selected threshold.", "", "## Table 6. Stable held-out motif associations", "", *stable_table(stability), "", "## Table 7. Cross-standard mapping summary", "", f"The four catalogs contain {mapping['catalog_entries']} inventory-specific motifs, grouped into {mapping['pipeline_families']} pipeline-equivalent and {mapping['structural_families']} relation-agnostic structural families. Structural families—not raw hashes—are the defensible unit for broad cross-standard comparison.", "", "## Table 8. Domain-conditioned analysis", "", f"Training selected {domain['candidate_hypotheses']} domain–motif hypotheses; {domain['replicated_primary']} replicated on test and {domain['robust_both_cohorts']} also survived the identity-clean sensitivity cohort. {domain['shared_family_interactions']} exact-label motif families produced recurring robust domain interactions in multiple standards. This supports substantial inventory specificity alongside a smaller shared structural signal.", "", "## Segmentation target", "", segmentation_statement, "", "## Scope", "", "F1 for label 1 (the human class) is the primary reported predictive metric. The LR/XGBoost representation models and LR granularity fits retain ROC-AUC validation selection and validation-only threshold selection. The RF granularity comparison uses Kim-style fixed 100-tree random forests over MF+WAD with a fixed 0.5 threshold. No test-set tuning was performed. All outcomes are parser-conditional measurements on matched HC3 documents. They compare measurement consequences of relation inventories; they do not rank linguistic correctness or parser quality.", ""]

    report_lines = [
        "# Canonical UniRST2 four-standard research report", "",
        "## Executive summary", "",
        "The consolidated experiment compares PCC, full eRST, full RST-DT, and NLDT on exactly the same train, validation, and test identities and on a shared EDU segmentation target. This removes segmentation as a source of cross-standard variation in the primary comparison.", "",
        f"M3-frequency XGBoost test F1 ranges from {min(xgb_m3_f1):.3f} to {max(xgb_m3_f1):.3f}. Against structural features, its paired F1 gain ranges from {min(xgb_frequency_gains):.3f} to {max(xgb_frequency_gains):.3f}.", "",
        f"The fixed-tree LR MF+WAD projection gives a fine-minus-coarse F1 difference of {min(lr_mf_wad_gains):+.3f} to {max(lr_mf_wad_gains):+.3f}; the Kim-style RF MF+WAD comparison gives {min(granularity_gains):+.3f} to {max(granularity_gains):+.3f}.", "",
        f"The stable-association screen finds {stable_total} stable candidate–representation associations: {stable_counts['presence']} presence, {stable_counts['frequency']} frequency, and {stable_counts['depth']} conditional-depth effects.", "",
        "## Canonical design", "",
        f"The common cohort contains {cohort['splits']['train']['common_rows']:,} train, {cohort['splits']['validation']['common_rows']:,} validation, and {cohort['splits']['test']['common_rows']:,} test documents. Identity order and labels are exact across all four matrices. Catalogs remain training-only. Binary F1 for label 1 (human) is reported for comparability with Kim et al. LR/XGBoost models use validation ROC-AUC selection and validation-only thresholds; the Kim-style fine/coarse random forests use the fixed 0.5 threshold.", "",
        "## Predictive results", "", *model_table(model_rows), "",
        "![Canonical model comparison](docs/figures/canonical/model_f1_comparison.png)", "",
        "## Paired representation effects", "", *paired_table(paired_rows), "",
        "![Paired frequency gain](docs/figures/canonical/paired_frequency_f1_gain.png)", "",
        "## Fixed-tree granularity", "", "LR with MF and MF+WAD", "", *lr_granularity_table(lr_granular, lr_interactions), "",
        "The paired MF versus MF+WAD contrast estimates whether adding depth changes LR's fine-minus-coarse gain. Both use the same model selection, threshold selection, matched test identities, and bootstrap resamples.", "",
        "Kim-style RF with MF+WAD", "", *granularity_table(granular), "",
        "The RF comparison follows Kim et al.'s motif-classifier feature construction and uses a fixed 0.5 threshold. Coarse WAD is occurrence-weighted from its merged fine motifs. Across both LR representations and RF, documents, segmentation, tree topology, attachments, nuclearity handling, and motif occurrence mass remain fixed; only relation labels and projected motif dimensions change.", "",
        "## Stable motif associations", "", *stable_table(stability), "",
        "The table summarizes the primary test cohort. The following figures use the identity-clean sensitivity cohort and its stability screen.", "",
        "![Stable motif effects](docs/figures/canonical/stable_motif_domain_effects.png)", "",
        "![Stable presence odds ratios](docs/figures/canonical/stable_presence_odds_ratios.png)", "",
        "The odds ratios compare motif-presence odds in human and generated documents within each domain. The color scale is centered on OR = 1; a 0.5 count correction is applied only if a two-by-two table has a zero cell.", "",
        "![Relaxed 3-of-5 presence odds ratios](docs/figures/canonical/relaxed_presence_odds_ratios_3of5.png)", "",
        "The relaxed sensitivity view requires robust same-direction presence effects in at least three of five domains, with no robust effect in the opposite direction; it is not a replacement for the primary stability screen.", "",
        "## Cross-standard and domain interpretation", "",
        f"The catalogs contain {mapping['catalog_entries']} inventory-specific motifs. Exact relation-labelled overlap is strongest between inventories that share labels, while all four standards occupy common relation-agnostic structural families. Domain analysis retained {domain['robust_both_cohorts']} robust identity-clean interactions, with {domain['shared_family_interactions']} recurring exact-label families across multiple standards. Thus the standards largely agree on structural templates while assigning much of the discriminative detail through inventory-specific relation distinctions.", "",
        "![Cross-standard family overlap](docs/figures/canonical/cross_standard_family_overlap.png)", "",
        "## Conclusions", "",
        "1. Local discourse motifs contribute information beyond document-scale structure across all four standards.",
        f"2. {granularity_conclusion}",
        "3. Stable signal is associated with motif occurrence and frequency, not conditional depth.",
        "4. Cross-standard overlap is clearest at the structural-family level; exact relation-labelled motifs are often inventory-specific.",
        "5. These results concern the consequences of measurement choices, not the ontological superiority of a relation inventory.", "",
        "## Reproducibility", "",
        "Canonical artifacts are under `artifacts/canonical`. The common-cohort manifest records every source and artifact hash; superseded and side-study outputs are indexed under `archive/`.", "",
    ]
    tables = "\n".join(table_lines)
    report = "\n".join(report_lines)
    atomic_text(args.tables, tables)
    atomic_text(args.report, report)
    atomic_text(root / "report.md", report.replace("(docs/figures/", "(../../docs/figures/"))
    print("wrote canonical reports")


if __name__ == "__main__":
    main()
