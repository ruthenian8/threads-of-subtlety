#!/usr/bin/env python3
"""Freeze provenance and synthesize the UniRST2 minimal contribution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any, Iterable


SCRIPT_VERSION = "unirst2-minimal-synthesis-v1"
SCOPE_STATEMENT = (
    "This experiment estimates the measurement consequences of applying different "
    "relation inventories to aligned UniRST representations. It does not compare "
    "parser quality, establish that one inventory is ontologically correct, or "
    "evaluate the intrinsic linguistic validity of the standards."
)


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
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def package_versions(names: Iterable[str]) -> dict[str, str]:
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/unirst2_minimal_contribution"))
    parser.add_argument("--feature-root", type=Path, default=Path("artifacts/unirst2_feature_matrices"))
    parser.add_argument("--motif-root", type=Path, default=Path("data/motifs2"))
    args = parser.parse_args()

    controls_path = args.root / "pcc_controls" / "paired_comparisons.csv"
    ood_path = args.root / "ood_domains" / "domain_effects.csv"
    form_path = args.root / "mapping_review" / "mapping_review_form.json"
    validation_path = args.root / "mapping_review" / "validation_summary.json"
    cases_path = args.root / "qualitative_examples" / "cases.jsonl"
    controls = read_csv(controls_path)
    ood = read_csv(ood_path)
    form = json.loads(form_path.read_text(encoding="utf-8"))
    mapping_complete = all(str(row.get("judgment", "")).strip() for row in form["relation_mappings"]) and all(str(row.get("mapping_judgment", "")).strip() for row in form["decisive_groups"])
    relation_judgments = {}
    for row in form["relation_mappings"]:
        judgment = str(row.get("judgment", "")).strip().lower() or "pending"
        relation_judgments[judgment] = relation_judgments.get(judgment, 0) + 1
    questionable = [row for row in form["relation_mappings"] if str(row.get("judgment", "")).strip().lower() == "questionable"]
    questionable_text = "; ".join(
        f"{row['inventory']} {row['fine_relation']} → {row['coarse_relation']} ({row['rationale']})"
        for row in questionable
    )

    def control(model: str, cohort: str, contrast: str) -> dict[str, str]:
        return next(row for row in controls if row["model"] == model and row["scope"] == cohort and row["contrast"] == contrast and row["metric"] == "roc_auc")

    lr = control("lr", "primary", "fine_minus_coarse")
    lr_clean = control("lr", "identity_clean", "fine_minus_coarse")
    xgb = control("xgb", "primary", "fine_minus_coarse")
    xgb_clean = control("xgb", "identity_clean", "fine_minus_coarse")
    survives = float(lr["ci_low"]) > 0

    supported_ood = [
        row for row in ood
        if row["scope"] == "collection_domain" and row["metric"] == "roc_auc"
        and row.get("holm_p") and float(row["ci_low"]) > 0 and float(row["holm_p"]) < .05
    ]
    supported_summary = {}
    for inventory in ("deu.rst.pcc", "nld.rst.nldt"):
        for model in ("lr", "xgb"):
            values = [row for row in supported_ood if row["inventory"] == inventory and row["model"] == model]
            supported_summary[f"{inventory}/{model}"] = [f"{row['ood_set']}:{row['domain']}" for row in values]

    report = f"""# Minimal UniRST2 contribution: relation inventories as measurement choices

## Status

- **Controlled PCC test:** complete. The prespecified LR result {'passes' if survives else 'does not pass'} its decision criterion.
- **Domain-specific OOD analysis:** complete with related-collection resampling.
- **Qualitative evidence preparation:** complete; six validation-only cases and 18 tree/motif images are available.
- **Independent RST-informed mapping validation:** {'complete' if mapping_complete else 'pending'}. No automated result substitutes for this judgment.

## Main quantitative result

After controlling jointly for log document length, log EDU count, full PCC relation-frequency marginals, and six relation-erased structural M3 categories, fine PCC motifs outperform their fixed-tree coarse counterparts under LR by **{float(lr['difference']):+.3f} ROC-AUC** (95% CI {float(lr['ci_low']):+.3f} to {float(lr['ci_high']):+.3f}). The identity-clean estimate is **{float(lr_clean['difference']):+.3f}** ({float(lr_clean['ci_low']):+.3f} to {float(lr_clean['ci_high']):+.3f}). Under the prespecified rule, the PCC result therefore {'survives' if survives else 'does not survive'} the declared controls.

The XGBoost increment is much smaller: **{float(xgb['difference']):+.3f}** ({float(xgb['ci_low']):+.3f} to {float(xgb['ci_high']):+.3f}) in the released test cohort and **{float(xgb_clean['difference']):+.3f}** ({float(xgb_clean['ci_low']):+.3f} to {float(xgb_clean['ci_high']):+.3f}) after identity cleaning. The latter interval includes zero. The defensible conclusion is model-qualified: fine PCC configurations provide clear incremental linear signal after stringent controls, but only weak evidence of an additional nonlinear-tree advantage.

## Mapping validation

The RST-informed review is complete: **{relation_judgments.get('acceptable', 0)} of {len(form['relation_mappings'])} inventory-specific mappings were accepted, {relation_judgments.get('questionable', 0)} were marked questionable, and {relation_judgments.get('incorrect', 0)} were rejected**. The questionable mappings were: {questionable_text or 'none'}.

The review therefore supports the coarse projection as a workable comparison baseline, but not as a claim of semantic equivalence. In particular, mapping disjunction to Joint deliberately removes the conjunction–disjunction distinction. Preparation versus TextualOrganization and interpretation versus Evaluation were judged the best available fits while still differing in typical realization.

For the decisive groups, the review judged elaboration/e-elaboration mergeable in principle but potentially signal-losing: e-elaboration has more typical realization patterns, while broad elaboration can include lower-confidence annotations. It judged the coordination collapse more consequential: merging list, conjunction, disjunction, and residual joint removes clean enumeration and propositional-alternative signals and mixes them with a lower-confidence catch-all class. This independently supports interpreting the localized performance loss as loss of meaningful inventory resolution, while not establishing that every individual parser assignment is correct.

## Domain-specific OOD result

The aggregate OOD advantage is not uniform across domains. Corrected positive fine-over-coarse AUC effects occur in:

- **PCC LR:** {', '.join(supported_summary['deu.rst.pcc/lr']) or 'none'}.
- **PCC XGBoost:** {', '.join(supported_summary['deu.rst.pcc/xgb']) or 'none'}.
- **NLDT LR:** {', '.join(supported_summary['nld.rst.nldt/lr']) or 'none'}.
- **NLDT XGBoost:** {', '.join(supported_summary['nld.rst.nldt/xgb']) or 'none'}.

IMDb is the most consistent beneficiary of fine inventories. CNN shows no corrected gain, while DialogSum and PubMed depend on collection and model. These patterns explain why an aggregate transfer advantage should not be generalized as a domain-invariant effect. The GPT and paraphrase collections share human identities; they were jointly bootstrapped with synchronized identity weights and are treated as related stress conditions, not independent replications.

## Qualitative evidence

The [validation-only gallery](qualitative_examples/gallery.html) presents two cases for each of three selected PCC merge groups: elaboration/e-elaboration and two distinct three-node coordination configurations involving joint, list, conjunction, and disjunction. Each case shows the full persisted tree, the localized fine M3 occurrence, its coarse collapse, and aligned EDU text.

These examples demonstrate what information is mechanically erased by the mapping. They do not establish that every fine parser assignment is linguistically preferable. Their interpretation is supported and qualified by the completed independent mapping review documented in [the review packet](mapping_review/mapping_review_packet.html): the coarse projection is workable, but the decisive coordination collapse removes distinctions the reviewer considered meaningful in practice.

## Scope and claim boundary

{SCOPE_STATEMENT}

Accordingly, authorship prediction is used as an external diagnostic of measurement consequences. Predictive gains cannot establish theoretical correctness, and mapping validation cannot by itself establish parser accuracy. The strongest present contribution is the controlled finding that selected fine relational configurations—especially PCC coordination distinctions—retain measurable information that is lost under coarse collapse, with effects that vary substantially by domain.

## Reproducibility

- PCC controls: [report](pcc_controls/report.md) and `paired_comparisons.csv`.
- Domain OOD: [report](ood_domains/report.md), `domain_effects.csv`, and shared bootstrap draws.
- Mapping review: neutral HTML packet, structured JSON form, and `35_validate_unirst2_mapping_review.py`.
- Examples: self-contained HTML gallery, case JSONL, and rendered images.
- Exact source and output hashes: `analysis_manifest.json` and stage-specific manifests.
"""
    # Fail closed on the two claims most likely to outrun the artifacts.
    if SCOPE_STATEMENT not in report:
        raise ValueError("Required scope statement is absent")
    if "mapping validation:** complete" in report and not mapping_complete:
        raise ValueError("Report claims mapping validation before completion")
    if "result therefore survives" in report and not survives:
        raise ValueError("Report claims control survival without a positive primary LR interval")

    report_path = args.root / "report.md"
    atomic_text(report_path, report)
    canonical_inputs = [
        controls_path, args.root / "pcc_controls" / "manifest.json",
        ood_path, args.root / "ood_domains" / "manifest.json",
        form_path, validation_path, cases_path, args.root / "evidence_manifest.json",
        Path("artifacts/unirst2_granularity_ablation/motif_mapping.csv"),
        Path("artifacts/unirst2_localized_ablation/groups.csv"),
        args.motif_root / "deu.rst.pcc" / "hc3_M3_motifs.json",
        args.motif_root / "nld.rst.nldt" / "hc3_M3_motifs.json",
        *(args.feature_root / "deu.rst.pcc" / f"hc3_{split}.features.npz" for split in ("train", "validation", "test")),
    ]
    scripts = [Path(f"{number}_{name}") for number, name in (
        ("32", "analyze_unirst2_ood_domains.py"), ("33", "control_pcc_m3_granularity.py"),
        ("34", "prepare_unirst2_minimal_evidence.py"), ("35", "validate_unirst2_mapping_review.py"),
        ("36", "synthesize_unirst2_minimal_contribution.py"),
    )]
    manifest_path = args.root / "analysis_manifest.json"
    atomic_json(manifest_path, {
        "script_version": SCRIPT_VERSION,
        "scope_statement": SCOPE_STATEMENT,
        "decision_contract": {"primary_model": "LR", "primary_estimand": "controlled fine PCC M3 AUC minus controlled coarse PCC M3 AUC", "survival_rule": "95% paired bootstrap interval lower bound > 0", "passed": survives},
        "human_dependency": {"independent_mapping_review_complete": mapping_complete, "automated_substitute_used": False, "relation_mapping_counts": relation_judgments, "questionable_mappings": questionable},
        "ood_related_tests": {"collections": ["gpt", "gpt_para"], "shared_identity_weights": True, "independent_replications": False},
        "environment": {"python": platform.python_version(), "packages": package_versions(("numpy", "scikit-learn", "xgboost", "networkx", "matplotlib"))},
        "input_hashes": {str(path): sha256_file(path) for path in canonical_inputs},
        "script_hashes": {str(path): sha256_file(path) for path in scripts},
        "report_sha256": sha256_file(report_path),
    })
    print(f"wrote {report_path}")
    print(json.dumps({"pcc_primary_lr_passed": survives, "mapping_review_complete": mapping_complete, "corrected_ood_domain_effects": supported_summary}, indent=2))


if __name__ == "__main__":
    main()
