#!/usr/bin/env python3
"""Build, resume, verify, and report the canonical four-standard experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from tos.analysis.granularity import granularity_dir


DEFAULT_CONFIG = Path("configs/canonical.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"inventories", "paths", "settings", "expected"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"Configuration misses: {sorted(missing)}")
    return value


def module_command(module: str, *arguments: object) -> list[str]:
    return [sys.executable, "-m", module, *(str(value) for value in arguments)]


def stage_commands(config: dict[str, Any], resume: bool) -> list[tuple[str, list[str]]]:
    paths = config["paths"]
    settings = config["settings"]
    inventories = config["inventories"]
    root = Path(paths["artifact_root"])
    raw = Path(paths["raw_feature_root"])
    common = ["--inventories", *inventories]
    force = [] if resume else ["--force"]
    cohort_mode = ["--reuse-existing"] if resume else []
    bootstrap = [
        "--iterations", settings["bootstrap_iterations"],
        "--batch-size", settings["bootstrap_batch_size"],
    ]
    commands: list[tuple[str, list[str]]] = [
        ("raw feature matrices", module_command(
            "scripts.canonical.build_feature_matrices",
            "--root", paths["data_root"], "--motif-root", paths["motif_root"],
            "--output-dir", raw, *common, *force,
        )),
        ("matched cohort", module_command(
            "scripts.canonical.prepare_canonical_cohort",
            "--source-dir", raw, "--motif-dir", paths["motif_root"],
            "--output-dir", root / "feature_matrices", *common, *cohort_mode,
        )),
        ("pooled relation frequencies", module_command(
            "scripts.canonical.aggregate_relation_frequencies",
            "--matrix-root", root / "feature_matrices",
            "--output-root", root / "relation_frequencies",
        )),
        ("feature models", module_command(
            "scripts.canonical.evaluate_feature_models",
            "--feature-dir", root / "feature_matrices",
            "--output-dir", root / "model_comparison", *common,
            "--seed", settings["model_seed"], "--n-jobs", settings["n_jobs"], *force,
        )),
        ("paired comparisons", module_command(
            "scripts.canonical.compare_paired_performance",
            "--prediction-dir", root / "model_comparison",
            "--output-dir", root / "paired_comparison", *common, *bootstrap,
            "--seed", settings["bootstrap_seed"], *force,
        )),
        ("motif associations", module_command(
            "scripts.canonical.analyze_motif_associations",
            "--feature-dir", root / "feature_matrices",
            "--model-dir", root / "model_comparison", "--motif-dir", paths["motif_root"],
            "--output-dir", root / "motif_associations", *common, *bootstrap,
            "--seed", settings["bootstrap_seed"], *force,
        )),
        ("robustness", module_command(
            "scripts.canonical.run_robustness_analysis",
            "--feature-dir", root / "feature_matrices",
            "--prediction-dir", root / "model_comparison",
            "--association-dir", root / "motif_associations",
            "--output-dir", root / "robustness", *common, *bootstrap,
            "--seed", settings["bootstrap_seed"], *force,
        )),
        ("cross-standard mapping", module_command(
            "scripts.canonical.map_cross_standard_motifs",
            "--motif-dir", paths["motif_root"],
            "--stability-file", root / "robustness" / "motif_stability.csv",
            "--output-dir", root / "cross_standard_mapping", *force,
        )),
        ("PCC/NLDT granularity", module_command(
            "scripts.canonical.ablate_m3_granularity",
            "--feature-root", root / "feature_matrices", "--motif-root", paths["motif_root"],
            "--full-model-root", root / "model_comparison",
            "--output-root", root / "granularity" / granularity_dir("deu.rst.pcc"),
            *bootstrap, "--seed", settings["model_seed"], "--n-jobs", settings["n_jobs"],
        )),
        ("eRST granularity", module_command(
            "scripts.canonical.ablate_full_inventory_granularity",
            "--inventory", "eng.erst.gum.new",
            "--feature-root", root / "feature_matrices", "--motif-root", paths["motif_root"],
            "--fine-model-root", root / "model_comparison",
            "--output-dir", root / "granularity" / granularity_dir("eng.erst.gum.new"), *bootstrap,
            "--seed", settings["model_seed"], "--n-jobs", settings["n_jobs"],
        )),
        ("RST-DT granularity", module_command(
            "scripts.canonical.ablate_full_inventory_granularity",
            "--inventory", "eng.rst.rstdt.new", "--mapping-json", paths["rstdt_mapping"],
            "--feature-root", root / "feature_matrices", "--motif-root", paths["motif_root"],
            "--fine-model-root", root / "model_comparison",
            "--output-dir", root / "granularity" / granularity_dir("eng.rst.rstdt.new"), *bootstrap,
            "--seed", settings["model_seed"], "--n-jobs", settings["n_jobs"],
        )),
        ("RF MF+WAD granularity", module_command(
            "scripts.canonical.compare_rf_granularity",
            "--feature-root", root / "feature_matrices",
            "--granularity-root", root / "granularity",
            "--mapping-file", root / "granularity" / granularity_dir("deu.rst.pcc") / "motif_mapping.csv",
            "--output-dir", root / "granularity" / "rf_mf_wad",
            *bootstrap, "--seed", settings["model_seed"],
            "--bootstrap-seed", settings["bootstrap_seed"],
            "--n-jobs", settings["n_jobs"], *force,
        )),
        ("LR MF+WAD granularity", module_command(
            "scripts.canonical.compare_lr_wad_granularity",
            "--feature-root", root / "feature_matrices",
            "--granularity-root", root / "granularity",
            "--mapping-file", root / "granularity" / granularity_dir("deu.rst.pcc") / "motif_mapping.csv",
            "--prediction-root", root / "model_comparison",
            "--output-dir", root / "granularity" / "lr_mf_wad",
            *bootstrap,
            "--seed", settings["model_seed"],
            "--bootstrap-seed", settings["bootstrap_seed"],
            "--n-jobs", settings["n_jobs"], *force,
        )),
        ("motif distributions", module_command(
            "scripts.canonical.compare_motif_distributions",
            "--feature-root", root / "feature_matrices", "--motif-root", paths["motif_root"],
            "--mapping-members", root / "cross_standard_mapping" / "members.csv",
            "--cohort-manifest", root / "feature_matrices" / "manifest.json",
            "--output-dir", root / "motif_distributions",
            "--bootstrap-iterations", settings["bootstrap_iterations"],
            "--bootstrap-batch-size", 200, "--seed", settings["distribution_seed"],
        )),
        ("domain distinctions", module_command(
            "scripts.canonical.analyze_domain_motif_distinctions",
            "--feature-dir", root / "feature_matrices", "--motif-dir", paths["motif_root"],
            "--family-members", root / "cross_standard_mapping" / "members.csv",
            "--coarse-mapping", root / "granularity" / "pcc_nldt" / "motif_mapping.csv",
            "--output-dir", root / "domain_distinctions", *bootstrap,
            "--seed", settings["domain_seed"],
        )),
        ("figures", module_command(
            "scripts.canonical.visualize_results",
            "--model-dir", root / "model_comparison", "--paired-dir", root / "paired_comparison",
            "--robustness-dir", root / "robustness",
            "--mapping-dir", root / "cross_standard_mapping",
            "--output-dir", root / "figures", "--dpi", settings["figure_dpi"], *force,
        )),
    ]
    return commands


def run_stages(config: dict[str, Any], resume: bool) -> None:
    for name, command in stage_commands(config, resume):
        print(f"\n== {name} ==", flush=True)
        subprocess.run(command, check=True)
    run_report(config)
    write_run_manifest(config)


def run_report(config: dict[str, Any]) -> None:
    root = config["paths"]["artifact_root"]
    subprocess.run(module_command(
        "scripts.canonical.synthesize_results", "--artifact-root", root,
        "--report", "UNIRST2_RESEARCH_REPORT.md",
        "--tables", "PAPER_RELEVANT_RESULTS_TABLES.md",
    ), check=True)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def headline(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["paths"]["artifact_root"])
    cohort = json.loads((root / "feature_matrices" / "manifest.json").read_text())
    return {
        "rows": {split: cohort["splits"][split]["common_rows"] for split in ("train", "validation", "test")},
        "model_summary_rows": len(csv_rows(root / "model_comparison" / "summary.csv")),
        "paired_summary_rows": len(csv_rows(root / "paired_comparison" / "summary.csv")),
        "stability_rows": len(csv_rows(root / "motif_associations" / "stability.csv")),
        "robustness_model_rows": len(csv_rows(root / "robustness" / "model_metrics.csv")),
        "rf_granularity_rows": len(csv_rows(root / "granularity" / "rf_mf_wad" / "paired_comparisons.csv")),
        "lr_wad_granularity_rows": len(csv_rows(root / "granularity" / "lr_mf_wad" / "paired_comparisons.csv")),
    }


def artifact_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path not in {
            root / "manifest.json",
            root / "frozen_pre_refactor_manifest.json",
        }:
            yield path


def git_revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def manifest_value(config: dict[str, Any], kind: str) -> dict[str, Any]:
    root = Path(config["paths"]["artifact_root"])
    source_paths = [
        Path("run_canonical.py"),
        Path("configs/canonical.json"),
        Path(config["paths"]["rstdt_mapping"]),
        *sorted(Path("scripts/canonical").glob("*.py")),
        *sorted(Path("tos").glob("*.py")),
        *sorted(Path("tos/analysis").glob("*.py")),
    ]
    package_versions = {}
    for package in ("numpy", "scipy", "pandas", "scikit-learn", "xgboost", "networkx", "matplotlib"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    parents = [
        Path(config["paths"]["raw_feature_root"]) / "manifest.json",
        root / "feature_matrices" / "manifest.json",
    ]
    return {
        "schema_version": 1,
        "kind": kind,
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions,
        "config": config,
        "config_sha256": sha256_file(DEFAULT_CONFIG),
        "commands": {name: command for name, command in stage_commands(config, resume=False)},
        "sources": {str(path): sha256_file(path) for path in source_paths},
        "parent_manifests": {str(path): sha256_file(path) for path in parents},
        "headline": headline(config),
        "files": {
            str(path.relative_to(root)): {"size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in artifact_files(root)
        },
    }


def write_run_manifest(config: dict[str, Any]) -> None:
    root = Path(config["paths"]["artifact_root"])
    atomic_json(root / "manifest.json", manifest_value(config, "canonical_run"))


def freeze(config: dict[str, Any]) -> None:
    root = Path(config["paths"]["artifact_root"])
    atomic_json(root / "frozen_pre_refactor_manifest.json", manifest_value(config, "pre_refactor_snapshot"))
    print(f"wrote {root / 'frozen_pre_refactor_manifest.json'}")


def verify(config: dict[str, Any]) -> None:
    root = Path(config["paths"]["artifact_root"])
    observed = headline(config)
    if observed != config["expected"]:
        raise ValueError(f"Headline mismatch: expected {config['expected']}, observed {observed}")
    inventories = set(config["inventories"])
    for relative, column in (
        ("model_comparison/summary.csv", "inventory"),
        ("paired_comparison/summary.csv", "inventory"),
        ("motif_associations/stability.csv", "inventory"),
        ("robustness/model_metrics.csv", "inventory"),
        ("granularity/rf_mf_wad/paired_comparisons.csv", "inventory"),
        ("granularity/lr_mf_wad/paired_comparisons.csv", "inventory"),
    ):
        values = {row[column] for row in csv_rows(root / relative)}
        if values != inventories:
            raise ValueError(f"Inventory mismatch in {relative}: {sorted(values)}")
    for report in (Path("UNIRST2_RESEARCH_REPORT.md"), Path("PAPER_RELEVANT_RESULTS_TABLES.md"), root / "report.md"):
        text = report.read_text(encoding="utf-8").lower()
        forbidden = [token for token in ("out-of-domain", "human detectors", "mage-ood") if token in text]
        if forbidden:
            raise ValueError(f"Excluded outcomes occur in {report}: {forbidden}")
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stored.get("config") != config:
            raise ValueError("Canonical manifest configuration differs from the requested configuration")
        current_files = {str(path.relative_to(root)) for path in artifact_files(root)}
        recorded_files = set(stored.get("files", {}))
        if current_files != recorded_files:
            raise ValueError(
                f"Canonical file inventory differs: missing={sorted(recorded_files - current_files)}, "
                f"unrecorded={sorted(current_files - recorded_files)}"
            )
        current_manifest = manifest_value(config, stored.get("kind", "canonical_run"))
        for key in ("config_sha256", "sources", "parent_manifests"):
            if stored.get(key) != current_manifest.get(key):
                raise ValueError(f"Canonical manifest {key} differs from the active workflow")
        for relative, record in stored.get("files", {}).items():
            path = root / relative
            if not path.is_file() or path.stat().st_size != record["size"] or sha256_file(path) != record["sha256"]:
                raise ValueError(f"Artifact verification failed: {path}")
    print(f"verified canonical run: {observed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "resume", "verify", "report", "freeze"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.artifact_root is not None:
        config["paths"]["artifact_root"] = str(args.artifact_root)
    if args.command == "build":
        run_stages(config, resume=False)
    elif args.command == "resume":
        run_stages(config, resume=True)
    elif args.command == "verify":
        verify(config)
    elif args.command == "report":
        run_report(config)
        write_run_manifest(config)
    else:
        freeze(config)


if __name__ == "__main__":
    main()
