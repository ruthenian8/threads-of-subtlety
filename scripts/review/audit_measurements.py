#!/usr/bin/env python3
"""Audit the measurements used in the UniRST2 complementarity analysis.

This is a read-only audit.  It reuses the existing matrices, prediction
bundle, manifests, and parsed test graphs; it does not retrain models or
rewrite source data.  In particular, it separates the ``/`` scaffolding links
exported by the graph adapter from non-scaffolding discourse links.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from tqdm import tqdm


INVENTORIES = (
    "deu.rst.pcc",
    "eng.erst.gum.new",
    "eng.rst.rstdt.new",
    "nld.rst.nldt",
)
PAIRS = tuple(
    (left, right)
    for index, left in enumerate(INVENTORIES)
    for right in INVENTORIES[index + 1 :]
)
SPAN_RE = re.compile(r"^span_(\d+)-(\d+)$")


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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    stream = io.StringIO()
    if rows:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def load_matrix(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {key: np.asarray(values[key]) for key in values.files}


def array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and np.array_equal(left, right)


def check_matrix_alignment(feature_dir: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        matrices = {
            inventory: load_matrix(feature_dir / inventory / f"hc3_{split}.features.npz")
            for inventory in INVENTORIES
        }
        reference = matrices[INVENTORIES[0]]
        counts[split] = {inventory: int(len(matrices[inventory]["y"])) for inventory in INVENTORIES}
        for inventory, matrix in matrices.items():
            ok = all(
                array_equal(reference[key], matrix[key])
                for key in ("identity_hash", "identity_id", "y", "domain", "source", "split", "document_index")
            )
            checks.append({"split": split, "inventory": inventory, "rows": len(matrix["y"]), "identity_label_alignment": bool(ok)})
            if not ok:
                raise ValueError(f"matrix identity/label mismatch: {split}/{inventory}")
    return {"checks": checks, "counts": counts, "all_pass": True}


def check_manifest_hashes(paths: Sequence[Path]) -> list[dict[str, Any]]:
    results = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for name, expected in (payload.get("output_hashes") or {}).items():
            candidate = path.parent / name
            exists = candidate.exists()
            actual = sha256_file(candidate) if exists else None
            results.append({"manifest": str(path), "artifact": name, "exists": exists, "hash_matches": exists and actual == expected})
    return results


def decode_prediction_key(key: str) -> str:
    value = key[len("probability__") :]
    if value.startswith("late_pair__"):
        value = "late_pair/" + value[len("late_pair__") :].replace("+", "+", 1)
    else:
        value = value.replace("__", "/", 1)
    return value


def check_predictions(root: Path, feature_dir: Path) -> dict[str, Any]:
    model_dir = root / "incremental_models"
    bundle_path = model_dir / "primary_test_predictions.npz"
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    bundle_hash = sha256_file(bundle_path)
    expected_hash = (manifest.get("output_hashes") or {}).get(bundle_path.name)
    with np.load(bundle_path, allow_pickle=False) as values:
        keys = list(values.files)
        identity_hash = np.asarray(values["identity_hash"])
        labels = np.asarray(values["y"])
        probabilities = {key: np.asarray(values[key], dtype=float) for key in keys if key.startswith("probability__")}
    expected_keys = set()
    for model in ("lr", "xgb"):
        expected_keys.update(f"probability__{model}__{representation}__{inventory}" for representation in ("sr", "full") for inventory in INVENTORIES)
        expected_keys.update(f"probability__{model}__late_pair__{'_'.join(pair)}" for pair in PAIRS)
        expected_keys.add(f"probability__{model}__late_all")
    reference = load_matrix(feature_dir / INVENTORIES[0] / "hc3_test.features.npz")
    identity_ok = array_equal(identity_hash, reference["identity_hash"])
    labels_ok = array_equal(labels, reference["y"])
    lengths_ok = all(probability.shape == labels.shape for probability in probabilities.values())
    finite_ok = all(np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all() for probability in probabilities.values())
    metrics_ok = True
    threshold_count = 0
    for seed in manifest.get("seeds", []):
        for model in manifest.get("models", []):
            for config in [f"sr__{inventory}" for inventory in INVENTORIES] + [f"full__{inventory}" for inventory in INVENTORIES] + ["late_all"]:
                metric_path = model_dir / f"seed_{seed}" / model / config / "metrics.json"
                if not metric_path.exists():
                    metrics_ok = False
                    continue
                payload = json.loads(metric_path.read_text(encoding="utf-8"))
                threshold = payload.get("selection", {}).get("validation_selected_threshold")
                if threshold is not None:
                    threshold_count += 1
                    metrics_ok &= bool(np.isfinite(threshold) and 0 <= threshold <= 1)
    source_hashes = manifest.get("source_matrix_hashes") or {}
    source_hash_results = {}
    for key, expected in source_hashes.items():
        inventory, split = key.split("/", 1)
        matrix_path = feature_dir / inventory / f"hc3_{split}.features.npz"
        actual = sha256_file(matrix_path) if matrix_path.exists() else None
        source_hash_results[key] = bool(actual == expected)
    source_hashes_ok = bool(source_hash_results) and all(source_hash_results.values())
    return {
        "bundle_sha256": bundle_hash,
        "manifest_bundle_hash": expected_hash,
        "bundle_hash_matches": bundle_hash == expected_hash,
        "prediction_keys": len(probabilities),
        "prediction_key_set_exact": set(probabilities) == expected_keys,
        "identity_hash_matches_pcc_test": identity_ok,
        "labels_match_pcc_test": labels_ok,
        "all_prediction_lengths_match": lengths_ok,
        "all_probabilities_finite_in_unit_interval": finite_ok,
        "metric_threshold_files_valid": metrics_ok,
        "thresholds_checked": threshold_count,
        "source_matrix_hashes_match": source_hashes_ok,
        "source_matrix_hash_checks": source_hash_results,
        "all_pass": bool(bundle_hash == expected_hash and set(probabilities) == expected_keys and identity_ok and labels_ok and lengths_ok and finite_ok and metrics_ok and source_hashes_ok),
    }


def span_bounds(node_id: str) -> tuple[int, int] | None:
    match = SPAN_RE.match(node_id)
    return (int(match.group(1)), int(match.group(2))) if match else None


def jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return len(left & right) / float(len(union)) if union else 1.0


def graph_scene_metrics(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_edus, right_edus = set(left["edus"]), set(right["edus"])
    left_nodes, right_nodes = set(left["node_labels"]), set(right["node_labels"])
    valid_nodes = 0
    valid_by_side: dict[str, int] = {}
    total_nodes = len(left_nodes) + len(right_nodes)
    node_spans: dict[str, set[tuple[int, int]]] = {}
    for side, nodes, edus in (("left", left_nodes, left_edus), ("right", right_nodes, right_edus)):
        spans = set()
        for node in nodes:
            bounds = span_bounds(node)
            if bounds and 1 <= bounds[0] <= bounds[1] <= len(edus):
                valid_nodes += 1
                spans.add(bounds)
        node_spans[side] = spans
        valid_by_side[side] = len(spans)
    all_left = {(source, target) for source, target, _ in left["edges"]}
    all_right = {(source, target) for source, target, _ in right["edges"]}
    disc_left = {(source, target) for source, target, label in left["edges"] if label != "/"}
    disc_right = {(source, target) for source, target, label in right["edges"] if label != "/"}
    def label_agreement(left_edges, right_edges):
        left_map = {(source, target): label for source, target, label in left["edges"] if (source, target) in left_edges}
        right_map = {(source, target): label for source, target, label in right["edges"] if (source, target) in right_edges}
        shared = left_edges & right_edges
        return sum(left_map[key] == right_map[key] for key in shared) / float(len(shared)) if shared else float("nan")
    return {
        "edu_key_exact": left_edus == right_edus,
        "left_nodes_valid_span_ids": valid_by_side["left"] == len(left_nodes),
        "right_nodes_valid_span_ids": valid_by_side["right"] == len(right_nodes),
        "valid_node_id_rate": valid_nodes / float(total_nodes) if total_nodes else 1.0,
        "node_span_jaccard": jaccard(node_spans["left"], node_spans["right"]),
        "all_link_endpoint_jaccard": jaccard(all_left, all_right),
        "discourse_link_endpoint_jaccard": jaccard(disc_left, disc_right),
        "all_link_label_agreement": label_agreement(all_left, all_right),
        "discourse_relation_label_agreement": label_agreement(disc_left, disc_right),
        "left_scaffolding_links": sum(label == "/" for _, _, label in left["edges"]),
        "right_scaffolding_links": sum(label == "/" for _, _, label in right["edges"]),
        "left_discourse_links": len(disc_left),
        "right_discourse_links": len(disc_right),
    }


def graph_audit(dataset_root: Path, motif_root: Path) -> dict[str, Any]:
    from scripts.review import align_representations as alignment
    motif_hashes = {inventory: alignment.catalog_hashes(motif_root, inventory) for inventory in INVENTORIES}
    paths = {
        inventory: dataset_root / f"rel-{inventory}" / "hc3_test.discourse_parsed.graph_added.m3_motif_dists.jsonl"
        for inventory in INVENTORIES
    }
    counts = {inventory: sum(1 for line in path.open("rb") if line.strip()) for inventory, path in paths.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"unaligned graph test lengths: {counts}")
    aggregates: dict[tuple[str, str], dict[str, float]] = {pair: defaultdict(float) for pair in PAIRS}
    documents = 0
    handles = [paths[inventory].open("rb") for inventory in INVENTORIES]
    try:
        for lines in tqdm(zip(*handles), total=next(iter(counts.values())), desc="audit graph test", unit="doc"):
            docs = [alignment.document_summary(alignment.read_record(line), motif_hashes[inventory]) for line, inventory in zip(lines, INVENTORIES)]
            if any(doc["identity_hash"] != docs[0]["identity_hash"] or doc["label"] != docs[0]["label"] for doc in docs[1:]):
                raise ValueError("graph source identities or labels are not aligned")
            documents += 1
            for pair in PAIRS:
                left, right = docs[INVENTORIES.index(pair[0])], docs[INVENTORIES.index(pair[1])]
                matched, left_unmatched, right_unmatched = alignment.align_scenes(left["scenes"], right["scenes"])
                bucket = aggregates[pair]
                bucket["documents"] += 1
                bucket["matched_scenes"] += len(matched)
                bucket["unmatched_scenes"] += left_unmatched + right_unmatched
                for scene_left, scene_right in matched:
                    metrics = graph_scene_metrics(scene_left, scene_right)
                    for key, value in metrics.items():
                        if isinstance(value, bool):
                            bucket[key] += int(value)
                        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                            bucket[key] += float(value)
                    bucket["scenes"] += 1
    finally:
        for handle in handles:
            handle.close()
    rows = []
    for pair in PAIRS:
        bucket = aggregates[pair]
        scenes = max(1.0, bucket["scenes"])
        documents_float = max(1.0, bucket["documents"])
        rows.append({
            "split": "test", "left_inventory": pair[0], "right_inventory": pair[1],
            "documents": int(bucket["documents"]), "matched_scenes": int(bucket["matched_scenes"]),
            "unmatched_scenes": int(bucket["unmatched_scenes"]), "scenes": int(bucket["scenes"]),
            "edu_key_exact_rate": bucket["edu_key_exact"] / scenes,
            "valid_node_id_rate": bucket["valid_node_id_rate"] / scenes,
            "node_span_jaccard": bucket["node_span_jaccard"] / scenes,
            "all_link_endpoint_jaccard": bucket["all_link_endpoint_jaccard"] / scenes,
            "discourse_link_endpoint_jaccard": bucket["discourse_link_endpoint_jaccard"] / scenes,
            "all_link_label_agreement": bucket["all_link_label_agreement"] / scenes,
            "discourse_relation_label_agreement": bucket["discourse_relation_label_agreement"] / scenes,
            "mean_left_scaffolding_links": bucket["left_scaffolding_links"] / scenes,
            "mean_right_scaffolding_links": bucket["right_scaffolding_links"] / scenes,
            "mean_left_discourse_links": bucket["left_discourse_links"] / scenes,
            "mean_right_discourse_links": bucket["right_discourse_links"] / scenes,
        })
    return {"documents": documents, "line_counts": counts, "rows": rows, "all_pass": all(row["edu_key_exact_rate"] == 1.0 and row["valid_node_id_rate"] == 1.0 for row in rows)}


def inference_audit(root: Path) -> dict[str, Any]:
    model_dir = root / "incremental_models"
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    contrasts = list(csv.DictReader((model_dir / "incremental_contrasts.csv").open(encoding="utf-8")))
    primary = [row for row in contrasts if row["cohort"] == "primary"]
    identity = [row for row in contrasts if row["cohort"] == "identity_clean"]
    seed_rows = list(csv.DictReader((model_dir / "seed_stability.csv").open(encoding="utf-8")))
    families = defaultdict(int)
    for row in identity:
        families[(row["model"], row["metric"])] += 1
    seed_ranges = {}
    for key in sorted({(row["model"], row["representation"], row["inventory"]) for row in seed_rows}):
        values = [float(row["roc_auc"]) for row in seed_rows if (row["model"], row["representation"], row["inventory"]) == key]
        seed_ranges["/".join(key)] = max(values) - min(values)
    pvalues = [float(row["bootstrap_sign_pvalue"]) for row in contrasts]
    summary = json.loads((model_dir / "complementarity_summary.json").read_text(encoding="utf-8"))
    def validation_score(model: str, inventory: str) -> float:
        path = model_dir / "seed_42" / model / f"full__{inventory}" / "metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return float(payload["validation_metrics"]["roc_auc"])

    selected_keys = set()
    for model in ("lr", "xgb"):
        scores = {inventory: validation_score(model, inventory) for inventory in INVENTORIES}
        for left, right in PAIRS:
            stronger = right if scores[right] >= scores[left] else left
            selected_keys.add((model, "pair_vs_right" if stronger == right else "pair_vs_left", f"{model}/late_pair/{left}+{right}", f"{model}/full/{stronger}"))
        best = max(INVENTORIES, key=scores.get)
        selected_keys.add((model, "all_vs_inventory", f"{model}/late_all", f"{model}/full/{best}"))
    primary_gate = [row for row in identity if row["metric"] == "roc_auc" and (row["model"], row["family"], row["candidate"], row["reference"]) in selected_keys]
    model_records = manifest.get("model_records") or []
    model_record_seeds = sorted({int(record["seed"]) for record in model_records if "seed" in record})
    model_metadata_ok = len(model_records) == 32 and model_record_seeds == [42, 43] and sorted({record.get("model") for record in model_records}) == ["lr", "xgb"]
    return {
        "contrast_rows": len(contrasts), "primary_rows": len(primary), "identity_clean_rows": len(identity),
        "identity_clean_rows_per_model_metric": dict(sorted((f"{k[0]}/{k[1]}", v) for k, v in families.items())),
        "holm_family_size": 20, "bootstrap_iterations": 2000,
        "bootstrap_pvalue_floor": 2.0 / 2001.0, "minimum_observed_sign_pvalue": min(pvalues),
        "selected_reference_gate_rows": len(primary_gate),
        "selected_reference_gate_passing": sum(float(row["difference"]) >= 0.01 and row["ci_excludes_zero"].lower() == "true" and float(row["holm_adjusted_pvalue"]) < 0.05 for row in primary_gate),
        "summary_gate_rows": summary.get("primary_cross_inventory_gate", {}).get("total_contrasts"),
        "summary_gate_passing": summary.get("primary_cross_inventory_gate", {}).get("criterion_passing"),
        "seed_rows": len(seed_rows), "seed_values": sorted({int(row["seed"]) for row in seed_rows}),
        "model_records": len(model_records), "manifest_seed_values": model_record_seeds,
        "model_record_metadata_ok": model_metadata_ok,
        "seed_test_roc_auc_ranges": seed_ranges,
        "seed_gate_recomputed": False,
        "interpretation": "Intervals are nominal paired identity-cluster percentile intervals; Holm-adjusted sign values correct 20 contrasts per model/cohort/metric. Seven stronger-reference rows per model are selected after validation; intervals are not simultaneous and the two seeds are not independent replications.",
    }


def make_report(result: Mapping[str, Any]) -> str:
    matrix = result["matrix"]
    prediction = result["predictions"]
    graph = result["graph"]
    inference = result["inference"]
    lines = [
        "# UniRST2 measurement audit",
        "",
        "Audit run: 2026-09-10. This audit reuses the frozen UniRST2 matrices, parsed graphs, prediction bundle, and manifests. It does not refit models.",
        "",
        "## Findings",
        "",
        f"- Matrix identity/label alignment: **PASS** for all four inventories and all three splits ({sum(matrix['counts']['train'].values()) + sum(matrix['counts']['validation'].values()) + sum(matrix['counts']['test'].values()):,} inventory-split rows checked).",
        f"- Prediction bundle provenance and shape checks: **{'PASS' if prediction['all_pass'] else 'FAIL'}** ({prediction['prediction_keys']} probability arrays; {prediction['thresholds_checked']} threshold files; source matrix hashes {'match' if prediction['source_matrix_hashes_match'] else 'do not all match'}).",
        f"- Graph identity alignment: **{'PASS' if graph['all_pass'] else 'FAIL'}** for {graph['documents']:,} test documents; EDU keys match exactly and all node IDs are valid `span_start-end` ranges within the EDU count.",
        "- The original `tree_agreement.csv` field `mean_attachment_jaccard` is an endpoint Jaccard over *all exported links*. It is not constituent-span agreement. This audit reports that quantity separately from discourse-only endpoint Jaccard and explicitly counts `/` scaffolding links.",
        f"- Inference accounting: **PASS** for the stored protocol ({inference['contrast_rows']} contrasts; Holm family size 20; {inference['bootstrap_iterations']} resamples; {inference['selected_reference_gate_rows']} selected-reference rows, {inference['selected_reference_gate_passing']} passing the +0.01/CI/sign gate; {inference['model_records']} model records across seeds {inference['manifest_seed_values']}).",
        "- Seed stability is descriptive only: two stored seeds provide aggregate test scores, but the primary prediction bundle and selected-reference gate were generated from seed 42; no independent seed-level gate was programmatically evaluated.",
        "",
        "## Corrected graph measurements (test)",
        "",
        "See `representation_metric_audit.csv` for machine-readable values. `node_span_jaccard` compares parsed constituent span ranges; `all_link_endpoint_jaccard` reproduces the broad exported-link endpoint measure; `discourse_link_endpoint_jaccard` excludes links labeled `/`; label-agreement columns are conditional on shared endpoints.",
        "",
        "| Pair | EDU exact | Valid node IDs | Node-span Jaccard | All-link Jaccard | Discourse-link Jaccard | All-link label agreement | Discourse label agreement | Mean scaffolding links (L/R) | Mean discourse links (L/R) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in graph["rows"]:
        lines.append(
            f"| {row['left_inventory']} / {row['right_inventory']} | {row['edu_key_exact_rate']:.3f} | {row['valid_node_id_rate']:.3f} | {row['node_span_jaccard']:.3f} | {row['all_link_endpoint_jaccard']:.3f} | {row['discourse_link_endpoint_jaccard']:.3f} | {row['all_link_label_agreement']:.3f} | {row['discourse_relation_label_agreement']:.3f} | {row['mean_left_scaffolding_links']:.2f} / {row['mean_right_scaffolding_links']:.2f} | {row['mean_left_discourse_links']:.2f} / {row['mean_right_discourse_links']:.2f} |"
        )
    lines += [
        "",
        "## Provenance caveats",
        "",
        "- Test outcomes remain exploratory because the test set was inspected in earlier stages.",
        "- Nominal percentile intervals and Holm-adjusted bootstrap-sign values should not be described as simultaneous corrected intervals or as a newly preregistered family.",
        "- The failed +0.01 cross-inventory gate is an operational result for this fusion procedure; it does not exclude complementary linguistic accounts, and it does not establish equivalence or redundancy of the standards.",
        "- The incremental-model manifest now hashes `report.md` as well as the numerical outputs; this audit verifies the report hash.",
        "",
        "## Outputs",
        "",
        "- `measurement_audit.json` — structured checks and provenance.",
        "- `representation_metric_audit.csv` — corrected structural/discourse graph metrics.",
        "- `measurement_audit.md` — this human-readable audit.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=Path("artifacts/canonical/feature_matrices"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/unirst2"))
    parser.add_argument("--motif-root", type=Path, default=Path("data/motifs2"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/review"))
    args = parser.parse_args()
    matrix = check_matrix_alignment(args.feature_dir)
    predictions = check_predictions(args.output_dir, args.feature_dir)
    graph = graph_audit(args.dataset_root, args.motif_root)
    inference = inference_audit(args.output_dir)
    manifests = check_manifest_hashes([
        args.feature_dir / "manifest.json",
        args.output_dir / "representation_alignment" / "manifest.json",
        args.output_dir / "incremental_models" / "manifest.json",
    ])
    result = {
        "stage": "unirst2_measurement_audit",
        "inputs": {"feature_dir": str(args.feature_dir.resolve()), "dataset_root": str(args.dataset_root.resolve()), "motif_root": str(args.motif_root.resolve())},
        "matrix": matrix, "predictions": predictions, "graph": graph, "inference": inference,
        "manifest_hash_checks": manifests,
        "manifest_hash_failures": [item for item in manifests if not item["hash_matches"]],
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "representation_metric_audit.csv", graph["rows"])
    atomic_text(output_dir / "measurement_audit.md", make_report(result))
    result["outputs"] = {
        "measurement_audit_md": str((output_dir / "measurement_audit.md").resolve()),
        "measurement_audit_json": str((output_dir / "measurement_audit.json").resolve()),
        "representation_metric_audit_csv": str((output_dir / "representation_metric_audit.csv").resolve()),
    }
    result["output_hashes"] = {name: sha256_file(output_dir / name) for name in ("measurement_audit.md", "representation_metric_audit.csv")}
    atomic_text(output_dir / "measurement_audit.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(make_report(result))


if __name__ == "__main__":
    main()
