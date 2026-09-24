#!/usr/bin/env python3
"""Align and compare serialized UniRST2 discourse representations.

The four inventories are parsed from the existing graph-plus-M3 files in lock
step.  Only compact summaries and aggregate comparisons are written; source
corpora and model artifacts are never modified.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from tqdm import tqdm


INVENTORIES = (
    "deu.rst.pcc",
    "eng.erst.gum.new",
    "eng.rst.rstdt.new",
    "nld.rst.nldt",
)
SPLITS = ("train", "validation", "test")
PAIRS = tuple(
    (left, right)
    for index, left in enumerate(INVENTORIES)
    for right in INVENTORIES[index + 1 :]
)


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
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    stream = io.StringIO()
    if rows:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def read_record(line: bytes) -> Mapping[str, Any]:
    value = line.decode("utf-8").strip()
    if not value:
        raise ValueError("empty record")
    record = json.loads(value) if value.startswith("{\"") else eval(value, {"__builtins__": {}}, {})
    if not isinstance(record, Mapping):
        raise ValueError("record is not a mapping")
    return record


def source_domain(source: str) -> str:
    value = source
    for suffix in ("_human", "_llm", "_gpt4", "_machine", "_para"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value or source


def base_identity(text: str, source: str, label: int) -> str:
    payload = json.dumps([text, source, label], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def catalog_hashes(motif_root: Path, inventory: str) -> List[str]:
    path = motif_root / inventory / "hc3_selected-motif-hashes.generated.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = list(payload.get("m3") or [])
    if not values:
        raise ValueError(f"empty M3 catalog: {path}")
    return values


def scene_summary(scene: Mapping[str, Any], motif_hashes: Sequence[str]) -> Dict[str, Any]:
    text = str(scene.get("text") or "")
    graph = scene.get("graph_dict") or {}
    links = graph.get("links") or graph.get("edges") or []
    nodes = graph.get("nodes") or []
    edges = [
        (
            str(edge.get("source")),
            str(edge.get("target")),
            str(edge.get("label_0", edge.get("label", ""))),
        )
        for edge in links
        if isinstance(edge, Mapping)
    ]
    node_labels = {
        str(node.get("id")): str(node.get("label_0", node.get("label", "")))
        for node in nodes
        if isinstance(node, Mapping) and node.get("id") is not None
    }
    motif_values = ((scene.get("motif_dists") or {}).get("m3") or {}).get("raw") or []
    motif_counts = {
        motif_hashes[index]: float(value)
        for index, value in enumerate(motif_values)
        if index < len(motif_hashes) and isinstance(value, (int, float)) and float(value) > 0
    }
    return {
        "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "segments": [int(value) for value in (scene.get("segments") or [])],
        "edus": sorted(str(value) for value in (scene.get("edus") or {}).keys()),
        "edges": edges,
        "node_labels": node_labels,
        "motif_counts": motif_counts,
    }


def document_summary(record: Mapping[str, Any], motif_hashes: Sequence[str]) -> Dict[str, Any]:
    text = str(record.get("text") or "")
    source = str(record.get("source", record.get("src", "unknown")))
    label = int(record.get("label"))
    trees = record.get("scene_discourse_trees") or {}
    if isinstance(trees, Mapping):
        values = list(trees.values())
    else:
        values = list(trees)
    scenes = [scene_summary(value, motif_hashes) for value in values if isinstance(value, Mapping)]
    occurrence = Counter()
    for scene in scenes:
        occurrence[scene["text_hash"]] += 1
        scene["scene_occurrence"] = occurrence[scene["text_hash"]] - 1
    return {
        "identity_hash": base_identity(text, source, label),
        "source": source,
        "domain": source_domain(source),
        "label": label,
        "scenes": scenes,
    }


def align_scenes(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> Tuple[List[Tuple[Mapping[str, Any], Mapping[str, Any]]], int, int]:
    right_index: Dict[Tuple[str, int], Mapping[str, Any]] = {
        (scene["text_hash"], int(scene["scene_occurrence"])): scene for scene in right
    }
    matched = []
    used = set()
    for scene in left:
        key = (scene["text_hash"], int(scene["scene_occurrence"]))
        counterpart = right_index.get(key)
        if counterpart is not None:
            matched.append((scene, counterpart))
            used.add(key)
    return matched, len(left) - len(matched), len(right) - len(used)


def scene_pair_metrics(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    left_attachments = {(edge[0], edge[1]) for edge in left["edges"]}
    right_attachments = {(edge[0], edge[1]) for edge in right["edges"]}
    shared = left_attachments & right_attachments
    union = left_attachments | right_attachments
    left_labels = {(edge[0], edge[1]): edge[2] for edge in left["edges"]}
    right_labels = {(edge[0], edge[1]): edge[2] for edge in right["edges"]}
    label_agreement = sum(left_labels[key] == right_labels[key] for key in shared)
    node_ids = set(left["node_labels"]) & set(right["node_labels"])
    nuclearity_agreement = sum(left["node_labels"][key] == right["node_labels"][key] for key in node_ids)
    left_motifs, right_motifs = set(left["motif_counts"]), set(right["motif_counts"])
    common_motifs = left_motifs & right_motifs
    return {
        "segmentation_exact": bool(left["segments"] == right["segments"] and left["edus"] == right["edus"]),
        "left_edu_count": len(left["edus"]),
        "right_edu_count": len(right["edus"]),
        "left_edges": len(left_attachments),
        "right_edges": len(right_attachments),
        "shared_edges": len(shared),
        "left_only_edges": len(left_attachments - right_attachments),
        "right_only_edges": len(right_attachments - left_attachments),
        "attachment_jaccard": len(shared) / float(len(union)) if union else 1.0,
        "shared_edge_label_agreement": label_agreement / float(len(shared)) if shared else float("nan"),
        "shared_node_label_agreement": nuclearity_agreement / float(len(node_ids)) if node_ids else float("nan"),
        "shared_motifs": sorted(common_motifs),
        "left_only_motifs": sorted(left_motifs - right_motifs),
        "right_only_motifs": sorted(right_motifs - left_motifs),
        "shared_motif_l1": float(sum(abs(left["motif_counts"][key] - right["motif_counts"][key]) for key in common_motifs)),
    }


def document_pair_metrics(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    matched, left_unmatched, right_unmatched = align_scenes(left["scenes"], right["scenes"])
    scene_metrics = [scene_pair_metrics(a, b) for a, b in matched]
    def total(key: str) -> float:
        return float(sum(metric[key] for metric in scene_metrics))
    def mean(key: str) -> float:
        values = [metric[key] for metric in scene_metrics if np.isfinite(metric[key])]
        return float(np.mean(values)) if values else float("nan")
    return {
        "matched_scenes": len(matched),
        "left_unmatched_scenes": left_unmatched,
        "right_unmatched_scenes": right_unmatched,
        "segmentation_exact": bool(scene_metrics) and all(metric["segmentation_exact"] for metric in scene_metrics) and not left_unmatched and not right_unmatched,
        "left_edu_count": int(total("left_edu_count")),
        "right_edu_count": int(total("right_edu_count")),
        "left_edges": int(total("left_edges")),
        "right_edges": int(total("right_edges")),
        "shared_edges": int(total("shared_edges")),
        "left_only_edges": int(total("left_only_edges")),
        "right_only_edges": int(total("right_only_edges")),
        "attachment_jaccard": mean("attachment_jaccard"),
        "shared_edge_label_agreement": mean("shared_edge_label_agreement"),
        "shared_node_label_agreement": mean("shared_node_label_agreement"),
        "shared_motif_l1": total("shared_motif_l1"),
        "shared_motif_count": int(len(set().union(*(set(metric["shared_motifs"]) for metric in scene_metrics))) if scene_metrics else 0),
        "left_only_motif_count": int(len(set().union(*(set(metric["left_only_motifs"]) for metric in scene_metrics))) if scene_metrics else 0),
        "right_only_motif_count": int(len(set().union(*(set(metric["right_only_motifs"]) for metric in scene_metrics))) if scene_metrics else 0),
        "shared_motifs": sorted(set().union(*(set(metric["shared_motifs"]) for metric in scene_metrics)) if scene_metrics else set()),
        "left_only_motifs": sorted(set().union(*(set(metric["left_only_motifs"]) for metric in scene_metrics)) if scene_metrics else set()),
        "right_only_motifs": sorted(set().union(*(set(metric["right_only_motifs"]) for metric in scene_metrics)) if scene_metrics else set()),
    }


def relation_pairs(left: Mapping[str, Any], right: Mapping[str, Any]) -> Counter:
    matched, _, _ = align_scenes(left["scenes"], right["scenes"])
    counts = Counter()
    for a, b in matched:
        left_edges = {(edge[0], edge[1]): edge[2] for edge in a["edges"]}
        right_edges = {(edge[0], edge[1]): edge[2] for edge in b["edges"]}
        for key in left_edges.keys() & right_edges.keys():
            counts[(left_edges[key], right_edges[key])] += 1
    return counts


def load_train_hashes(feature_dir: Path, inventory: str) -> set:
    path = feature_dir / inventory / "hc3_train.features.npz"
    with np.load(path, allow_pickle=False) as values:
        return set(values["identity_hash"].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/unirst2"))
    parser.add_argument("--motif-root", type=Path, default=Path("data/motifs2"))
    parser.add_argument("--feature-dir", type=Path, default=Path("artifacts/canonical/feature_matrices"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/review/representation_alignment"))
    parser.add_argument("--inventories", nargs="+", choices=INVENTORIES, default=list(INVENTORIES))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    args = parser.parse_args()
    if tuple(args.inventories) != INVENTORIES:
        raise ValueError("stage 3 requires all inventories in canonical order")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    motif_hashes = {inventory: catalog_hashes(args.motif_root, inventory) for inventory in INVENTORIES}
    train_identity_hashes = load_train_hashes(args.feature_dir, INVENTORIES[0])
    pair_counters: Dict[Tuple[str, str], Counter] = {pair: Counter() for pair in PAIRS}
    aggregate: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    document_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    input_hashes: Dict[str, str] = {}

    for split in args.splits:
        paths = {
            inventory: args.dataset_root / f"rel-{inventory}" / f"hc3_{split}.discourse_parsed.graph_added.m3_motif_dists.jsonl"
            for inventory in INVENTORIES
        }
        for path in paths.values():
            if not path.exists():
                raise FileNotFoundError(path)
            input_hashes[f"{split}/{path.parent.name}"] = sha256_file(path)
        totals = [count_lines(paths[inventory]) for inventory in INVENTORIES]
        if len(set(totals)) != 1:
            raise ValueError(f"unaligned line counts for {split}: {totals}")
        handles = [paths[inventory].open("rb") for inventory in INVENTORIES]
        try:
            for lines in tqdm(zip(*handles), total=totals[0], desc=f"align {split}", unit="doc"):
                documents = [document_summary(read_record(line), motif_hashes[inventory]) for line, inventory in zip(lines, INVENTORIES)]
                reference = documents[0]
                if any(document["identity_hash"] != reference["identity_hash"] or document["label"] != reference["label"] for document in documents[1:]):
                    raise ValueError(f"identity/label mismatch during {split}")
                for pair_index, (left, right) in enumerate(PAIRS):
                    left_doc, right_doc = documents[INVENTORIES.index(left)], documents[INVENTORIES.index(right)]
                    metrics = document_pair_metrics(left_doc, right_doc)
                    row = {
                        "split": split,
                        "left_inventory": left,
                        "right_inventory": right,
                        "identity_hash": reference["identity_hash"],
                        "domain": reference["domain"],
                        "label": reference["label"],
                        "identity_clean": reference["identity_hash"] not in train_identity_hashes,
                        **metrics,
                    }
                    if split != "train":
                        document_rows[split].append(row)
                    for key in ("matched_scenes", "left_unmatched_scenes", "right_unmatched_scenes", "left_edu_count", "right_edu_count", "left_edges", "right_edges", "shared_edges", "left_only_edges", "right_only_edges", "shared_motif_l1"):
                        aggregate[(split, left, right)][key] += float(metrics[key])
                    for key in ("segmentation_exact",):
                        aggregate[(split, left, right)][key] += int(metrics[key])
                    for key in ("attachment_jaccard", "shared_edge_label_agreement", "shared_node_label_agreement"):
                        if np.isfinite(metrics[key]):
                            aggregate[(split, left, right)][f"{key}_sum"] += metrics[key]
                            aggregate[(split, left, right)][f"{key}_n"] += 1
                    if split == "train":
                        pair_counters[(left, right)].update(relation_pairs(left_doc, right_doc))
        finally:
            for handle in handles:
                handle.close()

    segmentation_rows, tree_rows = [], []
    for (split, left, right), values in sorted(aggregate.items()):
        records = sum(
            1
            for row in document_rows[split]
            if row["left_inventory"] == left and row["right_inventory"] == right
        ) if split != "train" else None
        if records is None:
            # Train has no retained document rows; recover its common count from
            # the aligned source line count.
            records = count_lines(args.dataset_root / f"rel-{left}" / f"hc3_{split}.discourse_parsed.graph_added.m3_motif_dists.jsonl")
        segmentation_rows.append({
            "split": split, "left_inventory": left, "right_inventory": right,
            "records": records,
            "exact_segmentation_records": int(values["segmentation_exact"]),
            "exact_segmentation_rate": float(values["segmentation_exact"] / records),
            "mean_matched_scenes": float(values["matched_scenes"] / records),
            "mean_left_unmatched_scenes": float(values["left_unmatched_scenes"] / records),
            "mean_right_unmatched_scenes": float(values["right_unmatched_scenes"] / records),
        })
        tree_rows.append({
            "split": split, "left_inventory": left, "right_inventory": right,
            "records": records,
            "mean_attachment_jaccard": float(values["attachment_jaccard_sum"] / values["attachment_jaccard_n"]) if values["attachment_jaccard_n"] else float("nan"),
            "mean_shared_edge_label_agreement": float(values["shared_edge_label_agreement_sum"] / values["shared_edge_label_agreement_n"]) if values["shared_edge_label_agreement_n"] else float("nan"),
            "mean_shared_node_label_agreement": float(values["shared_node_label_agreement_sum"] / values["shared_node_label_agreement_n"]) if values["shared_node_label_agreement_n"] else float("nan"),
            "mean_shared_edges": float(values["shared_edges"] / records),
            "mean_left_only_edges": float(values["left_only_edges"] / records),
            "mean_right_only_edges": float(values["right_only_edges"] / records),
        })

    relation_rows = []
    for left, right in PAIRS:
        counts = pair_counters[(left, right)]
        left_totals, right_totals = Counter(), Counter()
        for (left_label, right_label), count in counts.items():
            left_totals[left_label] += count
            right_totals[right_label] += count
        for (left_label, right_label), count in sorted(counts.items()):
            relation_rows.append({
                "split": "train", "left_inventory": left, "right_inventory": right,
                "left_relation": left_label, "right_relation": right_label,
                "matched_attachment_count": count,
                "right_given_left": count / float(left_totals[left_label]) if left_totals[left_label] else 0.0,
                "left_given_right": count / float(right_totals[right_label]) if right_totals[right_label] else 0.0,
            })

    difference_rows = document_rows["test"] + document_rows["validation"]
    for row in difference_rows:
        for key in ("shared_motifs", "left_only_motifs", "right_only_motifs"):
            row[key] = json.dumps(row[key], separators=(",", ":"))
    write_csv(args.output_dir / "segmentation_audit.csv", segmentation_rows)
    write_csv(args.output_dir / "tree_agreement.csv", tree_rows)
    write_csv(args.output_dir / "relation_correspondence.csv", relation_rows)
    write_csv(args.output_dir / "motif_representation_differences.csv", difference_rows)
    parquet_path = args.output_dir / "motif_representation_differences.parquet"
    parquet_written = False
    try:
        import pandas as pd
        temporary = parquet_path.with_name(parquet_path.name + ".tmp")
        pd.DataFrame(difference_rows).to_parquet(temporary, index=False)
        os.replace(temporary, parquet_path)
        parquet_written = True
    except ImportError:
        pass

    output_names = ["segmentation_audit.csv", "tree_agreement.csv", "relation_correspondence.csv", "motif_representation_differences.csv"]
    if parquet_written:
        output_names.append("motif_representation_differences.parquet")
    summary = {
        "stage": "unirst2_discourse_representation_alignment",
        "inventories": list(INVENTORIES), "splits": list(args.splits), "pairs": [list(pair) for pair in PAIRS],
        "identity": "sha256(canonical JSON [text, source, label])",
        "scene_alignment": "exact scene-text hash plus occurrence; unmatched scenes retained as diagnostics",
        "relation_correspondence_source": "training split only; matching attachment endpoints after exact scene alignment",
        "nuclearity_policy": "node labels are compared diagnostically; existing motif counts do not separate nuclearity variants",
        "input_hashes": input_hashes,
        "output_hashes": {name: sha256_file(args.output_dir / name) for name in output_names},
        "outputs": {name: str((args.output_dir / name).resolve()) for name in output_names},
        "document_difference_rows": len(difference_rows),
    }
    atomic_json(args.output_dir / "manifest.json", summary)
    report = [
        "# UniRST2 discourse representation alignment", "",
        "Representations were compared on identical text/source/label identities using exact scene-text matching.",
        "Relation correspondence tables are learned from training only. Identical relation strings are not treated as semantic equivalence.",
        "",
        "The per-document difference table includes validation and test documents and marks the exact-identity-clean cohort.",
        "Nuclearity agreement is diagnostic only because the operative M3 feature identity omits node attributes.", "",
    ]
    atomic_text(args.output_dir / "report.md", "\n".join(report))
    print(json.dumps({"document_difference_rows": len(difference_rows), "relation_rows": len(relation_rows)}, indent=2))


if __name__ == "__main__":
    main()
