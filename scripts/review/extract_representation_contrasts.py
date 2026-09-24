#!/usr/bin/env python3
"""Find recurring, auditable differences between aligned UniRST2 parses.

Training relation correspondence is used to freeze high-confidence mapping
rules.  Validation establishes which difference types recur.  Test passages
are then selected by a representation-only rule; authorship labels, model
scores, and motif coefficient signs are never used for selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx
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
MAPPING_MIN_SUPPORT = 100
MAPPING_MIN_CONDITIONAL = 0.50
VALIDATION_MIN_DOCS = 10
MAX_PER_EVENT_DOMAIN = 2
MAX_CANDIDATES = 120
MAX_PER_PAIR = MAX_CANDIDATES // len(PAIRS)
MAX_PER_PAIR_CATEGORY = max(1, MAX_PER_PAIR // 5)


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


def load_motif_catalog(motif_root: Path, inventory: str) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    selected_path = motif_root / inventory / "hc3_selected-motif-hashes.generated.json"
    selected = list(json.loads(selected_path.read_text(encoding="utf-8")).get("m3") or [])
    catalog_path = motif_root / inventory / "hc3_M3_motifs.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    return selected, {motif_hash: catalog[motif_hash] for motif_hash in selected if motif_hash in catalog}


def source_domain(source: str) -> str:
    value = source
    for suffix in ("_human", "_llm", "_gpt4", "_machine", "_para"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value or source


def rich_document(record: Mapping[str, Any], alignment: Any, motif_hashes: Sequence[str]) -> dict[str, Any]:
    text = str(record.get("text") or "")
    source = str(record.get("source", record.get("src", "unknown")))
    label = int(record.get("label"))
    trees = record.get("scene_discourse_trees") or {}
    values = list(trees.values()) if isinstance(trees, Mapping) else list(trees)
    scenes = []
    occurrence = Counter()
    for raw_scene in values:
        if not isinstance(raw_scene, Mapping):
            continue
        scene = alignment.scene_summary(raw_scene, motif_hashes)
        scene["text"] = str(raw_scene.get("text") or "")
        scene["edus_text"] = {
            str(key): str(value)
            for key, value in (raw_scene.get("edus") or {}).items()
        }
        scene["scene_occurrence"] = occurrence[scene["text_hash"]]
        occurrence[scene["text_hash"]] += 1
        scenes.append(scene)
    return {
        "identity_hash": alignment.base_identity(text, source, label),
        "source": source,
        "domain": source_domain(source),
        "label": label,
        "text": text,
        "scenes": scenes,
    }


def discourse_edges(scene: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    return {
        (source, target): label
        for source, target, label in scene["edges"]
        if label != "/"
    }


def node_span_set(scene: Mapping[str, Any]) -> set[str]:
    return set(scene.get("node_labels") or {})


def exact_motif_occurrences(scene: Mapping[str, Any], motif_hash: str, catalog: Mapping[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    pattern = catalog.get(motif_hash)
    if not pattern:
        return []
    actual_graph = nx.DiGraph()
    for node in scene.get("node_labels", {}):
        actual_graph.add_node(node, label_0=scene["node_labels"][node])
    for source, target, label in scene.get("edges", []):
        actual_graph.add_edge(source, target, label_0=label)
    pattern_graph = nx.DiGraph()
    for node in pattern.get("nodes") or []:
        pattern_graph.add_node(str(node.get("id")), label_0=str(node.get("label_0", node.get("label", ""))))
    for edge in pattern.get("links") or pattern.get("edges") or []:
        pattern_graph.add_edge(
            str(edge.get("source")), str(edge.get("target")),
            label_0=str(edge.get("label_0", edge.get("label", ""))),
        )
    if not pattern_graph.nodes or not pattern_graph.edges:
        return []
    node_match = nx.algorithms.isomorphism.categorical_node_match("label_0", "")
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("label_0", "")
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(
        actual_graph, pattern_graph, node_match=node_match, edge_match=edge_match
    )
    occurrences = []
    for mapping in matcher.subgraph_isomorphisms_iter():
        actual_nodes = sorted(str(node) for node in mapping)
        occurrences.append({
            "node_ids": actual_nodes,
            "edu_spans": sorted(node for node in actual_nodes if node.startswith("span_")),
        })
        if len(occurrences) >= limit:
            break
    return occurrences


def make_relation_rules(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], Counter]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    tables: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in rows:
        left, right = row["left_inventory"], row["right_inventory"]
        left_relation, right_relation = row["left_relation"], row["right_relation"]
        if left_relation == "/" or right_relation == "/":
            continue
        tables[(left, right)][(left_relation, right_relation)] += int(row["matched_attachment_count"])
    rules = {}
    for pair, counts in tables.items():
        left_totals = Counter()
        right_totals = Counter()
        for (left_relation, right_relation), count in counts.items():
            left_totals[left_relation] += count
            right_totals[right_relation] += count
        top_by_left: dict[str, tuple[str, int, float]] = {}
        for (left_relation, right_relation), count in counts.items():
            conditional = count / float(left_totals[left_relation])
            current = top_by_left.get(left_relation)
            if current is None or (conditional, count, right_relation) > (current[2], current[1], current[0]):
                top_by_left[left_relation] = (right_relation, count, conditional)
        reverse_candidates: dict[str, set[str]] = defaultdict(set)
        for left_relation, (right_relation, count, conditional) in top_by_left.items():
            if count >= MAPPING_MIN_SUPPORT and conditional >= MAPPING_MIN_CONDITIONAL:
                reverse_candidates[right_relation].add(left_relation)
        rules[pair] = {
            "top_by_left": top_by_left,
            "reverse_candidates": reverse_candidates,
            "left_totals": left_totals,
            "right_totals": right_totals,
        }
    return rules, tables


def align_pair(left: Mapping[str, Any], right: Mapping[str, Any], alignment: Any) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    matched, _, _ = alignment.align_scenes(left["scenes"], right["scenes"])
    return matched


def scene_events(left: Mapping[str, Any], right: Mapping[str, Any], rules: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    left_edges, right_edges = discourse_edges(left), discourse_edges(right)
    shared = set(left_edges) & set(right_edges)
    events: list[dict[str, Any]] = []
    top_by_left = (rules or {}).get("top_by_left", {})
    reverse_candidates = (rules or {}).get("reverse_candidates", {})
    relation_pairs = Counter()
    for endpoint in sorted(shared):
        left_relation, right_relation = left_edges[endpoint], right_edges[endpoint]
        if left_relation == right_relation:
            continue
        relation_pairs[(left_relation, right_relation)] += 1
        expected = top_by_left.get(left_relation)
        if expected and expected[1] >= MAPPING_MIN_SUPPORT and expected[2] >= MAPPING_MIN_CONDITIONAL:
            expected_relation = expected[0]
            if right_relation == expected_relation:
                if len(reverse_candidates.get(right_relation, set())) > 1:
                    category = "many_to_one_granularity"
                elif left_relation != right_relation:
                    category = "one_to_one_renaming"
                else:
                    category = "same_label_control"
            else:
                category = "residual_relation_disagreement"
        else:
            category = "unmapped_relation_disagreement"
        events.append({
            "category": category,
            "left_relation": left_relation,
            "right_relation": right_relation,
            "endpoint": list(endpoint),
            "expected_right_relation": expected[0] if expected else None,
        })
    left_only = sorted(set(left_edges) - set(right_edges))
    right_only = sorted(set(right_edges) - set(left_edges))
    if left_only or right_only:
        events.append({
            "category": "attachment_difference",
            "left_only_endpoints": [list(endpoint) for endpoint in left_only],
            "right_only_endpoints": [list(endpoint) for endpoint in right_only],
        })
    shared_nodes = set(left["node_labels"]) & set(right["node_labels"])
    nuclearity = [
        (node, left["node_labels"][node], right["node_labels"][node])
        for node in sorted(shared_nodes)
        if left["node_labels"][node] != right["node_labels"][node]
    ]
    if nuclearity:
        events.append({"category": "nuclearity_difference", "nodes": [list(item) for item in nuclearity]})
    left_spans, right_spans = node_span_set(left), node_span_set(right)
    if left_spans != right_spans:
        events.append({
            "category": "constituent_span_difference",
            "left_only_nodes": sorted(left_spans - right_spans),
            "right_only_nodes": sorted(right_spans - left_spans),
        })
    summary = {
        "relation_pairs": {f"{left_label} -> {right_label}": count for (left_label, right_label), count in relation_pairs.items()},
        "left_only_endpoints": [list(endpoint) for endpoint in left_only],
        "right_only_endpoints": [list(endpoint) for endpoint in right_only],
        "nuclearity_differences": [list(item) for item in nuclearity],
        "left_node_ids": sorted(left_spans),
        "right_node_ids": sorted(right_spans),
    }
    return events, summary


def event_key(event: Mapping[str, Any]) -> tuple[str, str, str]:
    category = str(event["category"])
    if category in {"one_to_one_renaming", "many_to_one_granularity", "residual_relation_disagreement", "unmapped_relation_disagreement"}:
        return category, str(event.get("left_relation")), str(event.get("right_relation"))
    return category, "*", "*"


def process_split(
    split: str,
    dataset_root: Path,
    alignment: Any,
    motif_hashes: Mapping[str, Sequence[str]],
    rules: Mapping[tuple[str, str], Mapping[str, Any]],
    catalogs: Mapping[str, Mapping[str, Any]],
    collect_candidates: bool,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, tuple[str, str, str]], dict[str, Any]]]:
    paths = {
        inventory: dataset_root / f"rel-{inventory}" / f"hc3_{split}.discourse_parsed.graph_added.m3_motif_dists.jsonl"
        for inventory in INVENTORIES
    }
    counts = {inventory: sum(1 for line in paths[inventory].open("rb") if line.strip()) for inventory in INVENTORIES}
    if len(set(counts.values())) != 1:
        raise ValueError(f"unaligned {split} counts: {counts}")
    aggregates: dict[tuple[str, str, tuple[str, str, str]], dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    handles = [paths[inventory].open("rb") for inventory in INVENTORIES]
    try:
        for lines in tqdm(zip(*handles), total=next(iter(counts.values())), desc=f"contrasts {split}", unit="doc"):
            documents = [
                rich_document(alignment.read_record(line), alignment, motif_hashes[inventory])
                for line, inventory in zip(lines, INVENTORIES)
            ]
            reference = documents[0]
            if any(document["identity_hash"] != reference["identity_hash"] or document["label"] != reference["label"] for document in documents[1:]):
                raise ValueError(f"identity/label mismatch in {split}")
            for pair in PAIRS:
                left_doc = documents[INVENTORIES.index(pair[0])]
                right_doc = documents[INVENTORIES.index(pair[1])]
                matched = align_pair(left_doc, right_doc, alignment)
                document_events: list[dict[str, Any]] = []
                scene_evidence: list[dict[str, Any]] = []
                for left_scene, right_scene in matched:
                    events, evidence = scene_events(left_scene, right_scene, rules.get(pair, {}))
                    if events:
                        scene_evidence.append({
                            "scene_text": left_scene["text"],
                            "scene_occurrence": left_scene["scene_occurrence"],
                            "events": events,
                            "evidence": evidence,
                            "left_scene": left_scene,
                            "right_scene": right_scene,
                        })
                        document_events.extend(events)
                seen_keys = set()
                for event in document_events:
                    key = event_key(event)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    aggregate_key = (pair[0], pair[1], key)
                    bucket = aggregates.setdefault(aggregate_key, {"documents": set(), "event_occurrences": 0, "domains": Counter()})
                    bucket["documents"].add(reference["identity_hash"])
                    bucket["event_occurrences"] += 1
                    bucket["domains"][reference["domain"]] += 1
                if collect_candidates and scene_evidence:
                    candidates.append({
                        "identity_hash": reference["identity_hash"],
                        "source": reference["source"],
                        "domain": reference["domain"],
                        "text": reference["text"],
                        "pair": list(pair),
                        "label": reference["label"],
                        "scene_evidence": scene_evidence,
                    })
    finally:
        for handle in handles:
            handle.close()
    return candidates, aggregates


def choose_shortlist(validation_aggregates: Mapping[tuple[str, str, tuple[str, str, str]], Mapping[str, Any]]) -> set[tuple[str, str, tuple[str, str, str]]]:
    shortlist = set()
    for pair in PAIRS:
        rows = [
            (key, bucket)
            for key, bucket in validation_aggregates.items()
            if key[:2] == pair and len(bucket["documents"]) >= VALIDATION_MIN_DOCS
        ]
        by_category: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
        for key, bucket in rows:
            by_category[key[2][0]].append((key, bucket))
        for category, category_rows in by_category.items():
            category_rows.sort(key=lambda item: (-len(item[1]["documents"]), -item[1]["event_occurrences"], item[0][2]))
            limit = 3 if category in {"one_to_one_renaming", "many_to_one_granularity", "residual_relation_disagreement", "unmapped_relation_disagreement"} else 1
            shortlist.update(key for key, _ in category_rows[:limit])
    return shortlist


def build_contrast_rows(
    validation_aggregates: Mapping[tuple[str, str, tuple[str, str, str]], Mapping[str, Any]],
    test_aggregates: Mapping[tuple[str, str, tuple[str, str, str]], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    all_keys = sorted(set(validation_aggregates) | set(test_aggregates))
    for pair_left, pair_right, key in all_keys:
        validation = validation_aggregates.get((pair_left, pair_right, key), {"documents": set(), "event_occurrences": 0, "domains": Counter()})
        test = test_aggregates.get((pair_left, pair_right, key), {"documents": set(), "event_occurrences": 0, "domains": Counter()})
        rows.append({
            "left_inventory": pair_left,
            "right_inventory": pair_right,
            "category": key[0],
            "left_relation": key[1],
            "right_relation": key[2],
            "validation_documents": len(validation["documents"]),
            "validation_event_occurrences": validation["event_occurrences"],
            "validation_domains": ";".join(sorted(validation["domains"])),
            "test_documents": len(test["documents"]),
            "test_event_occurrences": test["event_occurrences"],
            "test_domains": ";".join(sorted(test["domains"])),
            "selection_rule": f"validation_documents >= {VALIDATION_MIN_DOCS}; representation-only",
        })
    return rows


def select_candidates(
    raw_candidates: Sequence[Mapping[str, Any]],
    shortlist: set[tuple[str, str, tuple[str, str, str]]],
    validation_aggregates: Mapping[tuple[str, str, tuple[str, str, str]], Mapping[str, Any]],
    catalogs: Mapping[str, Mapping[str, Any]],
    motif_members: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    prepared = []
    for candidate in raw_candidates:
        evidence_by_key = defaultdict(list)
        for scene in candidate["scene_evidence"]:
            for event in scene["events"]:
                evidence_by_key[event_key(event)].append((scene, event))
        eligible = []
        pair = tuple(candidate["pair"])
        for key, evidence in evidence_by_key.items():
            aggregate_key = (pair[0], pair[1], key)
            if aggregate_key in shortlist:
                eligible.append((aggregate_key, evidence))
        if not eligible:
            continue
        eligible.sort(key=lambda item: (-len(validation_aggregates[item[0]]["documents"]), item[0][2]))
        chosen_key, chosen_evidence = eligible[0]
        scene, event = sorted(chosen_evidence, key=lambda item: (item[0]["scene_text"], item[0]["scene_occurrence"]))[0]
        left_scene, right_scene = scene["left_scene"], scene["right_scene"]
        prepared.append({
            "identity_hash": candidate["identity_hash"],
            "source": candidate["source"],
            "domain": candidate["domain"],
            "text": candidate["text"],
            "pair": list(pair),
            "category": chosen_key[2][0],
            "left_relation": chosen_key[2][1],
            "right_relation": chosen_key[2][2],
            "validation_document_support": len(validation_aggregates[chosen_key]["documents"]),
            "selection_rule": "representation-only; shortlist frozen from validation recurrence",
            "scene_text": scene["scene_text"],
            "scene_occurrence": scene["scene_occurrence"],
            "edus": [
                {"id": key, "text": value}
                for key, value in sorted(left_scene.get("edus_text", {}).items())
            ],
            "event": event,
            "evidence": scene["evidence"],
            "left_analysis": {
                "nodes": left_scene["node_labels"],
                "edges": [list(edge) for edge in left_scene["edges"]],
            },
            "right_analysis": {
                "nodes": right_scene["node_labels"],
                "edges": [list(edge) for edge in right_scene["edges"]],
            },
            "_left_scene": left_scene,
            "_right_scene": right_scene,
        })
    # Deterministic cap, balancing pair/category/domain and preventing a single
    # prompt/domain from filling the review pool.
    prepared.sort(key=lambda row: (tuple(row["pair"]), row["category"], row["domain"], row["identity_hash"], row["scene_occurrence"]))
    counts = Counter()
    pair_counts = Counter()
    category_counts = Counter()
    pair_domain_counts = Counter()
    output = []

    def append_row(row: Mapping[str, Any]) -> None:
        pair_key = tuple(row["pair"])
        key = (pair_key, row["category"], row["domain"])
        if pair_counts[pair_key] >= MAX_PER_PAIR or category_counts[(pair_key, row["category"])] >= MAX_PER_PAIR_CATEGORY or counts[key] >= MAX_PER_EVENT_DOMAIN:
            return
        counts[key] += 1
        pair_counts[pair_key] += 1
        category_counts[(pair_key, row["category"])] += 1
        pair_domain_counts[(pair_key, row["domain"])] += 1
        output.append(row)

    # First allocate a small quota to every available category in every pair,
    # so residual disagreements cannot be crowded out by attachment cases.
    categories_by_pair = defaultdict(set)
    for row in prepared:
        categories_by_pair[tuple(row["pair"])].add(row["category"])
    for pair in PAIRS:
        for category in sorted(categories_by_pair[pair]):
            category_rows = [
                row for row in prepared
                if tuple(row["pair"]) == pair and row["category"] == category
            ]
            while category_rows and category_counts[(pair, category)] < MAX_PER_PAIR_CATEGORY:
                category_rows.sort(key=lambda row: (pair_domain_counts[(pair, row["domain"])], row["domain"], row["identity_hash"], row["scene_occurrence"]))
                row = category_rows.pop(0)
                append_row(row)
    # Fill any unused pair quota using the remaining deterministic order.
    for row in prepared:
        before = len(output)
        append_row(row)
        if len(output) >= MAX_CANDIDATES:
            break
        if len(output) == before and pair_counts[tuple(row["pair"])] >= MAX_PER_PAIR:
            continue
    enriched = []
    for index, row in enumerate(output):
        row = dict(row)
        left_scene = row.pop("_left_scene")
        right_scene = row.pop("_right_scene")
        motif_payload = {}
        for inventory, scene_value in ((row["pair"][0], left_scene), (row["pair"][1], right_scene)):
            nonzero = sorted(
                ((motif_hash, float(count)) for motif_hash, count in scene_value["motif_counts"].items() if float(count) > 0),
                key=lambda item: (-item[1], item[0]),
            )[:4]
            motif_payload[inventory] = [
                {
                    "motif_hash": motif_hash,
                    "raw_count": count,
                    "pipeline_family": motif_members.get((inventory, motif_hash), {}).get("pipeline_family"),
                    "description": motif_members.get((inventory, motif_hash), {}).get("description"),
                    "occurrences": exact_motif_occurrences(scene_value, motif_hash, catalogs[inventory]),
                }
                for motif_hash, count in nonzero
            ]
        row["motif_occurrences"] = motif_payload
        row["candidate_id"] = f"contrast-{index + 1:04d}"
        enriched.append(row)
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/unirst2"))
    parser.add_argument("--motif-root", type=Path, default=Path("data/motifs2"))
    parser.add_argument("--correspondence", type=Path, default=Path("artifacts/review/representation_alignment/relation_correspondence.csv"))
    parser.add_argument("--mapping-members", type=Path, default=Path("artifacts/canonical/cross_standard_mapping/members.csv"))
    parser.add_argument("--alignment-manifest", type=Path, default=Path("artifacts/review/representation_alignment/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/review/representation_contrasts"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from scripts.review import align_representations as alignment
    motif_hashes, catalogs = {}, {}
    for inventory in INVENTORIES:
        motif_hashes[inventory], catalogs[inventory] = load_motif_catalog(args.motif_root, inventory)
    rules, correspondence_counts = make_relation_rules(args.correspondence)
    motif_members = {}
    for row in csv.DictReader(args.mapping_members.open(encoding="utf-8")):
        motif_members[(row["inventory"], row["motif_hash"])] = row
    validation_candidates, validation_aggregates = process_split("validation", args.dataset_root, alignment, motif_hashes, rules, catalogs, False)
    test_candidates, test_aggregates = process_split("test", args.dataset_root, alignment, motif_hashes, rules, catalogs, True)
    shortlist = choose_shortlist(validation_aggregates)
    rows = build_contrast_rows(validation_aggregates, test_aggregates)
    selected = select_candidates(test_candidates, shortlist, validation_aggregates, catalogs, motif_members)
    for index, row in enumerate(rows):
        key = (row["left_inventory"], row["right_inventory"], (row["category"], row["left_relation"], row["right_relation"]))
        row["shortlisted_from_validation"] = key in shortlist
    write_csv(args.output_dir / "representation_contrasts.csv", rows)
    jsonl = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected)
    atomic_text(args.output_dir / "candidate_passages.jsonl", jsonl)
    manifest = {
        "stage": "unirst2_representation_contrasts",
        "inventories": list(INVENTORIES),
        "splits": ["validation", "test"],
        "segmentation": "existing graph-added M3 parses; exact scene-text alignment",
        "relation_rules": {
            "source": str(args.correspondence.resolve()),
            "min_support": MAPPING_MIN_SUPPORT,
            "min_conditional": MAPPING_MIN_CONDITIONAL,
            "excluded_scaffolding_label": "/",
        },
        "selection": {
            "validation_min_documents": VALIDATION_MIN_DOCS,
            "max_per_event_domain": MAX_PER_EVENT_DOMAIN,
            "max_per_pair": MAX_PER_PAIR,
            "max_per_pair_category": MAX_PER_PAIR_CATEGORY,
            "max_candidates": MAX_CANDIDATES,
            "uses_labels_for_selection": False,
            "uses_model_scores_for_selection": False,
            "uses_motif_coefficients_for_selection": False,
            "shortlist_event_count": len(shortlist),
            "candidate_count": len(selected),
        },
        "source_hashes": {
            "correspondence": sha256_file(args.correspondence),
            "mapping_members": sha256_file(args.mapping_members),
            "alignment_manifest": sha256_file(args.alignment_manifest),
            "alignment_script": sha256_file(Path(alignment.__file__)),
            "motif_catalogs": {
                inventory: sha256_file(args.motif_root / inventory / "hc3_M3_motifs.json")
                for inventory in INVENTORIES
            },
            "motif_selection_manifests": {
                inventory: sha256_file(args.motif_root / inventory / "hc3_selected-motif-hashes.generated.json")
                for inventory in INVENTORIES
            },
        },
        "output_hashes": {
            "representation_contrasts.csv": sha256_file(args.output_dir / "representation_contrasts.csv"),
            "candidate_passages.jsonl": sha256_file(args.output_dir / "candidate_passages.jsonl"),
        },
    }
    report = [
        "# UniRST2 representation contrasts", "",
        f"The validation-frozen rules produced {len(rows)} contrast summaries and {len(selected)} deterministic test candidate passages.",
        "Relation rules exclude `/` scaffolding links and are learned from the training correspondence table. Validation recurrence, not label or model information, determines the test shortlist.",
        "",
        f"- validation shortlist events: {len(shortlist)}",
        f"- candidate passages: {len(selected)}",
        f"- exact constituent-span differences in candidates: {sum(row['category'] == 'constituent_span_difference' for row in selected)}",
        f"- attachment differences in candidates: {sum(row['category'] == 'attachment_difference' for row in selected)}",
        f"- nuclearity differences in candidates: {sum(row['category'] == 'nuclearity_difference' for row in selected)}",
        f"- one-to-one renaming candidates: {sum(row['category'] == 'one_to_one_renaming' for row in selected)}",
        f"- many-to-one granularity candidates: {sum(row['category'] == 'many_to_one_granularity' for row in selected)}",
        f"- residual relation disagreements: {sum(row['category'] == 'residual_relation_disagreement' for row in selected)}",
        f"- unmapped relation disagreements: {sum(row['category'] == 'unmapped_relation_disagreement' for row in selected)}",
        "",
        "`candidate_passages.jsonl` contains text and both aligned analyses for later partial-blind review. It omits authorship labels from the candidate records; the source parse label is retained only in the audit computation and never used for selection.",
    ]
    atomic_text(args.output_dir / "report.md", "\n".join(report) + "\n")
    manifest["output_hashes"]["report.md"] = sha256_file(args.output_dir / "report.md")
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"contrast_rows": len(rows), "shortlist_events": len(shortlist), "candidate_passages": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
