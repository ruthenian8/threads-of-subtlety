#!/usr/bin/env python3
"""Prepare mapping-review and decisive PCC motif-example packages."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import networkx as nx
import numpy as np
from tqdm import tqdm

from rst_graph_visualizer import RENDERER_VERSION, render_graph_png


INVENTORY = "deu.rst.pcc"
DECISIVE = ("ba99d080", "4540b7da", "19497a19")
SCRIPT_VERSION = "unirst2-minimal-evidence-v1"


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


def graph_from_data(value: Mapping[str, Any]) -> nx.DiGraph:
    return nx.node_link_graph(dict(value))


def analysis_from_graph(graph: nx.DiGraph, highlight: set[str] | None = None) -> dict[str, Any]:
    highlight = highlight or set()
    nodes = {}
    for node, data in graph.nodes(data=True):
        label = str(data.get("label_0", "unknown"))
        nodes[str(node)] = f"{label} ★" if str(node) in highlight else label
    edges = []
    for source, target, data in graph.edges(data=True):
        label = str(data.get("label_0", ""))
        if str(source) in highlight and str(target) in highlight and label != "/":
            label += " ★"
        edges.append([str(source), str(target), label])
    return {"nodes": nodes, "edges": edges}


def coarsen_graph(graph: nx.DiGraph, relation_map: Mapping[str, str]) -> nx.DiGraph:
    result = graph.copy()
    for _, _, data in result.edges(data=True):
        relation = str(data.get("label_0"))
        if relation != "/":
            data["label_0"] = relation_map[relation]
    return result


def first_occurrence(full: nx.DiGraph, motif: nx.DiGraph) -> set[str]:
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(
        full, motif,
        edge_match=lambda left, right: left.get("label_0") == right.get("label_0"),
    )
    try:
        mapping = next(matcher.subgraph_isomorphisms_iter())
    except StopIteration:
        return set()
    return {str(node) for node in mapping}


def read_records(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in tqdm(handle, desc="load validation records", unit="doc"):
            if line.strip():
                rows.append(json.loads(line) if line.lstrip().startswith('{"') else ast.literal_eval(line))
    return rows


def mapping_packet(mapping_path: Path, groups_path: Path, catalog_root: Path, output_root: Path) -> dict[str, Any]:
    with mapping_path.open(newline="", encoding="utf-8") as handle:
        mappings = list(csv.DictReader(handle))
    with groups_path.open(newline="", encoding="utf-8") as handle:
        groups = list(csv.DictReader(handle))
    relation_pairs = sorted({(row["inventory"], relation, coarse) for row in mappings for relation, coarse in zip(row["fine_relations"].split("|"), row["coarse_relations"].split("|")) if relation != "(structural only)"})
    form = {
        "instructions": "Complete independently without consulting model scores or test outcomes. Validate semantic defensibility of each fine-to-coarse mapping and then interpret the selected merge groups.",
        "status": "pending_independent_rst_informed_annotator",
        "relation_mappings": [
            {"inventory": inventory, "fine_relation": fine, "coarse_relation": coarse, "judgment": "", "proposed_replacement": "", "rationale": "", "confidence": ""}
            for inventory, fine, coarse in relation_pairs
        ],
        "decisive_groups": [],
    }
    decisive_rows = [row for row in groups if any(row["coarse_hash"].startswith(prefix) for prefix in DECISIVE)]
    for row in decisive_rows:
        form["decisive_groups"].append({
            "inventory": row["inventory"], "group_id": row["group_id"], "fine_relations": row["fine_relations"].split(";"),
            "coarse_relations": row["coarse_relations"], "mapping_judgment": "", "interpretation": "", "confidence": "",
        })
    review_dir = output_root / "mapping_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    form_path = review_dir / "mapping_review_form.json"
    atomic_json(form_path, form)
    relation_rows = "".join(f"<tr><td>{html.escape(i)}</td><td>{html.escape(f)}</td><td>{html.escape(c)}</td><td></td><td></td></tr>" for i, f, c in relation_pairs)
    group_rows = "".join(f"<tr><td>{html.escape(row['inventory'])}</td><td>{html.escape(row['group_id'])}</td><td>{html.escape(row['fine_relations'])}</td><td>{html.escape(row['coarse_relations'])}</td><td></td></tr>" for row in decisive_rows)
    packet = f"""<!doctype html><meta charset='utf-8'><title>Independent coarse-mapping review</title>
<style>body{{font:16px sans-serif;max-width:1200px;margin:auto;padding:2rem}}table{{border-collapse:collapse;width:100%;margin-bottom:2rem}}td,th{{border:1px solid #aaa;padding:.4rem;vertical-align:top}}.notice{{background:#fff3cd;padding:1rem}}</style>
<h1>Independent RST-informed review of coarse mappings</h1><p class='notice'>Classifier outcomes, authorship labels, and test statistics are intentionally absent. Consult primary relation definitions as needed; do not infer correctness from the target label name alone.</p>
<h2>Task A: relation mappings</h2><p>For every row, judge whether collapsing the fine relation into the stated coarse class is semantically defensible for this controlled ablation. Record acceptable/questionable/incorrect, a replacement if needed, and a short rationale in the JSON form.</p>
<table><tr><th>Inventory</th><th>Fine relation</th><th>Proposed coarse class</th><th>Judgment</th><th>Rationale</th></tr>{relation_rows}</table>
<h2>Task B: selected merge groups</h2><p>Explain what information is lost by each collapse, without judging whether that information predicts authorship.</p>
<table><tr><th>Inventory</th><th>Group</th><th>Fine motif relations</th><th>Collapsed motif</th><th>Interpretation</th></tr>{group_rows}</table>"""
    packet_path = review_dir / "mapping_review_packet.html"
    atomic_text(packet_path, packet)
    return {"form": str(form_path), "packet": str(packet_path), "relation_mapping_count": len(relation_pairs), "group_count": len(decisive_rows)}


def choose_examples(matrix_path: Path, groups_path: Path, per_group: int) -> list[dict[str, Any]]:
    with np.load(matrix_path, allow_pickle=False) as values:
        X = np.asarray(values["X_m3_frequency"], dtype=np.float32)
        labels = values["y"].astype(int)
        domains = values["domain"].astype(str)
        identities = values["identity_hash"].astype(str)
    with groups_path.open(newline="", encoding="utf-8") as handle:
        groups = [row for row in csv.DictReader(handle) if row["inventory"] == INVENTORY and any(row["coarse_hash"].startswith(prefix) for prefix in DECISIVE)]
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for group in groups:
        indices = [int(value) for value in group["fine_indices"].split(";")]
        relations = group["fine_relations"].split(";")
        candidates = []
        for fine_index, relation in zip(indices, relations):
            for row_index in np.argsort(X[:, fine_index])[::-1]:
                if X[row_index, fine_index] <= 0:
                    break
                candidates.append((float(X[row_index, fine_index]), int(row_index), fine_index, relation))
        group_selected = []
        while len(group_selected) < per_group:
            eligible = [candidate for candidate in candidates if identities[candidate[1]] not in used]
            if not eligible:
                break
            prior_relations = {x["fine_relation"] for x in group_selected}
            prior_labels = {x["label"] for x in group_selected}
            prior_domains = {x["domain"] for x in group_selected}
            value, row_index, fine_index, relation = max(
                eligible,
                key=lambda candidate: (
                    candidate[3] not in prior_relations,
                    int(labels[candidate[1]]) not in prior_labels,
                    domains[candidate[1]] not in prior_domains,
                    candidate[0],
                    -candidate[1],
                ),
            )
            identity = identities[row_index]
            group_selected.append({"group_id": group["group_id"], "coarse_hash": group["coarse_hash"], "coarse_relations": group["coarse_relations"], "row_index": row_index, "identity_hash": identity, "domain": domains[row_index], "label": int(labels[row_index]), "fine_index": fine_index, "fine_relation": relation, "frequency": value})
            used.add(identity)
        selected.extend(group_selected)
    return selected


def example_gallery(selected: list[dict[str, Any]], records_path: Path, catalog_path: Path, relation_map: Mapping[str, str], output_root: Path, dpi: int) -> dict[str, Any]:
    records = read_records(records_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    with np.load(Path("artifacts/unirst2_feature_matrices") / INVENTORY / "hc3_validation.features.npz", allow_pickle=False) as values:
        hashes = values["m3_hashes"].astype(str)
    gallery_dir = output_root / "qualitative_examples"
    image_dir = gallery_dir / "images"
    cases = []
    for case_number, item in enumerate(selected, 1):
        record = records[item["row_index"]]
        motif_hash = hashes[item["fine_index"]]
        motif = graph_from_data(catalog[motif_hash])
        chosen_scene = None
        occurrence: set[str] = set()
        trees = record.get("scene_discourse_trees") or {}
        scene_values = trees.values() if isinstance(trees, Mapping) else trees
        for scene in scene_values:
            raw = np.asarray((scene.get("motif_dists") or {}).get("m3", {}).get("raw", []))
            if len(raw) <= item["fine_index"] or raw[item["fine_index"]] <= 0:
                continue
            full = graph_from_data(scene["graph_dict"])
            occurrence = first_occurrence(full, motif)
            if occurrence:
                chosen_scene = scene
                break
        if chosen_scene is None:
            raise ValueError(f"Could not localize motif {motif_hash} in validation row {item['row_index']}")
        full = graph_from_data(chosen_scene["graph_dict"])
        coarse = coarsen_graph(motif, relation_map)
        case_id = f"example-{case_number:02d}"
        paths = {
            "tree": image_dir / f"{case_id}_tree.png",
            "fine": image_dir / f"{case_id}_fine.png",
            "coarse": image_dir / f"{case_id}_coarse.png",
        }
        render_graph_png(analysis_from_graph(full, occurrence), paths["tree"], title=f"{case_id}: complete validation tree; ★ marks localized motif", dpi=dpi)
        render_graph_png(analysis_from_graph(motif), paths["fine"], title=f"{case_id}: fine PCC M3 ({item['fine_relation']})", min_width=8, dpi=dpi)
        render_graph_png(analysis_from_graph(coarse), paths["coarse"], title=f"{case_id}: coarse collapse ({item['coarse_relations']})", min_width=8, dpi=dpi)
        edus = chosen_scene.get("edus") or {}
        cases.append({**item, "case_id": case_id, "motif_hash": str(motif_hash), "text": record.get("text", ""), "edus": edus, "localized_nodes": sorted(occurrence), "images": {key: str(path.relative_to(gallery_dir)) for key, path in paths.items()}})
    cases_path = gallery_dir / "cases.jsonl"
    atomic_text(cases_path, "".join(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in cases))
    sections = []
    for case in cases:
        edu_rows = "".join(f"<tr class={'hit' if key in case['localized_nodes'] else ''}><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>" for key, value in sorted(case["edus"].items(), key=lambda item: int(str(item[0]).split("_")[1].split("-")[0])))
        images = "".join(f"<figure><img src='{html.escape(path)}'><figcaption>{html.escape(kind)}</figcaption></figure>" for kind, path in case["images"].items())
        sections.append(f"<section><h2>{case['case_id']}: {html.escape(case['fine_relation'])} → {html.escape(case['coarse_relations'])}</h2><p>Validation domain: {html.escape(case['domain'])}. Selection used motif frequency and coverage; this is an illustrative, not prevalence-estimating, case. Starred nodes localize one exact occurrence.</p><div class='images'>{images}</div><details><summary>Aligned EDU text</summary><table>{edu_rows}</table></details></section>")
    gallery = """<!doctype html><meta charset='utf-8'><title>Decisive PCC motif examples</title><style>body{font:16px sans-serif;max-width:1500px;margin:auto;padding:2rem}.images{display:grid;grid-template-columns:2fr 1fr 1fr;gap:1rem}img{max-width:100%;cursor:zoom-in;border:1px solid #aaa}img.expanded{max-width:none;width:auto;position:relative;z-index:2;cursor:zoom-out}.hit{background:#fff3a8}table{border-collapse:collapse}td{border:1px solid #bbb;padding:.3rem}section{border-top:2px solid #555;margin-top:2rem}</style><h1>Validation-only PCC fine/coarse motif examples</h1><p>These cases illustrate what the selected distinctions encode in persisted parser outputs. They do not establish linguistic correctness or population prevalence.</p>""" + "".join(sections) + "<script>document.querySelectorAll('img').forEach(x=>x.addEventListener('click',()=>x.classList.toggle('expanded')))</script>"
    gallery_path = gallery_dir / "gallery.html"
    atomic_text(gallery_path, gallery)
    return {"gallery": str(gallery_path), "cases": str(cases_path), "case_count": len(cases), "renderer": RENDERER_VERSION, "image_count": len(cases) * 3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-path", type=Path, default=Path("artifacts/unirst2_granularity_ablation/motif_mapping.csv"))
    parser.add_argument("--groups-path", type=Path, default=Path("artifacts/unirst2_localized_ablation/groups.csv"))
    parser.add_argument("--motif-root", type=Path, default=Path("data/motifs2"))
    parser.add_argument("--feature-root", type=Path, default=Path("artifacts/unirst2_feature_matrices"))
    parser.add_argument("--records-path", type=Path, default=Path("data/unirst2/rel-deu.rst.pcc/hc3_validation.discourse_parsed.graph_added.m3_motif_dists.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/unirst2_minimal_contribution"))
    parser.add_argument("--per-group", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()
    import importlib.util
    spec = importlib.util.spec_from_file_location("granularity_for_evidence", Path("28_ablate_unirst2_m3_granularity.py")); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    packet = mapping_packet(args.mapping_path, args.groups_path, args.motif_root, args.output_root)
    selected = choose_examples(args.feature_root / INVENTORY / "hc3_validation.features.npz", args.groups_path, args.per_group)
    catalog_path = args.motif_root / INVENTORY / "hc3_M3_motifs.json"
    gallery = example_gallery(selected, args.records_path, catalog_path, module.RELATION_MAP, args.output_root, args.dpi)
    inputs = (args.mapping_path, args.groups_path, args.feature_root / INVENTORY / "hc3_validation.features.npz", args.records_path, catalog_path)
    manifest_path = args.output_root / "evidence_manifest.json"
    atomic_json(manifest_path, {"script_version": SCRIPT_VERSION, "mapping_review": packet, "qualitative_gallery": gallery, "selection": "validation only; top observed fine-motif frequency with unique identities and coverage preference", "input_hashes": {str(path): sha256_file(path) for path in inputs}})
    print(json.dumps({"mapping_review": packet, "qualitative_gallery": gallery}, indent=2))


if __name__ == "__main__":
    main()
