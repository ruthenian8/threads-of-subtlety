#!/usr/bin/env python3
"""Prepare a deterministic, partially blinded UniRST2 review package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PAIRS = (
    ("deu.rst.pcc", "eng.erst.gum.new"),
    ("deu.rst.pcc", "eng.rst.rstdt.new"),
    ("deu.rst.pcc", "nld.rst.nldt"),
    ("eng.erst.gum.new", "eng.rst.rstdt.new"),
    ("eng.erst.gum.new", "nld.rst.nldt"),
    ("eng.rst.rstdt.new", "nld.rst.nldt"),
)
CATEGORIES = (
    "attachment_difference",
    "one_to_one_renaming",
    "many_to_one_granularity",
    "residual_relation_disagreement",
    "unmapped_relation_disagreement",
)
DOMAINS = ("finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai")
SEED = "unirst2-review-v1"
TARGET_CASES = 60
SPAN_RE = re.compile(r"^span_(\d+)-(\d+)$")


def span_sort_key(value: Any) -> tuple[int, int, str]:
    """Sort span identifiers by numeric start/end indices, not text order."""
    text = str(value)
    match = SPAN_RE.match(text)
    if match:
        return int(match.group(1)), int(match.group(2)), text
    return 10**9, 10**9, text


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


def tie_key(row: Mapping[str, Any]) -> str:
    payload = "|".join((SEED, str(row["identity_hash"]), str(row["pair"]), str(row["category"])))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_clusters(path: Path) -> dict[str, str]:
    identity_to_cluster = {}
    if not path.exists():
        return identity_to_cluster
    for line in path.open(encoding="utf-8"):
        payload = json.loads(line)
        for member in payload.get("members", []):
            identity_to_cluster[str(member["identity_hash"])] = str(payload["cluster_id"])
    return identity_to_cluster


def load_candidates(path: Path, clusters_path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str], set[str]]:
    clusters = load_clusters(clusters_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_ids: set[str] = set()
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        identity = str(row["identity_hash"])
        candidate_ids.add(identity)
        if identity in clusters:
            continue
        grouped[identity].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: (tuple(row["pair"]), row["category"], row["domain"], tie_key(row)))
    return grouped, clusters, candidate_ids


def choose_rows(grouped: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    pair_counts = Counter()
    domain_counts = Counter()
    category_counts = Counter()
    pair_domain_counts = Counter()

    def add(row: Mapping[str, Any]) -> bool:
        identity = str(row["identity_hash"])
        if identity in used or len(selected) >= TARGET_CASES:
            return False
        used.add(identity)
        copied = dict(row)
        selected.append(copied)
        pair = tuple(row["pair"])
        pair_counts[pair] += 1
        domain_counts[row["domain"]] += 1
        category_counts[row["category"]] += 1
        pair_domain_counts[(pair, row["domain"])] += 1
        return True

    # Satisfy coverage requirements before optimizing the remaining sample.
    for pair in PAIRS:
        options = [row for rows in grouped.values() for row in rows if tuple(row["pair"]) == pair and row["identity_hash"] not in used]
        if options:
            add(min(options, key=tie_key))
    for domain in DOMAINS:
        options = [row for rows in grouped.values() for row in rows if row["domain"] == domain and row["identity_hash"] not in used]
        if options:
            add(min(options, key=tie_key))
    for category in CATEGORIES:
        options = [row for rows in grouped.values() for row in rows if row["category"] == category and row["identity_hash"] not in used]
        if options:
            add(min(options, key=tie_key))

    while len(selected) < min(TARGET_CASES, len(grouped)):
        options = [row for rows in grouped.values() for row in rows if row["identity_hash"] not in used]
        if not options:
            break
        def score(row: Mapping[str, Any]) -> tuple[Any, ...]:
            pair = tuple(row["pair"])
            # Larger deficits first; hash is a deterministic final tie-break.
            return (
                int(pair_counts[pair] < 8),
                int(domain_counts[row["domain"]] < 10),
                int(category_counts[row["category"]] < 10),
                int(pair_domain_counts[(pair, row["domain"])] < 2),
                -pair_counts[pair],
                -domain_counts[row["domain"]],
                -category_counts[row["category"]],
                tie_key(row),
            )
        add(max(options, key=score))
    selected.sort(key=lambda row: tie_key(row))
    return selected


def neutralize(selected: Sequence[Mapping[str, Any]], labels_by_identity: Mapping[str, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases, answer_key = [], []
    for index, row in enumerate(selected, 1):
        case_id = f"case-{index:03d}"
        flip = int(hashlib.sha256(f"{SEED}|order|{row['identity_hash']}".encode()).hexdigest()[:2], 16) % 2
        left_is_a = flip == 0
        left, right = row["left_analysis"], row["right_analysis"]
        analysis_a, analysis_b = (left, right) if left_is_a else (right, left)
        cases.append({
            "case_id": case_id,
            "text": row["text"],
            "edus": row["edus"],
            "analysis_a": analysis_a,
            "analysis_b": analysis_b,
            "instruction": "Compare analyses A and B for the passage below. Relation names are native labels and may reveal the annotation system; do not infer authorship from the passage.",
        })
        answer_key.append({
            "case_id": case_id,
            "identity_hash": row["identity_hash"],
            "source": row["source"],
            "domain": row["domain"],
            "label": int(labels_by_identity[row["identity_hash"]]),
            "pair": row["pair"],
            "category": row["category"],
            "left_relation": row.get("left_relation"),
            "right_relation": row.get("right_relation"),
            "analysis_a_inventory": row["pair"][0] if left_is_a else row["pair"][1],
            "analysis_b_inventory": row["pair"][1] if left_is_a else row["pair"][0],
            "event": row.get("event"),
            "validation_document_support": row.get("validation_document_support"),
            "motif_occurrences": row.get("motif_occurrences"),
        })
    return cases, answer_key


def render_analysis(analysis: Mapping[str, Any]) -> str:
    nodes = "".join(
        f"<li><code>{html.escape(str(node))}</code>: {html.escape(str(label))}</li>"
        for node, label in sorted((analysis.get("nodes") or {}).items(), key=lambda item: span_sort_key(item[0]))
    )
    edges = "".join(
        f"<tr><td><code>{html.escape(str(edge[0]))}</code></td><td><code>{html.escape(str(edge[1]))}</code></td><td>{html.escape(str(edge[2]))}</td></tr>"
        for edge in sorted(
            analysis.get("edges") or [],
            key=lambda edge: (span_sort_key(edge[0]), span_sort_key(edge[1]), str(edge[2])),
        )
    )
    return f"<h4>Nodes</h4><ul>{nodes}</ul><h4>Links</h4><table><tr><th>Source</th><th>Target</th><th>Label</th></tr>{edges}</table>"


def render_tree_panel(title: str, image_path: str | None) -> str:
    """Render an optional graph image that is visible when the case opens."""
    if not image_path:
        return ""
    escaped = html.escape(str(image_path), quote=True)
    alt = html.escape(f"{title} rendered relation graph", quote=True)
    return (
        "<figure class='tree-panel'>"
        f"<figcaption>{html.escape(title)} — rendered graph (PNG)</figcaption>"
        f"<img loading='lazy' src='{escaped}' alt='{alt}'>"
        "</figure>"
    )


def render_graph_details(title: str, analysis: Mapping[str, Any]) -> str:
    """Keep the tabular graph representation available but collapsible."""
    return (
        f"<details class='graph-details'><summary>{html.escape(title)}"
        " — nodes and links</summary>"
        f"{render_analysis(analysis)}</details>"
    )


def render_html(cases: Sequence[Mapping[str, Any]]) -> str:
    sections = []
    for case in cases:
        edus = "".join(
            f"<tr><td><code>{html.escape(str(edu['id']))}</code></td><td>{html.escape(str(edu['text']))}</td></tr>"
            for edu in sorted(case["edus"], key=lambda edu: span_sort_key(edu["id"]))
        )
        tree_images = case.get("tree_images") or {}
        sections.append(
            f"<section class='case'><h2>{html.escape(case['case_id'])}</h2>"
            f"<p>{html.escape(case['instruction'])}</p><h3>Passage</h3><p class='text'>{html.escape(case['text'])}</p>"
            f"<h3>Aligned EDUs</h3><table><tr><th>ID</th><th>Text</th></tr>{edus}</table>"
            f"<label class='notes-label' for='notes-{html.escape(case['case_id'], quote=True)}'>Notes</label>"
            f"<textarea class='notes' id='notes-{html.escape(case['case_id'], quote=True)}' rows='4' "
            "placeholder='Record your observations for this case...'></textarea>"
            f"<div class='analyses'><div><h3>Analysis A</h3>"
            f"{render_tree_panel('Analysis A', tree_images.get('analysis_a'))}"
            f"{render_graph_details('Analysis A', case['analysis_a'])}</div>"
            f"<div><h3>Analysis B</h3>"
            f"{render_tree_panel('Analysis B', tree_images.get('analysis_b'))}"
            f"{render_graph_details('Analysis B', case['analysis_b'])}</div></div></section>"
        )
    return """<!doctype html><html><head><meta charset='utf-8'><title>UniRST2 blinded review</title>
<style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;line-height:1.4}.case{border-top:3px solid #333;margin:3rem 0;padding-top:1rem}.text{white-space:pre-wrap;background:#f4f4f4;padding:1rem}table{border-collapse:collapse;width:100%;margin:.5rem 0 1rem}th,td{border:1px solid #bbb;text-align:left;padding:.35rem;vertical-align:top}th{background:#eee}.notes-label{display:block;font-weight:600;margin-top:1rem}.notes{box-sizing:border-box;display:block;width:100%;margin:.35rem 0 1rem;padding:.55rem;border:1px solid #aaa;border-radius:.25rem;font:inherit;resize:vertical}.analyses{display:grid;grid-template-columns:1fr 1fr;gap:2rem}.analyses>div{min-width:0}.tree-panel,.graph-details{margin:.75rem 0;border:1px solid #bbb;border-radius:.25rem;padding:.4rem .7rem;background:#fafafa}.graph-details summary{cursor:pointer;font-weight:600}.tree-panel figcaption{font-weight:600}.tree-panel img{display:block;width:100%;height:auto;margin-top:.75rem;background:#fff;border:1px solid #ddd;cursor:zoom-in}.tree-panel img.expanded{position:relative;width:200%;max-width:200%;margin-left:-50%;cursor:zoom-out;z-index:3;box-shadow:0 8px 28px rgba(0,0,0,.28)}code{font-size:.9em}@media(max-width:800px){.analyses{grid-template-columns:1fr}.tree-panel img.expanded{width:100%;max-width:100%;margin-left:0}}</style></head><body>
<h1>UniRST2 blinded representation review</h1><p>This package is partially blinded. Native relation labels may reveal the annotation system. Authorship labels, domains, model scores, and inventory identifiers are withheld here.</p>
""" + "\n".join(sections) + """</body><script>
document.querySelectorAll('.tree-panel img').forEach(function (image) {
  image.addEventListener('click', function () {
    image.classList.toggle('expanded');
  });
});
</script></html>
"""


def codebook() -> str:
    return """# UniRST2 representation review codebook

## Scope

Review each passage independently. Analysis A and Analysis B are native
representations of the same shared EDU segmentation. Inventory identity and
authorship label are withheld, but native relation names may partially reveal
the annotation system. Do not guess authorship or use relation-name familiarity
as a quality criterion.

## Record for each case

1. **Textual support:** Are the discourse links in each analysis plausibly
   supported by the passage? Record `A`, `B`, `both`, `neither`, or `uncertain`
   for the overall judgment and optionally per disputed link.
2. **Well-formedness:** Does each tree appear internally coherent and aligned
   to the EDUs? Record `A`, `B`, `both`, `neither`, or `uncertain`.
3. **Defensibility:** Are both analyses defensible descriptions of the passage?
   Use `both`, `one_only`, `neither`, or `uncertain`.
4. **Difference type:** Select all that apply: `mere_renaming`,
   `granularity`, `attachment`, `nuclearity`, `alternative_interpretation`,
   `parser_or_alignment_error`, `no_meaningful_difference`, or `uncertain`.
5. **Coherence distinction:** If the analyses differ meaningfully, describe
   in plain language what each analysis highlights (for example temporal
   organization, causal explanation, evaluation, attribution, or discourse
   structure). Do not infer a standard's theoretical meaning from the label
   alone.
6. **Confidence:** `high`, `medium`, or `low`, with a short justification.

Use `neither` and `uncertain` freely. These data are purposively sampled and
do not estimate population prevalence. Adjudication should preserve the
original free-text rationale and record disagreements rather than forcing a
consensus.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", type=Path, default=Path("artifacts/review/representation_contrasts/candidate_passages.jsonl"))
    parser.add_argument("--clusters-path", type=Path, default=Path("artifacts/review/near_duplicate_clusters.jsonl"))
    parser.add_argument("--feature-path", type=Path, default=Path("artifacts/canonical/feature_matrices/deu.rst.pcc/hc3_test.features.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/review/human_review"))
    args = parser.parse_args()
    grouped, clusters, candidate_ids = load_candidates(args.candidate_path, args.clusters_path)
    with np.load(args.feature_path, allow_pickle=False) as values:
        labels_by_identity = {
            str(identity): int(label)
            for identity, label in zip(values["identity_hash"], values["y"])
        }
    missing_labels = set(grouped) - set(labels_by_identity)
    if missing_labels:
        raise ValueError(f"candidate identities missing from feature labels: {len(missing_labels)}")
    selected = choose_rows(grouped)
    cases, answer_key = neutralize(selected, labels_by_identity)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = args.output_dir / "blinded_cases.jsonl"
    key_path = args.output_dir / "sealed_answer_key.jsonl"
    html_path = args.output_dir / "blinded_cases.html"
    codebook_path = args.output_dir / "codebook.md"
    sampling_path = args.output_dir / "sampling_manifest.json"
    atomic_text(cases_path, "".join(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in cases))
    atomic_text(key_path, "".join(json.dumps(key, ensure_ascii=False, sort_keys=True) + "\n" for key in answer_key))
    os.chmod(key_path, 0o600)
    atomic_text(html_path, render_html(cases))
    atomic_text(codebook_path, codebook())
    pair_counts = Counter(tuple(row["pair"]) for row in selected)
    domain_counts = Counter(row["domain"] for row in selected)
    category_counts = Counter(row["category"] for row in selected)
    manifest = {
        "stage": "unirst2_blinded_review_package",
        "sampling_seed": SEED,
        "source": str(args.candidate_path.resolve()),
        "source_sha256": sha256_file(args.candidate_path),
        "near_duplicate_source": str(args.clusters_path.resolve()),
        "near_duplicate_source_sha256": sha256_file(args.clusters_path),
        "feature_label_source": str(args.feature_path.resolve()),
        "feature_label_source_sha256": sha256_file(args.feature_path),
        "candidate_pool_before_cluster_exclusion": len(candidate_ids),
        "excluded_identity_count": len(candidate_ids & set(clusters)),
        "available_unique_identities_after_exclusion": len(grouped),
        "selected_cases": len(cases),
        "coverage": {
            "pairs": {" / ".join(pair): count for pair, count in sorted(pair_counts.items())},
            "domains": dict(sorted(domain_counts.items())),
            "categories": dict(sorted(category_counts.items())),
        },
        "blinding": {
            "withheld": ["inventory identifiers", "pair identifiers", "domain", "source", "authorship label", "model scores", "motif coefficient signs"],
            "partial_blinding_caveat": "Native relation labels and structural conventions may reveal the annotation system.",
            "answer_key_permissions": "0600",
        },
    }
    manifest["output_hashes"] = {
        path.name: sha256_file(path)
        for path in (cases_path, key_path, html_path, codebook_path)
    }
    atomic_json(sampling_path, manifest)
    print(json.dumps({"selected_cases": len(cases), "unique_identities": len(grouped), "pairs": {" / ".join(pair): count for pair, count in pair_counts.items()}, "domains": dict(domain_counts), "categories": dict(category_counts)}, indent=2, default=str))


if __name__ == "__main__":
    main()
