#!/usr/bin/env python3
"""Map exact and relation-agnostic M3 motif families across RST inventories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import networkx as nx
from tqdm import tqdm


INVENTORIES = (
    "deu.rst.pcc",
    "eng.erst.gum.new",
    "eng.rst.rstdt.new",
    "nld.rst.nldt",
)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_text(rows: Iterable[Mapping[str, Any]]) -> str:
    rows = list(rows)
    fields = list(rows[0]) if rows else []
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def to_graph(data: Mapping[str, Any]) -> nx.DiGraph:
    graph = nx.node_link_graph(dict(data))
    if not graph.is_directed() or graph.is_multigraph():
        raise ValueError("M3 catalogs must contain directed simple graphs")
    if len(graph) != 3:
        raise ValueError(f"Expected a three-node motif, found {len(graph)} nodes")
    return graph


def exact_isomorphic(left: nx.DiGraph, right: nx.DiGraph) -> bool:
    return nx.is_isomorphic(
        left,
        right,
        node_match=lambda a, b: a.get("label_0") == b.get("label_0"),
        edge_match=lambda a, b: a.get("label_0") == b.get("label_0"),
    )


def pipeline_isomorphic(left: nx.DiGraph, right: nx.DiGraph) -> bool:
    """Match the extraction/distribution identity: edge labels, no node attrs."""
    return nx.is_isomorphic(
        left,
        right,
        edge_match=lambda a, b: a.get("label_0") == b.get("label_0"),
    )


def structural_isomorphic(left: nx.DiGraph, right: nx.DiGraph) -> bool:
    """Ignore discourse-relation names but retain structural '/' edge identity."""
    return nx.is_isomorphic(
        left,
        right,
        node_match=lambda a, b: a.get("label_0") == b.get("label_0"),
        edge_match=lambda a, b: (a.get("label_0") == "/")
        == (b.get("label_0") == "/"),
    )


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)

    def groups(self) -> List[List[int]]:
        grouped: Dict[int, List[int]] = {}
        for index in range(len(self.parent)):
            grouped.setdefault(self.find(index), []).append(index)
        return sorted(grouped.values(), key=lambda group: min(group))


def isomorphic_groups(
    graphs: Sequence[nx.DiGraph], matcher
) -> List[List[int]]:
    union = UnionFind(len(graphs))
    for left, right in itertools.combinations(range(len(graphs)), 2):
        if matcher(graphs[left], graphs[right]):
            union.union(left, right)
    return union.groups()


def motif_description(graph: nx.DiGraph, *, structural: bool = False) -> str:
    nodes = sorted(str(value.get("label_0", "?")) for _, value in graph.nodes(data=True))
    relations = sorted(
        str(value.get("label_0", "?"))
        for _, _, value in graph.edges(data=True)
        if value.get("label_0") != "/"
    )
    if structural:
        relation_text = f"{len(relations)} rhetorical edge(s)"
    else:
        relation_text = "|".join(relations) if relations else "(structural only)"
    return "nodes=" + "|".join(nodes) + "; relations=" + relation_text


def load_catalogs(
    motif_root: Path, audit_dir: Path
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    members: List[Dict[str, Any]] = []
    provenance: Dict[str, str] = {}
    for inventory in tqdm(INVENTORIES, desc="validate catalogs", unit="inventory"):
        catalog_path = motif_root / inventory / "hc3_M3_motifs.json"
        selected_path = (
            motif_root / inventory / "hc3_selected-motif-hashes.generated.json"
        )
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        selected = json.loads(selected_path.read_text(encoding="utf-8"))["m3"]
        if len(selected) != len(set(selected)):
            raise ValueError(f"Duplicate selected hashes in {selected_path}")
        missing = [motif_hash for motif_hash in selected if motif_hash not in catalog]
        if missing:
            raise ValueError(f"{inventory} catalog misses {len(missing)} selected motifs")
        inventory_members = []
        for motif_hash in selected:
            graph_data = catalog[motif_hash]
            graph = to_graph(graph_data)
            inventory_members.append(
                {
                    "inventory": inventory,
                    "motif_hash": motif_hash,
                    "graph_data": graph_data,
                    "graph": graph,
                    "description": motif_description(graph),
                }
            )
        catalog_hash = sha256_file(catalog_path)
        selected_hash = sha256_file(selected_path)
        provenance[f"{inventory}/catalog"] = catalog_hash
        provenance[f"{inventory}/selected"] = selected_hash
        atomic_write_json(
            audit_dir / f"{inventory}.json",
            {
                "inventory": inventory,
                "catalog": str(catalog_path.resolve()),
                "catalog_sha256": catalog_hash,
                "selected_manifest": str(selected_path.resolve()),
                "selected_manifest_sha256": selected_hash,
                "selected_motifs": len(selected),
                "all_selected_hashes_resolved": True,
                "all_graphs_directed_simple_three_node": True,
            },
        )
        members.extend(inventory_members)
    return members, provenance


def load_stability(path: Path) -> Tuple[List[Dict[str, str]], str]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    stable = [row for row in rows if row["stable"] == "True"]
    return stable, sha256_file(path)


def family_summaries(
    members: Sequence[Mapping[str, Any]],
    groups: Sequence[Sequence[int]],
    prefix: str,
    stable_lookup: Mapping[Tuple[str, str], Sequence[Mapping[str, str]]],
    *,
    structural: bool,
    ignore_nodes: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
    ordered_groups = sorted(
        groups,
        key=lambda group: min(
            (members[index]["motif_hash"], members[index]["inventory"])
            for index in group
        ),
    )
    summaries: List[Dict[str, Any]] = []
    assignments: Dict[int, str] = {}
    for number, group in enumerate(ordered_groups, 1):
        family_id = f"{prefix}-{number:03d}"
        for index in group:
            assignments[index] = family_id
        family_members = [members[index] for index in group]
        stable = [
            effect
            for member in family_members
            for effect in stable_lookup.get(
                (member["inventory"], member["motif_hash"]), ()
            )
        ]
        representative = family_members[0]
        node_signatures = sorted(
            {
                "|".join(
                    sorted(
                        str(value.get("label_0", "?"))
                        for _, value in member["graph"].nodes(data=True)
                    )
                )
                for member in family_members
            }
        )
        description = motif_description(
            representative["graph"], structural=structural
        )
        if ignore_nodes:
            relation_text = description.split("; relations=", 1)[1]
            description = (
                "nodes=(ignored; variants="
                + " / ".join(node_signatures)
                + "); relations="
                + relation_text
            )
        summaries.append(
            {
                "family_id": family_id,
                "member_count": len(group),
                "inventory_count": len(
                    {member["inventory"] for member in family_members}
                ),
                "inventories": sorted(
                    {member["inventory"] for member in family_members}
                ),
                "motif_hashes": sorted(
                    {member["motif_hash"] for member in family_members}
                ),
                "relation_signatures": sorted(
                    {member["description"].split("; relations=", 1)[1] for member in family_members}
                ),
                "node_signatures": node_signatures,
                "description": description,
                "stable_association_count": len(stable),
                "stable_inventory_count": len(
                    {effect["inventory"] for effect in stable}
                ),
                "stable_associations": sorted(
                    {
                        f"{effect['inventory']}:{effect['representation']}:{effect['stable_direction']}"
                        for effect in stable
                    }
                ),
                "representative_graph": representative["graph_data"],
            }
        )
    return summaries, assignments


def verify_repeated_hashes(
    members: Sequence[Mapping[str, Any]],
    pipeline_assignment: Mapping[int, str],
    nuclearity_assignment: Mapping[int, str],
) -> Dict[str, Any]:
    by_hash: Dict[str, List[int]] = {}
    for index, member in enumerate(members):
        by_hash.setdefault(member["motif_hash"], []).append(index)
    repeated = {key: value for key, value in by_hash.items() if len(value) > 1}
    collisions = []
    nuclearity_ambiguities = []
    for motif_hash, indices in repeated.items():
        pipeline_families = {pipeline_assignment[index] for index in indices}
        nuclearity_families = {nuclearity_assignment[index] for index in indices}
        if len(pipeline_families) != 1:
            collisions.append(
                {
                    "motif_hash": motif_hash,
                    "inventories": sorted(
                        {members[index]["inventory"] for index in indices}
                    ),
                    "pipeline_families": sorted(pipeline_families),
                }
            )
        if len(nuclearity_families) != 1:
            nuclearity_ambiguities.append(
                {
                    "motif_hash": motif_hash,
                    "inventories": sorted(
                        {members[index]["inventory"] for index in indices}
                    ),
                    "nuclearity_families": sorted(nuclearity_families),
                }
            )
    pipeline_family_hashes: Dict[str, set] = {}
    for index, family in pipeline_assignment.items():
        pipeline_family_hashes.setdefault(family, set()).add(
            members[index]["motif_hash"]
        )
    different_hash_equivalents = {
        family: sorted(hashes)
        for family, hashes in pipeline_family_hashes.items()
        if len(hashes) > 1
    }
    return {
        "unique_hashes": len(by_hash),
        "repeated_hash_groups": len(repeated),
        "repeated_hash_occurrences": sum(len(indices) for indices in repeated.values()),
        "repeated_hashes_all_pipeline_isomorphic": not collisions,
        "hash_collision_count": len(collisions),
        "hash_collisions": collisions,
        "nuclearity_ambiguity_count": len(nuclearity_ambiguities),
        "nuclearity_ambiguities": nuclearity_ambiguities,
        "different_hash_pipeline_equivalence_count": len(different_hash_equivalents),
        "different_hash_pipeline_equivalences": different_hash_equivalents,
    }


def render_report(
    members: Sequence[Mapping[str, Any]],
    pipeline_families: Sequence[Mapping[str, Any]],
    exact: Sequence[Mapping[str, Any]],
    structural: Sequence[Mapping[str, Any]],
    verification: Mapping[str, Any],
    stable_count: int,
) -> str:
    cross_pipeline = [
        family for family in pipeline_families if family["inventory_count"] >= 2
    ]
    cross_exact = [family for family in exact if family["inventory_count"] >= 2]
    cross_structural = [
        family for family in structural if family["inventory_count"] >= 2
    ]
    stable_pipeline = [
        family
        for family in pipeline_families
        if family["stable_inventory_count"] >= 2
    ]
    stable_exact = [
        family for family in exact if family["stable_inventory_count"] >= 2
    ]
    stable_structural = [
        family for family in structural if family["stable_inventory_count"] >= 2
    ]
    lines = [
        "# UniRST2 cross-standard M3 motif mapping",
        "",
        f"The four selected catalogs contain {len(members)} inventory-specific entries "
        f"and {verification['unique_hashes']} unique hashes. Pipeline mapping reproduces "
        "the feature identity by preserving edge relation labels while ignoring node "
        "attributes. Nuclearity-aware mapping additionally preserves node labels; "
        "structural mapping preserves topology, "
        "node nuclearity, and structural-versus-rhetorical edge type while ignoring the "
        "relation name.",
        "",
        "## Verification",
        "",
        f"- Repeated hash groups: {verification['repeated_hash_groups']}",
        f"- Hash collisions: {verification['hash_collision_count']}",
        f"- Repeated hashes with multiple nuclearity variants: {verification['nuclearity_ambiguity_count']}",
        f"- Pipeline-equivalent families spanning multiple standards: {len(cross_pipeline)}",
        f"- Nuclearity-aware families spanning multiple standards: {len(cross_exact)}",
        f"- Relation-agnostic structural families spanning multiple standards: {len(cross_structural)}",
        f"- Robust stable associations projected into the map: {stable_count}",
        "",
        (
            f"All repeated hashes are isomorphic under the pipeline's declared edge-only "
            f"identity, so no WL collision was found under the operative feature definition. "
            f"However, {verification['nuclearity_ambiguity_count']} hashes cover multiple "
            "node-nuclearity variants because extraction explicitly omits node attributes. "
            "Nuclearity-specific interpretations must therefore use the stricter mapping."
        ),
        "",
        "## Pipeline-equivalent families stable in multiple standards",
        "",
        "| Family | Standards | Hashes | Stable associations | Description |",
        "|---|---:|---|---|---|",
    ]
    for family in stable_pipeline:
        lines.append(
            f"| {family['family_id']} | {family['stable_inventory_count']} | "
            f"`{'`, `'.join(family['motif_hashes'])}` | "
            f"{'<br>'.join(family['stable_associations'])} | {family['description']} |"
        )
    if not stable_pipeline:
        lines.append("| — | 0 | — | — | No pipeline family was stable in multiple standards |")
    lines.extend(
        [
            "",
            "## Nuclearity-aware families stable in multiple standards",
            "",
            "| Family | Standards | Hashes | Stable associations | Description |",
            "|---|---:|---|---|---|",
        ]
    )
    for family in stable_exact:
        lines.append(
            f"| {family['family_id']} | {family['stable_inventory_count']} | "
            f"`{'`, `'.join(family['motif_hashes'])}` | "
            f"{'<br>'.join(family['stable_associations'])} | {family['description']} |"
        )
    if not stable_exact:
        lines.append("| — | 0 | — | — | No nuclearity-aware family was stable in multiple standards |")
    lines.extend(
        [
            "",
            "## Structural families stable in multiple standards",
            "",
            "| Family | Standards | Relation variants | Stable associations | Structure |",
            "|---|---:|---|---|---|",
        ]
    )
    for family in stable_structural:
        lines.append(
            f"| {family['family_id']} | {family['stable_inventory_count']} | "
            f"{'<br>'.join(family['relation_signatures'])} | "
            f"{'<br>'.join(family['stable_associations'])} | {family['description']} |"
        )
    if not stable_structural:
        lines.append("| — | 0 | — | — | No structural family was stable in multiple standards |")
    lines.extend(
        [
            "",
            "Pipeline families are the faithful unit for the existing feature results. "
            "The nuclearity-aware mapping is a diagnostic because the current features "
            "do not distinguish its variants. Structural mapping establishes broader "
            "comparability, but it does not assert "
            "that differently named relations are semantically equivalent. Interpretation "
            "should use pipeline families for claims about the computed features and the "
            "nuclearity-aware families only to diagnose distinctions those features omit. "
            "Structural families identify candidate inventory-specific realizations for "
            "follow-up.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motif-dir", default="data/motifs2")
    parser.add_argument(
        "--stability-file",
        default="artifacts/canonical/robustness/motif_stability.csv",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/canonical/cross_standard_mapping"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    motif_root = Path(args.motif_dir)
    stability_path = Path(args.stability_file)
    output_dir = Path(args.output_dir)
    provenance = {
        "stability": sha256_file(stability_path),
        **{
            f"{inventory}/{kind}": sha256_file(
                motif_root
                / inventory
                / (
                    "hc3_M3_motifs.json"
                    if kind == "catalog"
                    else "hc3_selected-motif-hashes.generated.json"
                )
            )
            for inventory in INVENTORIES
            for kind in ("catalog", "selected")
        },
    }
    manifest_path = output_dir / "manifest.json"
    required = (
        output_dir / "members.csv",
        output_dir / "pipeline_families.json",
        output_dir / "exact_families.json",
        output_dir / "structural_families.json",
        output_dir / "verification.json",
        output_dir / "report.md",
    )
    if manifest_path.exists() and all(path.exists() for path in required) and not args.force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("provenance") == provenance:
            print(f"skipping complete mapping; wrote {output_dir / 'report.md'}")
            return

    members, loaded_provenance = load_catalogs(
        motif_root, output_dir / "inventory_audit"
    )
    if any(provenance[key] != value for key, value in loaded_provenance.items()):
        raise ValueError("Catalog provenance changed during mapping")
    stable, stability_hash = load_stability(stability_path)
    if stability_hash != provenance["stability"]:
        raise ValueError("Stability provenance changed during mapping")
    stable_lookup: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in stable:
        stable_lookup.setdefault((row["inventory"], row["motif_hash"]), []).append(row)

    graphs = [member["graph"] for member in members]
    pipeline_groups = isomorphic_groups(
        graphs, lambda left, right: pipeline_isomorphic(left, right)
    )
    exact_groups = isomorphic_groups(
        graphs, lambda left, right: exact_isomorphic(left, right)
    )
    structural_groups = isomorphic_groups(
        graphs, lambda left, right: structural_isomorphic(left, right)
    )
    pipeline_families, pipeline_assignment = family_summaries(
        members,
        pipeline_groups,
        "pipeline",
        stable_lookup,
        structural=False,
        ignore_nodes=True,
    )
    exact, exact_assignment = family_summaries(
        members, exact_groups, "exact", stable_lookup, structural=False
    )
    structural, structural_assignment = family_summaries(
        members, structural_groups, "structural", stable_lookup, structural=True
    )
    verification = verify_repeated_hashes(
        members, pipeline_assignment, exact_assignment
    )
    atomic_write_json(output_dir / "verification.json", verification)
    ambiguous_hashes = {
        ambiguity["motif_hash"]
        for ambiguity in verification["nuclearity_ambiguities"]
    }

    member_rows = []
    for index, member in enumerate(members):
        effects = stable_lookup.get((member["inventory"], member["motif_hash"]), ())
        member_rows.append(
            {
                "inventory": member["inventory"],
                "motif_hash": member["motif_hash"],
                "pipeline_family": pipeline_assignment[index],
                "exact_family": exact_assignment[index],
                "structural_family": structural_assignment[index],
                "nuclearity_ambiguous_hash": member["motif_hash"] in ambiguous_hashes,
                "robust_stable": bool(effects),
                "stable_associations": ";".join(
                    sorted(
                        f"{effect['representation']}:{effect['stable_direction']}"
                        for effect in effects
                    )
                ),
                "description": member["description"],
            }
        )
    atomic_write_text(output_dir / "members.csv", csv_text(member_rows))
    atomic_write_json(
        output_dir / "pipeline_families.json", {"families": pipeline_families}
    )
    atomic_write_text(output_dir / "pipeline_families.csv", csv_text(
        [{key: (";".join(value) if isinstance(value, list) else value)
          for key, value in family.items() if key != "representative_graph"}
         for family in pipeline_families]
    ))
    atomic_write_json(output_dir / "exact_families.json", {"families": exact})
    atomic_write_text(output_dir / "exact_families.csv", csv_text(
        [{key: (";".join(value) if isinstance(value, list) else value)
          for key, value in family.items() if key != "representative_graph"}
         for family in exact]
    ))
    atomic_write_json(
        output_dir / "structural_families.json", {"families": structural}
    )
    atomic_write_text(output_dir / "structural_families.csv", csv_text(
        [{key: (";".join(value) if isinstance(value, list) else value)
          for key, value in family.items() if key != "representative_graph"}
         for family in structural]
    ))
    report = render_report(
        members, pipeline_families, exact, structural, verification, len(stable)
    )
    atomic_write_text(output_dir / "report.md", report)
    atomic_write_json(
        manifest_path,
        {
            "provenance": provenance,
            "catalog_entries": len(members),
            "pipeline_families": len(pipeline_families),
            "exact_families": len(exact),
            "structural_families": len(structural),
            "robust_stable_associations": len(stable),
            "report_sha256": sha256_file(output_dir / "report.md"),
        },
    )
    print(f"wrote {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
