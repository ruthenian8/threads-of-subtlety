"""Deterministic, lossless-enough PNG rendering for UniRST graph records.

The review data stores a relation graph rather than an RS3 document.  An RS3
round-trip is not always semantics-preserving for these records (notably for
parallel links and the explicit ``/`` dominance links), so this module draws
the graph directly.  Every node is labelled, every dominance link is drawn,
and every non-dominance link carries its relation label.  The EDU table in the
review HTML remains the source for the full EDU text.

The module deliberately keeps the renderer independent from the review
sampling code.  It can therefore also be used to inspect a single graph:

    rstenv/bin/python rst_graph_visualizer.py --input graph.json --output graph.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SPAN_RE = re.compile(r"^span_(\d+)-(\d+)$")
RENDERER_VERSION = "direct-graph-v1"


def _span_key(node: str) -> tuple[int, int, str]:
    match = SPAN_RE.match(str(node))
    if match:
        return int(match.group(1)), int(match.group(2)), str(node)
    return 10**9, 10**9, str(node)


def _span_bounds(node: str) -> tuple[int, int]:
    match = SPAN_RE.match(str(node))
    if match:
        return int(match.group(1)), int(match.group(2))
    # Unknown IDs are still rendered rather than silently dropped.
    digest = int(hashlib.sha256(str(node).encode("utf-8")).hexdigest()[:8], 16)
    return digest % 100000, digest % 100000


def _edge_parts(analysis: Mapping[str, Any]) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    dominance: list[tuple[str, str]] = []
    relations: list[tuple[str, str, str]] = []
    for raw in analysis.get("edges") or []:
        if len(raw) < 3:
            continue
        source, target, label = (str(raw[0]), str(raw[1]), str(raw[2]))
        if label == "/":
            dominance.append((source, target))
        else:
            relations.append((source, target, label))
    return dominance, relations


def _depths(nodes: Sequence[str], dominance: Sequence[tuple[str, str]]) -> tuple[dict[str, int], list[str]]:
    """Compute leaf-to-root levels from child→parent dominance links."""
    parent: dict[str, str] = {}
    warnings: list[str] = []
    for child, ancestor in dominance:
        previous = parent.get(child)
        if previous is not None and previous != ancestor:
            warnings.append(f"node {child} has multiple dominance parents")
        else:
            parent[child] = ancestor

    memo: dict[str, int] = {}
    visiting: set[str] = set()

    def visit(node: str) -> int:
        if node in memo:
            return memo[node]
        if node in visiting:
            warnings.append(f"dominance cycle involving {node}")
            return 0
        visiting.add(node)
        level = visit(parent[node]) + 1 if node in parent else 0
        visiting.remove(node)
        memo[node] = level
        return level

    for node in nodes:
        visit(node)
    return memo, warnings


def _layout(analysis: Mapping[str, Any]) -> tuple[dict[str, tuple[float, float]], list[str]]:
    nodes = sorted({str(node) for node in (analysis.get("nodes") or {}).keys()}, key=_span_key)
    dominance, _ = _edge_parts(analysis)
    levels, warnings = _depths(nodes, dominance)

    # Put spans over their covered EDU interval.  A deterministic small spread
    # keeps equal-span or equal-level nodes fully readable without randomness.
    buckets: dict[tuple[int, float], list[str]] = defaultdict(list)
    for node in nodes:
        start, end = _span_bounds(node)
        buckets[(levels[node], (start + end) / 2.0)].append(node)
    positions: dict[str, tuple[float, float]] = {}
    for (level, centre), bucket in sorted(buckets.items()):
        bucket.sort(key=_span_key)
        spread = max(0.45, 0.85 * (len(bucket) - 1))
        for index, node in enumerate(bucket):
            offset = (index * 0.85) - spread / 2.0
            positions[node] = (centre + offset, float(level))
    return positions, warnings


def render_graph_png(
    analysis: Mapping[str, Any],
    output_path: Path,
    *,
    title: str = "",
    min_width: float = 24.0,
    dpi: int = 220,
) -> dict[str, Any]:
    """Render one graph and return auditable rendering metadata.

    The figure uses a large canvas and ``bbox_inches='tight'`` so labels and
    relation arrows are not clipped.  The returned metadata is suitable for a
    manifest and includes every input edge count and any structural warnings.
    """
    node_labels = {str(node): str(label) for node, label in (analysis.get("nodes") or {}).items()}
    nodes = sorted(node_labels, key=_span_key)
    dominance, relations = _edge_parts(analysis)
    positions, warnings = _layout(analysis)
    missing = {
        endpoint
        for edge in [*dominance, *[(s, t) for s, t, _ in relations]]
        for endpoint in edge
        if endpoint not in node_labels
    }
    if missing:
        warnings.append(f"edges reference nodes absent from node map: {sorted(missing)}")
        for endpoint in sorted(missing, key=_span_key):
            node_labels[endpoint] = "unknown"
            nodes.append(endpoint)
            positions[endpoint] = (float(len(positions)), 0.0)

    max_level = max((position[1] for position in positions.values()), default=1.0)
    max_span = max((_span_bounds(node)[1] for node in nodes), default=1)
    width = max(min_width, max_span * 0.95)
    height = max(12.0, 3.2 + max_level * 2.25)
    figure, axis = plt.subplots(figsize=(width, height), constrained_layout=False)
    axis.set_axis_off()
    axis.set_xlim(-1.5, max(max_span + 1.5, width))
    axis.set_ylim(-1.4, max_level + 1.7)

    role_colours = {"Nucleus": "#b9d7ff", "Satellite": "#ffd4a8", "root": "#c8c8c8"}
    for node in nodes:
        role = node_labels.get(node, "unknown")
        colour = role_colours.get(role, "#e8e8e8")
        x, y = positions[node]
        axis.scatter([x], [y], s=1150 if _span_bounds(node)[0] != _span_bounds(node)[1] else 800,
                     c=colour, edgecolors="#333333", linewidths=1.0, zorder=4)
        axis.text(x, y, f"{node}\n{role}", ha="center", va="center", fontsize=7.2,
                  zorder=5, clip_on=False)

    # Draw dominance links parent→child, reversing the stored child→parent
    # direction only for visual tree readability.  Draw relation links in the
    # stored direction and curve them so reciprocal links remain observable.
    pair_seen: defaultdict[tuple[str, str], int] = defaultdict(int)

    def arrow(source: str, target: str, *, colour: str, style: str, rad: float, width_: float = 1.25) -> None:
        if source not in positions or target not in positions:
            return
        sx, sy = positions[source]
        tx, ty = positions[target]
        axis.annotate(
            "", xy=(tx, ty), xytext=(sx, sy),
            arrowprops={
                "arrowstyle": "-|>",
                "color": colour,
                "lw": width_,
                "linestyle": style,
                "shrinkA": 22,
                "shrinkB": 22,
                "connectionstyle": f"arc3,rad={rad:.3f}",
            }, zorder=2,
        )

    for child, parent in dominance:
        key = tuple(sorted((child, parent)))
        offset = pair_seen[key]
        pair_seen[key] += 1
        arrow(parent, child, colour="#777777", style="-", rad=0.04 * (offset + 1), width_=1.1)

    relation_texts: list[tuple[float, float, str, float]] = []
    for source, target, label in relations:
        key = tuple(sorted((source, target)))
        offset = pair_seen[key]
        pair_seen[key] += 1
        rad = 0.12 * ((offset % 3) - 1)
        arrow(source, target, colour="#b33a3a", style="--", rad=rad, width_=1.45)
        if source in positions and target in positions:
            sx, sy = positions[source]
            tx, ty = positions[target]
            mx, my = (sx + tx) / 2.0, (sy + ty) / 2.0
            dx, dy = tx - sx, ty - sy
            length = math.hypot(dx, dy) or 1.0
            # Offset labels away from the edge's centre line.
            normal = (-dy / length, dx / length)
            relation_texts.append((mx + normal[0] * (0.16 + abs(rad)), my + normal[1] * (0.16 + abs(rad)), label, rad))
    for x, y, label, _ in relation_texts:
        axis.text(x, y, label, fontsize=6.5, ha="center", va="center", color="#7d1111",
                  bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "#d9a0a0", "alpha": 0.92},
                  zorder=6, clip_on=False)

    if title:
        axis.set_title(title, fontsize=14, pad=18)
    axis.text(0.0, 1.01, f"{len(nodes)} nodes · {len(dominance)} dominance links · {len(relations)} relation links",
              transform=axis.transAxes, ha="left", va="bottom", fontsize=9, color="#444444")
    legend = [
        Patch(facecolor="#b9d7ff", edgecolor="#333", label="Nucleus"),
        Patch(facecolor="#ffd4a8", edgecolor="#333", label="Satellite"),
        Patch(facecolor="#c8c8c8", edgecolor="#333", label="Root / span"),
        Line2D([0], [0], color="#777", lw=1.2, label="Dominance (stored `/`)"),
        Line2D([0], [0], color="#b33a3a", lw=1.5, ls="--", label="RST relation (labelled)"),
    ]
    axis.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.95)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp.png")
    try:
        figure.savefig(temporary, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "renderer": RENDERER_VERSION,
        "node_count": len(nodes),
        "dominance_edge_count": len(dominance),
        "relation_edge_count": len(relations),
        "warnings": warnings,
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON object containing nodes and edges")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    analysis = json.loads(args.input.read_text(encoding="utf-8"))
    metadata = render_graph_png(analysis, args.output, title=args.title, dpi=args.dpi)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
