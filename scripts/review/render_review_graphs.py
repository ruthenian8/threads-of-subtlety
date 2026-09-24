#!/usr/bin/env python3
"""Render the blinded UniRST2 review graphs as direct, fully-labelled PNGs.

This is intentionally a direct graph renderer rather than a lossy graph→RS3
round-trip: the stored records contain explicit ``/`` dominance links and
relation links, including reciprocal links.  Both are retained in the image.
The script augments the existing review package and regenerates its HTML; it
does not touch the sealed answer key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from rst_graph_visualizer import RENDERER_VERSION, render_graph_png


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_cases(cases: list[dict[str, Any]]) -> str:
    """Serialize cases without previously generated image paths."""
    stripped = []
    for case in cases:
        copied = dict(case)
        copied.pop("tree_images", None)
        stripped.append(copied)
    return "".join(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in stripped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=Path("artifacts/review/human_review"))
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--skip-existing", action="store_true", help="reuse existing PNGs when present")
    args = parser.parse_args()

    cases_path = args.review_dir / "blinded_cases.jsonl"
    html_path = args.review_dir / "blinded_cases.html"
    sampling_path = args.review_dir / "sampling_manifest.json"
    png_dir = args.review_dir / "tree_pngs"
    manifest_path = args.review_dir / "tree_render_manifest.json"
    cases = load_jsonl(cases_path)
    base_source = canonical_cases(cases).encode("utf-8")
    base_source_sha256 = hashlib.sha256(base_source).hexdigest()

    rendered: dict[str, dict[str, Any]] = {}
    enriched: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        case_id = str(case["case_id"])
        copied = dict(case)
        tree_images: dict[str, str] = {}
        for side in ("analysis_a", "analysis_b"):
            filename = f"{case_id}_{side}.png"
            output_path = png_dir / filename
            relative_path = f"tree_pngs/{filename}"
            if args.skip_existing and output_path.exists() and output_path.stat().st_size > 0:
                metadata = {
                    "renderer": RENDERER_VERSION,
                    "sha256": sha256_file(output_path),
                    "reused": True,
                }
            else:
                metadata = render_graph_png(
                    case[side],
                    output_path,
                    title=f"{case_id} · {side.replace('_', ' ').title()}",
                    dpi=args.dpi,
                )
                metadata["reused"] = False
            tree_images[side] = relative_path
            rendered[relative_path] = metadata
        copied["tree_images"] = tree_images
        enriched.append(copied)
        if index == 1 or index % 10 == 0 or index == len(cases):
            print(f"rendered {index}/{len(cases)} cases", flush=True)

    atomic_text(cases_path, "".join(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in enriched))
    from scripts.review import prepare_blinded_review as review_renderer
    atomic_text(html_path, review_renderer.render_html(enriched))

    tree_manifest = {
        "stage": "unirst2_blinded_review_graph_rendering",
        "renderer": RENDERER_VERSION,
        "source_cases_sha256_without_tree_images": base_source_sha256,
        "case_count": len(enriched),
        "image_count": len(rendered),
        "images": rendered,
        "output_paths": {"png_directory": str(png_dir), "html": str(html_path), "cases": str(cases_path)},
    }
    atomic_json(manifest_path, tree_manifest)

    if sampling_path.exists():
        sampling = json.loads(sampling_path.read_text(encoding="utf-8"))
        output_hashes = dict(sampling.get("output_hashes") or {})
        output_hashes.update({
            cases_path.name: sha256_file(cases_path),
            html_path.name: sha256_file(html_path),
            manifest_path.name: sha256_file(manifest_path),
        })
        sampling["output_hashes"] = output_hashes
        sampling["tree_render_manifest"] = manifest_path.name
        atomic_json(sampling_path, sampling)

    print(json.dumps({
        "cases": len(enriched),
        "images": len(rendered),
        "png_dir": str(png_dir),
        "html": str(html_path),
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
