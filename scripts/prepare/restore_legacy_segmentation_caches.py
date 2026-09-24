#!/usr/bin/env python3
"""Restore legacy shared HC3 EDU segmentations from PCC/NLDT parsed trees.

The existing schema-v2 caches provide canonical HC3 scene ordering and keys.
This utility replaces only their segmentation fields with the matching legacy
PCC tree values, after requiring exact PCC/NLDT agreement for each source tree.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pickle
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


DOMAINS = ("finance", "medicine", "open_qa", "wiki_csai")
LABELS = ("human", "machine")
PCC = "deu.rst.pcc"
NLDT = "nld.rst.nldt"
DEFAULT_ROOT = Path("data/unirst")
DEFAULT_OUTPUT = DEFAULT_ROOT / "segments-deu.rst.pcc.2"
EDU_KEY = re.compile(r"^span_(\d+)-(\d+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_record(line: str, path: Path, line_number: int) -> Mapping[str, Any]:
    try:
        record = ast.literal_eval(line)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{path}:{line_number}: invalid serialized record") from exc
    if not isinstance(record, Mapping):
        raise ValueError(f"{path}:{line_number}: record is not a mapping")
    return record


def ordered_scene_trees(document: Mapping[str, Any], path: Path, line_number: int):
    trees = document.get("scene_discourse_trees")
    if not isinstance(trees, Mapping):
        raise ValueError(f"{path}:{line_number}: missing scene_discourse_trees mapping")
    try:
        return sorted(trees.items(), key=lambda item: int(item[0]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{line_number}: non-numeric scene-tree key") from exc


def ordered_edus(tree: Mapping[str, Any], where: str) -> List[str]:
    edus = tree.get("edus")
    if not isinstance(edus, Mapping) or not edus:
        raise ValueError(f"{where}: missing EDU mapping")
    indexed: List[Tuple[int, str]] = []
    for key, value in edus.items():
        match = EDU_KEY.match(str(key))
        if match is None or match.group(1) != match.group(2):
            raise ValueError(f"{where}: unexpected EDU key {key!r}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{where}: empty or non-text EDU at {key!r}")
        indexed.append((int(match.group(1)), value))
    indexed.sort(key=lambda pair: pair[0])
    if [index for index, _ in indexed] != list(range(1, len(indexed) + 1)):
        raise ValueError(f"{where}: EDU indices are not contiguous from 1")
    return [value for _, value in indexed]


def segmentation_from_tree(
    tree: Mapping[str, Any], where: str
) -> Dict[str, Any]:
    text = tree.get("text")
    tokenized = tree.get("tokenized")
    segments = tree.get("segments")
    if not isinstance(text, str):
        raise ValueError(f"{where}: scene text is not a string")
    if not isinstance(tokenized, list) or not all(isinstance(x, str) for x in tokenized):
        raise ValueError(f"{where}: tokenized field is not a list of strings")
    if not isinstance(segments, list) or not all(type(x) is int for x in segments):
        raise ValueError(f"{where}: segments field is not a list of integer boundaries")
    edus = ordered_edus(tree, where)
    if len(segments) != len(edus):
        raise ValueError(f"{where}: {len(segments)} boundaries for {len(edus)} EDUs")
    if not tokenized or not segments or segments[-1] != len(tokenized) - 1:
        raise ValueError(f"{where}: final EDU boundary does not cover all tokens")
    if any(a >= b for a, b in zip(segments, segments[1:])):
        raise ValueError(f"{where}: EDU boundaries are not strictly increasing")
    if segments[0] < 0 or segments[-1] >= len(tokenized):
        raise ValueError(f"{where}: EDU boundary is outside tokenized sequence")
    return {
        "text": text,
        "edus": edus,
        "tokenized": list(tokenized),
        "segments": list(segments),
        "status": "ok",
        "error": None,
    }


def cache_scene_key(scene: Mapping[str, Any], where: str) -> str:
    key = scene.get("scene_key")
    text = scene.get("text")
    if not isinstance(key, str) or not isinstance(text, str):
        raise ValueError(f"{where}: scaffold scene lacks scene_key or text")
    expected = ":".join(
        str(scene.get(field))
        for field in ("dataset", "split", "document_index", "scene_index")
    )
    if key != expected:
        raise ValueError(f"{where}: scene_key {key!r} disagrees with {expected!r}")
    return key


def collect_trusted_segmentations(
    root: Path, domain: str
) -> Tuple[Dict[str, Dict[str, Any]], Counter, Dict[str, Any]]:
    trusted: Dict[str, Dict[str, Any]] = {}
    source_text_counts: Counter = Counter()
    unavailable_occurrences = 0
    unavailable_texts = set()
    document_count = 0
    scene_count = 0
    source_paths: List[Path] = []

    for label in LABELS:
        pcc_path = root / f"rel-{PCC}" / f"hc3_{domain}_{label}.discourse_parsed.jsonl"
        nldt_path = root / f"rel-{NLDT}" / f"hc3_{domain}_{label}.discourse_parsed.jsonl"
        if not pcc_path.is_file() or not nldt_path.is_file():
            raise FileNotFoundError(f"Missing PCC/NLDT source pair for {domain}_{label}")
        source_paths.extend((pcc_path, nldt_path))

        with pcc_path.open(encoding="utf-8") as pcc_handle, nldt_path.open(
            encoding="utf-8"
        ) as nldt_handle:
            line_number = 0
            while True:
                pcc_line = pcc_handle.readline()
                nldt_line = nldt_handle.readline()
                if not pcc_line and not nldt_line:
                    break
                line_number += 1
                if not pcc_line or not nldt_line:
                    raise ValueError(f"PCC/NLDT line-count mismatch in {domain}_{label}")
                pcc_doc = read_record(pcc_line, pcc_path, line_number)
                nldt_doc = read_record(nldt_line, nldt_path, line_number)
                document_count += 1

                for field in ("text", "source", "label", "scenes"):
                    if pcc_doc.get(field) != nldt_doc.get(field):
                        raise ValueError(
                            f"{domain}_{label} record {line_number}: PCC/NLDT "
                            f"document field {field!r} differs"
                        )
                pcc_trees = ordered_scene_trees(pcc_doc, pcc_path, line_number)
                nldt_trees = ordered_scene_trees(nldt_doc, nldt_path, line_number)
                if [key for key, _ in pcc_trees] != [key for key, _ in nldt_trees]:
                    raise ValueError(
                        f"{domain}_{label} record {line_number}: PCC/NLDT scene keys differ"
                    )
                scene_texts = pcc_doc["scenes"]
                if not isinstance(scene_texts, list) or len(scene_texts) != len(pcc_trees):
                    raise ValueError(
                        f"{domain}_{label} record {line_number}: document scenes do not "
                        "match tree count"
                    )

                for scene_index, ((_, pcc_tree), (_, nldt_tree)) in enumerate(
                    zip(pcc_trees, nldt_trees)
                ):
                    where = f"{pcc_path}:{line_number}:scene {scene_index}"
                    if pcc_tree.get("text") != nldt_tree.get("text"):
                        raise ValueError(f"{where}: PCC/NLDT scene text differs")
                    if pcc_tree.get("text") != scene_texts[scene_index]:
                        raise ValueError(f"{where}: tree text differs from document scene")
                    pcc_has_edus = bool(pcc_tree.get("edus"))
                    nldt_has_edus = bool(nldt_tree.get("edus"))
                    if not pcc_has_edus and not nldt_has_edus:
                        if (
                            pcc_tree.get("parsed") != "NONE"
                            or nldt_tree.get("parsed") != "NONE"
                            or pcc_tree.get("tokenized") != []
                            or nldt_tree.get("tokenized") != []
                            or pcc_tree.get("segments") != []
                            or nldt_tree.get("segments") != []
                        ):
                            raise ValueError(
                                f"{where}: missing EDUs are not represented as a "
                                "matching NONE parse in both standards"
                            )
                        scene_count += 1
                        source_text_counts[pcc_tree["text"]] += 1
                        unavailable_occurrences += 1
                        unavailable_texts.add(pcc_tree["text"])
                        continue
                    if pcc_has_edus != nldt_has_edus:
                        raise ValueError(f"{where}: only one standard has EDU values")
                    pcc_seg = segmentation_from_tree(pcc_tree, where)
                    nldt_seg = segmentation_from_tree(
                        nldt_tree, f"{nldt_path}:{line_number}:scene {scene_index}"
                    )
                    if pcc_seg != nldt_seg:
                        raise ValueError(f"{where}: PCC/NLDT scene segmentations differ")
                    scene_count += 1
                    source_text_counts[pcc_seg["text"]] += 1
                    prior = trusted.get(pcc_seg["text"])
                    if prior is not None:
                        if prior != pcc_seg:
                            raise ValueError(
                                f"{where}: duplicate scene text has conflicting segmentation"
                            )
                    else:
                        trusted[pcc_seg["text"]] = pcc_seg

    return trusted, source_text_counts, {
        "documents": document_count,
        "source_scene_occurrences": scene_count,
        "unique_scene_texts": len(source_text_counts),
        "recoverable_unique_scene_texts": len(trusted),
        "duplicate_scene_occurrences": sum(
            count - 1 for count in source_text_counts.values() if count > 1
        ),
        "unavailable_edu_occurrences": unavailable_occurrences,
        "unique_texts_without_recoverable_edus": len(unavailable_texts - set(trusted)),
        "source_files": [str(path) for path in source_paths],
        "source_sha256": {str(path): sha256_file(path) for path in source_paths},
    }


def restore_one_domain(root: Path, output_dir: Path, domain: str) -> Dict[str, Any]:
    trusted, source_text_counts, source_info = collect_trusted_segmentations(root, domain)
    scaffold_path = root / "segments-deu.rst.pcc" / f"hc3_{domain}.pkl"
    if not scaffold_path.is_file():
        raise FileNotFoundError(scaffold_path)
    with scaffold_path.open("rb") as handle:
        scaffold = pickle.load(handle)
    if not isinstance(scaffold, dict) or not isinstance(scaffold.get("scenes"), list):
        raise ValueError(f"{scaffold_path}: unsupported cache structure")
    old_metadata = scaffold.get("metadata")
    if not isinstance(old_metadata, Mapping):
        raise ValueError(f"{scaffold_path}: missing cache metadata")

    old_scenes = scaffold["scenes"]
    old_texts = [scene.get("text") for scene in old_scenes]
    # The source lookup is unique by text. Match multiplicity against both
    # corpus occurrences and scaffold scenes to detect omissions or extras.
    scaffold_text_counts = Counter(old_texts)
    if scaffold_text_counts != source_text_counts:
        raise ValueError(f"{domain}: scaffold scene-text multiset differs from PCC trees")

    if not old_scenes:
        raise ValueError(f"{scaffold_path}: empty cache")
    restored_scenes: List[Dict[str, Any]] = []
    used_keys = set()
    for index, old_scene in enumerate(old_scenes):
        where = f"{scaffold_path}:scene {index}"
        key = cache_scene_key(old_scene, where)
        if key in used_keys:
            raise ValueError(f"{where}: duplicate scene_key {key!r}")
        used_keys.add(key)
        text = old_scene["text"]
        segmentation = trusted.get(text)
        if segmentation is None:
            # Both source standards have a NONE parse and no stored EDU data
            # for this scene. Preserve it as an explicit failed cache entry;
            # never substitute the newer independently generated segmentation.
            restored_scenes.append(
                {
                    **old_scene,
                    "edus": [],
                    "tokenized": [],
                    "segments": [],
                    "status": "error",
                    "error": "PCC and NLDT source trees both have parsed='NONE' "
                    "and contain no recoverable EDU segmentation",
                }
            )
        else:
            restored_scenes.append(
                {
                    **old_scene,
                    "edus": list(segmentation["edus"]),
                    "tokenized": list(segmentation["tokenized"]),
                    "segments": list(segmentation["segments"]),
                    "status": "ok",
                    "error": None,
                }
            )

    metadata = {
        "schema_version": 1,
        "hf_model_name": old_metadata.get("hf_model_name"),
        "hf_model_version": old_metadata.get("hf_model_version"),
        "segment_relinventory": PCC,
        "scene_keys": [cache_scene_key(scene, "restored scene") for scene in restored_scenes],
        "scene_text_hashes": [
            hashlib.sha256(scene["text"].encode("utf-8")).hexdigest()
            for scene in restored_scenes
        ],
    }
    restored = {"metadata": metadata, "scenes": restored_scenes}
    if len(metadata["scene_keys"]) != len(metadata["scene_text_hashes"]):
        raise AssertionError("cache metadata key/hash lengths differ")
    for scene in restored_scenes:
        if scene["status"] == "error":
            if scene["edus"] or scene["tokenized"] or scene["segments"]:
                raise AssertionError(f"{domain}: failed scene has partial segmentation")
            continue
        if len(scene["edus"]) != len(scene["segments"]):
            raise AssertionError(f"{domain}: invalid restored segmentation")
        if scene["segments"][-1] != len(scene["tokenized"]) - 1:
            raise AssertionError(f"{domain}: restored segmentation does not cover scene")

    return {
        "payload": restored,
        "scaffold_path": scaffold_path,
        "source_info": source_info,
        "counts": {
            "scaffold_scenes": len(old_scenes),
            "restored_scenes": len(restored_scenes),
            "unique_source_scene_texts": len(source_text_counts),
            "recoverable_unique_scene_texts": len(trusted),
            "unresolved_scenes": sum(
                scene["status"] != "ok" for scene in restored_scenes
            ),
        },
    }


def atomic_pickle_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def atomic_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.root = args.root.resolve()
    args.output_dir = args.output_dir.resolve()
    outputs = [args.output_dir / f"hc3_{domain}.pkl" for domain in DOMAINS]
    manifest_path = args.output_dir / "restoration_manifest.json"
    existing = [path for path in [*outputs, manifest_path] if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Refusing to overwrite existing restoration artifacts; pass --force: "
            + ", ".join(str(path) for path in existing)
        )

    restored = {
        domain: restore_one_domain(args.root, args.output_dir, domain)
        for domain in DOMAINS
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "purpose": "Restore PCC/NLDT-shared legacy EDU boundaries for HC3 domains",
        "schema_version": 1,
        "segment_relinventory": PCC,
        "domains": {},
    }
    for domain, item in restored.items():
        output_path = args.output_dir / f"hc3_{domain}.pkl"
        atomic_pickle_dump(item["payload"], output_path)
        with output_path.open("rb") as handle:
            reopened = pickle.load(handle)
        if reopened != item["payload"]:
            raise ValueError(f"Serialized cache failed round-trip validation: {output_path}")
        reopened_scenes = reopened["scenes"]
        reopened_metadata = reopened["metadata"]
        if reopened_metadata.get("schema_version") != 1:
            raise ValueError(f"Unexpected schema version in {output_path}")
        if len(reopened_scenes) != item["counts"]["scaffold_scenes"]:
            raise ValueError(f"Scene count changed during serialization: {output_path}")
        expected_keys = [cache_scene_key(scene, str(output_path)) for scene in reopened_scenes]
        expected_hashes = [
            hashlib.sha256(scene["text"].encode("utf-8")).hexdigest()
            for scene in reopened_scenes
        ]
        if (
            reopened_metadata.get("scene_keys") != expected_keys
            or reopened_metadata.get("scene_text_hashes") != expected_hashes
        ):
            raise ValueError(f"Cache metadata failed round-trip validation: {output_path}")
        manifest["domains"][domain] = {
            **item["counts"],
            **item["source_info"],
            "scaffold_file": str(item["scaffold_path"]),
            "scaffold_sha256": sha256_file(item["scaffold_path"]),
            "output_file": str(output_path),
            "output_sha256": sha256_file(output_path),
            "schema_version": 1,
        }
    atomic_json_dump(manifest, manifest_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
