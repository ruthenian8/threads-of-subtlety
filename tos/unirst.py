"""UniRST integration for the Threads of Subtlety preprocessing pipeline.

The public :mod:`isanlp_rst` API exposes two operations that are important for
this project: parsing raw text and parsing a supplied sequence of EDUs.  This
module uses the former exactly once for segmentation and the latter for every
relation inventory, so all inventory-specific predictions share boundaries.
"""

from __future__ import annotations

import gc
import hashlib
import os
import pickle
import re
import tempfile
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from .tos_dataset import Document, SceneDiscourseTree


DEFAULT_REL_INVENTORIES = (
    "eng.erst.gum",
    "eng.rst.rstdt",
    "deu.rst.pcc",
    "nld.rst.nldt",
)


class UniRSTAdapter:
    """Adapter around the public ``isanlp_rst.parser.Parser`` API."""

    cache_schema_version = 1

    def __init__(
        self,
        segment_relinventory: str = "deu.rst.pcc",
        relation_inventories: Sequence[str] = DEFAULT_REL_INVENTORIES,
        hf_model_name: str = "tchewik/isanlp_rst_v3",
        hf_model_version: str = "unirst",
        cuda_device: int = -1,
        parser_factory: Optional[Callable[..., Any]] = None,
    ):
        self.segment_relinventory = segment_relinventory
        self.relation_inventories = tuple(relation_inventories)
        self.hf_model_name = hf_model_name
        self.hf_model_version = hf_model_version
        self.cuda_device = cuda_device
        self._parser_factory = parser_factory
        self._segmenter = None

    def _get_parser_factory(self) -> Callable[..., Any]:
        if self._parser_factory is None:
            from isanlp_rst.parser import Parser

            self._parser_factory = Parser
        return self._parser_factory

    def make_parser(self, relinventory: str) -> Any:
        """Create one parser for a single relation inventory."""
        parser_factory = self._get_parser_factory()
        return parser_factory(
            hf_model_name=self.hf_model_name,
            hf_model_version=self.hf_model_version,
            relinventory=relinventory,
            cuda_device=self.cuda_device,
        )

    def segmenter(self) -> Any:
        if self._segmenter is None:
            self._segmenter = self.make_parser(self.segment_relinventory)
        return self._segmenter

    def release_segmenter(self) -> None:
        self._segmenter = None
        gc.collect()

    @staticmethod
    def _tree_root(result: Mapping[str, Any]) -> Any:
        roots = result.get("rst")
        if not roots:
            raise ValueError("UniRST returned no RST tree")
        return roots[0]

    @classmethod
    def _collect_leaf_texts(cls, unit: Any, output: List[str]) -> None:
        left = getattr(unit, "left", None)
        right = getattr(unit, "right", None)
        if left is None and right is None:
            text = getattr(unit, "text", None)
            if not isinstance(text, str) or not text.strip():
                raise ValueError("UniRST returned an empty EDU leaf")
            # from_edus() joins supplied EDUs with one space.  Canonicalizing
            # whitespace here makes the persisted segmentation replayable.
            output.append(" ".join(text.split()))
            return
        if left is not None:
            cls._collect_leaf_texts(left, output)
        if right is not None:
            cls._collect_leaf_texts(right, output)

    @classmethod
    def extract_edus(cls, result: Mapping[str, Any]) -> List[str]:
        edus: List[str] = []
        cls._collect_leaf_texts(cls._tree_root(result), edus)
        if not edus:
            raise ValueError("UniRST returned no EDU leaves")
        return edus

    @staticmethod
    def _scene_token_metadata(edus: Sequence[str]) -> Dict[str, Any]:
        tokenized: List[str] = []
        segments: List[int] = []
        for edu in edus:
            edu_tokens = re.findall(r"\S+", edu)
            if not edu_tokens:
                raise ValueError("An EDU contains no non-whitespace tokens")
            tokenized.extend(edu_tokens)
            segments.append(len(tokenized) - 1)
        return {"tokenized": tokenized, "segments": segments}

    def segment_scene(self, text: str) -> Dict[str, Any]:
        result = self.segmenter()(text)
        edus = self.extract_edus(result)
        metadata = self._scene_token_metadata(edus)
        return {
            "text": text,
            "edus": edus,
            **metadata,
            "status": "ok",
            "error": None,
        }

    @staticmethod
    def _normalise_nuclearity(value: Any) -> str:
        value = str(value or "NN").upper().replace("-", "")
        if value in {"NS", "SN", "NN", "SS"}:
            return value
        if len(value) == 2 and set(value) <= {"N", "S"}:
            return value
        raise ValueError(f"Unsupported UniRST nuclearity: {value!r}")

    @staticmethod
    def _normalise_relation(value: Any) -> str:
        relation = str(value or "span").strip()
        relation = re.sub(r"[\s,:]+", "-", relation)
        return relation or "span"

    @classmethod
    def _format_relation_labels(cls, unit: Any) -> tuple[str, str, str, str]:
        nuclearity = cls._normalise_nuclearity(getattr(unit, "nuclearity", "NN"))
        relation = cls._normalise_relation(getattr(unit, "relation", "span"))
        left_type = "Nucleus" if nuclearity[0] == "N" else "Satellite"
        right_type = "Nucleus" if nuclearity[1] == "N" else "Satellite"

        # DMRST attaches a relation to a satellite and uses ``span`` on the
        # nucleus.  Multinuclear and multisatellite relations are represented
        # on both children because there is no single nucleus.
        if nuclearity == "NS":
            left_relation, right_relation = "span", relation
        elif nuclearity == "SN":
            left_relation, right_relation = relation, "span"
        else:
            left_relation = right_relation = relation
        return left_type, left_relation, right_type, right_relation

    @classmethod
    def _constituency_spans(cls, unit: Any, start_edu: int = 1) -> tuple[int, List[str]]:
        left = getattr(unit, "left", None)
        right = getattr(unit, "right", None)
        if left is None and right is None:
            return start_edu + 1, []
        if left is None or right is None:
            raise ValueError("UniRST returned a non-binary internal node")

        next_edu, left_spans = cls._constituency_spans(left, start_edu)
        end_edu, right_spans = cls._constituency_spans(right, next_edu)
        left_end = next_edu - 1
        right_end = end_edu - 1
        left_type, left_relation, right_type, right_relation = cls._format_relation_labels(unit)
        current = (
            f"({start_edu}:{left_type}={left_relation}:{left_end},"
            f"{next_edu}:{right_type}={right_relation}:{right_end})"
        )
        # Root must be first for create_graph_from_const_format(); descendants
        # can follow in any order, so emit the current node before its children.
        return end_edu, [current, *left_spans, *right_spans]

    @classmethod
    def to_constituency_format(cls, result: Mapping[str, Any], num_edus: int) -> str:
        if num_edus < 2:
            return "NONE"
        root = cls._tree_root(result)
        end_edu, spans = cls._constituency_spans(root)
        if end_edu - 1 != num_edus:
            raise ValueError(
                f"UniRST tree covers {end_edu - 1} EDUs, expected {num_edus}"
            )
        return " ".join(spans)

    @staticmethod
    def scene_tree(text: str, edus: Sequence[str], result: Mapping[str, Any]) -> SceneDiscourseTree:
        from .tos_dataset import SceneDiscourseTree

        metadata = UniRSTAdapter._scene_token_metadata(edus)
        edu_map = {
            f"span_{index}-{index}": edu for index, edu in enumerate(edus, start=1)
        }
        return SceneDiscourseTree(
            text=text,
            tokenized=metadata["tokenized"],
            segments=metadata["segments"],
            edus=edu_map,
            parsed=UniRSTAdapter.to_constituency_format(result, len(edus)),
            graph_dict=None,
            graph_networkx=None,
            motif_dists=None,
        )


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_pickle_dump(value: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=".tmp-", suffix=".pkl", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _cache_metadata(
    adapter: UniRSTAdapter, scene_specs: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    return {
        "schema_version": adapter.cache_schema_version,
        "hf_model_name": adapter.hf_model_name,
        "hf_model_version": adapter.hf_model_version,
        "segment_relinventory": adapter.segment_relinventory,
        "scene_keys": [spec["scene_key"] for spec in scene_specs],
        "scene_text_hashes": [_text_hash(spec["text"]) for spec in scene_specs],
    }


def _valid_cache(
    cached: Any, adapter: UniRSTAdapter, scene_specs: Sequence[Mapping[str, Any]]
) -> bool:
    if not isinstance(cached, dict) or "metadata" not in cached or "scenes" not in cached:
        return False
    return cached["metadata"] == _cache_metadata(adapter, scene_specs) and len(
        cached["scenes"]
    ) == len(scene_specs)


def load_or_create_segmentation_cache(
    adapter: UniRSTAdapter,
    scene_specs: Sequence[Mapping[str, Any]],
    path: str,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Load a validated segmentation pickle or create it once."""
    if not force and os.path.exists(path):
        with open(path, "rb") as handle:
            cached = pickle.load(handle)
        if _valid_cache(cached, adapter, scene_specs):
            return cached["scenes"]

    scenes: List[Dict[str, Any]] = []
    for spec in scene_specs:
        try:
            segmented = adapter.segment_scene(spec["text"])
        except Exception as exc:  # Persist failures so interrupted jobs are inspectable.
            segmented = {
                "text": spec["text"],
                "edus": [],
                "tokenized": [],
                "segments": [],
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        scenes.append({**spec, **segmented})

    _atomic_pickle_dump(
        {"metadata": _cache_metadata(adapter, scene_specs), "scenes": scenes}, path
    )
    return scenes


def tree_from_prediction(
    scene: Mapping[str, Any], result: Mapping[str, Any]
) -> SceneDiscourseTree:
    return UniRSTAdapter.scene_tree(scene["text"], scene["edus"], result)


def write_jsonl_documents(documents: Iterable[Document], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(f"{document.model_dump(mode='json')}\n")


def build_scene_lookup(scenes: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {scene["scene_key"]: scene for scene in scenes}
