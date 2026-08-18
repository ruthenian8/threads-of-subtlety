"""UniRST integration for the Threads of Subtlety preprocessing pipeline.

The public :mod:`isanlp_rst` API parses either raw text or a supplied sequence
of EDUs.  Raw-text parsing also performs expensive discourse-tree decoding,
so segmentation calls the universal parser's learned segmenter directly.  The
public EDU-based parser is then used for every relation inventory, ensuring
that all inventory-specific predictions share the cached boundaries.
"""

from __future__ import annotations

import gc
import hashlib
import os
import pickle
import re
import tempfile
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from tqdm.auto import tqdm

if TYPE_CHECKING:
    from .tos_dataset import Document, SceneDiscourseTree


DEFAULT_REL_INVENTORIES = (
    "eng.erst.gum",
    "eng.rst.rstdt",
    "deu.rst.pcc",
    "nld.rst.nldt",
)

SEGMENTATION_CHECKPOINT_INTERVAL = 100


class UniRSTAdapter:
    """Adapter around UniRST segmentation and EDU-based parsing."""

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

            self._install_du_converter_alignment_fix()
            self._parser_factory = Parser
        return self._parser_factory

    @staticmethod
    def _install_du_converter_alignment_fix() -> None:
        """Replace UniRST's non-terminating EDU text alignment globally.

        Both ``parse_rst`` and ``parse_from_edus`` instantiate ``DUConverter``
        after inference.  Until the upstream implementation is fixed, install
        the bounded equivalent before constructing a production parser so the
        parsing phase cannot enter the same one-word-EDU loop as segmentation.
        """
        from isanlp_rst.utils.du_converter import DUConverter

        DUConverter.fix_segmented_strings = staticmethod(
            UniRSTAdapter._fix_segmented_strings
        )

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
    def _normalise_edus(edus: Iterable[str]) -> List[str]:
        if isinstance(edus, (str, bytes)):
            raise TypeError("UniRST EDUs must be an iterable of strings, not text")
        normalised: List[str] = []
        for edu in edus:
            if not isinstance(edu, str) or not edu.strip():
                raise ValueError("UniRST returned an empty EDU")
            normalised.append(" ".join(edu.split()))
        if not normalised:
            raise ValueError("UniRST returned no EDUs")
        return normalised

    @staticmethod
    def _align_predicted_segments(
        predicted_segments: Sequence[str], gold_tokens: Sequence[str]
    ) -> List[int]:
        """Map predicted segment lengths to exclusive gold-token boundaries.

        UniRST's ``DUConverter.fix_segmented_strings`` can move its token
        cursor backwards when a predicted EDU consists of exactly one word,
        after which its unbounded alignment loop never makes progress.  This
        bounded implementation requires every EDU to consume at least one
        complete gold token and fails explicitly on an unalignable boundary.
        """
        boundaries: List[int] = []
        start_token = 0

        for segment_index, segment in enumerate(predicted_segments):
            target_length = len("".join(segment.split()))
            if target_length == 0:
                raise ValueError(f"Predicted EDU {segment_index} is empty")
            if start_token >= len(gold_tokens):
                raise ValueError(
                    f"Predicted EDU {segment_index} starts after all gold tokens"
                )

            candidate_length = 0
            end_token = start_token
            while end_token < len(gold_tokens) and candidate_length < target_length:
                candidate_length += len("".join(gold_tokens[end_token].split()))
                end_token += 1

            if candidate_length < target_length:
                raise ValueError(
                    f"Unable to align predicted EDU {segment_index}: requires "
                    f"{target_length} non-whitespace characters, but only "
                    f"{candidate_length} remain"
                )
            if candidate_length != target_length:
                raise ValueError(
                    f"Predicted EDU {segment_index} ends inside a gold token: "
                    f"expected {target_length} non-whitespace characters, "
                    f"reached {candidate_length}"
                )

            boundaries.append(end_token)
            start_token = end_token

        if start_token != len(gold_tokens):
            raise ValueError(
                f"Segmentation left {len(gold_tokens) - start_token} gold tokens unused"
            )
        return boundaries

    @staticmethod
    def _fix_segmented_strings(
        predicted_segments: Sequence[str], gold_tokens: Sequence[str]
    ) -> List[str]:
        """Return gold-token text grouped by bounded predicted boundaries."""
        boundaries = UniRSTAdapter._align_predicted_segments(
            predicted_segments, gold_tokens
        )
        fixed_segments: List[str] = []
        start_token = 0
        for end_token in boundaries:
            fixed_segments.append(" ".join(gold_tokens[start_token:end_token]).strip())
            start_token = end_token
        return fixed_segments

    @staticmethod
    def _segment_with_unirst_predictor(predictor: Any, text: str) -> List[str]:
        """Run UniRST's learned EDU segmenter without decoding an RST tree.

        ``isanlp_rst.parser.Parser`` currently has no public segmentation-only
        method: ``Parser.__call__`` always runs ``parse_rst``.  This method
        mirrors the tokenization and EDU alignment portions of ``parse_rst``,
        but calls the model encoder directly and never invokes tree or relation
        decoding.  Attribute checks make upstream API incompatibilities fail
        explicitly instead of silently falling back to the expensive path.
        """
        try:
            import razdel
            import torch
            from isanlp_rst.universal_parser.src.parser.data import Data
        except ImportError as exc:  # pragma: no cover - installation failure
            raise RuntimeError(
                "UniRST segmentation requires razdel, torch, and isanlp_rst"
            ) from exc

        model = getattr(predictor, "model", None)
        encoder = getattr(model, "encoder", None)
        tokenizer = getattr(predictor, "tokenizer", None)
        tokenize = getattr(predictor, "tokenize", None)
        if not callable(encoder) or tokenizer is None or not callable(tokenize):
            raise RuntimeError(
                "The installed isanlp_rst version does not expose the "
                "universal parser segmentation components expected by this adapter"
            )

        razdel_tokens = list(razdel.tokenize(text))
        word_tokens = [token.text for token in razdel_tokens]
        word_offsets = [(token.start, token.stop) for token in razdel_tokens]
        if not word_tokens:
            raise ValueError("UniRST could not tokenize the scene")

        # parse_rst() returns a single dummy EDU for fewer than three tokens
        # without running the model.  Preserve that behavior here.
        if len(word_tokens) < 3:
            start, end = word_offsets[0][0], word_offsets[-1][1]
            return [text[start:end]]

        input_data = Data(
            input_sentences=[word_tokens],
            edu_breaks=[[]],
            decoder_input=[[]],
            relation_label=[[]],
            parsing_breaks=[[]],
            golden_metric=[[]],
        )
        batch = tokenize(input_data)

        with torch.inference_mode():
            encoder_output = encoder(
                batch.input_sentences,
                batch.entity_ids,
                batch.entity_position_ids,
                batch.edu_breaks,
                sent_breaks=batch.sent_breaks,
                is_test=True,
                dataset_index=batch.dataset_index,
            )

        if not isinstance(encoder_output, (tuple, list)) or len(encoder_output) < 4:
            raise RuntimeError("UniRST encoder did not return predicted EDU boundaries")
        predicted_batches = encoder_output[3]
        if not predicted_batches or len(predicted_batches) != 1:
            raise RuntimeError("UniRST returned an invalid segmentation batch")

        input_ids = batch.input_sentences[0]
        predicted_breaks = [int(index) for index in predicted_batches[0]]
        if (
            not predicted_breaks
            or predicted_breaks[-1] != len(input_ids) - 1
            or any(
                current <= previous
                for previous, current in zip(predicted_breaks, predicted_breaks[1:])
            )
            or predicted_breaks[0] < 0
        ):
            raise ValueError(
                "UniRST returned invalid or incomplete predicted EDU boundaries"
            )

        # Align subword segments to complete Razdel tokens with a bounded
        # cursor, then slice the original scene so punctuation and whitespace
        # are preserved without invoking DUConverter's unbounded loop.
        subword_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        predicted_segments: List[str] = []
        previous_break = 0
        for predicted_break in predicted_breaks:
            predicted_segments.append(
                "".join(subword_tokens[previous_break : predicted_break + 1])
                .replace("▁", " ")
                .strip()
            )
            previous_break = predicted_break + 1

        word_boundaries = UniRSTAdapter._align_predicted_segments(
            predicted_segments, word_tokens
        )
        edus: List[str] = []
        start_token = 0
        for end_token in word_boundaries:
            start_character = word_offsets[start_token][0]
            end_character = word_offsets[end_token - 1][1]
            edus.append(text[start_character:end_character])
            start_token = end_token
        return edus

    def _segment_edus(self, text: str) -> List[str]:
        parser = self.segmenter()

        # Prefer a future/public segmentation-only API when available.  The
        # currently pinned UniRST implementation uses the predictor path below.
        segment_edus = getattr(parser, "segment_edus", None)
        if callable(segment_edus):
            return self._normalise_edus(segment_edus(text))

        predictor = getattr(parser, "predictor", None)
        if predictor is None:
            raise RuntimeError(
                "The UniRST parser exposes neither segment_edus() nor predictor internals"
            )
        return self._normalise_edus(
            self._segment_with_unirst_predictor(predictor, text)
        )

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
        edus = self._segment_edus(text)
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
    if not _valid_cache_prefix(cached, adapter, scene_specs):
        return False
    return len(cached["scenes"]) == len(scene_specs)


def _valid_cache_prefix(
    cached: Any, adapter: UniRSTAdapter, scene_specs: Sequence[Mapping[str, Any]]
) -> bool:
    """Return whether a cache contains a valid completed prefix."""
    if not isinstance(cached, dict) or "metadata" not in cached or "scenes" not in cached:
        return False
    if cached["metadata"] != _cache_metadata(adapter, scene_specs):
        return False
    cached_scenes = cached["scenes"]
    if not isinstance(cached_scenes, list) or len(cached_scenes) > len(scene_specs):
        return False
    return all(
        isinstance(cached_scene, dict)
        and cached_scene.get("scene_key") == scene_spec["scene_key"]
        and cached_scene.get("text") == scene_spec["text"]
        for cached_scene, scene_spec in zip(cached_scenes, scene_specs)
    )


def load_or_create_segmentation_cache(
    adapter: UniRSTAdapter,
    scene_specs: Sequence[Mapping[str, Any]],
    path: str,
    force: bool = False,
    progress_desc: Optional[str] = None,
    checkpoint_interval: int = SEGMENTATION_CHECKPOINT_INTERVAL,
) -> List[Dict[str, Any]]:
    """Load a validated segmentation pickle or create it incrementally.

    Checkpoints are atomically written in bounded batches, allowing an
    interrupted run to resume without repeatedly serializing the full scene
    list for every scene.
    """
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    metadata = _cache_metadata(adapter, scene_specs)
    scenes: List[Dict[str, Any]] = []
    if not force and os.path.exists(path):
        with open(path, "rb") as handle:
            cached = pickle.load(handle)
        if _valid_cache(cached, adapter, scene_specs):
            return cached["scenes"]
        if _valid_cache_prefix(cached, adapter, scene_specs):
            scenes = list(cached["scenes"])

    remaining_specs = scene_specs[len(scenes) :]
    specs = (
        tqdm(
            remaining_specs,
            desc=progress_desc,
            unit="scene",
            total=len(scene_specs),
            initial=len(scenes),
        )
        if progress_desc
        else remaining_specs
    )
    for scene_index, spec in enumerate(specs, start=len(scenes) + 1):
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
        if scene_index % checkpoint_interval == 0:
            _atomic_pickle_dump({"metadata": metadata, "scenes": scenes}, path)

    if not scenes and not scene_specs:
        _atomic_pickle_dump({"metadata": metadata, "scenes": scenes}, path)
    elif scenes and len(scenes) % checkpoint_interval != 0:
        _atomic_pickle_dump({"metadata": metadata, "scenes": scenes}, path)
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
