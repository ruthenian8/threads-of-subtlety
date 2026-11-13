"""Dipo Ettin parsers behind the existing shared-EDU preprocessing interface."""
from __future__ import annotations

import json
import os
import tempfile
import warnings
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence

from .unirst import SceneSplitter, UniRSTAdapter

DIPO_REVISION = "4debc92e41140e5bcd9aadb987a297244c0fb14a"
GUM = "eng.erst.gum-12.1.0.ettin"
RSTDT = "eng.rst.rstdt-coarse.ettin"
MODELS = {
    GUM: ("larc-iu/dmrst-ettin-400m-gum-12.1.0", "e33906a27f89b2703df8ab19e471aa82141d62cb"),
    RSTDT: ("larc-iu/dmrst-ettin-400m-rstdt-coarse", "26594833545ef713d14e2ea6e9a7670c881c143a"),
}
DEFAULT_REL_INVENTORIES = (GUM, RSTDT)


class EttinSceneSplitter:
    """Keep documents fitting one GUM encoder window intact; split longer ones."""

    def __init__(self, adapter):
        self.adapter = adapter
        self._fallback = None

    def split(self, document):
        from dipo.rst.parsers.common.encoding import encode_with_offsets, _window_sentinels
        from dipo.rst.parsers.common.whitespace import normalize_whitespace

        model = self.adapter.segmenter().model
        normalized, _ = normalize_whitespace(document, drop_cjk_gaps=model.config.split_cjk_tokens)
        ids, _ = encode_with_offsets(model.tokenizer, normalized, split_cjk=model.config.split_cjk_tokens)
        if model.config.encoder_window_size is not None:
            limit = model.config.encoder_window_size
        else:
            prefix, suffix = _window_sentinels(model.tokenizer)
            limit = model.max_length - len(prefix) - len(suffix)
        if len(ids) <= limit:
            return [document] if document.strip() else []
        if self._fallback is None:
            self._fallback = SceneSplitter()
        return self._fallback.split(document)


def legacy_result(tree: Any, edus: Sequence[str]) -> dict:
    """Convert binary parsing actions without changing labels or nuclearity."""
    if list(tree.edu_strings) != list(edus):
        raise ValueError("Dipo changed the supplied EDU text or order")
    actions = tree.parsing_actions()
    if len(actions) != len(edus) - 1:
        raise ValueError("Dipo returned an incomplete binary tree")
    coverage = [(SimpleNamespace(text=text, left=None, right=None), i, i + 1)
                for i, text in enumerate(edus)]
    for split, nuclearity, relation in reversed(actions):
        if not 0 < split < len(edus) or nuclearity not in {"NS", "SN", "NN"}:
            raise ValueError("Dipo returned an invalid split or nuclearity")
        left, begin, left_end = coverage[split - 1]
        right, right_begin, end = coverage[split]
        if left_end != split or right_begin != split:
            raise ValueError("Dipo returned overlapping or repeated splits")
        if not isinstance(relation, str) or not relation.strip():
            raise ValueError("Dipo returned an empty relation")
        node = SimpleNamespace(left=left, right=right, nuclearity=nuclearity, relation=relation)
        for i in range(begin, end):
            coverage[i] = (node, begin, end)
    root, begin, end = coverage[0]
    if begin != 0 or end != len(edus):
        raise ValueError("Dipo tree does not cover all supplied EDUs")
    return {"rst": [root]}


class EttinParser:
    def __init__(self, model: Any):
        self.model = model.eval()

    def segment_edus(self, text: str) -> list[str]:
        import torch
        from dipo.rst.parsers.common.encoding import encode_with_offsets
        from dipo.rst.parsers.common.whitespace import line_start_breaks, normalize_whitespace
        from dipo.rst.parsers.dmrst.modeling_dmrst import edus_from_breaks, with_forced_breaks

        model = self.model
        if model.segmenter is None:
            raise RuntimeError("The Ettin checkpoint has no trained segmenter")
        source, char_map = normalize_whitespace(text, drop_cjk_gaps=model.config.split_cjk_tokens)
        ids, offsets = encode_with_offsets(model.tokenizer, source, split_cjk=model.config.split_cjk_tokens)
        if not ids:
            raise ValueError("Dipo tokenized the scene to zero tokens")
        with torch.inference_mode():
            inputs = torch.tensor(ids, dtype=torch.long, device=model.device)
            embeddings = model.layer_norm(model._encode_tokens(inputs).float())
            breaks = model.segmenter.predict_breaks(embeddings)
            breaks = with_forced_breaks(text, char_map, offsets, breaks, line_start_breaks(text))
            _, edus = edus_from_breaks(text, char_map, offsets, breaks)
        edus = UniRSTAdapter._normalise_edus(edus)
        if "".join("".join(edus).split()) != "".join(text.split()):
            raise ValueError("Dipo segmentation changed the scene's non-whitespace text")
        return edus

    def from_edus(self, edus: Sequence[str]) -> dict:
        import torch
        from dipo.rst.data.tree import RstTree

        edus = UniRSTAdapter._normalise_edus(edus)
        # predict() reads only the supplied EDUs; placeholder topology is discarded.
        label = self.model.config.relation_types[0][0]
        supplied = RstTree.from_parsing_actions(
            [(i, "NS", label) for i in range(1, len(edus))], edus,
            relation_types=self.model.config.relation_types,
        )
        with torch.inference_mode():
            predicted = self.model.predict(supplied)
        return legacy_result(predicted, edus)

    def from_edus_batch(self, documents: Sequence[Sequence[str]], batch_size: int = 8) -> list:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        warnings.warn("Pinned Dipo runs documents individually; --batch-size bounds scheduling, not GPU batching.")
        return [self.from_edus(edus) for edus in documents]


class EttinAdapter(UniRSTAdapter):
    cache_schema_version = "ettin-dipo-v1"

    def __init__(
        self,
        segment_relinventory: str = GUM,
        relation_inventories: Sequence[str] = DEFAULT_REL_INVENTORIES,
        cuda_device: int = -1,
        model_loader: Optional[Callable[..., Any]] = None,
    ):
        if segment_relinventory not in MODELS or any(key not in MODELS for key in relation_inventories):
            raise ValueError("Unknown Ettin model key")
        model_id, revision = MODELS[segment_relinventory]
        super().__init__(
            segment_relinventory=segment_relinventory,
            relation_inventories=relation_inventories,
            hf_model_name=model_id, hf_model_version=revision, cuda_device=cuda_device,
        )
        self.model_loader = model_loader
        self.segmentation_provenance = {
            "backend": "dipo", "revision": DIPO_REVISION,
            "model": model_id, "model_revision": revision,
            "break_at_paragraphs": True,
        }

    def make_parser(self, relinventory: str) -> EttinParser:
        if self.model_loader is None:
            from dipo.rst.parsers.dmrst.modeling_dmrst import DMRSTParser
            loader = DMRSTParser.from_pretrained
        else:
            loader = self.model_loader
        model_id, revision = MODELS[relinventory]
        device = "cpu" if self.cuda_device < 0 else f"cuda:{self.cuda_device}"
        return EttinParser(loader(model_id, revision=revision, device=device))

    def segment_scenes(self, texts: Sequence[str], batch_size: int = 8) -> list:
        if batch_size > 1:
            warnings.warn("Pinned Dipo segments documents individually; no native segmentation batching.")
        return super().segment_scenes(texts, batch_size=batch_size)

    def ensure_output_provenance(self, output_dir: str, shard_count: Optional[int] = None) -> None:
        """Prevent existence-based resumption from silently mixing parser settings."""
        expected = {
            "backend": "dipo", "revision": DIPO_REVISION,
            "scene_policy": "whole-document-if-single-gum-window-v1",
            "models": {key: {"id": repo, "revision": rev} for key, (repo, rev) in MODELS.items()},
            "segmentation": self.segmentation_provenance,
        }
        if shard_count is not None:
            expected["sharding"] = {"method": "round-robin-v1", "shard_count": shard_count}
        path = os.path.join(output_dir, "ettin_provenance.json")
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                if json.load(handle) != expected:
                    raise ValueError("Output provenance differs; choose a new --output-dir")
            return
        if any(name.startswith("rel-") or name == "paired" for name in os.listdir(output_dir)):
            raise ValueError("Existing parses lack Ettin provenance; choose a new --output-dir")
        fd, temporary = tempfile.mkstemp(prefix=".ettin-", dir=output_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(expected, handle, indent=2)
                handle.write("\n")
            # Publish the complete manifest without overwriting another worker's
            # concurrently published provenance.
            try:
                os.link(temporary, path)
            except FileExistsError:
                with open(path, encoding="utf-8") as handle:
                    if json.load(handle) != expected:
                        raise ValueError("Output provenance differs; choose a new --output-dir")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
