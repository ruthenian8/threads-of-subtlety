#!/usr/bin/env python3
"""Reformat the current filled-case notes into the review codebook schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_INPUT = Path(
    "artifacts/review/human_review/filled_cases_notes.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/review/human_review/filled_cases_codebook.json"
)
DEFAULT_CODEBOOK = Path(
    "artifacts/review/human_review/codebook.md"
)
EXPECTED_INPUT_SHA256 = (
    "0ceaefe859bfb4c45e183716cace09c9d3b708feeac4c7d78f2f3d0657bf4a0a"
)
EXPECTED_CODEBOOK_SHA256 = (
    "ff12106db24e3f2afa6257b5e9060821d6ea7feaeb4671436d9992f461990070"
)

TEXTUAL_SUPPORT = {"A", "B", "both", "neither", "uncertain"}
DEFENSIBILITY = {"both", "one_only", "neither", "uncertain"}
DIFFERENCE_TYPES = {
    "mere_renaming",
    "granularity",
    "attachment",
    "nuclearity",
    "alternative_interpretation",
    "parser_or_alignment_error",
    "no_meaningful_difference",
    "uncertain",
}

# case_id: (textual support, defensibility, difference types,
#           coherence distinction, confidence justification)
Decision = Tuple[str, str, List[str], str, str]
DEFAULT_CONFIDENCE = "High by requested default; the note gives a specific link-level rationale."
EXPLICIT_CONFIDENCE = "The note explicitly records high confidence."

DECISIONS: Dict[str, Decision] = {
    "case-001": (
        "uncertain", "uncertain",
        ["nuclearity", "alternative_interpretation", "parser_or_alignment_error"],
        "A preserves the judged centrality of spans 17–19, whereas B supplies a closer causal reading. The preferred analysis would combine A's nuclearity with a result/reason relation.",
        DEFAULT_CONFIDENCE,
    ),
    "case-003": (
        "both", "both", ["alternative_interpretation"],
        "At spans 13–14, A highlights circumstance and B highlights manner; both construals are equally plausible.",
        DEFAULT_CONFIDENCE,
    ),
    "case-004": (
        "A", "one_only", ["alternative_interpretation", "parser_or_alignment_error"],
        "A's elaboration analysis is preferable. B's restatement analysis incorrectly treats a partial elaboration of span 1 as a restatement and is attributed to parser bias.",
        DEFAULT_CONFIDENCE,
    ),
    "case-005": (
        "both", "both", ["mere_renaming", "no_meaningful_difference"],
        "A's purpose and B's enablement labels are appropriate within their respective conventions and contextually equivalent.",
        DEFAULT_CONFIDENCE,
    ),
    "case-006": (
        "both", "both", ["granularity", "alternative_interpretation"],
        "A's purpose reading at spans 2–3 is preferable to B's e-elaboration under the shared segmentation. A's interpretation reading at spans 4–5 is more specific than B's result reading, although the causal reading remains complementary.",
        DEFAULT_CONFIDENCE,
    ),
    "case-007": (
        "A", "one_only", ["alternative_interpretation", "parser_or_alignment_error"],
        "A correctly treats spans 1–2 as a reason for spans 3–4. B's enablement reading incorrectly implies direct facilitation of further action.",
        DEFAULT_CONFIDENCE,
    ),
    "case-008": (
        "both", "both", ["granularity", "alternative_interpretation"],
        "B's restatement readings at the cited span pairs are semantically more accurate than A's summary or elaboration readings, although the latter remain broadly defensible.",
        DEFAULT_CONFIDENCE,
    ),
    "case-009": (
        "both", "both", ["granularity", "alternative_interpretation"],
        "A's comparison reading is more precise than B's adversative reading, but both are contextually valid.",
        DEFAULT_CONFIDENCE,
    ),
    "case-010": (
        "A", "one_only", ["alternative_interpretation", "parser_or_alignment_error", "uncertain"],
        "For spans 9–15 versus 16–17, A's joint reading is preferable because the context does not signal enumeration; B's list reading is unsupported. The preceding note fragment about volitional cause is incomplete.",
        "High by requested default for the explicit joint/list judgment; one note fragment is incomplete.",
    ),
    "case-011": (
        "both", "both", ["granularity"],
        "A's attribution reading at spans 2–3 is more specific and preferable to B's preparation reading.",
        DEFAULT_CONFIDENCE,
    ),
    "case-015": (
        "both", "both", ["granularity", "alternative_interpretation"],
        "A's explanation is more precise than B's cause reading for spans 1 versus 2–3, while a motivation label would also fit inventories that provide it. For spans 2–3, B's conjunction is preferable to A's temporal reading because temporality is not foregrounded.",
        DEFAULT_CONFIDENCE,
    ),
    "case-018": (
        "B", "one_only", ["alternative_interpretation", "parser_or_alignment_error"],
        "For spans 1–3 versus span 4, B's explanation reading is supported and A's purpose reading is not.",
        DEFAULT_CONFIDENCE,
    ),
    "case-019": (
        "both", "both", ["mere_renaming"],
        "The analyses express the same organization/preparation distinction using inventory-specific terminology.",
        EXPLICIT_CONFIDENCE,
    ),
    "case-020": (
        "both", "both", ["alternative_interpretation"],
        "A's explanation and B's cause readings at spans 3–4 versus span 5 are both valid causal construals.",
        DEFAULT_CONFIDENCE,
    ),
    "case-021": (
        "B", "one_only", ["granularity", "alternative_interpretation", "parser_or_alignment_error"],
        "B's general cause reading is preferable. A's volitional-cause reading is too specific because the causal link is not volitional.",
        DEFAULT_CONFIDENCE,
    ),
    "case-023": (
        "both", "both", ["granularity", "alternative_interpretation"],
        "A's organization-heading reading and B's preparation reading are both accurate under their conventions; organization-heading is more precise because it does not overlap other preparation functions.",
        DEFAULT_CONFIDENCE,
    ),
    "case-024": (
        "both", "both", ["granularity", "alternative_interpretation"],
        "A is consistently more specific: manner-means is preferable to B's circumstance at spans 4–5, and explanation is preferable to B's cause at spans 1–3 versus 4–5.",
        DEFAULT_CONFIDENCE,
    ),
    "case-025": (
        "B", "one_only", ["granularity", "alternative_interpretation", "parser_or_alignment_error"],
        "B's reason reading is preferable to A's topic-comment for spans 1–2 versus 3–6, with motivation noted as the ideal available label. B's purpose is also more accurate than A's enablement because span 4 does not enable span 3.",
        DEFAULT_CONFIDENCE,
    ),
    "case-026": (
        "B", "one_only", ["granularity", "alternative_interpretation", "parser_or_alignment_error"],
        "B's organization-heading readings are more specific than A's elaborations. A's solutionhood and B's cause are both applicable, with solutionhood more specific. At spans 17 versus 18–19, B's explanation is correct and A's volitional-cause reading is not.",
        DEFAULT_CONFIDENCE,
    ),
    "case-027": (
        "B", "one_only", ["alternative_interpretation", "parser_or_alignment_error"],
        "At spans 3–4 versus span 5, B's attribution reading is supported and A's enablement reading is not.",
        DEFAULT_CONFIDENCE,
    ),
    "case-028": (
        "both", "both", ["mere_renaming"],
        "The elaboration/restatement difference is judged to be an inventory-specific renaming without a meaningful representational difference.",
        EXPLICIT_CONFIDENCE,
    ),
    "case-029": (
        "both", "both", ["alternative_interpretation"],
        "For spans 6–13 versus 14–17, A's result reading and B's elaboration reading are both applicable.",
        DEFAULT_CONFIDENCE,
    ),
    "case-031": (
        "neither", "neither", ["granularity", "alternative_interpretation", "parser_or_alignment_error"],
        "At spans 2 versus 3–4, A's evidence and B's cause are compatible, with evidence more accurate. At spans 5–6, neither A's background nor B's e-elaboration is adequate; manner-means is preferred.",
        DEFAULT_CONFIDENCE,
    ),
    "case-033": (
        "A", "one_only", ["alternative_interpretation", "parser_or_alignment_error"],
        "A's condition readings are more accurate than B's circumstance readings because the nuclei describe hypothetical rather than factual situations.",
        DEFAULT_CONFIDENCE,
    ),
    "case-034": (
        "both", "both", ["mere_renaming", "granularity", "alternative_interpretation"],
        "A's explanation is preferable to B's cause at spans 1 versus 2–12. Background/circumstance and background/cause provide compatible readings at the other cited links, while enablement/purpose is effectively a translation between conventions.",
        DEFAULT_CONFIDENCE,
    ),
    "case-035": (
        "B", "one_only", ["alternative_interpretation", "parser_or_alignment_error"],
        "A's comparison is more accurate than B's concession at spans 1–2. Elsewhere B is preferable: conjunction rather than temporal at spans 4–5, and circumstance rather than temporal at spans 3 versus 4–5.",
        DEFAULT_CONFIDENCE,
    ),
    "case-036": (
        "both", "both", ["mere_renaming"],
        "The observed relation differences are judged to be mere renaming between inventory conventions.",
        EXPLICIT_CONFIDENCE,
    ),
    "case-037": (
        "both", "both", ["mere_renaming"],
        "The observed relation differences are judged to be mere renaming between inventory conventions.",
        EXPLICIT_CONFIDENCE,
    ),
    "case-038": (
        "both", "both", ["granularity", "alternative_interpretation"],
        "A's organization reading at spans 1 versus 2–3 is more fitting than B's background reading.",
        DEFAULT_CONFIDENCE,
    ),
    "case-041": (
        "both", "both", ["mere_renaming"],
        "The preparation/organization difference is judged to be mere renaming between inventory conventions.",
        EXPLICIT_CONFIDENCE,
    ),
    "case-043": (
        "neither", "neither", ["granularity", "alternative_interpretation", "parser_or_alignment_error"],
        "For spans 1–7 versus span 8, B's textual-organization reading is strictly preferable to A's reason reading. For spans 1–4 versus 5–7, neither A's reason nor B's enablement is adequate.",
        DEFAULT_CONFIDENCE,
    ),
    "case-044": (
        "both", "both", ["mere_renaming"],
        "The sequence/temporal difference is judged to be mere renaming between inventory conventions.",
        EXPLICIT_CONFIDENCE,
    ),
    "case-045": (
        "both", "both", ["granularity", "alternative_interpretation"],
        "B is preferable at both cited links: same-unit is more fitting than A's conjunction for spans 1–2 versus 3, and adversative is more fitting than A's joint for spans 1–3 versus 4.",
        DEFAULT_CONFIDENCE,
    ),
    "case-048": (
        "both", "both", ["granularity", "nuclearity", "alternative_interpretation"],
        "A's textual-organization readings are preferable to B's attribution and elaboration readings, although either analysis is acceptable at spans 1–6 versus 7. The mononuclear organization is preferable to the multinuclear alternative.",
        DEFAULT_CONFIDENCE,
    ),
    "case-051": (
        "B", "one_only", ["granularity", "parser_or_alignment_error"],
        "At spans 3 versus 4–6, B's attribution reading is correct and A's preparation reading is not.",
        DEFAULT_CONFIDENCE,
    ),
    "case-054": (
        "both", "both", ["mere_renaming"],
        "The sequence/temporal difference is judged to be mere renaming between inventory conventions.",
        EXPLICIT_CONFIDENCE,
    ),
    "case-056": (
        "both", "both", ["alternative_interpretation"],
        "At spans 7–8, A's cause and B's background readings are both applicable.",
        DEFAULT_CONFIDENCE,
    ),
    "case-057": (
        "both", "both", ["mere_renaming"],
        "The observed relation differences are judged to be mere renaming between inventory conventions.",
        EXPLICIT_CONFIDENCE,
    ),
    "case-012": (
        "neither", "neither",
        ["attachment", "alternative_interpretation", "parser_or_alignment_error"],
        "Both analyses use the wrong shared grouping at spans 5 versus 6–7; the preferred attachment groups spans 5–6 against span 7. At spans 8–9 versus 10, A's adversative and B's comparison readings are both compatible.",
        DEFAULT_CONFIDENCE,
    ),
    "case-014": (
        "A", "one_only", ["mere_renaming", "alternative_interpretation", "parser_or_alignment_error"],
        "At spans 7–8 versus 9–10, A's conjunction is correct and B's same-unit is not. A's interpretation and B's context are both applicable at spans 6–12 versus 13–16, while evidence/explanation is treated as a terminological correspondence.",
        DEFAULT_CONFIDENCE,
    ),
    "case-016": (
        "both", "both", ["mere_renaming"],
        "The volitional-causal/cause difference is judged to be a translation between inventory conventions.",
        DEFAULT_CONFIDENCE,
    ),
    "case-017": (
        "both", "both", ["granularity", "alternative_interpretation"],
        "A's joint and B's same-unit readings at spans 4–5 versus 6 are both correct. A's preparation and B's organization readings at span 1 versus spans 2–7 are both applicable, although preparation is the less direct interpretation.",
        DEFAULT_CONFIDENCE,
    ),
    "case-046": (
        "A", "one_only", ["granularity", "alternative_interpretation", "parser_or_alignment_error"],
        "A's organization readings are correct at span 1 versus 2 and spans 1–8 versus 9. B's preparation and summary readings at those links are not supported.",
        DEFAULT_CONFIDENCE,
    ),
    "case-049": (
        "both", "both", ["mere_renaming", "granularity"],
        "Both analyses are defensible variants of the same causal/result organization; B's non-volitional-causal label records the distinction at finer granularity.",
        EXPLICIT_CONFIDENCE,
    ),
    "case-053": (
        "A", "one_only", ["alternative_interpretation", "parser_or_alignment_error"],
        "At spans 2–3, A's context reading is appropriate and B's temporal reading is not because the situation is hypothetical.",
        DEFAULT_CONFIDENCE,
    ),
    "case-055": (
        "A", "one_only",
        ["attachment", "nuclearity", "granularity", "alternative_interpretation", "parser_or_alignment_error"],
        "Span 1 should be a satellite in a relation covering the full text body. A's organization is correct at span 1 versus 2–3 and at spans 1–6/1–7 versus 7/8. B's preparation is only a fallback imposed by its inventory, while B's evaluation readings are incorrect because the appropriate relation is unavailable.",
        DEFAULT_CONFIDENCE,
    ),
    "case-058": (
        "both", "both", ["mere_renaming"],
        "The observed relation differences are judged to be mere renaming between inventory conventions.",
        EXPLICIT_CONFIDENCE,
    ),
    "case-059": (
        "B", "one_only", ["alternative_interpretation", "parser_or_alignment_error"],
        "At spans 7–8 versus 9–10, B's adversative reading is correct and A's comparison reading reflects parser bias. At spans 1–2 versus 3–10, A's elaboration and B's context readings are both applicable.",
        DEFAULT_CONFIDENCE,
    ),
    "case-060": (
        "both", "both", ["mere_renaming", "alternative_interpretation"],
        "At spans 1–2 versus 3–24, A's background reading is correct; B's preparation is theoretically applicable but less accurate. The remaining cited differences are judged to be mere renaming.",
        DEFAULT_CONFIDENCE,
    ),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_decisions() -> None:
    for case_id, decision in DECISIONS.items():
        support, defensibility, difference_types, distinction, justification = decision
        if support not in TEXTUAL_SUPPORT:
            raise ValueError(f"{case_id}: invalid textual support {support}")
        if defensibility not in DEFENSIBILITY:
            raise ValueError(f"{case_id}: invalid defensibility {defensibility}")
        invalid_types = set(difference_types) - DIFFERENCE_TYPES
        if invalid_types or not difference_types:
            raise ValueError(f"{case_id}: invalid difference types {invalid_types}")
        if not distinction.strip() or not justification.strip():
            raise ValueError(f"{case_id}: missing rationale")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--codebook", type=Path, default=DEFAULT_CODEBOOK)
    parser.add_argument(
        "--allow-changed-input",
        action="store_true",
        help="Allow formatting notes whose hash differs from the reviewed snapshot.",
    )
    args = parser.parse_args()

    input_hash = sha256_file(args.input)
    if input_hash != EXPECTED_INPUT_SHA256 and not args.allow_changed_input:
        raise ValueError(
            "filled notes changed after the coding decisions were prepared; "
            "review the changes before using --allow-changed-input"
        )
    codebook_hash = sha256_file(args.codebook)
    if codebook_hash != EXPECTED_CODEBOOK_SHA256:
        raise ValueError("codebook changed after the coding decisions were prepared")
    validate_decisions()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    notes = {case["case_id"]: case["value"] for case in source["cases"]}
    completed = {case_id for case_id, value in notes.items() if value.strip()}
    if completed != set(DECISIONS):
        raise ValueError(
            f"coded case mismatch: missing={sorted(completed - set(DECISIONS))}, "
            f"stale={sorted(set(DECISIONS) - completed)}"
        )

    annotations = []
    for case_id in sorted(DECISIONS, key=lambda value: int(value.rsplit("-", 1)[1])):
        support, defensibility, difference_types, distinction, justification = DECISIONS[case_id]
        annotations.append(
            {
                "case_id": case_id,
                "textual_support": support,
                "well_formedness": "both",
                "defensibility": defensibility,
                "difference_types": difference_types,
                "coherence_distinction": distinction,
                "confidence": "high",
                "confidence_justification": justification,
                "original_note": notes[case_id],
            }
        )

    unannotated = sorted(
        set(notes) - completed,
        key=lambda value: int(value.rsplit("-", 1)[1]),
    )
    payload = {
        "format": "UniRST2 representation review codebook",
        "codebook": str(args.codebook),
        "codebook_sha256": codebook_hash,
        "source_notes": str(args.input),
        "source_notes_sha256": input_hash,
        "assumptions": {
            "well_formedness": "both unless explicitly contradicted",
            "confidence": "high unless explicitly contradicted",
            "empty_notes": "listed as unannotated; no substantive judgments imputed",
            "scope": "overall judgments summarize the links discussed in each note; unmentioned graph differences were not adjudicated",
        },
        "completed_case_count": len(annotations),
        "unannotated_case_count": len(unannotated),
        "unannotated_case_ids": unannotated,
        "annotations": annotations,
    }
    atomic_json(args.output, payload)
    print(
        f"Saved {len(annotations)} codebook annotations; "
        f"left {len(unannotated)} empty cases unannotated: {args.output}"
    )


if __name__ == "__main__":
    main()
