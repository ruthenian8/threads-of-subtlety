#!/usr/bin/env python3
"""Extract saved review-note fields from filled_cases.html into JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_INPUT = Path(
    "artifacts/review/human_review/filled_cases.html"
)
DEFAULT_OUTPUT = Path(
    "artifacts/review/human_review/filled_cases_notes.json"
)
CASE_FIELD = re.compile(r"^notes-(case-(\d+))$")


class TextFieldParser(HTMLParser):
    """Collect textarea contents and text-like input values in document order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.fields: List[Tuple[str, str]] = []
        self._textarea_id: Optional[str] = None
        self._textarea_chunks: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attributes: Dict[str, str] = {
            name: value or "" for name, value in attrs
        }
        if tag == "textarea":
            if self._textarea_id is not None:
                raise ValueError("nested textarea elements are not supported")
            field_id = attributes.get("id", "")
            if CASE_FIELD.fullmatch(field_id):
                self._textarea_id = field_id
                self._textarea_chunks = []
        elif tag == "input":
            field_id = attributes.get("id", "")
            input_type = attributes.get("type", "text").lower()
            if CASE_FIELD.fullmatch(field_id) and input_type in {
                "text",
                "search",
                "email",
                "url",
                "tel",
            }:
                self.fields.append((field_id, attributes.get("value", "")))

    def handle_data(self, data: str) -> None:
        if self._textarea_id is not None:
            self._textarea_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "textarea" and self._textarea_id is not None:
            self.fields.append(
                (self._textarea_id, "".join(self._textarea_chunks))
            )
            self._textarea_id = None
            self._textarea_chunks = []

    def close(self) -> None:
        super().close()
        if self._textarea_id is not None:
            raise ValueError(f"unclosed textarea: {self._textarea_id}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def extract_fields(path: Path) -> List[dict]:
    parser = TextFieldParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()

    seen = set()
    cases = []
    for field_id, value in parser.fields:
        if field_id in seen:
            raise ValueError(f"duplicate note field id: {field_id}")
        seen.add(field_id)
        match = CASE_FIELD.fullmatch(field_id)
        if match is None:
            raise ValueError(f"unexpected note field id: {field_id}")
        cases.append(
            {
                "case_id": match.group(1),
                "input_id": field_id,
                "value": value,
            }
        )

    if not cases:
        raise ValueError(f"no case note fields found in {path}")
    cases.sort(key=lambda item: int(CASE_FIELD.fullmatch(item["input_id"]).group(2)))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    cases = extract_fields(args.input)
    payload = {
        "source_html": str(args.input),
        "source_sha256": sha256_file(args.input),
        "field_count": len(cases),
        "nonempty_field_count": sum(bool(case["value"].strip()) for case in cases),
        "cases": cases,
    }
    atomic_json(args.output, payload)
    print(
        f"Saved {payload['field_count']} note fields "
        f"({payload['nonempty_field_count']} nonempty) to {args.output}"
    )


if __name__ == "__main__":
    main()
