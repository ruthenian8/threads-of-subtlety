#!/usr/bin/env python3
"""Validate an independently completed UniRST2 coarse-mapping review form."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


ALLOWED = {"acceptable", "questionable", "incorrect"}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("form", nargs="?", type=Path, default=Path("artifacts/unirst2_minimal_contribution/mapping_review/mapping_review_form.json"))
    parser.add_argument("--allow-pending", action="store_true", help="report incomplete fields without failing")
    parser.add_argument("--output", type=Path, default=Path("artifacts/unirst2_minimal_contribution/mapping_review/validation_summary.json"))
    args = parser.parse_args()
    data = json.loads(args.form.read_text(encoding="utf-8"))
    errors = []
    judgments = []
    for section, key in (("relation_mappings", "judgment"), ("decisive_groups", "mapping_judgment")):
        for index, row in enumerate(data.get(section, [])):
            value = str(row.get(key, "")).strip().lower()
            if not value:
                errors.append(f"{section}[{index}].{key}: pending")
            elif section == "relation_mappings" and value not in ALLOWED:
                errors.append(f"{section}[{index}].{key}: invalid {value!r}")
            elif section == "decisive_groups" and value not in ALLOWED:
                # The group task requests an interpretation, so a substantive
                # free-text judgment is valid even when it is not categorical.
                judgments.append("qualified_free_text")
            else:
                judgments.append(value)
            if value in {"questionable", "incorrect"} and not str(row.get("rationale", row.get("interpretation", ""))).strip():
                errors.append(f"{section}[{index}]: rationale required for {value}")
    relation_counts = Counter(str(row.get("judgment", "")).strip().lower() for row in data.get("relation_mappings", []))
    summary = {
        "status": "complete" if not errors else "pending_or_invalid",
        "counts": dict(Counter(judgments)),
        "relation_mapping_counts": dict(relation_counts),
        "decisive_group_count": len(data.get("decisive_groups", [])),
        "issues": errors,
    }
    atomic_json(args.output, summary)
    print(json.dumps(summary, indent=2))
    if errors and not args.allow_pending:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
