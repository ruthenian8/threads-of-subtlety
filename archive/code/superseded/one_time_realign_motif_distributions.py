"""Restore motif-distribution rows to their graph-added source order."""

import argparse
import ast
from collections import defaultdict, deque
from pathlib import Path
import os

from tqdm import tqdm


def document_identity(serialized: str):
    """Read text/source/label without evaluating the full document record."""
    serialized = serialized.rstrip("\r\n")
    prefix = "{'text': "
    scenes_marker = ", 'scenes': "
    source_marker = ", 'source': "
    label_marker = ", 'label': "
    text_end = serialized.find(scenes_marker, len(prefix))
    source_at = serialized.rfind(source_marker)
    label_at = serialized.rfind(label_marker)
    if not (
        serialized.startswith(prefix)
        and serialized.endswith("}")
        and text_end > 0
        and source_at > text_end
        and label_at > source_at
    ):
        raise ValueError("Unexpected serialized Document layout")
    return (
        ast.literal_eval(serialized[len(prefix) : text_end]),
        ast.literal_eval(
            serialized[source_at + len(source_marker) : label_at]
        ),
        ast.literal_eval(serialized[label_at + len(label_marker) : -1]),
    )


def count_documents(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def realign_file(source_path: Path, distribution_path: Path) -> None:
    offsets = defaultdict(deque)
    with distribution_path.open("rb") as distributions:
        while True:
            offset = distributions.tell()
            line = distributions.readline()
            if not line:
                break
            if line.strip():
                offsets[document_identity(line.decode("utf-8"))].append(offset)

    temporary_path = distribution_path.with_suffix(
        distribution_path.suffix + ".realigning"
    )
    expected = count_documents(source_path)
    with source_path.open(encoding="utf-8") as source, distribution_path.open(
        "rb"
    ) as distributions, temporary_path.open("wb") as output:
        for source_line in tqdm(
            source,
            total=expected,
            desc=distribution_path.name,
            unit="doc",
        ):
            if not source_line.strip():
                continue
            identity = document_identity(source_line)
            if not offsets[identity]:
                raise ValueError(
                    f"Missing distribution row for identity in {source_path}"
                )
            distributions.seek(offsets[identity].popleft())
            output.write(distributions.readline())
        output.flush()
        os.fsync(output.fileno())

    remaining = sum(len(identity_offsets) for identity_offsets in offsets.values())
    if remaining:
        temporary_path.unlink()
        raise ValueError(
            f"Distribution file has {remaining} unmatched rows: {distribution_path}"
        )
    os.replace(temporary_path, distribution_path)
    print(f"realigned {distribution_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/unirst2")
    parser.add_argument("--feature-set", default="m3")
    args = parser.parse_args()

    root = Path(args.root)
    distribution_paths = sorted(
        root.glob(
            "rel-*/hc3_*.discourse_parsed.graph_added."
            f"{args.feature_set}_motif_dists.jsonl"
        )
    )
    if not distribution_paths:
        raise FileNotFoundError(f"No distribution files found below {root}")

    for distribution_path in distribution_paths:
        source_path = Path(
            str(distribution_path).replace(
                f".{args.feature_set}_motif_dists.jsonl", ".jsonl"
            )
        )
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        realign_file(source_path, distribution_path)


if __name__ == "__main__":
    main()
