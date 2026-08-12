#!/usr/bin/env python3
"""Normalize the text field in a JSONL speech manifest.

The normalizer lowercases text, removes angle-bracket annotation tokens such
as ``<COMMA>`` and ``<PERIOD>``, and collapses the whitespace left behind.
All other JSON fields are preserved.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


ANNOTATION_RE = re.compile(r"<[^<>\r\n]+>")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> tuple[str, list[str]]:
    """Return normalized text and the annotation tokens that were removed."""
    annotations = ANNOTATION_RE.findall(text)
    without_annotations = ANNOTATION_RE.sub(" ", text)
    normalized = WHITESPACE_RE.sub(" ", without_annotations.lower()).strip()
    return normalized, annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input JSONL manifest")
    parser.add_argument("output", type=Path, help="Output JSONL manifest")
    parser.add_argument(
        "--text-field",
        default="text",
        help="JSON field containing the transcript (default: text)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file",
    )
    return parser.parse_args()


def load_record(line: str, line_number: int, text_field: str) -> dict[str, Any]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"line {line_number}: invalid JSON: {error}") from error

    if not isinstance(record, dict):
        raise ValueError(f"line {line_number}: expected a JSON object")
    if text_field not in record:
        raise ValueError(f"line {line_number}: missing {text_field!r} field")
    if not isinstance(record[text_field], str):
        raise ValueError(
            f"line {line_number}: {text_field!r} must contain a string"
        )
    return record


def normalize_manifest(
    input_path: Path,
    output_path: Path,
    text_field: str,
    overwrite: bool,
) -> None:
    input_path = input_path.resolve()
    output_path = output_path.resolve()

    if input_path == output_path:
        raise ValueError("input and output paths must be different")
    if not input_path.is_file():
        raise FileNotFoundError(f"input manifest does not exist: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_path}; pass --overwrite to replace it"
        )
    if not output_path.parent.is_dir():
        raise FileNotFoundError(
            f"output directory does not exist: {output_path.parent}"
        )

    rows = 0
    changed_rows = 0
    empty_text_rows = 0
    removed_annotations: Counter[str] = Counter()
    temporary_path: Path | None = None

    try:
        with input_path.open("r", encoding="utf-8") as source:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as destination:
                temporary_path = Path(destination.name)

                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        raise ValueError(f"line {line_number}: empty JSONL line")

                    record = load_record(line, line_number, text_field)
                    original_text = record[text_field]
                    normalized_text, annotations = normalize_text(original_text)
                    record[text_field] = normalized_text

                    rows += 1
                    changed_rows += normalized_text != original_text
                    empty_text_rows += normalized_text == ""
                    removed_annotations.update(annotations)

                    json.dump(record, destination, ensure_ascii=False)
                    destination.write("\n")

                destination.flush()
                os.fsync(destination.fileno())

        if rows == 0:
            raise ValueError("input manifest is empty")

        os.chmod(temporary_path, input_path.stat().st_mode & 0o777)
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(f"Input rows: {rows}")
    print(f"Output rows: {rows}")
    print(f"Changed rows: {changed_rows}")
    print(f"Empty normalized texts: {empty_text_rows}")
    print(f"Removed annotations: {sum(removed_annotations.values())}")
    for annotation, count in removed_annotations.most_common():
        print(f"  {annotation}: {count}")
    print(f"Output: {output_path}")


def main() -> None:
    args = parse_args()
    normalize_manifest(
        input_path=args.input,
        output_path=args.output,
        text_field=args.text_field,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
