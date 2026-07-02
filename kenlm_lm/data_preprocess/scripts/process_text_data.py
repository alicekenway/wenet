#!/usr/bin/env python3
"""Clean generated text files.

For every regular file in the input directory:
  - drop empty lines
  - optionally convert text to lower or upper case
  - optionally apply one or more regex replacements
  - discard lines without '#'
  - split lines by '#', writing only the part before the first '#'
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Pattern


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process text files by removing empty lines and trimming labels after '#'."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing input text files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where processed files will be written.",
    )
    parser.add_argument(
        "--case",
        choices=("none", "lower", "upper"),
        default="none",
        help="Case conversion to apply before writing output. Default: none.",
    )
    parser.add_argument(
        "--replace",
        nargs=2,
        action="append",
        metavar=("REGEX", "REPLACEMENT"),
        default=[],
        help=(
            "Regex replacement to apply to each line. "
            "Can be used more than once. Example: --replace '\\$\\$' ''"
        ),
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for input and output files. Default: utf-8.",
    )
    parser.add_argument(
        "--errors",
        choices=("strict", "replace", "ignore"),
        default="replace",
        help="How to handle encoding errors. Default: replace.",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Also process hidden files such as .DS_Store. Default: skip hidden files.",
    )
    return parser.parse_args()


def convert_case(text: str, case_mode: str) -> str:
    if case_mode == "lower":
        return text.lower()
    if case_mode == "upper":
        return text.upper()
    return text


def compile_replacements(
    replacements: list[list[str]],
) -> list[tuple[Pattern[str], str]]:
    compiled_replacements = []
    for pattern, replacement in replacements:
        try:
            compiled_replacements.append((re.compile(pattern), replacement))
        except re.error as exc:
            raise SystemExit(f"Invalid regex pattern {pattern!r}: {exc}") from exc
    return compiled_replacements


def apply_replacements(
    text: str,
    replacements: list[tuple[Pattern[str], str]],
) -> str:
    for pattern, replacement in replacements:
        text = pattern.sub(replacement, text)
    return text


def process_file(
    input_file: Path,
    output_file: Path,
    case_mode: str,
    replacements: list[tuple[Pattern[str], str]],
    encoding: str,
    errors: str,
) -> tuple[int, int, int]:
    total_lines = 0
    written_lines = 0
    skipped_lines = 0

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with input_file.open("r", encoding=encoding, errors=errors) as src, output_file.open(
        "w", encoding=encoding, errors=errors
    ) as dst:
        for raw_line in src:
            total_lines += 1
            line = raw_line.strip()
            if not line:
                skipped_lines += 1
                continue

            line = convert_case(line, case_mode)
            line = apply_replacements(line, replacements)
            if "#" not in line:
                skipped_lines += 1
                continue

            text, _ = line.split("#", 1)
            dst.write(f"{text.rstrip()}\n")
            written_lines += 1

    return total_lines, written_lines, skipped_lines


def main() -> int:
    args = parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    replacements = compile_replacements(args.replace)

    input_files = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.is_file() and (args.include_hidden or not path.name.startswith("."))
    )
    processed_files = 0
    total_lines = 0
    written_lines = 0
    skipped_lines = 0

    for input_file in input_files:
        output_file = args.output_dir / input_file.name
        file_total, file_written, file_skipped = process_file(
            input_file=input_file,
            output_file=output_file,
            case_mode=args.case,
            replacements=replacements,
            encoding=args.encoding,
            errors=args.errors,
        )
        processed_files += 1
        total_lines += file_total
        written_lines += file_written
        skipped_lines += file_skipped

    print(f"Processed files: {processed_files}")
    print(f"Read lines: {total_lines}")
    print(f"Written lines: {written_lines}")
    print(f"Skipped lines: {skipped_lines}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
