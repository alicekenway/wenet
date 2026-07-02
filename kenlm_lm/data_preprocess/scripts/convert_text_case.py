#!/usr/bin/env python3
"""Convert text files to uppercase or lowercase."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a text file, or all text files in a directory, to upper/lower case."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input text file or directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output text file or directory.",
    )
    parser.add_argument(
        "--case",
        required=True,
        choices=("upper", "lower"),
        help="Convert text to uppercase or lowercase.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When input is a directory, also process files in subdirectories.",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Also process hidden files and files in hidden directories.",
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
    return parser.parse_args()


def convert_case(text: str, case_mode: str) -> str:
    if case_mode == "upper":
        return text.upper()
    return text.lower()


def is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def iter_input_files(
    input_dir: Path,
    output_dir: Path,
    recursive: bool,
    include_hidden: bool,
) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    output_dir = output_dir.resolve()
    input_files = []

    for path in sorted(input_dir.glob(pattern)):
        if not path.is_file():
            continue
        if not include_hidden and is_hidden(path.relative_to(input_dir)):
            continue
        if path.resolve().is_relative_to(output_dir):
            continue
        input_files.append(path)

    return input_files


def convert_file(
    input_file: Path,
    output_file: Path,
    case_mode: str,
    encoding: str,
    errors: str,
) -> tuple[int, int]:
    if input_file.resolve() == output_file.resolve():
        raise SystemExit(f"Input and output must be different files: {input_file}")

    line_count = 0
    character_count = 0

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with input_file.open("r", encoding=encoding, errors=errors) as src, output_file.open(
        "w", encoding=encoding, errors=errors
    ) as dst:
        for line in src:
            converted_line = convert_case(line, case_mode)
            dst.write(converted_line)
            line_count += 1
            character_count += len(converted_line)

    return line_count, character_count


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    case_mode: str,
    recursive: bool,
    include_hidden: bool,
    encoding: str,
    errors: str,
) -> tuple[int, int, int]:
    file_count = 0
    line_count = 0
    character_count = 0

    input_files = iter_input_files(input_dir, output_dir, recursive, include_hidden)
    for input_file in input_files:
        relative_path = input_file.relative_to(input_dir)
        output_file = output_dir / relative_path
        file_lines, file_characters = convert_file(
            input_file=input_file,
            output_file=output_file,
            case_mode=case_mode,
            encoding=encoding,
            errors=errors,
        )
        file_count += 1
        line_count += file_lines
        character_count += file_characters

    return file_count, line_count, character_count


def main() -> int:
    args = parse_args()

    if args.input.is_file():
        if args.output.exists() and args.output.is_dir():
            output_file = args.output / args.input.name
        else:
            output_file = args.output

        line_count, character_count = convert_file(
            input_file=args.input,
            output_file=output_file,
            case_mode=args.case,
            encoding=args.encoding,
            errors=args.errors,
        )
        print("Processed files: 1")
        print(f"Written lines: {line_count}")
        print(f"Written characters: {character_count}")
        print(f"Output: {output_file}")
        return 0

    if args.input.is_dir():
        if args.output.exists() and not args.output.is_dir():
            raise SystemExit(f"Output must be a directory when input is a directory: {args.output}")
        if args.input.resolve() == args.output.resolve():
            raise SystemExit(f"Input and output must be different directories: {args.input}")

        args.output.mkdir(parents=True, exist_ok=True)
        file_count, line_count, character_count = convert_directory(
            input_dir=args.input,
            output_dir=args.output,
            case_mode=args.case,
            recursive=args.recursive,
            include_hidden=args.include_hidden,
            encoding=args.encoding,
            errors=args.errors,
        )
        print(f"Processed files: {file_count}")
        print(f"Written lines: {line_count}")
        print(f"Written characters: {character_count}")
        print(f"Output: {args.output}")
        return 0

    raise SystemExit(f"Input does not exist: {args.input}")


if __name__ == "__main__":
    raise SystemExit(main())
