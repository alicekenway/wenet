#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def visible_char(ch: str) -> str:
    """Make whitespace characters visible in output."""
    if ch == " ":
        return "<space>"
    if ch == "\t":
        return "<tab>"
    if ch == "\n":
        return "<newline>"
    if ch == "\r":
        return "<carriage_return>"
    return ch


def main():
    parser = argparse.ArgumentParser(
        description="Extract unique words or characters from a JSONL file."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output text file path",
    )
    parser.add_argument(
        "--field",
        "-f",
        default="txt",
        help="JSON field containing text. Default: txt",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["words", "chars"],
        default="words",
        help="Extraction mode: words or chars. Default: words",
    )
    parser.add_argument(
        "--lower",
        action="store_true",
        help="Convert text to lowercase before extraction",
    )
    parser.add_argument(
        "--keep-whitespace",
        action="store_true",
        help="In chars mode, keep whitespace characters and output them as visible labels",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    unique_items = set()

    with input_path.open("r", encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skip invalid JSON at line {line_num}: {e}")
                continue

            if args.field not in item:
                print(f"Warning: line {line_num} does not contain field '{args.field}', skipped")
                continue

            text = item[args.field]

            if not isinstance(text, str):
                print(f"Warning: field '{args.field}' at line {line_num} is not string, skipped")
                continue

            if args.lower:
                text = text.lower()

            if args.mode == "words":
                items = text.split()
                unique_items.update(items)

            elif args.mode == "chars":
                for ch in text:
                    if ch.isspace() and not args.keep_whitespace:
                        continue

                    if args.keep_whitespace:
                        unique_items.add(visible_char(ch))
                    else:
                        unique_items.add(ch)

    sorted_items = sorted(unique_items)

    with output_path.open("w", encoding="utf-8") as fout:
        for item in sorted_items:
            fout.write(item + "\n")

    print(f"Done. Found {len(sorted_items)} unique {args.mode}.")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()