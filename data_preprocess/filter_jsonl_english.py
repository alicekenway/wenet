#!/usr/bin/env python3
import argparse
import json
import string
from pathlib import Path


# ASCII punctuation:
# !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
#
# We remove all of them except single quote.
PUNCT_TO_REMOVE = "".join(ch for ch in string.punctuation if ch != "'")
PUNCT_TRANS_TABLE = str.maketrans("", "", PUNCT_TO_REMOVE)


def contains_digit(text: str) -> bool:
    return any("0" <= ch <= "9" for ch in text)


def is_valid_english_text(text: str) -> bool:
    """
    After punctuation removal and lowercasing,
    only allow:
      - lowercase a-z
      - space-like whitespace
      - single quote '
    """
    for ch in text:
        if "a" <= ch <= "z":
            continue
        if ch == "'":
            continue
        if ch.isspace():
            continue
        return False
    return True


def clean_spaces(text: str) -> str:
    """
    Collapse repeated whitespace into one space.
    Also strip leading/trailing spaces.
    """
    return " ".join(text.split())


def process_text(text: str) -> str | None:
    """
    Return processed text if valid.
    Return None if this line should be discarded.
    """

    # Rule 2: uppercase -> lowercase
    text = text.lower()

    # Rule 4: discard if contains digit
    if contains_digit(text):
        return None

    # Rule 1: remove English ASCII punctuation except single quote
    text = text.translate(PUNCT_TRANS_TABLE)

    # Normalize spaces after punctuation removal
    text = clean_spaces(text)

    # Empty text is useless, discard it
    if not text:
        return None

    # Rule 3: discard if contains non-English characters
    if not is_valid_english_text(text):
        return None

    return text


def main():
    parser = argparse.ArgumentParser(
        description="Filter JSONL text field to clean English-only text."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output filtered JSONL file",
    )
    parser.add_argument(
        "--field",
        "-f",
        default="txt",
        help="Text field name in JSONL. Default: txt",
    )
    parser.add_argument(
        "--discard-log",
        default=None,
        help="Optional path to write discarded lines with reasons.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    discard_log_path = Path(args.discard_log) if args.discard_log else None

    total = 0
    kept = 0
    discarded = 0

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        flog = discard_log_path.open("w", encoding="utf-8") if discard_log_path else None

        try:
            for line_num, line in enumerate(fin, start=1):
                total += 1
                raw_line = line.rstrip("\n")

                if not raw_line.strip():
                    discarded += 1
                    if flog:
                        flog.write(f"{line_num}\tempty line\t{raw_line}\n")
                    continue

                try:
                    item = json.loads(raw_line)
                except json.JSONDecodeError as e:
                    discarded += 1
                    if flog:
                        flog.write(f"{line_num}\tinvalid json: {e}\t{raw_line}\n")
                    continue

                if args.field not in item:
                    discarded += 1
                    if flog:
                        flog.write(f"{line_num}\tmissing field {args.field}\t{raw_line}\n")
                    continue

                text = item[args.field]

                if not isinstance(text, str):
                    discarded += 1
                    if flog:
                        flog.write(f"{line_num}\tfield is not string\t{raw_line}\n")
                    continue

                processed_text = process_text(text)

                if processed_text is None:
                    discarded += 1
                    if flog:
                        flog.write(f"{line_num}\trejected text\t{text}\n")
                    continue

                item[args.field] = processed_text

                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept += 1

        finally:
            if flog:
                flog.close()

    print(f"Total lines:     {total}")
    print(f"Kept lines:      {kept}")
    print(f"Discarded lines: {discarded}")
    print(f"Output:          {output_path}")

    if discard_log_path:
        print(f"Discard log:     {discard_log_path}")


if __name__ == "__main__":
    main()