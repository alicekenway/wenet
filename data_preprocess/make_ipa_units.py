#!/usr/bin/env python3

from __future__ import annotations

import argparse
import collections
import unicodedata
from pathlib import Path


SPECIAL_TOKENS = [
    ("<blank>", 0),
    ("<unk>", 1),
    ("<sos/eos>", 2),
]

RESERVED = {token for token, _ in SPECIAL_TOKENS}


def normalize_phone(phone: str) -> str:
    """
    Normalize canonically equivalent Unicode IPA strings.

    For example, a precomposed character and a base character followed by
    a combining diacritic should not accidentally become separate units.
    """
    return unicodedata.normalize("NFC", phone.strip())


def read_phones(text_path: Path) -> collections.Counter[str]:
    """
    Expected input format:

        utterance_id phone1 phone2 phone3 ...

    Each whitespace-separated item after the utterance ID is one IPA unit.
    """
    counts: collections.Counter[str] = collections.Counter()

    with text_path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            fields = raw_line.strip().split()

            if not fields:
                continue

            if len(fields) == 1:
                raise ValueError(
                    f"{text_path}:{line_number}: "
                    f"utterance {fields[0]!r} has no phoneme labels"
                )

            utterance_id = fields[0]

            for raw_phone in fields[1:]:
                phone = normalize_phone(raw_phone)

                if not phone:
                    continue

                if phone in RESERVED:
                    raise ValueError(
                        f"{text_path}:{line_number}: utterance "
                        f"{utterance_id!r} contains reserved token {phone!r}"
                    )

                counts[phone] += 1

    if not counts:
        raise ValueError(f"No IPA units found in {text_path}")

    return counts


def write_units(
    output_path: Path,
    counts_path: Path,
    counts: collections.Counter[str],
) -> None:
    # Python's Unicode code-point ordering gives deterministic IDs.
    phones = sorted(counts)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for token, token_id in SPECIAL_TOKENS:
            output.write(f"{token} {token_id}\n")

        for token_id, phone in enumerate(phones, start=3):
            output.write(f"{phone} {token_id}\n")

    with counts_path.open("w", encoding="utf-8", newline="\n") as output:
        for phone, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            output.write(f"{phone}\t{count}\n")

    print(f"Number of IPA units: {len(phones)}")
    print(f"Total vocabulary size: {len(phones) + len(SPECIAL_TOKENS)}")
    print(f"Units file: {output_path}")
    print(f"Frequency report: {counts_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "text",
        type=Path,
        help="Training text: utterance_id followed by IPA units",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output units.txt path",
    )
    parser.add_argument(
        "--counts",
        type=Path,
        default=None,
        help="Optional IPA-unit frequency report",
    )
    args = parser.parse_args()

    counts_path = args.counts
    if counts_path is None:
        counts_path = args.output.with_suffix(".counts.txt")

    counts = read_phones(args.text)
    write_units(args.output, counts_path, counts)


if __name__ == "__main__":
    main()