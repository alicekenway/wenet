#!/usr/bin/env python3
"""Create a tab-separated text-to-phoneme dictionary from a text file.

The input contains one transcript per line.  The output retains every input
line, in order, as::

    <text>\t<space-separated phonemes>

Example:
    python text_to_phoneme_dict.py texts.txt text_phonemes.tsv \
        --language en-us --workers 8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Iterable, Iterator, List, Optional, Sequence, Tuple

from nemo_jsonl_to_phonemes import (
    DEFAULT_ESPEAK_LIBRARY,
    DEFAULT_LANGUAGE,
    DEFAULT_WORD_TOKEN,
    initialize_worker,
    parse_word_token,
    phonemize_batch,
)


Record = Tuple[int, str]


def read_batches(input_path: Path, batch_size: int) -> Iterator[List[Record]]:
    """Yield bounded, ordered batches without dropping blank input lines."""
    batch: List[Record] = []
    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            text = raw_line.rstrip("\r\n")
            if "\t" in text:
                raise ValueError(
                    f"Input line {line_number} contains a tab, which is not "
                    "valid in a tab-separated dictionary"
                )
            batch.append((line_number, text))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def write_completed_batch(
    output_file: object,
    batch: Sequence[Record],
    future: concurrent.futures.Future[List[str]],
) -> int:
    """Write one completed batch, retaining exact input ordering."""
    first_line, last_line = batch[0][0], batch[-1][0]
    try:
        phonemes = future.result()
    except Exception as error:
        raise RuntimeError(
            f"Failed to phonemize input lines {first_line}-{last_line}: {error}"
        ) from error
    if len(phonemes) != len(batch):
        raise RuntimeError(
            f"Worker returned {len(phonemes)} results for {len(batch)} input lines "
            f"{first_line}-{last_line}"
        )

    for (_, text), phoneme_text in zip(batch, phonemes):
        output_file.write(f"{text}\t{phoneme_text}\n")
    return len(batch)


def convert_text_file(
    input_path: Path,
    output_path: Path,
    espeak_library: str,
    language: str,
    word_token: Optional[str],
    workers: int,
    batch_size: int,
    pending_batches: int,
    progress_every: int,
) -> int:
    """Phonemize a line-oriented text file into a TSV dictionary."""
    if not input_path.is_file():
        raise ValueError(f"Input text file does not exist or is not a file: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different")
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if pending_batches < 1:
        raise ValueError("--pending-batches must be at least 1")
    if progress_every < 0:
        raise ValueError("--progress-every must be at least 0")
    if not language.strip():
        raise ValueError("--language must not be empty")
    if word_token is not None and (
        not word_token or any(character.isspace() for character in word_token)
    ):
        raise ValueError(
            "--word-token must be 'none' or a non-empty token without whitespace"
        )

    # Validate dependencies and configure the parent before processes are made.
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
    except ImportError as error:
        raise RuntimeError("phonemizer is not installed in this Python environment") from error
    EspeakWrapper.set_library(espeak_library)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending: Deque[Tuple[List[Record], concurrent.futures.Future[List[str]]]] = deque()
    records_written = 0
    next_progress = progress_every

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_worker,
        initargs=(espeak_library, language, word_token),
    ) as executor, output_path.open("w", encoding="utf-8") as output_file:
        for batch in read_batches(input_path, batch_size):
            future = executor.submit(phonemize_batch, [text for _, text in batch])
            pending.append((batch, future))

            if len(pending) >= pending_batches:
                completed_batch, completed_future = pending.popleft()
                records_written += write_completed_batch(
                    output_file, completed_batch, completed_future
                )
                if progress_every and records_written >= next_progress:
                    print(f"phonemized {records_written} lines", file=sys.stderr)
                    next_progress = (records_written // progress_every + 1) * progress_every

        while pending:
            completed_batch, completed_future = pending.popleft()
            records_written += write_completed_batch(
                output_file, completed_batch, completed_future
            )
            if progress_every and records_written >= next_progress:
                print(f"phonemized {records_written} lines", file=sys.stderr)
                next_progress = (records_written // progress_every + 1) * progress_every

    return records_written


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    default_workers = min(8, os.cpu_count() or 1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_text", type=Path, help="UTF-8 text file with one transcript per line")
    parser.add_argument("output_dict", type=Path, help="output TSV dictionary path")
    parser.add_argument(
        "--espeak-library", default=DEFAULT_ESPEAK_LIBRARY,
        help=f"path to libespeak-ng.so (default: {DEFAULT_ESPEAK_LIBRARY})",
    )
    parser.add_argument("-l", "--language", default=DEFAULT_LANGUAGE,
                        help=f"espeak language code (default: {DEFAULT_LANGUAGE})")
    parser.add_argument(
        "--word-token", type=parse_word_token, default=DEFAULT_WORD_TOKEN,
        help=("standalone word-boundary token, or 'none' to disable it "
              f"(default: {DEFAULT_WORD_TOKEN})"),
    )
    parser.add_argument("--workers", type=int, default=default_workers,
                        help=f"worker processes (default: {default_workers})")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="lines sent to each worker per task (default: 256)")
    parser.add_argument("--pending-batches", type=int, default=None,
                        help="maximum queued batches (default: workers * 2)")
    parser.add_argument("--progress-every", type=int, default=10000,
                        help="report progress every N lines; 0 disables it (default: 10000)")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    pending_batches = args.pending_batches if args.pending_batches is not None else args.workers * 2
    try:
        records_written = convert_text_file(
            input_path=args.input_text,
            output_path=args.output_dict,
            espeak_library=args.espeak_library,
            language=args.language,
            word_token=args.word_token,
            workers=args.workers,
            batch_size=args.batch_size,
            pending_batches=pending_batches,
            progress_every=args.progress_every,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"wrote {records_written} entries to {args.output_dict}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
