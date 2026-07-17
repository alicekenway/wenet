#!/usr/bin/env python3
"""Convert the text field in a NeMo JSONL manifest to phoneme tokens.

All fields other than the transcript field are preserved.  The converter uses
phonemizer's espeak backend and writes one independent phoneme per
space-separated CTC token.  By default, words are separated by the standalone
``▁`` token.  Pass ``--word-token none`` to omit word-boundary tokens.

Example:
    python nemo_jsonl_to_phonemes.py input.jsonl output.jsonl \
        --language en-us --word-token ▁ --workers 8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Deque, Iterable, Iterator, List, Optional, Sequence, Tuple


DEFAULT_ESPEAK_LIBRARY = "/home/jinyang_wang/.local/lib/libespeak-ng.so"
DEFAULT_LANGUAGE = "en-us"
DEFAULT_WORD_TOKEN = "▁"
# phonemizer does not allow its phone and word separators to both be a space.
# Use an internal private-use marker, then replace it after phonemization when
# the caller does not want an explicit word-boundary token.
DISABLED_WORD_TOKEN_MARKER = "\ue000"

_worker_language = DEFAULT_LANGUAGE
_worker_word_token: Optional[str] = DEFAULT_WORD_TOKEN


def initialize_worker(
    espeak_library: str,
    language: str,
    word_token: Optional[str],
) -> None:
    """Configure phonemizer independently inside each worker process."""
    global _worker_language, _worker_word_token

    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(espeak_library)
    _worker_language = language
    _worker_word_token = word_token


def phonemize_batch(texts: Sequence[str]) -> List[str]:
    """Phonemize one batch in a worker process."""
    from phonemizer import phonemize
    from phonemizer.separator import Separator

    # Avoid asking espeak to process empty transcripts.  This also guarantees
    # that an empty input transcript remains empty in the output manifest.
    nonempty_positions = [index for index, text in enumerate(texts) if text]
    if not nonempty_positions:
        return [""] * len(texts)

    nonempty_texts = [texts[index] for index in nonempty_positions]
    word_separator = (
        DISABLED_WORD_TOKEN_MARKER
        if _worker_word_token is None
        else f" {_worker_word_token} "
    )
    phonemes = phonemize(
        nonempty_texts,
        language=_worker_language,
        backend="espeak",
        separator=Separator(
            phone=" ",
            word=word_separator,
        ),
        strip=True,
        with_stress=False,
        njobs=1,
    )

    if isinstance(phonemes, str):
        phonemes = [phonemes]
    if _worker_word_token is None:
        phonemes = [
            phones.replace(DISABLED_WORD_TOKEN_MARKER, " ")
            for phones in phonemes
        ]
    if len(phonemes) != len(nonempty_texts):
        raise RuntimeError(
            "phonemizer returned a different number of transcripts "
            f"({len(phonemes)}) than it received ({len(nonempty_texts)})"
        )

    output = [""] * len(texts)
    for position, phones in zip(nonempty_positions, phonemes):
        output[position] = phones
    return output


Record = Tuple[int, dict[str, Any], str]


def read_batches(
    input_path: Path,
    text_key: str,
    batch_size: int,
) -> Iterator[List[Record]]:
    """Read and validate bounded batches from a NeMo JSONL manifest."""
    batch: List[Record] = []

    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            if not raw_line.strip():
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error.msg}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} must contain a JSON object"
                )
            if text_key not in record:
                raise ValueError(
                    f"Line {line_number} has no {text_key!r} field"
                )

            text = record[text_key]
            if not isinstance(text, str):
                raise ValueError(
                    f"Line {line_number} field {text_key!r} must be a string"
                )

            batch.append((line_number, record, text))
            if len(batch) == batch_size:
                yield batch
                batch = []

    if batch:
        yield batch


def write_completed_batch(
    output_file: Any,
    batch: Sequence[Record],
    future: concurrent.futures.Future[List[str]],
    text_key: str,
) -> int:
    """Wait for one batch and write it, preserving input record order."""
    first_line = batch[0][0]
    last_line = batch[-1][0]
    try:
        phoneme_texts = future.result()
    except Exception as error:
        raise RuntimeError(
            f"Failed to phonemize input lines {first_line}-{last_line}: {error}"
        ) from error

    if len(phoneme_texts) != len(batch):
        raise RuntimeError(
            f"Worker returned {len(phoneme_texts)} results for "
            f"{len(batch)} records on lines {first_line}-{last_line}"
        )

    for (_, record, _), phoneme_text in zip(batch, phoneme_texts):
        record[text_key] = phoneme_text
        json.dump(record, output_file, ensure_ascii=False)
        output_file.write("\n")

    return len(batch)


def convert_manifest(
    input_path: Path,
    output_path: Path,
    espeak_library: str,
    language: str,
    word_token: Optional[str],
    text_key: str,
    workers: int,
    batch_size: int,
    pending_batches: int,
    progress_every: int,
) -> int:
    """Convert a manifest with bounded memory and ordered parallel output."""
    if not input_path.is_file():
        raise ValueError(f"Input JSONL does not exist or is not a file: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output JSONL paths must be different")
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

    # Import in the parent first so a missing dependency produces a clear error
    # before the process pool is created.
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
    except ImportError as error:
        raise RuntimeError(
            "phonemizer is not installed in this Python environment"
        ) from error
    EspeakWrapper.set_library(espeak_library)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending: Deque[
        Tuple[List[Record], concurrent.futures.Future[List[str]]]
    ] = deque()
    records_written = 0
    next_progress = progress_every

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_worker,
        initargs=(espeak_library, language, word_token),
    ) as executor:
        with output_path.open("w", encoding="utf-8") as output_file:
            for batch in read_batches(input_path, text_key, batch_size):
                texts = [text for _, _, text in batch]
                future = executor.submit(phonemize_batch, texts)
                pending.append((batch, future))

                if len(pending) >= pending_batches:
                    completed_batch, completed_future = pending.popleft()
                    records_written += write_completed_batch(
                        output_file,
                        completed_batch,
                        completed_future,
                        text_key,
                    )
                    if progress_every > 0 and records_written >= next_progress:
                        print(
                            f"phonemized {records_written} records",
                            file=sys.stderr,
                        )
                        next_progress = (
                            records_written // progress_every + 1
                        ) * progress_every

            while pending:
                completed_batch, completed_future = pending.popleft()
                records_written += write_completed_batch(
                    output_file,
                    completed_batch,
                    completed_future,
                    text_key,
                )
                if progress_every > 0 and records_written >= next_progress:
                    print(
                        f"phonemized {records_written} records",
                        file=sys.stderr,
                    )
                    next_progress = (
                        records_written // progress_every + 1
                    ) * progress_every

    return records_written


def parse_word_token(value: str) -> Optional[str]:
    """Convert the CLI spelling 'none' to a disabled word separator."""
    return None if value.casefold() == "none" else value


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    default_workers = min(8, os.cpu_count() or 1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path, help="input NeMo JSONL manifest")
    parser.add_argument("output_jsonl", type=Path, help="output phoneme JSONL manifest")
    parser.add_argument(
        "--espeak-library",
        default=DEFAULT_ESPEAK_LIBRARY,
        help=f"path to libespeak-ng.so (default: {DEFAULT_ESPEAK_LIBRARY})",
    )
    parser.add_argument(
        "-l",
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"espeak language code (default: {DEFAULT_LANGUAGE})",
    )
    parser.add_argument(
        "--word-token",
        type=parse_word_token,
        default=DEFAULT_WORD_TOKEN,
        help=(
            "standalone word-boundary token, or 'none' to disable it "
            f"(default: {DEFAULT_WORD_TOKEN})"
        ),
    )
    parser.add_argument(
        "--text-key",
        default="text",
        help="JSON field containing the transcript (default: text)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"worker processes (default: {default_workers})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="transcripts sent to each worker per task (default: 256)",
    )
    parser.add_argument(
        "--pending-batches",
        type=int,
        default=None,
        help="maximum queued batches (default: workers * 2)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="report progress every N records; 0 disables it (default: 10000)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    pending_batches = (
        args.pending_batches
        if args.pending_batches is not None
        else args.workers * 2
    )

    try:
        records_written = convert_manifest(
            input_path=args.input_jsonl,
            output_path=args.output_jsonl,
            espeak_library=args.espeak_library,
            language=args.language,
            word_token=args.word_token,
            text_key=args.text_key,
            workers=args.workers,
            batch_size=args.batch_size,
            pending_batches=pending_batches,
            progress_every=args.progress_every,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        f"wrote {records_written} records to {args.output_jsonl}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
