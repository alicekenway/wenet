#!/usr/bin/env python3
"""Convert the text field in a NeMo JSONL manifest to phoneme tokens.

All fields other than the transcript field are preserved.  The default espeak
backend writes one independent phoneme per space-separated CTC token.  By
default, words are separated by the standalone ``▁`` token.  Pass
``--word-token none`` to omit word-boundary tokens.

The g2pw backend uses g2p-mix to convert Mandarin (and mixed Mandarin-English)
text to IPA with surface tone sandhi.  Its output is the space-separated
``G2PResult.phones`` sequence and does not contain word-boundary tokens.

Example:
    python nemo_jsonl_to_phonemes.py input.jsonl output.jsonl \
        --language en-us --word-token ▁ --workers 8

    python nemo_jsonl_to_phonemes.py input.jsonl output.jsonl \
        --backend g2pw --workers 1
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Iterable, Iterator, List, Optional, Sequence, Tuple


DEFAULT_ESPEAK_LIBRARY = "/home/jinyang_wang/.local/lib/libespeak-ng.so"
DEFAULT_BACKEND = "espeak"
DEFAULT_LANGUAGE = "en-us"
DEFAULT_WORD_TOKEN = "▁"
ESPEAK_BACKEND = "espeak"
G2PW_BACKEND = "g2pw"
SUPPORTED_BACKENDS = (ESPEAK_BACKEND, G2PW_BACKEND)
G2PW_WARMUP_TEXT = "测试 test"
DEFAULT_G2PW_OVERRIDES_PATH = Path(__file__).with_name(
    "g2pw_pinyin_overrides.json"
)
G2P_MIX_REQUIREMENT = (
    "g2p-mix[g2pw] @ "
    "git+https://github.com/pengzhendong/g2p-mix.git@"
    "36ea4d8c4fdb374a3fb8c260d58afc03436a348b"
)
# phonemizer does not allow its phone and word separators to both be a space.
# Use an internal private-use marker, then replace it after phonemization when
# the caller does not want an explicit word-boundary token.
DISABLED_WORD_TOKEN_MARKER = "\ue000"

_worker_backend = DEFAULT_BACKEND
_worker_language = DEFAULT_LANGUAGE
_worker_word_token: Optional[str] = DEFAULT_WORD_TOKEN
_worker_g2p: Any = None


G2PWPinyinOverrides = dict[Tuple[str, str], str]


def load_g2pw_pinyin_overrides(
    path: Path = DEFAULT_G2PW_OVERRIDES_PATH,
) -> G2PWPinyinOverrides:
    """Load and validate character-specific Pinyin corrections from JSON."""
    try:
        with path.open("r", encoding="utf-8") as mapping_file:
            raw_mapping = json.load(mapping_file)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"invalid G2PW override JSON in {path}: {error.msg}"
        ) from error
    except OSError as error:
        raise RuntimeError(
            f"cannot read G2PW override mapping {path}: {error}"
        ) from error

    if not isinstance(raw_mapping, dict):
        raise RuntimeError(
            f"G2PW override mapping {path} must contain a JSON object"
        )

    overrides: G2PWPinyinOverrides = {}
    for character, pronunciations in raw_mapping.items():
        if not isinstance(character, str) or len(character) != 1:
            raise RuntimeError(
                f"G2PW override key {character!r} in {path} must be one "
                "character"
            )
        if not isinstance(pronunciations, dict) or not pronunciations:
            raise RuntimeError(
                f"G2PW overrides for {character!r} in {path} must be a "
                "non-empty JSON object"
            )
        for source, replacement in pronunciations.items():
            if (
                not isinstance(source, str)
                or not source
                or not isinstance(replacement, str)
                or not replacement
                or source == replacement
            ):
                raise RuntimeError(
                    f"G2PW override {character!r}: {source!r} -> "
                    f"{replacement!r} in {path} must contain distinct, "
                    "non-empty string pronunciations"
                )
            overrides[(character, source)] = replacement
    return overrides


def apply_g2pw_pinyin_overrides(
    text: str,
    syllables: Sequence[Optional[str]],
    overrides: G2PWPinyinOverrides,
) -> Tuple[Optional[str], ...]:
    """Apply reviewed character-and-pronunciation G2PW corrections."""
    return tuple(
        overrides.get((text[index], syllable), syllable)
        if index < len(text)
        else syllable
        for index, syllable in enumerate(syllables)
    )


def create_g2pw_converter() -> Any:
    """Create the pinned g2p-mix Mandarin-to-IPA converter."""
    try:
        from g2p_mix import G2P
        from g2p_mix.backends import G2PWBackend
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "the IPA-capable g2p-mix G2P API is unavailable; install the "
            "pinned dependency from phonemie_tools/requirements-g2pw.txt "
            f"({G2P_MIX_REQUIREMENT})"
        ) from error

    overrides = load_g2pw_pinyin_overrides()

    class CorrectedG2PWBackend(G2PWBackend):
        """Normalize reviewed upstream G2PW syllables before validation."""

        def _convert(self, text: str) -> Tuple[Optional[str], ...]:
            syllables = super()._convert(text)
            return apply_g2pw_pinyin_overrides(text, syllables, overrides)

    try:
        converter = G2P(
            mode="mandarin",
            output="ipa",
            backend=CorrectedG2PWBackend(unknown_policy="strict"),
            unknown="strict",
            tone_sandhi=True,
        )
    except TypeError as error:
        raise RuntimeError(
            "the installed g2p-mix has the old API without IPA support; "
            "install the pinned dependency from "
            "phonemie_tools/requirements-g2pw.txt"
        ) from error
    except Exception as error:
        raise RuntimeError(
            f"failed to configure the g2p-mix G2PW backend: {error}"
        ) from error

    if getattr(converter, "output", None) != "ipa":
        raise RuntimeError(
            "the installed g2p-mix did not create an IPA converter; install "
            "the pinned dependency from phonemie_tools/requirements-g2pw.txt"
        )
    return converter


def format_exception_chain(error: BaseException) -> str:
    """Render an exception together with useful wrapped root causes."""
    messages = []
    current: Optional[BaseException] = error
    while current is not None:
        message = str(current) or type(current).__name__
        if not messages or message != messages[-1]:
            messages.append(message)
        current = current.__cause__ or current.__context__
    return " -> ".join(messages)


def initialize_g2pw_resources() -> None:
    """Populate G2PW and English caches in the current process.

    Both G2PW's ModelScope snapshot and g2p-en's NLTK resources are loaded
    lazily. If several fresh workers encounter their first non-empty input at
    once, their cache initialization can race and leave one worker observing a
    temporarily unavailable resource. A single mixed-language conversion in
    the parent makes those shared on-disk resources ready first.
    """
    converter = create_g2pw_converter()
    try:
        converter(G2PW_WARMUP_TEXT)
    except Exception as error:
        raise RuntimeError(
            "failed to initialize the G2PW and English resources: "
            f"{format_exception_chain(error)}"
        ) from error


def _initialize_g2pw_resources_child(connection: Any) -> None:
    """Run cache initialization and return any error to the parent."""
    try:
        initialize_g2pw_resources()
    except BaseException as error:
        connection.send(format_exception_chain(error))
    else:
        connection.send(None)
    finally:
        connection.close()


def warm_up_g2pw_resources(
    wait_interval: float = 0,
    on_wait: Optional[Callable[[], None]] = None,
) -> None:
    """Warm resources in an isolated process before creating ONNX workers.

    Loading ONNX Runtime in the parent before ProcessPoolExecutor forks can
    deadlock its children. A spawned, short-lived process prepares the shared
    caches and exits, leaving the parent safe to create the real worker pool.
    """
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_initialize_g2pw_resources_child,
        args=(child_connection,),
    )
    process.start()
    child_connection.close()
    if wait_interval > 0 and on_wait is not None:
        while process.is_alive():
            process.join(timeout=wait_interval)
            if process.is_alive():
                on_wait()
    else:
        process.join()

    error_message = (
        parent_connection.recv() if parent_connection.poll() else None
    )
    parent_connection.close()
    if error_message is not None:
        raise RuntimeError(
            "failed to initialize the G2PW and English resources before "
            f"starting parallel workers: {error_message}"
        )
    if process.exitcode != 0:
        raise RuntimeError(
            "the isolated G2PW resource initializer exited with status "
            f"{process.exitcode}"
        )


def initialize_worker(
    espeak_library: str,
    language: str,
    word_token: Optional[str],
    backend: str = DEFAULT_BACKEND,
) -> None:
    """Configure the selected phonemizer inside one worker process."""
    global _worker_backend, _worker_g2p, _worker_language, _worker_word_token

    _worker_backend = backend
    _worker_g2p = None
    if backend == G2PW_BACKEND:
        _worker_g2p = create_g2pw_converter()
        return
    if backend != ESPEAK_BACKEND:
        raise RuntimeError(f"unsupported phoneme backend: {backend}")

    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(espeak_library)
    _worker_language = language
    _worker_word_token = word_token


def phonemize_espeak_batch(texts: Sequence[str]) -> List[str]:
    """Phonemize one batch with espeak in a worker process."""
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


def phonemize_g2pw_batch(texts: Sequence[str]) -> List[str]:
    """Convert one batch to surface-tone IPA with g2p-mix and G2PW."""
    if _worker_g2p is None:
        raise RuntimeError("the G2PW worker was not initialized")

    output: List[str] = []
    for index, source_text in enumerate(texts):
        if not source_text:
            output.append("")
            continue
        try:
            result = _worker_g2p(source_text)
            phones = result.phones
        except Exception as error:
            preview = source_text[:120]
            if len(source_text) > len(preview):
                preview += "..."
            raise RuntimeError(
                f"g2p-mix failed for batch item {index + 1}: "
                f"{format_exception_chain(error)}; "
                f"text={preview!r}"
            ) from error

        if isinstance(phones, (str, bytes)) or not isinstance(phones, Sequence):
            raise RuntimeError(
                "g2p-mix returned phones that are not a non-string sequence "
                f"for batch item {index + 1}"
            )
        if any(not isinstance(phone, str) or not phone for phone in phones):
            raise RuntimeError(
                "g2p-mix returned an invalid phone for batch item "
                f"{index + 1}: {phones!r}"
            )
        if any(any(character.isspace() for character in phone) for phone in phones):
            raise RuntimeError(
                "g2p-mix returned a phone containing whitespace for batch item "
                f"{index + 1}: {phones!r}"
            )
        output.append(" ".join(phones))
    return output


def phonemize_batch(texts: Sequence[str]) -> List[str]:
    """Phonemize one batch with the backend configured for this worker."""
    if _worker_backend == G2PW_BACKEND:
        return phonemize_g2pw_batch(texts)
    if _worker_backend == ESPEAK_BACKEND:
        return phonemize_espeak_batch(texts)
    raise RuntimeError(f"unsupported phoneme backend: {_worker_backend}")


Record = Tuple[int, dict[str, Any], str]


def count_manifest_records(input_path: Path) -> int:
    """Count non-blank JSONL records for exact progress and ETA reporting."""
    with input_path.open("r", encoding="utf-8") as input_file:
        return sum(1 for raw_line in input_file if raw_line.strip())


def format_duration(seconds: float) -> str:
    """Format a non-negative duration as HH:MM:SS."""
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ProgressReporter:
    """Report record progress, elapsed time, throughput, and approximate ETA."""

    def __init__(
        self,
        total_records: int,
        progress_every: int,
        progress_interval: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.total_records = total_records
        self.progress_every = progress_every
        self.progress_interval = progress_interval
        self.clock = clock
        self.started_at = clock()
        self.last_report_at = self.started_at
        self.next_record_report = progress_every

    @property
    def enabled(self) -> bool:
        return self.progress_every > 0 or self.progress_interval > 0

    def report(
        self,
        completed_records: int,
        *,
        waiting: bool = False,
        force: bool = False,
    ) -> None:
        """Print a report when a record/time threshold has been reached."""
        if not self.enabled:
            return

        now = self.clock()
        record_due = (
            self.progress_every > 0
            and completed_records >= self.next_record_report
        )
        time_due = (
            self.progress_interval > 0
            and now - self.last_report_at >= self.progress_interval
        )
        if not (force or record_due or time_due):
            return

        elapsed = max(0.0, now - self.started_at)
        percentage = (
            100.0
            if self.total_records == 0
            else completed_records / self.total_records * 100.0
        )
        message = (
            f"progress: {completed_records}/{self.total_records} "
            f"({percentage:.1f}%); elapsed {format_duration(elapsed)}"
        )
        if completed_records > 0 and elapsed > 0:
            records_per_second = completed_records / elapsed
            remaining_records = max(
                0,
                self.total_records - completed_records,
            )
            eta = remaining_records / records_per_second
            message += (
                f"; {records_per_second:.2f} records/s; "
                f"ETA {format_duration(eta)}"
            )
        else:
            message += "; rate unavailable; ETA unavailable"
            if waiting:
                message += " (initializing model or processing first batch)"
        print(message, file=sys.stderr, flush=True)
        self.last_report_at = now
        if self.progress_every > 0:
            self.next_record_report = (
                completed_records // self.progress_every + 1
            ) * self.progress_every


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
    wait_interval: float = 0,
    on_wait: Optional[Callable[[], None]] = None,
) -> int:
    """Wait for one batch and write it, preserving input record order."""
    first_line = batch[0][0]
    last_line = batch[-1][0]
    try:
        if wait_interval > 0 and on_wait is not None:
            while True:
                try:
                    phoneme_texts = future.result(timeout=wait_interval)
                    break
                except concurrent.futures.TimeoutError:
                    on_wait()
        else:
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
    progress_interval: float = 30.0,
    backend: str = DEFAULT_BACKEND,
) -> int:
    """Convert a manifest with bounded memory and ordered parallel output."""
    if not input_path.is_file():
        raise ValueError(f"Input JSONL does not exist or is not a file: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output JSONL paths must be different")
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"--backend must be one of: {', '.join(SUPPORTED_BACKENDS)}"
        )
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if pending_batches < 1:
        raise ValueError("--pending-batches must be at least 1")
    if progress_every < 0:
        raise ValueError("--progress-every must be at least 0")
    if progress_interval < 0:
        raise ValueError("--progress-interval must be at least 0")

    progress_enabled = progress_every > 0 or progress_interval > 0
    if progress_enabled:
        print(f"counting records in {input_path}", file=sys.stderr, flush=True)
    total_records = count_manifest_records(input_path)
    reporter = ProgressReporter(
        total_records=total_records,
        progress_every=progress_every,
        progress_interval=progress_interval,
    )
    if progress_enabled:
        print(
            f"starting phonemization: {total_records} records; "
            f"backend={backend}; workers={workers}; batch_size={batch_size}",
            file=sys.stderr,
            flush=True,
        )
    # Import and configure once in the parent so dependency/API failures happen
    # before the output file or process pool is created.  The real G2PW model is
    # lazy and is only loaded by a worker on its first non-empty transcript.
    if backend == ESPEAK_BACKEND:
        if not language.strip():
            raise ValueError("--language must not be empty")
        if word_token is not None and (
            not word_token
            or any(character.isspace() for character in word_token)
        ):
            raise ValueError(
                "--word-token must be 'none' or a non-empty token without "
                "whitespace"
            )
        try:
            from phonemizer.backend.espeak.wrapper import EspeakWrapper
        except ImportError as error:
            raise RuntimeError(
                "phonemizer is not installed in this Python environment"
            ) from error
        EspeakWrapper.set_library(espeak_library)
    elif workers > 1:
        # g2p-mix initializes both its ModelScope and g2p-en/NLTK resources
        # lazily. Warm them once to avoid concurrent first-use downloads and
        # file reads when several worker processes start together.
        warm_up_g2pw_resources(
            wait_interval=progress_interval,
            on_wait=lambda: reporter.report(0, waiting=True, force=True),
        )
    else:
        create_g2pw_converter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending: Deque[
        Tuple[List[Record], concurrent.futures.Future[List[str]]]
    ] = deque()
    records_written = 0

    def write_pending_batch(
        output_file: Any,
        batch: Sequence[Record],
        future: concurrent.futures.Future[List[str]],
    ) -> None:
        nonlocal records_written
        records_written += write_completed_batch(
            output_file,
            batch,
            future,
            text_key,
            wait_interval=progress_interval,
            on_wait=lambda: reporter.report(
                records_written,
                waiting=True,
                force=True,
            ),
        )
        reporter.report(
            records_written,
            force=records_written == total_records,
        )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_worker,
        initargs=(espeak_library, language, word_token, backend),
    ) as executor:
        with output_path.open("w", encoding="utf-8") as output_file:
            for batch in read_batches(input_path, text_key, batch_size):
                texts = [text for _, _, text in batch]
                future = executor.submit(phonemize_batch, texts)
                pending.append((batch, future))

                if len(pending) >= pending_batches:
                    completed_batch, completed_future = pending.popleft()
                    write_pending_batch(
                        output_file,
                        completed_batch,
                        completed_future,
                    )

            while pending:
                completed_batch, completed_future = pending.popleft()
                write_pending_batch(
                    output_file,
                    completed_batch,
                    completed_future,
                )

    if total_records == 0:
        reporter.report(0, force=True)

    return records_written


def parse_word_token(value: str) -> Optional[str]:
    """Convert the CLI spelling 'none' to a disabled word separator."""
    return None if value.casefold() == "none" else value


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    default_espeak_workers = min(8, os.cpu_count() or 1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path, help="input NeMo JSONL manifest")
    parser.add_argument("output_jsonl", type=Path, help="output phoneme JSONL manifest")
    parser.add_argument(
        "--backend",
        choices=SUPPORTED_BACKENDS,
        default=DEFAULT_BACKEND,
        help=(
            "phoneme backend: espeak for the existing language-configurable "
            "path, or g2pw for Mandarin/mixed-text surface-tone IPA "
            f"(default: {DEFAULT_BACKEND})"
        ),
    )
    parser.add_argument(
        "--espeak-library",
        default=DEFAULT_ESPEAK_LIBRARY,
        help=(
            "path to libespeak-ng.so; used only by --backend espeak "
            f"(default: {DEFAULT_ESPEAK_LIBRARY})"
        ),
    )
    parser.add_argument(
        "-l",
        "--language",
        default=DEFAULT_LANGUAGE,
        help=(
            "espeak language code; used only by --backend espeak "
            f"(default: {DEFAULT_LANGUAGE})"
        ),
    )
    parser.add_argument(
        "--word-token",
        type=parse_word_token,
        default=DEFAULT_WORD_TOKEN,
        help=(
            "standalone espeak word-boundary token, or 'none' to disable it; "
            "not used by --backend g2pw "
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
        default=None,
        help=(
            "worker processes (default: 1 for g2pw; "
            f"{default_espeak_workers} for espeak)"
        ),
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
        help=(
            "report progress after every N completed records; 0 disables the "
            "record trigger (default: 10000)"
        ),
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help=(
            "report a heartbeat after this many seconds while waiting and "
            "include elapsed time, rate, and ETA; 0 disables the time trigger "
            "(default: 30)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    workers = args.workers
    if workers is None:
        workers = 1 if args.backend == G2PW_BACKEND else min(
            8, os.cpu_count() or 1
        )
    pending_batches = (
        args.pending_batches
        if args.pending_batches is not None
        else workers * 2
    )

    try:
        records_written = convert_manifest(
            input_path=args.input_jsonl,
            output_path=args.output_jsonl,
            espeak_library=args.espeak_library,
            language=args.language,
            word_token=args.word_token,
            text_key=args.text_key,
            workers=workers,
            batch_size=args.batch_size,
            pending_batches=pending_batches,
            progress_every=args.progress_every,
            progress_interval=args.progress_interval,
            backend=args.backend,
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
