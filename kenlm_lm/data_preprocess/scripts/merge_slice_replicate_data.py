#!/usr/bin/env python3
"""Merge, slice, shuffle, and replicate text data from a JSON config."""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select lines from one or more text data locations and merge them."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="JSON config file describing input datasets.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output text file path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shuffle. Default: no fixed seed.",
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


def load_config(config_path: Path) -> dict[str, dict[str, Any]]:
    try:
        with config_path.open("r", encoding="utf-8") as src:
            config = json.load(src)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON config {config_path}: {exc}") from exc

    if not isinstance(config, dict):
        raise SystemExit("Config must be a JSON object.")

    for name, item in config.items():
        if not isinstance(item, dict):
            raise SystemExit(f"Dataset {name!r} must be a JSON object.")
    return config


def get_replicate(name: str, item: dict[str, Any]) -> int:
    replicate = item.get("replicate", 1)
    if not isinstance(replicate, int) or replicate < 0:
        raise SystemExit(f"Dataset {name!r} replicate must be a non-negative integer.")
    return replicate


def get_range(name: str, item: dict[str, Any]) -> tuple[int | None, int | None]:
    selected_range = item.get("range", [])
    if selected_range in (None, []):
        return None, None

    if (
        not isinstance(selected_range, list)
        or len(selected_range) != 2
        or not all(isinstance(value, int) or value is None for value in selected_range)
    ):
        raise SystemExit(
            f"Dataset {name!r} range must be empty or like [start, end]."
        )

    start, end = selected_range
    if start is not None and start < 0:
        raise SystemExit(f"Dataset {name!r} range start must be >= 0.")
    if end is not None and end < 0:
        raise SystemExit(f"Dataset {name!r} range end must be >= 0.")
    if start is not None and end is not None and start > end:
        raise SystemExit(f"Dataset {name!r} range start must be <= end.")
    return start, end


def get_shuffle(name: str, item: dict[str, Any]) -> bool:
    shuffle = item.get("shuffle", False)
    if not isinstance(shuffle, bool):
        raise SystemExit(f"Dataset {name!r} shuffle must be true or false.")
    return shuffle


def expand_paths(name: str, item: dict[str, Any]) -> list[Path]:
    path_value = item.get("path")
    if isinstance(path_value, str):
        patterns = [path_value]
    elif isinstance(path_value, list) and all(
        isinstance(pattern, str) for pattern in path_value
    ):
        patterns = path_value
    else:
        raise SystemExit(
            f"Dataset {name!r} path must be a string or a list of strings."
        )

    paths: list[Path] = []
    for pattern in patterns:
        matched_paths = sorted(Path(path) for path in glob.glob(pattern))
        if not matched_paths:
            raise SystemExit(f"Dataset {name!r} path matched no files: {pattern}")
        paths.extend(path for path in matched_paths if path.is_file())

    if not paths:
        raise SystemExit(f"Dataset {name!r} contains no files.")
    return paths


def read_lines(path: Path, encoding: str, errors: str) -> list[str]:
    with path.open("r", encoding=encoding, errors=errors) as src:
        return [line.rstrip("\n") for line in src]


def read_range_lines(
    path: Path,
    start: int | None,
    end: int | None,
    encoding: str,
    errors: str,
) -> list[str]:
    start_index = start or 0
    selected_lines = []

    with path.open("r", encoding=encoding, errors=errors) as src:
        for index, line in enumerate(src):
            if index < start_index:
                continue
            if end is not None and index >= end:
                break
            selected_lines.append(line.rstrip("\n"))

    return selected_lines


def reservoir_sample_lines(
    path: Path,
    sample_size: int,
    rng: random.Random,
    encoding: str,
    errors: str,
) -> list[str]:
    if sample_size <= 0:
        return []

    reservoir = []
    with path.open("r", encoding=encoding, errors=errors) as src:
        for index, line in enumerate(src):
            line = line.rstrip("\n")
            if index < sample_size:
                reservoir.append(line)
                continue

            replacement_index = rng.randint(0, index)
            if replacement_index < sample_size:
                reservoir[replacement_index] = line

    return reservoir


def select_lines_from_file(
    path: Path,
    start: int | None,
    end: int | None,
    shuffle: bool,
    rng: random.Random,
    encoding: str,
    errors: str,
) -> list[str]:
    start_index = start or 0

    if not shuffle:
        return read_range_lines(path, start, end, encoding, errors)

    if end is not None:
        selected_lines = reservoir_sample_lines(path, end, rng, encoding, errors)
        rng.shuffle(selected_lines)
        return selected_lines[start_index:end]

    selected_lines = read_lines(path, encoding, errors)
    rng.shuffle(selected_lines)
    return selected_lines[start_index:end]


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    rng = random.Random(args.seed)

    dataset_count = 0
    file_count = 0
    selected_line_count = 0
    written_line_count = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding=args.encoding, errors=args.errors) as dst:
        for name, item in config.items():
            replicate = get_replicate(name, item)
            start, end = get_range(name, item)
            shuffle = get_shuffle(name, item)
            paths = expand_paths(name, item)

            dataset_count += 1
            for path in paths:
                selected_lines = select_lines_from_file(
                    path=path,
                    start=start,
                    end=end,
                    shuffle=shuffle,
                    rng=rng,
                    encoding=args.encoding,
                    errors=args.errors,
                )
                file_count += 1
                selected_line_count += len(selected_lines)

                for _ in range(replicate):
                    for line in selected_lines:
                        dst.write(f"{line}\n")
                        written_line_count += 1

    print(f"Datasets: {dataset_count}")
    print(f"Files: {file_count}")
    print(f"Selected lines before replicate: {selected_line_count}")
    print(f"Written lines: {written_line_count}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
