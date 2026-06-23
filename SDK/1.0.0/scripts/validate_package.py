#!/usr/bin/env python3
"""Validate a WeNet Lite SDK model package before deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


def read_symbols(path: Path) -> dict[int, str]:
    symbols: dict[int, str] = {}
    next_id = 0
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split()
        if len(cols) == 1:
            idx, symbol = next_id, cols[0]
        elif cols[0].lstrip("-").isdigit():
            idx, symbol = int(cols[0]), cols[1]
        elif cols[-1].lstrip("-").isdigit():
            idx, symbol = int(cols[-1]), cols[0]
        else:
            raise ValueError(f"{path}:{line_no}: cannot parse symbol id")
        if idx < 0:
            raise ValueError(f"{path}:{line_no}: negative symbol id")
        symbols[idx] = symbol
        next_id = max(next_id + 1, idx + 1)
    if not symbols:
        raise ValueError(f"{path}: symbol table is empty")
    return symbols


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checksum(model_dir: Path) -> list[str]:
    checksum = model_dir / "checksum.sha256"
    if not checksum.exists():
        return []
    errors: list[str] = []
    for line_no, raw in enumerate(checksum.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            errors.append(f"checksum.sha256:{line_no}: invalid sha256sum line")
            continue
        expected, rel = parts
        rel = rel.strip()
        if rel.startswith("*"):
            rel = rel[1:].strip()
        path = model_dir / rel
        if not path.exists():
            errors.append(f"checksum.sha256:{line_no}: missing target {rel}")
            continue
        actual = sha256(path)
        if actual != expected:
            errors.append(
                f"checksum.sha256:{line_no}: mismatch for {rel}: "
                f"expected {expected}, got {actual}"
            )
    return errors


def require_files(model_dir: Path, rels: Iterable[str]) -> list[str]:
    errors = []
    for rel in rels:
        if rel and not (model_dir / rel).exists():
            errors.append(f"missing required file: {rel}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument(
        "--require-onnx",
        action="store_true",
        help="Require ONNX model files to exist; useful for runtime packages.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    errors: list[str] = []
    if not model_dir.is_dir():
        print(f"not a directory: {model_dir}")
        return 1

    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        print("missing required file: manifest.json")
        return 1

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"manifest.json: {exc}")
        return 1

    vocab = manifest.get("vocab", {})
    decoder = manifest.get("decoder", {})
    onnx = manifest.get("onnx", {})

    tokens_rel = vocab.get("tokens", "tokens.txt")
    words_rel = vocab.get("words", "words.txt")
    graph_rel = decoder.get("graph", "TLG.fst")
    required = ["manifest.json", tokens_rel, words_rel, graph_rel]
    if args.require_onnx:
        required.append(onnx.get("encoder", "encoder.onnx"))
        ctc = onnx.get("ctc", "")
        if ctc:
            required.append(ctc)
    errors.extend(require_files(model_dir, required))

    if not errors:
        try:
            tokens = read_symbols(model_dir / tokens_rel)
            words = read_symbols(model_dir / words_rel)
            blank_id = int(vocab.get("blank_id", 0))
            if blank_id not in tokens:
                errors.append(f"blank_id {blank_id} is not in {tokens_rel}")
            if not words:
                errors.append(f"{words_rel} is empty")
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    errors.extend(validate_checksum(model_dir))
    if errors:
        for error in errors:
            print(error)
        return 1

    print("basic package files and metadata are valid")
    print(f"model_dir: {model_dir}")
    print(f"sample_rate: {manifest.get('sample_rate', 16000)}")
    print(f"feature_dim: {manifest.get('feature_dim', 80)}")
    print(f"decoder: {decoder.get('type', 'greedy_ctc')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
