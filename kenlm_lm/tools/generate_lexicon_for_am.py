#!/usr/bin/env python3
"""Generate an acoustic-model lexicon from words and model tokens."""

import argparse
import json
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LM_ROOT = SCRIPT_DIR.parent
REPO_ROOT = Path(os.environ.get("ROOT", LM_ROOT.parents[1])).resolve()


def read_symbols(path: Path):
    symbol_to_id = {}
    id_to_symbol = {}
    model_vocab_size = None
    with path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            line = line.rstrip()
            if not line:
                continue
            try:
                symbol, idx_text = line.rsplit(maxsplit=1)
                idx = int(idx_text)
            except ValueError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid symbol line") from exc
            if symbol in symbol_to_id:
                raise SystemExit(f"{path}:{line_no}: duplicate symbol {symbol}")
            if idx in id_to_symbol:
                raise SystemExit(f"{path}:{line_no}: duplicate id {idx}")
            symbol_to_id[symbol] = idx
            id_to_symbol[idx] = symbol
            if symbol.startswith("#") and model_vocab_size is None:
                model_vocab_size = idx
    if model_vocab_size is None:
        model_vocab_size = max(id_to_symbol) + 1
    return symbol_to_id, id_to_symbol, model_vocab_size


def utf8_byte_tokens(word: str):
    return [f"<0x{byte:02X}>" for byte in word.encode("utf-8")]


def is_byte_token(token: str) -> bool:
    if len(token) != 6 or not token.startswith("<0x") or not token.endswith(">"):
        return False
    try:
        int(token[3:5], 16)
    except ValueError:
        return False
    return True


def is_special_token(token: str) -> bool:
    return token.startswith("<") and token.endswith(">")


def read_words(path: Path):
    words = []
    with path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            line = line.rstrip()
            if not line:
                continue
            try:
                word, _ = line.rsplit(maxsplit=1)
            except ValueError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid words line") from exc
            words.append(word)
    return words


def valid_model_tokens(token_to_id, model_vocab_size):
    return {
        token
        for token, idx in token_to_id.items()
        if idx < model_vocab_size and not token.startswith("#")
    }


def build_case_insensitive_index(tokens, token_to_id):
    folded_to_token = {}
    for token in tokens:
        folded = token.casefold()
        current = folded_to_token.get(folded)
        if current is None or token_to_id[token] < token_to_id[current]:
            folded_to_token[folded] = token
    return folded_to_token


def resolve_token(token: str, token_set, folded_token_index=None):
    if token in token_set:
        return token, False
    if folded_token_index is None:
        return None, False
    matched = folded_token_index.get(token.casefold())
    if matched is None:
        return None, False
    return matched, True


def tokenize_bpe_longest_match(
    word: str,
    token_set,
    max_token_len: int,
    word_boundary: str,
    folded_token_index=None,
):
    text = word_boundary + word
    pieces = []
    pos = 0
    used_case_insensitive = False
    while pos < len(text):
        best = None
        best_next_pos = None
        end = min(len(text), pos + max_token_len)
        for next_pos in range(end, pos, -1):
            piece = text[pos:next_pos]
            matched, used_folded = resolve_token(piece, token_set, folded_token_index)
            if matched is not None:
                best = matched
                best_next_pos = next_pos
                used_case_insensitive = used_case_insensitive or used_folded
                break
        if best is None:
            return None, text[pos:], used_case_insensitive
        pieces.append(best)
        pos = best_next_pos
    return pieces, "", used_case_insensitive


def build_byte_lexicon(words, token_to_id, model_vocab_size, fout, ignore_case: bool):
    token_set = valid_model_tokens(token_to_id, model_vocab_size)
    folded_token_index = (
        build_case_insensitive_index(token_set, token_to_id) if ignore_case else None
    )
    direct = 0
    case_insensitive = 0
    byte_fallback = 0
    rejected = []
    for word in words:
        matched_word, used_folded = resolve_token(word, token_set, folded_token_index)
        if matched_word is not None:
            pieces = [matched_word]
            direct += 1
            if used_folded:
                case_insensitive += 1
        else:
            pieces = utf8_byte_tokens(word)
            missing = [piece for piece in pieces if piece not in token_to_id]
            if missing:
                rejected.append({"word": word, "missing": missing})
                continue
            byte_fallback += 1
        fout.write(word + " " + " ".join(pieces) + "\n")
    return {
        "direct_entries": direct,
        "byte_fallback_entries": byte_fallback,
        "bpe_entries": 0,
        "case_insensitive_entries": case_insensitive,
        "rejected": rejected,
    }


def build_bpe_lexicon(
    words,
    token_to_id,
    model_vocab_size,
    fout,
    word_boundary: str,
    allow_standalone_boundary: bool,
    ignore_case: bool,
):
    token_set = valid_model_tokens(token_to_id, model_vocab_size)
    token_set = {
        token
        for token in token_set
        if not is_special_token(token) and not is_byte_token(token)
    }
    if not allow_standalone_boundary:
        token_set.discard(word_boundary)
    if not token_set:
        raise SystemExit("no usable BPE tokens found in tokens.txt")
    if ignore_case:
        max_token_len = max(
            max(len(token), len(token.casefold())) for token in token_set
        )
    else:
        max_token_len = max(len(token) for token in token_set)
    folded_token_index = (
        build_case_insensitive_index(token_set, token_to_id) if ignore_case else None
    )

    one_piece = 0
    bpe = 0
    case_insensitive = 0
    rejected = []
    for word in words:
        if word == "<unk>" and word in token_to_id:
            fout.write(f"{word} {word}\n")
            one_piece += 1
            continue
        pieces, unmatched, used_folded = tokenize_bpe_longest_match(
            word, token_set, max_token_len, word_boundary, folded_token_index
        )
        if pieces is None:
            rejected.append({"word": word, "unmatched": unmatched})
            continue
        if used_folded:
            case_insensitive += 1
        if len(pieces) == 1:
            one_piece += 1
        else:
            bpe += 1
        fout.write(word + " " + " ".join(pieces) + "\n")
    return {
        "direct_entries": one_piece,
        "byte_fallback_entries": 0,
        "bpe_entries": bpe,
        "case_insensitive_entries": case_insensitive,
        "rejected": rejected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build strict Flashlight lexicon against AM tokens.txt."
    )
    parser.add_argument(
        "--words",
        default=str(LM_ROOT / "data/words.txt"),
    )
    parser.add_argument(
        "--tokens",
        default=str(
            REPO_ROOT
            / "model/sherpa-onnx-streaming-zipformer-ctc-zh-2025-06-30/tokens.txt"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(LM_ROOT / "data/lexicon.txt"),
    )
    parser.add_argument(
        "--report",
        default=str(LM_ROOT / "reports/lexicon_report.json"),
    )
    parser.add_argument(
        "--tokenization",
        choices=["byte", "bpe"],
        default="byte",
        help=(
            "How to spell LM words with AM tokens. Use 'byte' for byte-fallback "
            "models and 'bpe' for SentencePiece/BPE tokens with a word-boundary "
            "marker such as ▁."
        ),
    )
    parser.add_argument(
        "--bpe-word-boundary",
        default="▁",
        help="Word-boundary marker used by BPE/SentencePiece AM tokens.",
    )
    parser.add_argument(
        "--bpe-allow-standalone-boundary",
        action="store_true",
        help=(
            "Allow the standalone BPE boundary token as a lexicon spelling piece. "
            "By default it is avoided because the decoder often also uses it as "
            "the silence/separator token."
        ),
    )
    parser.add_argument(
        "--ignore-case",
        "--case-insensitive",
        dest="ignore_case",
        action="store_true",
        help=(
            "Match LM words to AM tokens without requiring the same letter case. "
            "The lexicon word is kept unchanged, and the emitted spelling uses "
            "the actual token text from tokens.txt."
        ),
    )
    parser.add_argument(
        "--allow-rejected",
        action="store_true",
        help=(
            "Exit successfully even if some words cannot be represented by the "
            "AM token set. The rejected examples are still written to the report."
        ),
    )
    args = parser.parse_args()

    token_to_id, _, model_vocab_size = read_symbols(Path(args.tokens))
    words = read_words(Path(args.words))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        if args.tokenization == "byte":
            stats = build_byte_lexicon(
                words, token_to_id, model_vocab_size, fout, args.ignore_case
            )
        else:
            stats = build_bpe_lexicon(
                words,
                token_to_id,
                model_vocab_size,
                fout,
                args.bpe_word_boundary,
                args.bpe_allow_standalone_boundary,
                args.ignore_case,
            )

    rejected = stats["rejected"]
    report = {
        "words": str(Path(args.words)),
        "tokens": str(Path(args.tokens)),
        "output": str(output_path),
        "tokenization": args.tokenization,
        "ignore_case": args.ignore_case,
        "word_count": len(words),
        "direct_entries": stats["direct_entries"],
        "byte_fallback_entries": stats["byte_fallback_entries"],
        "bpe_entries": stats["bpe_entries"],
        "case_insensitive_entries": stats["case_insensitive_entries"],
        "rejected_count": len(rejected),
        "rejected_examples": rejected[:50],
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if args.allow_rejected or not rejected else 1


if __name__ == "__main__":
    raise SystemExit(main())
