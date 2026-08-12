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


def tokenize_bpe_shortest(
    word: str,
    token_set,
    token_to_id,
    max_token_len: int,
    word_boundary: str,
    folded_token_index=None,
):
    """Return the globally shortest valid token spelling for a word."""
    text = word_boundary + word
    best = [None] * (len(text) + 1)
    best[0] = ([], False)
    for pos in range(len(text)):
        if best[pos] is None:
            continue
        prefix, prefix_used_folded = best[pos]
        end = min(len(text), pos + max_token_len)
        for next_pos in range(pos + 1, end + 1):
            surface = text[pos:next_pos]
            matched, used_folded = resolve_token(
                surface, token_set, folded_token_index
            )
            if matched is None:
                continue
            candidate = prefix + [matched]
            candidate_used_folded = prefix_used_folded or used_folded
            current = best[next_pos]
            candidate_key = (
                len(candidate),
                tuple(token_to_id[token] for token in candidate),
            )
            current_key = (
                (
                    len(current[0]),
                    tuple(token_to_id[token] for token in current[0]),
                )
                if current is not None
                else None
            )
            if current_key is None or candidate_key < current_key:
                best[next_pos] = (candidate, candidate_used_folded)
    if best[-1] is None:
        return None, text, False
    return best[-1][0], "", best[-1][1]


def tokenize_bpe_characters(
    word: str,
    token_set,
    word_boundary: str,
    allow_standalone_boundary: bool,
    folded_token_index=None,
):
    """Return a character spelling without requiring ▁ as a lexicon token."""
    if not word:
        return None, word_boundary, False
    pieces = []
    used_case_insensitive = False

    first_surface = word_boundary + word[0]
    first, used_folded = resolve_token(
        first_surface, token_set, folded_token_index
    )
    if first is not None:
        pieces.append(first)
        used_case_insensitive = used_case_insensitive or used_folded
    elif allow_standalone_boundary:
        boundary, boundary_folded = resolve_token(
            word_boundary, token_set, folded_token_index
        )
        character, character_folded = resolve_token(
            word[0], token_set, folded_token_index
        )
        if boundary is None or character is None:
            return None, first_surface, used_case_insensitive
        pieces.extend([boundary, character])
        used_case_insensitive = (
            used_case_insensitive or boundary_folded or character_folded
        )
    else:
        return None, first_surface, used_case_insensitive

    for character_surface in word[1:]:
        character, used_folded = resolve_token(
            character_surface, token_set, folded_token_index
        )
        if character is None:
            return None, character_surface, used_case_insensitive
        pieces.append(character)
        used_case_insensitive = used_case_insensitive or used_folded
    return pieces, "", used_case_insensitive


def load_sentencepiece_model(path: Path):
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise SystemExit(
            "--sentencepiece-model requires the Python sentencepiece package"
        ) from exc
    processor = spm.SentencePieceProcessor()
    if not processor.load(str(path)):
        raise SystemExit(f"failed to load SentencePiece model: {path}")
    return processor


def validate_sentencepiece_compatibility(processor, token_set, path: Path):
    ignored = {"<s>", "</s>"}
    pieces = {
        processor.id_to_piece(index)
        for index in range(processor.get_piece_size())
        if processor.id_to_piece(index) not in ignored
    }
    missing = sorted(pieces - token_set)
    coverage = 1.0 if not pieces else (len(pieces) - len(missing)) / len(pieces)
    if coverage < 0.99:
        raise SystemExit(
            f"{path}: SentencePiece/token coverage is only {coverage:.2%}; "
            "the tokenizer model does not match tokens.txt"
        )
    return {
        "piece_count": len(pieces),
        "covered_piece_count": len(pieces) - len(missing),
        "coverage": coverage,
        "missing_pieces": missing,
    }


def canonical_sentencepiece_spelling(
    processor,
    word: str,
    token_set,
    word_boundary: str,
    allow_standalone_boundary: bool,
):
    pieces = list(processor.encode(word, out_type=str))
    if not pieces:
        return None, ["<empty>"]
    missing = [piece for piece in pieces if piece not in token_set]
    if not allow_standalone_boundary and word_boundary in pieces:
        missing.append(word_boundary)
    if missing:
        return None, sorted(set(missing))
    return pieces, []


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
    sentencepiece_model=None,
    max_spellings: int = 1,
    add_character_fallback: bool = False,
):
    model_token_set = valid_model_tokens(token_to_id, model_vocab_size)
    token_set = {
        token
        for token in model_token_set
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

    processor = (
        load_sentencepiece_model(Path(sentencepiece_model))
        if sentencepiece_model
        else None
    )
    sentencepiece_compatibility = (
        validate_sentencepiece_compatibility(
            processor, model_token_set, Path(sentencepiece_model)
        )
        if processor is not None
        else None
    )

    one_piece = 0
    bpe = 0
    case_insensitive = 0
    canonical_entries = 0
    shortest_entries = 0
    character_fallback_entries = 0
    multi_spelling_words = 0
    entry_count = 0
    spelling_count_histogram = {}
    canonical_unavailable_count = 0
    canonical_unavailable = []
    rejected = []
    for word in words:
        if word == "<unk>" and word in token_to_id:
            fout.write(f"{word} {word}\n")
            one_piece += 1
            entry_count += 1
            spelling_count_histogram[1] = spelling_count_histogram.get(1, 0) + 1
            continue

        spellings = []
        spelling_keys = set()

        def add_spelling(pieces, source, used_folded=False):
            nonlocal one_piece, bpe, case_insensitive
            nonlocal canonical_entries, shortest_entries
            nonlocal character_fallback_entries
            if pieces is None or len(spellings) >= max_spellings:
                return
            key = tuple(pieces)
            if key in spelling_keys:
                return
            spelling_keys.add(key)
            spellings.append(pieces)
            if len(pieces) == 1:
                one_piece += 1
            else:
                bpe += 1
            if used_folded:
                case_insensitive += 1
            if source == "canonical":
                canonical_entries += 1
            elif source == "shortest":
                shortest_entries += 1
            elif source == "character":
                character_fallback_entries += 1

        if processor is not None:
            canonical, missing = canonical_sentencepiece_spelling(
                processor,
                word,
                token_set,
                word_boundary,
                allow_standalone_boundary,
            )
            if canonical is not None:
                add_spelling(canonical, "canonical")
            else:
                canonical_unavailable_count += 1
                if len(canonical_unavailable) < 50:
                    canonical_unavailable.append({"word": word, "missing": missing})

        shortest, unmatched, shortest_used_folded = tokenize_bpe_shortest(
            word,
            token_set,
            token_to_id,
            max_token_len,
            word_boundary,
            folded_token_index,
        )
        add_spelling(shortest, "shortest", shortest_used_folded)

        if add_character_fallback:
            characters, _, characters_used_folded = tokenize_bpe_characters(
                word,
                token_set,
                word_boundary,
                allow_standalone_boundary,
                folded_token_index,
            )
            add_spelling(characters, "character", characters_used_folded)

        if not spellings:
            rejected.append({"word": word, "unmatched": unmatched})
            continue
        if len(spellings) > 1:
            multi_spelling_words += 1
        spelling_count_histogram[len(spellings)] = (
            spelling_count_histogram.get(len(spellings), 0) + 1
        )
        for pieces in spellings:
            fout.write(word + " " + " ".join(pieces) + "\n")
            entry_count += 1
    return {
        "direct_entries": one_piece,
        "byte_fallback_entries": 0,
        "bpe_entries": bpe,
        "case_insensitive_entries": case_insensitive,
        "entry_count": entry_count,
        "canonical_entries": canonical_entries,
        "shortest_entries": shortest_entries,
        "character_fallback_entries": character_fallback_entries,
        "multi_spelling_words": multi_spelling_words,
        "spelling_count_histogram": spelling_count_histogram,
        "canonical_unavailable_count": canonical_unavailable_count,
        "canonical_unavailable_examples": canonical_unavailable,
        "sentencepiece_compatibility": sentencepiece_compatibility,
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
        "--sentencepiece-model",
        help=(
            "SentencePiece model used to generate the AM training labels. When "
            "set, its canonical spelling is emitted before fallback spellings."
        ),
    )
    parser.add_argument(
        "--bpe-max-spellings",
        type=int,
        default=3,
        help="Maximum unique BPE spellings emitted for each word.",
    )
    parser.add_argument(
        "--bpe-add-character-fallback",
        action="store_true",
        help=(
            "Also emit a character spelling when every required character token "
            "is available. The leading boundary is attached to the first letter "
            "unless --bpe-allow-standalone-boundary is enabled."
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
    if args.bpe_max_spellings < 1:
        parser.error("--bpe-max-spellings must be at least 1")
    if args.sentencepiece_model and args.tokenization != "bpe":
        parser.error("--sentencepiece-model requires --tokenization bpe")

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
                args.sentencepiece_model,
                args.bpe_max_spellings,
                args.bpe_add_character_fallback,
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
        "entry_count": stats.get("entry_count", len(words) - len(rejected)),
        "sentencepiece_model": args.sentencepiece_model,
        "bpe_max_spellings": args.bpe_max_spellings,
        "bpe_add_character_fallback": args.bpe_add_character_fallback,
        "canonical_entries": stats.get("canonical_entries", 0),
        "shortest_entries": stats.get("shortest_entries", 0),
        "character_fallback_entries": stats.get(
            "character_fallback_entries", 0
        ),
        "multi_spelling_words": stats.get("multi_spelling_words", 0),
        "spelling_count_histogram": stats.get("spelling_count_histogram", {}),
        "canonical_unavailable_examples": stats.get(
            "canonical_unavailable_examples", []
        ),
        "canonical_unavailable_count": stats.get(
            "canonical_unavailable_count", 0
        ),
        "sentencepiece_compatibility": stats.get("sentencepiece_compatibility"),
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
