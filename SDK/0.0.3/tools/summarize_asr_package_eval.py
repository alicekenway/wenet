#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def has_whitespace(text: str) -> bool:
    return any(ch.isspace() for ch in text)


def split_utf8_chars_no_whitespace(text: str) -> list[str]:
    return [ch for ch in text if not ch.isspace()]


def split_units(ref: str, hyp: str) -> tuple[list[str], list[str]]:
    if has_whitespace(ref) or has_whitespace(hyp):
        return ref.split(), hyp.split()
    return split_utf8_chars_no_whitespace(ref), split_utf8_chars_no_whitespace(hyp)


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, ref_unit in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, hyp_unit in enumerate(hyp, 1):
            sub = prev[j - 1] + (0 if ref_unit == hyp_unit else 1)
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            cur[j] = min(sub, ins, delete)
        prev = cur
    return prev[-1]


def first_text_value(row: dict, keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def wav_name(row: dict) -> str:
    wav = first_text_value(
        row, ["audiofile_path", "audio_filepath", "wav", "path", "file_name"]
    )
    return Path(wav).name if wav else ""


def finite_number(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize asr_package_eval JSONL output."
    )
    parser.add_argument("--input_json", "--input", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--text_key", default="")
    parser.add_argument(
        "--detail_json",
        default="",
        help="Optional JSONL output with added wer/ref_units/edit fields.",
    )
    args = parser.parse_args()

    input_json = Path(args.input_json)
    summary = Path(args.summary)
    detail_json = Path(args.detail_json) if args.detail_json else None

    rows = []
    total_ref_units = 0
    total_edits = 0
    total_audio_sec = 0.0
    total_decode_sec = 0.0
    failed = 0
    wrong = 0
    decode_mode = ""

    with input_json.open("r", encoding="utf-8") as fin:
      for line_no, line in enumerate(fin, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        ref = (
            str(row.get(args.text_key, ""))
            if args.text_key
            else first_text_value(row, ["text", "sentence", "transcript"])
        )
        hyp = str(row.get("hyp", ""))
        ref_units, hyp_units = split_units(ref, hyp)
        edits = edit_distance(ref_units, hyp_units)
        ref_count = len(ref_units)
        row_wer = edits / ref_count if ref_count else (0.0 if not hyp_units else 1.0)
        row["wer"] = row_wer
        row["ref_units"] = ref_count
        row["edits"] = edits

        total_ref_units += ref_count
        total_edits += edits
        total_audio_sec += finite_number(row.get("audio_sec"))
        total_decode_sec += finite_number(row.get("decode_sec"))
        failed += 1 if row.get("error") else 0
        wrong += 1 if edits > 0 or row.get("error") else 0
        if not decode_mode and row.get("decode_mode"):
            decode_mode = str(row["decode_mode"])
        rows.append(row)

    summary.parent.mkdir(parents=True, exist_ok=True)
    if detail_json is not None:
        detail_json.parent.mkdir(parents=True, exist_ok=True)
        with detail_json.open("w", encoding="utf-8") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    count = len(rows)
    wer = total_edits / total_ref_units if total_ref_units else 0.0
    ser = wrong / count if count else 0.0
    rtf = total_decode_sec / total_audio_sec if total_audio_sec else 0.0

    with summary.open("w", encoding="utf-8") as out:
        out.write(f"decode_mode: {decode_mode}\n")
        out.write(f"sentence_count: {count}\n")
        out.write(f"failed_count: {failed}\n")
        out.write(f"ser: {ser:.6f}\n")
        out.write(f"wer: {wer:.6f}\n")
        out.write(f"rtf: {rtf:.6f}\n")
        out.write(f"audio_sec: {total_audio_sec:.6f}\n")
        out.write(f"decode_sec: {total_decode_sec:.6f}\n")
        out.write(f"wrong_sentence_count: {wrong}\n\n")
        out.write("wrong_sentences:\n")
        for row in rows:
            if row.get("edits", 0) == 0 and not row.get("error"):
                continue
            ref = (
                str(row.get(args.text_key, ""))
                if args.text_key
                else first_text_value(row, ["text", "sentence", "transcript"])
            )
            out.write(f"- wav: {wav_name(row)}\n")
            out.write(f"  ref: {ref}\n")
            out.write(f"  hyp: {row.get('hyp', '')}\n")
            out.write(f"  wer: {row.get('wer', 0.0):.6f}\n")
            if row.get("error"):
                out.write(f"  error: {row['error']}\n")

    print(f"sentence_count {count}")
    print(f"failed_count {failed}")
    print(f"ser {ser:.6f}")
    print(f"wer {wer:.6f}")
    print(f"rtf {rtf:.6f}")
    print(f"summary {summary}")
    if detail_json is not None:
        print(f"detail_json {detail_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
