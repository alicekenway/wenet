#!/usr/bin/env python3
import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


def has_whitespace(text: str) -> bool:
    return any(ch.isspace() for ch in text)


def split_utf8_chars_no_whitespace(text: str) -> list[str]:
    return [ch for ch in text if not ch.isspace()]


def normalize_for_error_rate(text: str) -> str:
    return text.casefold()


def resolve_metric(metric: str, ref: str, hyp: str) -> str:
    if metric != "auto":
        return metric
    return "wer" if has_whitespace(ref) or has_whitespace(hyp) else "cer"


def split_units(ref: str, hyp: str, metric: str) -> tuple[list[str], list[str]]:
    if metric == "wer":
        return ref.split(), hyp.split()
    return split_utf8_chars_no_whitespace(ref), split_utf8_chars_no_whitespace(hyp)


@dataclass(frozen=True)
class EditStats:
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0

    @property
    def edits(self) -> int:
        return self.substitutions + self.insertions + self.deletions


@dataclass(frozen=True)
class RegexRule:
    pattern: re.Pattern
    replacement: str
    source: str


def parse_regex_rule(line: str, source: str) -> RegexRule:
    if "\t" in line:
        pattern, replacement = line.split("\t", 1)
    elif " -> " in line:
        pattern, replacement = line.split(" -> ", 1)
    else:
        pattern, replacement = line, ""
    pattern = pattern.strip()
    if not pattern:
        raise ValueError(f"{source}: empty regex pattern")
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"{source}: invalid regex {pattern!r}: {exc}") from exc
    return RegexRule(compiled, replacement, source)


def load_regex_rules(path: Path | None) -> list[RegexRule]:
    if path is None:
        return []
    rules = []
    with path.open("r", encoding="utf-8") as fin:
        for line_no, raw_line in enumerate(fin, 1):
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rules.append(parse_regex_rule(line, f"{path}:{line_no}"))
    return rules


def apply_regex_rules(text: str, rules: list[RegexRule]) -> str:
    for rule in rules:
        text = rule.pattern.sub(rule.replacement, text)
    return " ".join(text.split())


def edit_stats(ref: list[str], hyp: list[str]) -> EditStats:
    # Tie-break order prefers matching/substitution, then deletion, then insertion.
    # This gives stable, easy-to-read ASR-style S/D/I counts when alignments tie.
    dp: list[list[tuple[int, EditStats]]] = [
        [(0, EditStats()) for _ in range(len(hyp) + 1)]
        for _ in range(len(ref) + 1)
    ]
    for i in range(1, len(ref) + 1):
        prev_cost, prev_stats = dp[i - 1][0]
        dp[i][0] = (
            prev_cost + 1,
            EditStats(
                prev_stats.substitutions,
                prev_stats.insertions,
                prev_stats.deletions + 1,
            ),
        )
    for j in range(1, len(hyp) + 1):
        prev_cost, prev_stats = dp[0][j - 1]
        dp[0][j] = (
            prev_cost + 1,
            EditStats(
                prev_stats.substitutions,
                prev_stats.insertions + 1,
                prev_stats.deletions,
            ),
        )

    for i, ref_unit in enumerate(ref, 1):
        for j, hyp_unit in enumerate(hyp, 1):
            diag_cost, diag_stats = dp[i - 1][j - 1]
            if ref_unit == hyp_unit:
                diag = (diag_cost, diag_stats)
            else:
                diag = (
                    diag_cost + 1,
                    EditStats(
                        diag_stats.substitutions + 1,
                        diag_stats.insertions,
                        diag_stats.deletions,
                    ),
                )

            del_cost, del_stats = dp[i - 1][j]
            delete = (
                del_cost + 1,
                EditStats(
                    del_stats.substitutions,
                    del_stats.insertions,
                    del_stats.deletions + 1,
                ),
            )

            ins_cost, ins_stats = dp[i][j - 1]
            insert = (
                ins_cost + 1,
                EditStats(
                    ins_stats.substitutions,
                    ins_stats.insertions + 1,
                    ins_stats.deletions,
                ),
            )

            dp[i][j] = min(
                (diag, delete, insert),
                key=lambda item: (
                    item[0],
                    item[1].substitutions + item[1].deletions + item[1].insertions,
                ),
            )
    return dp[-1][-1][1]


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
        "--metric",
        choices=["auto", "wer", "cer"],
        default="auto",
        help="Error metric to compute in the summary. auto uses WER when either side has whitespace, otherwise CER.",
    )
    parser.add_argument(
        "--detail_json",
        default="",
        help="Optional JSONL output with added metric/error_rate/unit_count/edit/SID fields.",
    )
    parser.add_argument(
        "--ref_regex_rules",
        default="",
        help=(
            "Optional regex replacement rules applied to reference text before "
            "WER/CER calculation. Each non-comment line is PATTERN<TAB>REPLACEMENT; "
            "a one-column line means replace with empty string."
        ),
    )
    args = parser.parse_args()

    input_json = Path(args.input_json)
    summary = Path(args.summary)
    detail_json = Path(args.detail_json) if args.detail_json else None
    ref_regex_rules = load_regex_rules(
        Path(args.ref_regex_rules) if args.ref_regex_rules else None
    )

    rows = []
    total_ref_units = 0
    total_edits = 0
    total_substitutions = 0
    total_insertions = 0
    total_deletions = 0
    total_row_rtf = 0.0
    total_audio_sec = 0.0
    total_decode_sec = 0.0
    failed = 0
    wrong = 0
    decode_mode = ""
    row_metrics = []

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
            scored_ref = apply_regex_rules(ref, ref_regex_rules)
            hyp = str(row.get("hyp", ""))
            norm_ref = normalize_for_error_rate(scored_ref)
            norm_hyp = normalize_for_error_rate(hyp)
            row_metric = resolve_metric(args.metric, norm_ref, norm_hyp)
            ref_units, hyp_units = split_units(norm_ref, norm_hyp, row_metric)
            stats = edit_stats(ref_units, hyp_units)
            edits = stats.edits
            ref_count = len(ref_units)
            row_error_rate = (
                edits / ref_count if ref_count else (0.0 if not hyp_units else 1.0)
            )
            row["metric"] = row_metric
            row["error_rate"] = row_error_rate
            row["unit_count"] = ref_count
            row["edits"] = edits
            row["substitutions"] = stats.substitutions
            row["insertions"] = stats.insertions
            row["deletions"] = stats.deletions
            if scored_ref != ref:
                row["scored_ref"] = scored_ref
            row[row_metric] = row_error_rate

            total_ref_units += ref_count
            total_edits += edits
            total_substitutions += stats.substitutions
            total_insertions += stats.insertions
            total_deletions += stats.deletions
            total_row_rtf += finite_number(row.get("rtf"))
            total_audio_sec += finite_number(row.get("audio_sec"))
            total_decode_sec += finite_number(row.get("decode_sec"))
            failed += 1 if row.get("error") else 0
            wrong += 1 if edits > 0 or row.get("error") else 0
            if not decode_mode and row.get("decode_mode"):
                decode_mode = str(row["decode_mode"])
            row_metrics.append(row_metric)
            rows.append(row)

    summary.parent.mkdir(parents=True, exist_ok=True)
    if detail_json is not None:
        detail_json.parent.mkdir(parents=True, exist_ok=True)
        with detail_json.open("w", encoding="utf-8") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    count = len(rows)
    metric_name = args.metric
    if metric_name == "auto":
        unique_metrics = sorted(set(row_metrics))
        metric_name = unique_metrics[0] if len(unique_metrics) == 1 else "auto"
    metric_field = metric_name if metric_name in ("wer", "cer") else "error_rate"
    error_rate = total_edits / total_ref_units if total_ref_units else 0.0
    substitution_rate = (
        total_substitutions / total_ref_units if total_ref_units else 0.0
    )
    insertion_rate = total_insertions / total_ref_units if total_ref_units else 0.0
    deletion_rate = total_deletions / total_ref_units if total_ref_units else 0.0
    ser = wrong / count if count else 0.0
    rtf = (
        total_decode_sec / total_audio_sec
        if total_audio_sec
        else (total_row_rtf / count if count else 0.0)
    )

    with summary.open("w", encoding="utf-8") as out:
        out.write(f"decode_mode: {decode_mode}\n")
        out.write(f"metric: {metric_name}\n")
        if args.ref_regex_rules:
            out.write(f"ref_regex_rules: {args.ref_regex_rules}\n")
        out.write(f"sentence_count: {count}\n")
        out.write(f"failed_count: {failed}\n")
        out.write(f"ser: {ser:.6f}\n")
        out.write(f"{metric_field}: {error_rate:.6f}\n")
        out.write(f"substitutions: {total_substitutions}\n")
        out.write(f"insertions: {total_insertions}\n")
        out.write(f"deletions: {total_deletions}\n")
        out.write(f"substitution_rate: {substitution_rate:.6f}\n")
        out.write(f"insertion_rate: {insertion_rate:.6f}\n")
        out.write(f"deletion_rate: {deletion_rate:.6f}\n")
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
            if row.get("scored_ref") is not None:
                out.write(f"  scored_ref: {row['scored_ref']}\n")
            out.write(f"  hyp: {row.get('hyp', '')}\n")
            out.write(f"  metric: {row.get('metric', metric_name)}\n")
            out.write(f"  error_rate: {row.get('error_rate', 0.0):.6f}\n")
            out.write(f"  substitutions: {row.get('substitutions', 0)}\n")
            out.write(f"  insertions: {row.get('insertions', 0)}\n")
            out.write(f"  deletions: {row.get('deletions', 0)}\n")
            if row.get("error"):
                out.write(f"  error: {row['error']}\n")

    print(f"sentence_count {count}")
    print(f"failed_count {failed}")
    if args.ref_regex_rules:
        print(f"ref_regex_rules {args.ref_regex_rules}")
    print(f"ser {ser:.6f}")
    print(f"metric {metric_name}")
    print(f"{metric_field} {error_rate:.6f}")
    print(f"substitutions {total_substitutions}")
    print(f"insertions {total_insertions}")
    print(f"deletions {total_deletions}")
    print(f"substitution_rate {substitution_rate:.6f}")
    print(f"insertion_rate {insertion_rate:.6f}")
    print(f"deletion_rate {deletion_rate:.6f}")
    print(f"rtf {rtf:.6f}")
    print(f"summary {summary}")
    if detail_json is not None:
        print(f"detail_json {detail_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
