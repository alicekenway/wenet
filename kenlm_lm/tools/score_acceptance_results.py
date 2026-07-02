#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LM_ROOT = SCRIPT_DIR.parent
REPO_ROOT = Path(os.environ.get("ROOT", LM_ROOT.parents[1])).resolve()
DEFAULT_WAV_ROOT = (
    REPO_ROOT / "data/hf_wenetspeech_test_net/wenetspeech_test_net_sample_2000"
)


def normalize(text: str) -> str:
    return "".join(text.split())


def edit_distance(ref: str, hyp: str) -> int:
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, hc in enumerate(hyp, 1):
            sub = prev[j - 1] + (0 if rc == hc else 1)
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            cur[j] = min(sub, ins, delete)
        prev = cur
    return prev[-1]


def extract_log_value(log_text: str, prefix: str) -> str:
    for line in log_text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def read_refs(metadata: Path, wav_root: Path):
    refs = []
    wavs = []
    base = wav_root
    for line_no, line in enumerate(metadata.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        ref = item.get("text") or item.get("sentence") or item.get("transcript")
        if ref is None:
            raise SystemExit(f"{metadata}:{line_no}: no reference text field")
        wav = (
            item.get("audio_filepath")
            or item.get("audiofile_path")
            or item.get("wav")
            or item.get("path")
        )
        if wav is not None:
            wav_path = Path(wav)
            if not wav_path.is_absolute():
                wav_path = base / wav_path
            wavs.append(str(wav_path))
        else:
            wavs.append("")
        refs.append(ref)
    return refs, wavs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score SDK 0.0.3 acceptance logs against metadata refs."
    )
    parser.add_argument(
        "--metadata",
        default=str(REPO_ROOT / "test/0.0.3/metadata.sample100.jsonl"),
    )
    parser.add_argument(
        "--wav-root",
        default=str(DEFAULT_WAV_ROOT),
        help="Directory used to resolve relative wav paths in metadata.",
    )
    parser.add_argument(
        "--log-dir",
        default=str(REPO_ROOT / "test/0.0.3/acceptance"),
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "test/0.0.3/acceptance/eval.tsv"),
    )
    parser.add_argument(
        "--summary",
        default=str(REPO_ROOT / "test/0.0.3/acceptance/summary.txt"),
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    metadata = Path(args.metadata)
    log_dir = Path(args.log_dir)
    output = Path(args.output)
    summary = Path(args.summary)
    refs, wavs = read_refs(metadata, Path(args.wav_root))
    if args.limit > 0:
        refs = refs[: args.limit]
        wavs = wavs[: args.limit]

    total_ref_chars = 0
    total_greedy_edits = 0
    total_lm_edits = 0
    empty_greedy = 0
    empty_lm = 0
    lm_better = 0
    lm_worse = 0
    lm_equal = 0
    rows = []

    for idx, ref in enumerate(refs):
        log_path = log_dir / f"{idx}.log"
        if not log_path.exists():
            raise SystemExit(f"missing decode log: {log_path}")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        greedy = extract_log_value(log_text, "greedy text:")
        lm = extract_log_value(log_text, "hyp 0 mapped text:")
        raw = extract_log_value(log_text, "hyp 0 raw words:")
        search_rtf = extract_log_value(log_text, "search RTF:")
        total_rtf = extract_log_value(log_text, "total RTF:")

        ref_norm = normalize(ref)
        greedy_norm = normalize(greedy)
        lm_norm = normalize(lm)
        greedy_edits = edit_distance(ref_norm, greedy_norm)
        lm_edits = edit_distance(ref_norm, lm_norm)
        ref_chars = len(ref_norm)

        total_ref_chars += ref_chars
        total_greedy_edits += greedy_edits
        total_lm_edits += lm_edits
        empty_greedy += int(greedy_norm == "")
        empty_lm += int(lm_norm == "")
        if lm_edits < greedy_edits:
            lm_better += 1
            compare = "lm_better"
        elif lm_edits > greedy_edits:
            lm_worse += 1
            compare = "lm_worse"
        else:
            lm_equal += 1
            compare = "equal"

        rows.append(
            {
                "utt": idx,
                "wav": wavs[idx],
                "ref": ref_norm,
                "greedy": greedy_norm,
                "lm": lm_norm,
                "raw_lm_words": raw,
                "ref_chars": ref_chars,
                "greedy_edits": greedy_edits,
                "lm_edits": lm_edits,
                "greedy_cer": greedy_edits / ref_chars if ref_chars else 0.0,
                "lm_cer": lm_edits / ref_chars if ref_chars else 0.0,
                "compare": compare,
                "search_rtf": search_rtf,
                "total_rtf": total_rtf,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "utt",
        "wav",
        "ref",
        "greedy",
        "lm",
        "raw_lm_words",
        "ref_chars",
        "greedy_edits",
        "lm_edits",
        "greedy_cer",
        "lm_cer",
        "compare",
        "search_rtf",
        "total_rtf",
    ]
    with output.open("w", encoding="utf-8") as fout:
        fout.write("\t".join(fields) + "\n")
        for row in rows:
            fout.write("\t".join(str(row[field]) for field in fields) + "\n")

    greedy_cer = total_greedy_edits / total_ref_chars if total_ref_chars else 0.0
    lm_cer = total_lm_edits / total_ref_chars if total_ref_chars else 0.0
    summary_text = "\n".join(
        [
            f"utterances={len(rows)}",
            f"ref_chars={total_ref_chars}",
            f"greedy_edits={total_greedy_edits}",
            f"lm_edits={total_lm_edits}",
            f"greedy_cer={greedy_cer:.6f}",
            f"lm_cer={lm_cer:.6f}",
            f"empty_greedy={empty_greedy}",
            f"empty_lm={empty_lm}",
            f"lm_better={lm_better}",
            f"lm_worse={lm_worse}",
            f"lm_equal={lm_equal}",
            f"eval_tsv={output}",
        ]
    )
    summary.write_text(summary_text + "\n", encoding="utf-8")
    print(summary_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
