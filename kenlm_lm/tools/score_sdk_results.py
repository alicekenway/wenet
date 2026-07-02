#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LM_ROOT = SCRIPT_DIR.parent
REPO_ROOT = Path(os.environ.get("ROOT", LM_ROOT.parents[1])).resolve()


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


def read_refs(metadata: Path, limit: int):
    refs = []
    for line_no, line in enumerate(metadata.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        ref = item.get("text") or item.get("sentence") or item.get("transcript")
        if ref is None:
            raise SystemExit(f"{metadata}:{line_no}: no reference text field")
        refs.append(normalize(str(ref)))
        if limit > 0 and len(refs) >= limit:
            break
    return refs


def read_results(result_path: Path):
    hyps = {}
    for line_no, line in enumerate(result_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.rstrip("\n").split(maxsplit=1)
        if not parts:
            continue
        try:
            utt = int(parts[0])
        except ValueError as exc:
            raise SystemExit(f"{result_path}:{line_no}: bad utterance id") from exc
        hyps[utt] = normalize(parts[1] if len(parts) > 1 else "")
    return hyps


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score public SDK batch results against sampled metadata refs."
    )
    parser.add_argument(
        "--metadata",
        default=str(REPO_ROOT / "test/0.0.3/metadata.sample100.jsonl"),
    )
    parser.add_argument(
        "--result",
        default=str(REPO_ROOT / "test/0.0.3/sdk_batch_flashlight.txt"),
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "test/0.0.3/sdk_batch_flashlight_eval.tsv"),
    )
    parser.add_argument(
        "--summary",
        default=str(REPO_ROOT / "test/0.0.3/sdk_batch_flashlight_summary.txt"),
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    metadata = Path(args.metadata)
    result_path = Path(args.result)
    output = Path(args.output)
    summary = Path(args.summary)
    refs = read_refs(metadata, args.limit)
    hyps = read_results(result_path)

    total_ref_chars = 0
    total_edits = 0
    empty_hyp = 0
    missing = 0
    rows = []
    for utt, ref in enumerate(refs):
        hyp = hyps.get(utt, "")
        if utt not in hyps:
            missing += 1
        edits = edit_distance(ref, hyp)
        ref_chars = len(ref)
        total_ref_chars += ref_chars
        total_edits += edits
        empty_hyp += int(hyp == "")
        rows.append(
            {
                "utt": utt,
                "ref": ref,
                "sdk_lm": hyp,
                "ref_chars": ref_chars,
                "sdk_lm_edits": edits,
                "sdk_lm_cer": edits / ref_chars if ref_chars else 0.0,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["utt", "ref", "sdk_lm", "ref_chars", "sdk_lm_edits", "sdk_lm_cer"]
    with output.open("w", encoding="utf-8") as fout:
        fout.write("\t".join(fields) + "\n")
        for row in rows:
            fout.write("\t".join(str(row[field]) for field in fields) + "\n")

    cer = total_edits / total_ref_chars if total_ref_chars else 0.0
    summary_text = "\n".join(
        [
            f"utterances={len(rows)}",
            f"ref_chars={total_ref_chars}",
            f"sdk_lm_edits={total_edits}",
            f"sdk_lm_cer={cer:.6f}",
            f"empty_sdk_lm={empty_hyp}",
            f"missing_results={missing}",
            f"eval_tsv={output}",
        ]
    )
    summary.write_text(summary_text + "\n", encoding="utf-8")
    print(summary_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
