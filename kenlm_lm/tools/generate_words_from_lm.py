#!/usr/bin/env python3
"""Generate a words table from language-model text."""

import argparse
import collections
import json
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LM_ROOT = SCRIPT_DIR.parent
REPO_ROOT = Path(os.environ.get("ROOT", LM_ROOT.parents[1])).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Flashlight words.txt from word-segmented LM text."
    )
    parser.add_argument(
        "--lm-text",
        default=str(
            REPO_ROOT / "LM/wenet_lm/training/preprocess_data/wenetspeech_lm_char.txt"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(LM_ROOT / "data/words.txt"),
    )
    parser.add_argument(
        "--report",
        default=str(LM_ROOT / "reports/words_report.json"),
    )
    parser.add_argument("--unk", default="<unk>")
    args = parser.parse_args()

    counts = collections.Counter()
    input_path = Path(args.lm_text)
    line_count = 0
    token_count = 0
    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line_count += 1
            for token in line.strip().split():
                token_count += 1
                if token not in {"<s>", "</s>", args.unk}:
                    counts[token] += 1

    words = [args.unk]
    words.extend(
        token
        for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for idx, word in enumerate(words):
            fout.write(f"{word} {idx}\n")

    report = {
        "lm_text": str(input_path),
        "line_count": line_count,
        "token_count": token_count,
        "word_count": len(words),
        "unk": args.unk,
        "output": str(output_path),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
