#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WETEXT = ROOT / "SDK" / "0.0.9" / "third_party" / "WeTextProcessing"
sys.path.insert(0, str(WETEXT))
sys.path.insert(0, str(ROOT))

from ITN.english.inverse_normalizer import InverseNormalizer


CASES = {
    "F M one hundred point nine": "FM100.9",
    "M H three seventy": "MH370",
    "Call seven one one five four": "Call 71154",
    "one hundred": "100",
    "one hundred and one": "101",
}


def main():
    parser = argparse.ArgumentParser(description="Test or apply English ITN")
    parser.add_argument("--language", default="en", choices=["en"])
    parser.add_argument("--text")
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--input-text", type=Path)
    parser.add_argument("--output-text", type=Path)
    parser.add_argument("--fst-dir", type=Path, default=ROOT / "ITN" / "english" / "export")
    args = parser.parse_args()
    normalizer = InverseNormalizer(args.fst_dir)
    if args.text is not None:
        print(normalizer.normalize(args.text))
        return
    if args.input_text:
        if not args.output_text:
            parser.error("--output-text is required with --input-text")
        with args.input_text.open(encoding="utf-8") as source, args.output_text.open("w", encoding="utf-8") as target:
            for line in source:
                target.write(normalizer.normalize(line.rstrip("\n")) + "\n")
        return
    if args.input_jsonl:
        if not args.output_jsonl:
            parser.error("--output-jsonl is required with --input-jsonl")
        with args.input_jsonl.open(encoding="utf-8") as source, args.output_jsonl.open("w", encoding="utf-8") as target:
            for line in source:
                item = json.loads(line)
                item["raw_text"] = item["text"]
                item["text"] = normalizer.normalize(item["text"])
                target.write(json.dumps(item, ensure_ascii=False) + "\n")
        return
    for source, expected in CASES.items():
        actual = normalizer.normalize(source)
        if actual != expected:
            raise AssertionError(f"{source!r}: expected {expected!r}, got {actual!r}")
        print(f"PASS\t{source}\t{actual}")


if __name__ == "__main__":
    main()
