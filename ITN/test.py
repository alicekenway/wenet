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
    "R N B": "R&B",
    "F M one hundred point nine": "FM100.9",
    "tune to two eight point five": "tune to 28.5",
    "tune to three oh point five": "tune to 30.5",
    "tune to the f m station nine five point five": "tune to the FM station 95.5",
    "tune to the a m station six twelve": "tune to the AM station 612",
    "play nine oh point five": "play 90.5",
    "M H three seventy": "MH370",
    "Call seven one one five four": "Call 71154",
    "Dial four one five five five five zero one three two": "Dial 415-555-0132",
    "Dial plus eight one three one two three four five six seven eight": "Dial +81312345678",
    "Dial five five five one two three four": "Dial 5551234",
    "ten percent": "10%",
    "a hundred percent": "100%",
    "A C": "A/C",
    "open the sunshade to one half": "open the sunshade to 1/2",
    "open the sunshade to a half": "open the sunshade to 1/2",
    "open the sunshade to half": "open the sunshade to 1/2",
    "one third": "1/3",
    "a third": "1/3",
    "one quarter": "1/4",
    "a quarter": "1/4",
    "open the sunshade to quarter": "open the sunshade to 1/4",
    "change media sound to two": "change media sound to 2",
    "change media sound to twenty eight point five": "change media sound to 28.5",
    "decrease media volume by sixteen point five levels": "decrease media volume by 16.5 levels",
    "lower voice volume by forty one levels": "lower voice volume by 41 levels",
    "turn down voice volume by twenty three point five steps": "turn down voice volume by 23.5 steps",
    "make phone forty nine levels louder": "make phone 49 levels louder",
    "set ambient light brightness to twenty one percent": "set ambient light brightness to 21%",
    "set volume to ninety nine percent": "set volume to 99%",
    "increase fan speed by one levels": "increase fan speed by 1 levels",
    "increase seat heating one level": "increase seat heating 1 level",
    "set media volume to level a hundred": "set media volume to level 100",
    "go back one hundred seconds": "go back 100 seconds",
    "go back a hundred seconds": "go back 100 seconds",
    "switch the three sixty camera to front view": "switch the 360 camera to front view",
    "set driving mode to one pedal": "set driving mode to one pedal",
    "play the next one": "play the next one",
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
