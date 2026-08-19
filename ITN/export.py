#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WETEXT = ROOT / "SDK" / "0.0.9" / "third_party" / "WeTextProcessing"
sys.path.insert(0, str(WETEXT))
sys.path.insert(0, str(ROOT))

from ITN.english.inverse_normalizer import InverseNormalizer


def main():
    parser = argparse.ArgumentParser(description="Export project English ITN FSTs")
    parser.add_argument("--language", default="en", choices=["en"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ITN" / "english" / "export")
    args = parser.parse_args()
    normalizer = InverseNormalizer(args.output_dir, overwrite_cache=True)
    print(args.output_dir / "en_itn_tagger.fst")
    print(args.output_dir / "en_itn_verbalizer.fst")
    # Force a tiny composition so export failures surface here.
    assert normalizer.normalize("f m one hundred point nine") == "FM100.9"


if __name__ == "__main__":
    main()
