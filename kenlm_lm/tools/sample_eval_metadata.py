#!/usr/bin/env python3
import argparse
import json
import os
import random
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LM_ROOT = SCRIPT_DIR.parent
REPO_ROOT = Path(os.environ.get("ROOT", LM_ROOT.parents[1])).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deterministically sample metadata JSONL for SDK 0.0.3 eval."
    )
    parser.add_argument(
        "--input",
        default=str(
            REPO_ROOT
            / "data/hf_wenetspeech_test_net/wenetspeech_test_net_sample_2000/metadata.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "test/0.0.3/metadata.sample100.jsonl"),
    )
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20250625)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    lines = input_path.read_text(encoding="utf-8").splitlines()
    records = []
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{input_path}:{line_no}: invalid JSON: {exc}") from exc
        records.append(line)

    count = min(args.num, len(records))
    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(records)), count))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(records[index] + "\n" for index in indices), encoding="utf-8"
    )
    print(f"input_records={len(records)}")
    print(f"sampled_records={count}")
    print(f"seed={args.seed}")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
