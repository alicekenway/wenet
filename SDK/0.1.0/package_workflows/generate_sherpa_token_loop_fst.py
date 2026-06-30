#!/usr/bin/env python3
import argparse
import pathlib
import subprocess


def read_tokens(path):
    tokens = {}
    blank_id = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            token, sid = line.rsplit(maxsplit=1)
            tid = int(sid)
            tokens[tid] = token
            if token in ("<blk>", "<blank>"):
                blank_id = tid
    return tokens, blank_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fstcompile", default="fstcompile")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens, blank_id = read_tokens(args.tokens)
    if not tokens:
        raise RuntimeError("tokens.txt is empty")
    dense_size = max(tokens) + 1
    if len(tokens) != dense_size:
        raise RuntimeError("tokens.txt ids must be dense from 0")
    vocab_size = dense_size
    for tid in range(dense_size):
        if tokens[tid].startswith("#"):
            vocab_size = tid
            break

    fst_txt = out_dir / "TLG.txt"
    with open(fst_txt, "w", encoding="utf-8") as f:
        for tid in range(vocab_size):
            ilabel = tid + 1
            olabel = 0 if tid == blank_id else tid
            f.write(f"0 0 {ilabel} {olabel} 0.0\n")
        f.write("0 0.0\n")

    with open(out_dir / "words.txt", "w", encoding="utf-8") as f:
        f.write("<eps> 0\n")
        for tid in range(vocab_size):
            if tid == blank_id:
                continue
            f.write(f"{tokens[tid]} {tid}\n")

    with open(out_dir / "units.txt", "w", encoding="utf-8") as f:
        f.write("<eps> 0\n")
        for tid in range(vocab_size):
            f.write(f"{tokens[tid]} {tid + 1}\n")

    subprocess.run(
        [args.fstcompile, str(fst_txt), str(out_dir / "TLG.fst")],
        check=True,
    )


if __name__ == "__main__":
    main()
