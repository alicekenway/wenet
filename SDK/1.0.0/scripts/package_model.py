#!/usr/bin/env python3
"""Write manifest/config files for a WeNet Lite SDK model package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--feature_dim", type=int, default=80)
    parser.add_argument("--frame_length_ms", type=int, default=25)
    parser.add_argument("--frame_shift_ms", type=int, default=10)
    parser.add_argument("--subsampling_rate", type=int, default=4)
    parser.add_argument("--waveform_scale", type=float, default=32768.0)
    parser.add_argument("--encoder", default="encoder.onnx")
    parser.add_argument("--ctc", default="")
    parser.add_argument("--tokens", default="tokens.txt")
    parser.add_argument("--words", default="words.txt")
    parser.add_argument("--graph", default="TLG.fst")
    parser.add_argument("--decoder", default="ctc_wfst")
    parser.add_argument("--language_type", default="indo_european")
    parser.add_argument("--blank_skip_thresh", type=float, default=0.98)
    parser.add_argument("--beam", type=float, default=16.0)
    parser.add_argument("--lattice_beam", type=float, default=10.0)
    parser.add_argument("--max_active", type=int, default=7000)
    parser.add_argument("--min_active", type=int, default=200)
    parser.add_argument("--acoustic_scale", type=float, default=1.0)
    parser.add_argument("--lm_scale", type=float, default=1.0)
    parser.add_argument("--length_penalty", type=float, default=0.0)
    parser.add_argument("--nbest", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--num_left_chunks", type=int, default=4)
    parser.add_argument("--endpoint_silence_ms", type=int, default=800)
    parser.add_argument("--max_segment_ms", type=int, default=20000)
    parser.add_argument("--write_checksum", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "sdk_model_version": 1,
        "model_type": "wenet_ctc_streaming_onnx",
        "sample_rate": args.sample_rate,
        "feature_dim": args.feature_dim,
        "frame_length_ms": args.frame_length_ms,
        "frame_shift_ms": args.frame_shift_ms,
        "subsampling_rate": args.subsampling_rate,
        "waveform_scale": args.waveform_scale,
        "onnx": {
            "encoder": args.encoder,
            "ctc": args.ctc,
            "output_type": "log_prob",
            "input_names": {
                "chunk": "chunk",
                "offset": "offset",
                "att_cache": "att_cache",
                "cnn_cache": "cnn_cache",
            },
            "output_names": {
                "encoder_out": "output",
                "att_cache": "r_att_cache",
                "cnn_cache": "r_cnn_cache",
                "log_probs": "log_probs",
            },
        },
        "vocab": {
            "tokens": args.tokens,
            "words": args.words,
            "blank_id": 0,
            "sos_id": -1,
            "eos_id": -1,
        },
        "decoder": {
            "type": args.decoder,
            "graph": args.graph,
            "beam": args.beam,
            "lattice_beam": args.lattice_beam,
            "max_active": args.max_active,
            "min_active": args.min_active,
            "acoustic_scale": args.acoustic_scale,
            "lm_scale": args.lm_scale,
            "length_penalty": args.length_penalty,
            "blank_skip_thresh": args.blank_skip_thresh,
            "nbest": args.nbest,
        },
    }
    (model_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    (model_dir / "config.yaml").write_text(
        "\n".join(
            [
                "streaming:",
                f"  chunk_size: {args.chunk_size}",
                f"  num_left_chunks: {args.num_left_chunks}",
                f"  endpoint_silence_ms: {args.endpoint_silence_ms}",
                f"  max_segment_ms: {args.max_segment_ms}",
                "",
                "runtime:",
                "  intra_op_num_threads: 1",
                "  inter_op_num_threads: 1",
                "  enable_profiling: false",
                "",
                "postprocess:",
                "  lowercase: true",
                "  remove_bpe_marker: true",
                f"  language_type: {args.language_type}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    if args.write_checksum:
        rels = ["manifest.json", "config.yaml", args.encoder, args.tokens, args.words, args.graph]
        if args.ctc:
            rels.append(args.ctc)
        lines = []
        for rel in rels:
            path = model_dir / rel
            if path.exists():
                lines.append(f"{sha256(path)}  {rel}")
        (model_dir / "checksum.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote package metadata under {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
