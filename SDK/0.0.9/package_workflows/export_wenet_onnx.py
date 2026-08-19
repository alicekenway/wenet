#!/usr/bin/env python3
"""Export WeNet ONNX models with config-driven CMVN normalization."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required. Run this script with the WeNet Python environment."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a WeNet export config, preserving model-driven CMVN, "
            "then run wenet.bin.export_onnx_cpu."
        )
    )
    parser.add_argument("--model-dir", required=True, help="Model directory.")
    parser.add_argument("--output-dir", required=True, help="ONNX output directory.")
    parser.add_argument(
        "--wenet-dir",
        required=True,
        help=(
            "WeNet source/archive directory. This can be the source root "
            "containing wenet/bin/export_onnx_cpu.py, its parent, or the "
            "inner wenet package directory."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--num-decoding-left-chunks", type=int, default=16)
    parser.add_argument(
        "--reverse-weight",
        type=float,
        default=None,
        help="Override reverse_weight. Defaults to model_conf.reverse_weight.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run the WeNet exporter.",
    )
    parser.add_argument(
        "--skip-cmvn-verify",
        action="store_true",
        help="Do not verify that expected CMVN tensors exist in encoder.onnx.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only write train.export.yaml and print the export command.",
    )
    return parser.parse_args()


def abs_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def resolve_wenet_root(path: Path) -> Path:
    candidates = [path, path / "wenet", path.parent]
    for candidate in candidates:
        if (candidate / "wenet" / "bin" / "export_onnx_cpu.py").is_file():
            return candidate.resolve()
    raise SystemExit(
        f"Could not find wenet/bin/export_onnx_cpu.py from --wenet-dir {path}"
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fin:
        data = yaml.load(fin, Loader=yaml.FullLoader)
    if not isinstance(data, dict):
        raise SystemExit(f"Expected YAML mapping in {path}")
    return data


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        yaml.safe_dump(data, fout, allow_unicode=True, sort_keys=False)


def infer_is_json_cmvn(cmvn_path: Path) -> bool:
    with cmvn_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            stripped = line.lstrip()
            if stripped:
                return stripped.startswith("{")
    return False


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "on"}:
            return True
        if lowered in {"false", "no", "0", "off"}:
            return False
    return bool(value)


def resolve_cmvn_path(raw_path: str, model_dir: Path, config_dir: Path) -> Path:
    raw = Path(raw_path).expanduser()
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = [
            config_dir / raw,
            model_dir / raw,
            model_dir / raw.name,
            Path.cwd() / raw,
        ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    tried = "\n  ".join(str(candidate) for candidate in candidates)
    raise SystemExit(
        f"CMVN is configured, but the file was not found. Tried:\n  {tried}"
    )


def normalize_cmvn_config(
    configs: dict[str, Any], model_dir: Path, config_dir: Path
) -> bool:
    """Return True when CMVN is expected and should be verified in ONNX."""
    legacy_cmvn_file = configs.get("cmvn_file")
    legacy_is_json = configs.get("is_json_cmvn")

    cmvn = configs.get("cmvn")
    cmvn_conf = configs.get("cmvn_conf")

    if cmvn == "global_cmvn":
        if cmvn_conf is None:
            cmvn_conf = {}
        if not isinstance(cmvn_conf, dict):
            raise SystemExit("cmvn_conf must be a mapping when cmvn=global_cmvn")

        cmvn_file = cmvn_conf.get("cmvn_file") or legacy_cmvn_file
        if not cmvn_file:
            fallback = model_dir / "global_cmvn"
            if fallback.is_file():
                cmvn_file = str(fallback)
            else:
                raise SystemExit("cmvn=global_cmvn but no cmvn_file was provided")

        resolved = resolve_cmvn_path(str(cmvn_file), model_dir, config_dir)
        cmvn_conf["cmvn_file"] = str(resolved)
        if "is_json_cmvn" not in cmvn_conf:
            cmvn_conf["is_json_cmvn"] = (
                as_bool(legacy_is_json)
                if legacy_is_json is not None
                else infer_is_json_cmvn(resolved)
            )
        configs["cmvn_conf"] = cmvn_conf
        return True

    if legacy_cmvn_file:
        resolved = resolve_cmvn_path(str(legacy_cmvn_file), model_dir, config_dir)
        configs["cmvn"] = "global_cmvn"
        configs["cmvn_conf"] = {
            "cmvn_file": str(resolved),
            "is_json_cmvn": (
                as_bool(legacy_is_json)
                if legacy_is_json is not None
                else infer_is_json_cmvn(resolved)
            ),
        }
        return True

    return False


def get_reverse_weight(configs: dict[str, Any], override: float | None) -> float:
    if override is not None:
        return override
    model_conf = configs.get("model_conf") or {}
    if isinstance(model_conf, dict) and "reverse_weight" in model_conf:
        return float(model_conf["reverse_weight"])
    return 0.0


def run_export(
    python_bin: str,
    wenet_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    chunk_size: int,
    num_left_chunks: int,
    reverse_weight: float,
) -> None:
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(wenet_root)
        if not old_pythonpath
        else str(wenet_root) + os.pathsep + old_pythonpath
    )

    cmd = [
        python_bin,
        "-m",
        "wenet.bin.export_onnx_cpu",
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--chunk_size",
        str(chunk_size),
        "--num_decoding_left_chunks",
        str(num_left_chunks),
        "--reverse_weight",
        str(reverse_weight),
        "--output_dir",
        str(output_dir),
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(wenet_root), env=env, check=True)


def build_export_command(
    python_bin: str,
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    chunk_size: int,
    num_left_chunks: int,
    reverse_weight: float,
) -> list[str]:
    return [
        python_bin,
        "-m",
        "wenet.bin.export_onnx_cpu",
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--chunk_size",
        str(chunk_size),
        "--num_decoding_left_chunks",
        str(num_left_chunks),
        "--reverse_weight",
        str(reverse_weight),
        "--output_dir",
        str(output_dir),
    ]


def verify_cmvn_tensors(python_bin: str, encoder_path: Path) -> None:
    code = r"""
import sys
import onnx

model = onnx.load(sys.argv[1], load_external_data=False)
names = {item.name for item in model.graph.initializer}
missing = [name for name in ("global_cmvn.mean", "global_cmvn.istd")
           if name not in names]
if missing:
    print("Missing expected CMVN tensors:", ", ".join(missing), file=sys.stderr)
    sys.exit(1)
print("Verified CMVN tensors in encoder.onnx")
"""
    subprocess.run([python_bin, "-c", code, str(encoder_path)], check=True)


def main() -> None:
    args = parse_args()
    model_dir = abs_path(args.model_dir)
    output_dir = abs_path(args.output_dir)
    wenet_root = resolve_wenet_root(abs_path(args.wenet_dir))

    config_path = model_dir / "train.yaml"
    checkpoint_path = model_dir / "final.pt"
    if not config_path.is_file():
        raise SystemExit(f"Missing config: {config_path}")
    if not checkpoint_path.is_file():
        raise SystemExit(f"Missing checkpoint: {checkpoint_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    configs = load_yaml(config_path)
    cmvn_expected = normalize_cmvn_config(configs, model_dir, config_path.parent)
    reverse_weight = get_reverse_weight(configs, args.reverse_weight)

    export_config_path = output_dir / "train.export.yaml"
    dump_yaml(export_config_path, configs)

    print(f"Model dir: {model_dir}")
    print(f"WeNet dir: {wenet_root}")
    print(f"Output dir: {output_dir}")
    print(f"Export config: {export_config_path}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Num decoding left chunks: {args.num_decoding_left_chunks}")
    print(f"Reverse weight: {reverse_weight}")
    print(f"CMVN expected: {cmvn_expected}")

    if args.prepare_only:
        cmd = build_export_command(
            args.python,
            export_config_path,
            checkpoint_path,
            output_dir,
            args.chunk_size,
            args.num_decoding_left_chunks,
            reverse_weight,
        )
        print("Prepared export command:", " ".join(cmd))
        return

    run_export(
        args.python,
        wenet_root,
        export_config_path,
        checkpoint_path,
        output_dir,
        args.chunk_size,
        args.num_decoding_left_chunks,
        reverse_weight,
    )

    encoder_path = output_dir / "encoder.onnx"
    if cmvn_expected and not args.skip_cmvn_verify:
        verify_cmvn_tensors(args.python, encoder_path)


if __name__ == "__main__":
    main()
