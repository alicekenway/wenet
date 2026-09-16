#!/usr/bin/env python3
"""Export a NeMo ASR model to ONNX plus SDK package metadata.

The current SDK runtime in this repository loads the WeNet ONNX ABI:
encoder.onnx, ctc.onnx, decoder.onnx, WeNet metadata, and streaming cache
inputs such as chunk/offset/att_cache/cnn_cache. A standard NeMo ONNX export
does not have that ABI. This script therefore exports a NeMo CTC/RNNT model
cleanly and records the intended backend as nemo_ctc_onnxruntime, but it does
not fake a WeNet package.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_NEMO_PYTHON = Path("/home/jinyang_wang/miniforge3/envs/embedded_ASR/bin/python")
DEFAULT_BACKEND = "nemo_ctc_onnxruntime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a NeMo ASR model to ONNX and write metadata needed by a "
            "future NeMo ONNX SDK backend."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--model",
        "--nemo-model",
        dest="nemo_model",
        help="Path to a .nemo model file.",
    )
    source.add_argument(
        "--pretrained-name",
        help="NeMo pretrained model name, passed to ASRModel.from_pretrained.",
    )
    parser.add_argument("--output-dir", help="Output package directory.")
    parser.add_argument(
        "--onnx-name",
        default="model.onnx",
        help="ONNX filename inside output-dir.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used for export, for example cpu or cuda.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Override sample rate written to sdk_model.json.",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        help="Backend name written to sdk_model.json.",
    )
    parser.add_argument(
        "--skip-onnx-check",
        action="store_true",
        help="Skip onnx.checker validation after export.",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Only check that the Python environment has required packages.",
    )
    parser.add_argument(
        "--require-current-sdk-compatible",
        action="store_true",
        help="Exit non-zero after export if the current WeNet bridge cannot load it.",
    )
    return parser.parse_args()


def fail(message: str, code: int = 2) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(code)


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def abs_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def import_runtime_modules() -> dict[str, Any]:
    missing: list[str] = []
    modules: dict[str, Any] = {}

    try:
        import torch  # type: ignore

        modules["torch"] = torch
    except Exception as exc:  # pragma: no cover - depends on local env
        missing.append(f"torch ({type(exc).__name__}: {exc})")

    try:
        from nemo.collections.asr.models import ASRModel  # type: ignore

        modules["ASRModel"] = ASRModel
    except Exception as exc:  # pragma: no cover - depends on local env
        missing.append(f"nemo.collections.asr.models.ASRModel ({type(exc).__name__}: {exc})")

    try:
        import onnx  # type: ignore

        modules["onnx"] = onnx
    except Exception as exc:  # pragma: no cover - depends on local env
        missing.append(f"onnx ({type(exc).__name__}: {exc})")

    if missing:
        expected = str(DEFAULT_NEMO_PYTHON)
        script_path = Path(__file__).resolve()
        fail(
            "The export environment is missing required packages:\n  "
            + "\n  ".join(missing)
            + f"\nRun with the NeMo env, for example:\n  {expected} {script_path} ...",
            code=3,
        )
    return modules


def maybe_to_container(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf  # type: ignore

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return value


def nested_get(mapping: Any, keys: list[str]) -> Any:
    current = maybe_to_container(mapping)
    for key in keys:
        current = maybe_to_container(current)
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return maybe_to_container(current)


def load_model(args: argparse.Namespace, modules: dict[str, Any]) -> Any:
    ASRModel = modules["ASRModel"]
    torch = modules["torch"]
    device = torch.device(args.device)

    if args.nemo_model:
        model_path = abs_path(args.nemo_model)
        if not model_path.is_file():
            fail(f"NeMo model not found: {model_path}")
        model = ASRModel.restore_from(str(model_path), map_location=device)
    else:
        try:
            model = ASRModel.from_pretrained(args.pretrained_name, map_location=device)
        except TypeError:
            model = ASRModel.from_pretrained(args.pretrained_name)
            model = model.to(device)

    model = model.to(device)
    model.eval()
    if hasattr(model, "freeze"):
        model.freeze()
    return model


def export_to_onnx(model: Any, onnx_path: Path, opset: int) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    export_fn = model.export
    signature = inspect.signature(export_fn)
    params = signature.parameters

    args: list[Any] = []
    kwargs: dict[str, Any] = {}

    if "output" in params:
        kwargs["output"] = str(onnx_path)
    elif "output_file" in params:
        kwargs["output_file"] = str(onnx_path)
    elif "file_path" in params:
        kwargs["file_path"] = str(onnx_path)
    else:
        args.append(str(onnx_path))

    if "onnx_opset_version" in params:
        kwargs["onnx_opset_version"] = opset
    elif "opset_version" in params:
        kwargs["opset_version"] = opset
    elif "opset" in params:
        kwargs["opset"] = opset

    for optional_name, value in {
        "check_trace": False,
        "verbose": False,
        "do_constant_folding": True,
    }.items():
        if optional_name in params:
            kwargs[optional_name] = value

    print("Exporting ONNX:", onnx_path, flush=True)
    result = export_fn(*args, **kwargs)
    if not onnx_path.exists():
        # Some NeMo versions return one or more generated paths instead of
        # honoring the exact requested path. Keep the requested filename stable.
        returned_paths: list[Path] = []
        if isinstance(result, (str, os.PathLike)):
            returned_paths.append(Path(result))
        elif isinstance(result, (list, tuple)):
            returned_paths.extend(Path(item) for item in result if isinstance(item, (str, os.PathLike)))
        for returned in returned_paths:
            returned = returned.expanduser()
            if returned.is_file() and returned.suffix == ".onnx":
                returned.replace(onnx_path)
                break
    if not onnx_path.is_file():
        fail(f"NeMo export finished but ONNX file was not found: {onnx_path}")


def infer_vocabulary(model: Any) -> list[str]:
    decoder = getattr(model, "decoder", None)
    candidates = [
        getattr(decoder, "vocabulary", None),
        getattr(decoder, "vocab", None),
        nested_get(getattr(model, "cfg", None), ["decoder", "vocabulary"]),
        nested_get(getattr(model, "cfg", None), ["labels"]),
        nested_get(getattr(model, "cfg", None), ["model", "labels"]),
    ]

    tokenizer = getattr(model, "tokenizer", None)
    inner_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if inner_tokenizer is not None:
        if hasattr(inner_tokenizer, "get_vocab"):
            vocab = inner_tokenizer.get_vocab()
            if isinstance(vocab, dict):
                return [str(token) for token, _ in sorted(vocab.items(), key=lambda item: item[1])]
        if hasattr(inner_tokenizer, "get_piece_size") and hasattr(inner_tokenizer, "id_to_piece"):
            size = int(inner_tokenizer.get_piece_size())
            return [str(inner_tokenizer.id_to_piece(i)) for i in range(size)]

    for candidate in candidates:
        candidate = maybe_to_container(candidate)
        if isinstance(candidate, dict):
            return [str(token) for token, _ in sorted(candidate.items(), key=lambda item: item[1])]
        if isinstance(candidate, (list, tuple)) and candidate:
            return [str(token) for token in candidate]

    fail("Could not infer vocabulary from the NeMo model.")


def infer_blank_id(model: Any, vocab_size: int) -> int:
    decoder = getattr(model, "decoder", None)
    for name in ("blank_idx", "blank_id", "_blank_index"):
        value = getattr(decoder, name, None)
        if isinstance(value, int):
            return value
    value = nested_get(getattr(model, "cfg", None), ["decoder", "blank_idx"])
    if isinstance(value, int):
        return value
    # NeMo CTC models commonly use the last class as blank.
    return vocab_size


def printable_token(token: str) -> str:
    if token == " ":
        return "<space>"
    if token == "\t":
        return "<tab>"
    if token == "\n":
        return "<newline>"
    if token == "":
        return "<empty>"
    return token


def write_vocab_files(output_dir: Path, vocab: list[str], blank_id: int) -> None:
    tokens_path = output_dir / "tokens.txt"
    vocab_path = output_dir / "vocab.json"

    with tokens_path.open("w", encoding="utf-8") as fout:
        for idx, token in enumerate(vocab):
            fout.write(f"{printable_token(token)} {idx}\n")
        if blank_id >= len(vocab):
            fout.write(f"<blank> {blank_id}\n")

    with vocab_path.open("w", encoding="utf-8") as fout:
        json.dump(
            {
                "tokens": vocab,
                "blank_id": blank_id,
                "tokens_txt_escaping": {
                    " ": "<space>",
                    "\\t": "<tab>",
                    "\\n": "<newline>",
                    "": "<empty>",
                },
            },
            fout,
            ensure_ascii=False,
            indent=2,
        )
        fout.write("\n")


def infer_sample_rate(model: Any, override: int | None) -> int:
    if override:
        return override
    cfg = getattr(model, "cfg", None)
    for keys in (
        ["sample_rate"],
        ["preprocessor", "sample_rate"],
        ["model", "sample_rate"],
        ["model", "preprocessor", "sample_rate"],
    ):
        value = nested_get(cfg, keys)
        if isinstance(value, int) and value > 0:
            return value
    preprocessor = getattr(model, "preprocessor", None)
    value = getattr(preprocessor, "sample_rate", None)
    if isinstance(value, int) and value > 0:
        return value
    return 16000


def shape_to_json(value_info: Any) -> list[str | int]:
    dims = value_info.type.tensor_type.shape.dim
    result: list[str | int] = []
    for dim in dims:
        if dim.dim_param:
            result.append(str(dim.dim_param))
        elif dim.dim_value:
            result.append(int(dim.dim_value))
        else:
            result.append("?")
    return result


def onnx_io_summary(onnx_module: Any, onnx_path: Path, check_model: bool) -> dict[str, Any]:
    model = onnx_module.load(str(onnx_path), load_external_data=True)
    if check_model:
        onnx_module.checker.check_model(model)

    inputs = [
        {"name": item.name, "shape": shape_to_json(item)}
        for item in model.graph.input
    ]
    outputs = [
        {"name": item.name, "shape": shape_to_json(item)}
        for item in model.graph.output
    ]
    metadata = {item.key: item.value for item in model.metadata_props}
    return {"inputs": inputs, "outputs": outputs, "metadata": metadata}


def write_manifest(
    output_dir: Path,
    onnx_name: str,
    args: argparse.Namespace,
    model: Any,
    vocab: list[str],
    blank_id: int,
    sample_rate: int,
    io_summary: dict[str, Any],
) -> None:
    manifest = {
        "sdk_model_version": 1,
        "backend": args.backend,
        "model_type": "nemo_asr_onnx",
        "sample_rate": sample_rate,
        "unit_path": "tokens.txt",
        "runtime": {
            "onnx_file": onnx_name,
            "vocab_json": "vocab.json",
            "blank_id": blank_id,
            "vocab_size_without_blank": len(vocab),
            "model_class": type(model).__name__,
            "input_names": [item["name"] for item in io_summary["inputs"]],
            "output_names": [item["name"] for item in io_summary["outputs"]],
        },
        "compatibility": {
            "current_wenet_bridge_compatible": False,
            "reason": (
                "The current SDK bridge loads WeNet encoder.onnx/ctc.onnx/"
                "decoder.onnx and passes WeNet streaming cache tensors. "
                "This package requires a NeMo ONNX backend."
            ),
        },
    }
    with (output_dir / "sdk_model.json").open("w", encoding="utf-8") as fout:
        json.dump(manifest, fout, ensure_ascii=False, indent=2)
        fout.write("\n")

    with (output_dir / "onnx_io.json").open("w", encoding="utf-8") as fout:
        json.dump(io_summary, fout, ensure_ascii=False, indent=2)
        fout.write("\n")

    readme = output_dir / "README.nemo_export.txt"
    readme.write_text(
        "\n".join(
            [
                "NeMo ONNX export package",
                "",
                f"ONNX: {onnx_name}",
                "Vocabulary: tokens.txt and vocab.json",
                "Manifest: sdk_model.json",
                "",
                "Important:",
                "This is not loadable by the current WeNetRuntimeBridge.",
                "The current SDK expects a WeNet ONNX ABI with encoder.onnx,",
                "ctc.onnx, decoder.onnx, and streaming cache inputs.",
                "Add a NeMo ONNX backend in the SDK before using this package",
                "for embedded decoding.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    print(f"Python: {sys.executable}")
    print(f"Expected NeMo env: {DEFAULT_NEMO_PYTHON}")
    modules = import_runtime_modules()
    if args.check_env:
        print("Environment check passed.")
        return
    if not args.nemo_model and not args.pretrained_name:
        fail("Either --model/--nemo-model or --pretrained-name is required.")
    if not args.output_dir:
        fail("--output-dir is required.")

    output_dir = abs_path(args.output_dir)
    onnx_path = output_dir / args.onnx_name

    model = load_model(args, modules)
    export_to_onnx(model, onnx_path, args.opset)

    vocab = infer_vocabulary(model)
    blank_id = infer_blank_id(model, len(vocab))
    sample_rate = infer_sample_rate(model, args.sample_rate)
    write_vocab_files(output_dir, vocab, blank_id)

    io_summary = onnx_io_summary(
        modules["onnx"], onnx_path, check_model=not args.skip_onnx_check
    )
    write_manifest(
        output_dir,
        args.onnx_name,
        args,
        model,
        vocab,
        blank_id,
        sample_rate,
        io_summary,
    )

    print(f"Exported ONNX: {onnx_path}")
    print(f"Wrote manifest: {output_dir / 'sdk_model.json'}")
    print(f"Wrote vocabulary: {output_dir / 'tokens.txt'}")
    print(f"Blank id: {blank_id}")
    print(f"Sample rate: {sample_rate}")

    message = (
        "Generated NeMo ONNX package is not loadable by the current "
        "WeNetRuntimeBridge. Add a NeMo ONNX SDK backend before deploying it "
        "through this embedded framework."
    )
    warn(message)
    if args.require_current_sdk_compatible:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
