#!/usr/bin/env python3
"""Decode and score a WeNet ONNX model on a WeNet data directory."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import unicodedata
import wave


class PipelineError(RuntimeError):
    """An input or decoding error that should be shown without a traceback."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode a WeNet ONNX model, write results.jsonl, and calculate "
            "DEL/INS/SUB/WER/SER/RTF."
        )
    )
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--model-dir", type=Path,
                       help="ONNX runtime package/model directory")
    model.add_argument(
        "--onnx-model", type=Path,
        help="Single streaming WeNet CTC ONNX containing ctc_log_probs",
    )
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="WeNet data directory containing wav.scp and text")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for results.jsonl and summary.txt")
    parser.add_argument(
        "--decoder-bin", type=Path,
        help=(
            "Decoder executable. Auto-detected from SDK/*/build/asr_batch_decode "
            "or build/bin/decoder_main when omitted"
        ),
    )
    parser.add_argument(
        "--decoder-kind", choices=("auto", "sdk", "wenet"), default="auto",
        help="sdk=asr_batch_decode; wenet=runtime/core decoder_main",
    )
    parser.add_argument(
        "--scoring-unit", choices=("auto", "word", "char"), default="auto",
        help="auto uses character scoring for CJK references, otherwise words",
    )
    parser.add_argument("--case-sensitive", action="store_true",
                        help="Do not case-fold reference and hypothesis text")
    parser.add_argument("--unit-path", type=Path,
                        help="Token table for decoder_main (default: MODEL_DIR/units.txt)")
    parser.add_argument("--chunk-size", type=int, default=16,
                        help="decoder_main chunk size (default: 16)")
    parser.add_argument("--num-left-chunks", type=int, default=-1,
                        help="decoder_main left chunks (default: -1)")
    parser.add_argument("--thread-num", type=int, default=1,
                        help="decoder_main decode threads (default: 1)")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="decoder_main expected sample rate (default: 16000)")
    parser.add_argument("--num-bins", type=int, default=80,
                        help="decoder_main fbank bins (default: 80)")
    parser.add_argument("--chunk-ms", type=int, default=100,
                        help="SDK 1.x batch audio chunk size (default: 100)")
    parser.add_argument(
        "--contract", type=Path,
        help="Single-ONNX contract JSON (auto-detected beside --onnx-model)",
    )
    parser.add_argument(
        "--tokens", type=Path,
        help="Token-to-ID table required with --onnx-model",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "gpu"), default="auto",
        help="ONNX Runtime provider for --onnx-model (default: auto)",
    )
    parser.add_argument(
        "--waveform-scale", type=float, default=32768.0,
        help="Scale normalized audio before Kaldi fbank (WeNet default: 32768)",
    )
    parser.add_argument(
        "--token-separator", default=" ",
        help="Text placed between CTC tokens (default: one space)",
    )
    return parser.parse_args()


def ensure_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise PipelineError(f"{description} does not exist: {path}")
    return path


def read_keyed_text(path: Path, description: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw in enumerate(stream, 1):
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            fields = line.split(maxsplit=1)
            key = fields[0]
            value = fields[1].strip() if len(fields) == 2 else ""
            if key in rows:
                raise PipelineError(
                    f"duplicate key {key!r} in {description} at {path}:{line_number}"
                )
            rows[key] = value
    if not rows:
        raise PipelineError(f"{description} is empty: {path}")
    return rows


def resolve_audio_path(value: str, data_dir: Path, source: Path) -> Path:
    if value.rstrip().endswith("|"):
        raise PipelineError(
            f"pipe commands in wav.scp are not supported by the ONNX decoder: {value}"
        )
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return ensure_file(raw, "audio file")
    candidates = [Path.cwd() / raw, data_dir / raw, source.parent / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise PipelineError(
        f"audio file does not exist: {value!r} "
        f"(checked the current directory and {data_dir})"
    )


def read_data_list(path: Path, data_dir: Path) -> tuple[dict[str, Path], dict[str, str]]:
    wavs: dict[str, Path] = {}
    refs: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PipelineError(
                    f"{path} is not a raw JSON data.list at line {line_number}; "
                    "provide data_dir/wav.scp (shard lists cannot be decoded directly)"
                ) from exc
            key = str(row.get("key", ""))
            wav = row.get("wav") or row.get("audio_filepath")
            ref = row.get("txt", row.get("text"))
            if not key or wav is None or ref is None:
                raise PipelineError(
                    f"{path}:{line_number} must contain key, wav, and txt fields"
                )
            if key in wavs:
                raise PipelineError(f"duplicate key {key!r} in {path}:{line_number}")
            wavs[key] = resolve_audio_path(str(wav), data_dir, path)
            refs[key] = str(ref)
    if not wavs:
        raise PipelineError(f"data list is empty: {path}")
    return wavs, refs


def load_dataset(data_dir: Path) -> tuple[list[str], dict[str, Path], dict[str, str]]:
    data_dir = data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise PipelineError(f"data directory does not exist: {data_dir}")

    wav_file = next(
        (data_dir / name for name in ("wav.scp", "wav.list", "wav_list")
         if (data_dir / name).is_file()),
        None,
    )
    text_file = data_dir / "text"
    if wav_file is None:
        data_list = data_dir / "data.list"
        if not data_list.is_file():
            raise PipelineError(
                f"expected {data_dir}/wav.scp (or a raw JSON {data_dir}/data.list)"
            )
        wavs, list_refs = read_data_list(data_list, data_dir)
        refs = read_keyed_text(text_file, "reference text") if text_file.is_file() else list_refs
    else:
        wav_specs = read_keyed_text(wav_file, "wav list")
        if not text_file.is_file():
            raise PipelineError(f"reference text does not exist: {text_file}")
        refs = read_keyed_text(text_file, "reference text")
        wavs = {
            key: resolve_audio_path(value, data_dir, wav_file)
            for key, value in wav_specs.items()
        }

    missing_refs = [key for key in wavs if key not in refs]
    extra_refs = [key for key in refs if key not in wavs]
    if missing_refs or extra_refs:
        details = []
        if missing_refs:
            details.append(f"missing references for {missing_refs[:5]}")
        if extra_refs:
            details.append(f"references without audio for {extra_refs[:5]}")
        raise PipelineError("wav/text key mismatch: " + "; ".join(details))
    return list(wavs), wavs, refs


def version_key(path: Path) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", path.parts[-3] if len(path.parts) >= 3 else "")
    return tuple(int(number) for number in numbers)


def find_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def detect_decoder(model_dir: Path, requested: Path | None,
                   kind: str) -> tuple[Path, str]:
    if requested is not None:
        decoder = ensure_file(requested, "decoder executable")
        if not os.access(decoder, os.X_OK):
            raise PipelineError(f"decoder is not executable: {decoder}")
        if kind == "auto":
            if decoder.name == "decoder_main":
                kind = "wenet"
            elif decoder.name == "batch_files":
                kind = "sdk_v1"
            else:
                kind = ("sdk" if (model_dir / "sdk_model.json").is_file()
                        else "sdk_unpacked")
        elif kind == "sdk" and decoder.name == "batch_files":
            kind = "sdk_v1"
        elif kind == "sdk" and not (model_dir / "sdk_model.json").is_file():
            kind = "sdk_unpacked"
        return decoder, kind

    repo = find_repo_root()
    sdk_candidates = sorted(
        repo.glob("SDK/*/build/asr_batch_decode"), key=version_key, reverse=True
    )
    sdk_v1_candidates = sorted(
        repo.glob("SDK/*/build/batch_files"), key=version_key, reverse=True
    )
    wenet_candidates = [repo / "build/bin/decoder_main", repo / "build/decoder_main"]
    if kind in ("auto", "sdk") and (model_dir / "sdk_model.json").is_file():
        for candidate in sdk_candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate.resolve(), "sdk"
    if kind in ("auto", "sdk") and (model_dir / "manifest.json").is_file():
        for candidate in sdk_v1_candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate.resolve(), "sdk_v1"
    if kind in ("auto", "wenet"):
        for candidate in wenet_candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate.resolve(), "wenet"
    plain_onnx = all((model_dir / name).is_file()
                     for name in ("encoder.onnx", "ctc.onnx", "decoder.onnx"))
    if kind in ("auto", "sdk") and plain_onnx:
        for candidate in sdk_candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate.resolve(), "sdk_unpacked"
    if kind == "sdk" or (
        kind == "auto"
        and ((model_dir / "sdk_model.json").is_file()
             or (model_dir / "manifest.json").is_file())
    ):
        raise PipelineError(
            "no compatible SDK batch decoder was found; build asr_batch_decode "
            "or batch_files under SDK/<version>/build, or pass --decoder-bin"
        )
    raise PipelineError(
        "this is not an SDK model package (sdk_model.json is absent), and no "
        "build/bin/decoder_main was found. Build runtime/core with -DONNX=ON, "
        "or package the model for the SDK and pass --decoder-bin"
    )


def validate_model(model_dir: Path, decoder_kind: str, unit_path: Path | None) -> Path | None:
    model_dir = model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise PipelineError(f"model directory does not exist: {model_dir}")
    if decoder_kind == "sdk":
        ensure_file(model_dir / "sdk_model.json", "SDK model manifest")
        return None
    if decoder_kind == "sdk_v1":
        ensure_file(model_dir / "manifest.json", "SDK 1.x model manifest")
        return None
    for name in ("encoder.onnx", "ctc.onnx", "decoder.onnx"):
        ensure_file(model_dir / name, "ONNX model")
    default_unit_path = (model_dir / "units.txt" if (model_dir / "units.txt").is_file()
                         else model_dir / "tokens.txt")
    return ensure_file(unit_path or default_unit_path, "unit table")


def make_sdk_package(output_dir: Path, model_dir: Path, unit_path: Path,
                     args: argparse.Namespace, refs: dict[str, str]) -> Path:
    package_dir = output_dir / "runtime_model"
    package_dir.mkdir(parents=True, exist_ok=True)
    assets = {
        "encoder.onnx": model_dir / "encoder.onnx",
        "ctc.onnx": model_dir / "ctc.onnx",
        "decoder.onnx": model_dir / "decoder.onnx",
        "units.txt": unit_path,
    }
    for name, source in assets.items():
        destination = package_dir / name
        if destination.is_symlink() or destination.exists():
            destination.unlink()
        destination.symlink_to(source.resolve())
    language = "chs" if any(contains_cjk(ref) for ref in refs.values()) else "en"
    manifest = {
        "sdk_model_version": 1,
        "backend": "wenet_onnxruntime_static_wenet_dynamic_ort",
        "audio": {"sample_rate": args.sample_rate},
        "wenet": {"onnx_dir": ".", "unit_path": "units.txt"},
        "decode": {
            "chunk_size": args.chunk_size,
            "num_left_chunks": args.num_left_chunks,
            "nbest": 1,
        },
        "runtime": {"enable_continuous_decoding": True},
        "postprocess": {"language_type": language, "enable_timestamp": False},
    }
    (package_dir / "sdk_model.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return package_dir


def write_decoder_input(path: Path, keys: list[str], wavs: dict[str, Path]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for key in keys:
            wav = str(wavs[key])
            if any(character.isspace() for character in wav):
                raise PipelineError(
                    f"audio paths containing whitespace are unsupported by the decoder: {wav}"
                )
            stream.write(f"{key} {wav}\n")


def decoder_command(args: argparse.Namespace, decoder: Path, kind: str,
                    model_dir: Path, wav_scp: Path, hyp_path: Path,
                    unit_path: Path | None) -> list[str]:
    if kind in ("sdk", "sdk_unpacked"):
        return [str(decoder), "--model_dir", str(model_dir),
                "--wav_scp", str(wav_scp), "--result", str(hyp_path)]
    if kind == "sdk_v1":
        return [str(decoder), "--model_dir", str(model_dir),
                "--wav_list", str(wav_scp), "--chunk_ms", str(args.chunk_ms)]
    assert unit_path is not None
    return [
        str(decoder),
        f"--onnx_dir={model_dir}",
        f"--wav_scp={wav_scp}",
        f"--result={hyp_path}",
        f"--unit_path={unit_path}",
        f"--chunk_size={args.chunk_size}",
        f"--num_left_chunks={args.num_left_chunks}",
        f"--thread_num={args.thread_num}",
        f"--sample_rate={args.sample_rate}",
        f"--num_bins={args.num_bins}",
    ]


def run_decoder(command: list[str], log_path: Path,
                stdout_path: Path | None = None) -> tuple[float, str]:
    print("Running decoder:", " ".join(command), flush=True)
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        if stdout_path is None:
            completed = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT,
                text=True, check=False
            )
        else:
            with stdout_path.open("w", encoding="utf-8") as stdout:
                completed = subprocess.run(
                    command, stdout=stdout, stderr=log, text=True, check=False
                )
    elapsed = time.monotonic() - start
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        tail = "\n".join(log_text.splitlines()[-20:])
        raise PipelineError(
            f"decoder failed with exit code {completed.returncode}; see {log_path}\n{tail}"
        )
    return elapsed, log_text


def default_contract_path(model_path: Path) -> Path:
    name = model_path.name
    if name.endswith(".int8.onnx"):
        name = name[:-len(".int8.onnx")] + ".contract.json"
    elif name.endswith(".onnx"):
        name = name[:-len(".onnx")] + ".contract.json"
    else:
        name += ".contract.json"
    return model_path.with_name(name)


def load_single_contract(path: Path) -> dict:
    path = ensure_file(path, "single-ONNX contract")
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineError(f"invalid contract JSON {path}: {exc}") from exc
    required_objects = ("fbank", "inputs", "outputs")
    if not isinstance(contract, dict) or any(
        not isinstance(contract.get(name), dict) for name in required_objects
    ):
        raise PipelineError(
            f"contract must contain fbank, inputs, and outputs objects: {path}"
        )
    for name in ("chunk_frames", "chunk_stride_frames", "minimum_input_frames"):
        try:
            if int(contract[name]) < 1:
                raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineError(f"contract has invalid {name}: {path}") from exc
    if "features" not in contract["inputs"]:
        raise PipelineError(f"contract inputs.features is missing: {path}")
    if "ctc_log_probs" not in contract["outputs"]:
        raise PipelineError(f"contract outputs.ctc_log_probs is missing: {path}")
    caches = contract.get("cache_inputs", [])
    if not isinstance(caches, list):
        raise PipelineError(f"contract cache_inputs must be a list: {path}")
    return contract


def read_token_table(path: Path) -> dict[int, str]:
    path = ensure_file(path, "token table")
    tokens: dict[int, str] = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line:
                continue
            fields = line.rsplit(maxsplit=1)
            if len(fields) != 2:
                raise PipelineError(
                    f"invalid token table row at {path}:{line_number}: {line!r}"
                )
            token, raw_id = fields
            try:
                token_id = int(raw_id)
            except ValueError as exc:
                raise PipelineError(
                    f"invalid token ID at {path}:{line_number}: {raw_id!r}"
                ) from exc
            if token_id in tokens:
                raise PipelineError(f"duplicate token ID {token_id} at {path}:{line_number}")
            tokens[token_id] = token
    if not tokens:
        raise PipelineError(f"token table is empty: {path}")
    return tokens


def ctc_greedy_ids(frame_ids: list[int], blank_id: int) -> list[int]:
    result: list[int] = []
    previous: int | None = None
    for token_id in frame_ids:
        token_id = int(token_id)
        if token_id != blank_id and token_id != previous:
            result.append(token_id)
        previous = token_id
    return result


def _ort_numpy_dtype(tensor_type: str, np_module):
    mapping = {
        "tensor(float)": np_module.float32,
        "tensor(float16)": np_module.float16,
        "tensor(double)": np_module.float64,
        "tensor(int64)": np_module.int64,
        "tensor(int32)": np_module.int32,
        "tensor(bool)": np_module.bool_,
    }
    if tensor_type not in mapping:
        raise PipelineError(f"unsupported ONNX tensor type: {tensor_type}")
    return mapping[tensor_type]


def run_single_onnx(model_path: Path, contract: dict, tokens: dict[int, str],
                    keys: list[str], wavs: dict[str, Path], args: argparse.Namespace
                    ) -> tuple[dict[str, str], float, float, str]:
    try:
        import numpy as np
        import onnxruntime as ort
        import torch
        import torchaudio
    except ImportError as exc:
        raise PipelineError(
            "--onnx-model requires numpy, onnxruntime, torch, and torchaudio; "
            "run with a Python environment that provides them"
        ) from exc

    available = ort.get_available_providers()
    if args.device == "gpu":
        if "CUDAExecutionProvider" not in available:
            raise PipelineError(
                "--device gpu was requested, but CUDAExecutionProvider is unavailable"
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif args.device == "auto" and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    try:
        session = ort.InferenceSession(str(model_path), providers=providers)
    except Exception as exc:
        raise PipelineError(f"failed to load ONNX model {model_path}: {exc}") from exc
    inputs = {item.name: item for item in session.get_inputs()}
    outputs = {item.name: item for item in session.get_outputs()}
    feature_name = str(contract["inputs"]["features"])
    offset_name = contract["inputs"].get("offset")
    ctc_name = str(contract["outputs"]["ctc_log_probs"])
    cache_specs = contract.get("cache_inputs", [])
    attention = contract.get("attention_mask")
    constants = contract.get("constant_inputs", {})
    required_inputs = {feature_name, *constants.keys()}
    if offset_name:
        required_inputs.add(str(offset_name))
    for spec in cache_specs:
        required_inputs.add(str(spec["input"]))
    if attention:
        required_inputs.add(str(attention["input"]))
    missing_inputs = sorted(required_inputs - inputs.keys())
    required_outputs = {ctc_name, *(str(spec["output"]) for spec in cache_specs)}
    missing_outputs = sorted(required_outputs - outputs.keys())
    if missing_inputs or missing_outputs:
        raise PipelineError(
            f"model/contract mismatch: missing inputs={missing_inputs}, "
            f"missing outputs={missing_outputs}"
        )

    sample_rate = int(contract.get("sample_rate", 16000))
    fbank_conf = contract["fbank"]
    num_mel_bins = int(fbank_conf.get("num_mel_bins", 80))
    frame_length_ms = float(fbank_conf.get("frame_length_ms", 25.0))
    frame_shift_ms = float(fbank_conf.get("frame_shift_ms", 10.0))
    dither = float(fbank_conf.get("dither", 0.0))
    chunk_frames = int(contract["chunk_frames"])
    stride_frames = int(contract["chunk_stride_frames"])
    minimum_frames = int(contract["minimum_input_frames"])
    initial_offset = int(contract.get("initial_offset", 0))
    blank_id = int(contract.get("blank_id", 0))
    if args.waveform_scale <= 0 or not math.isfinite(args.waveform_scale):
        raise PipelineError("--waveform-scale must be a positive finite number")

    log_lines = [
        f"model {model_path}",
        f"providers {session.get_providers()}",
        f"inputs {[(item.name, item.type, item.shape) for item in session.get_inputs()]}",
        f"outputs {[(item.name, item.type, item.shape) for item in session.get_outputs()]}",
    ]
    hypotheses: dict[str, str] = {}
    total_audio = 0.0
    start_time = time.monotonic()
    for index, key in enumerate(keys, 1):
        try:
            waveform, source_rate = torchaudio.load(str(wavs[key]))
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if source_rate != sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, source_rate, sample_rate
                )
            waveform = waveform.to(torch.float32)
            total_audio += waveform.shape[1] / sample_rate
            minimum_samples = max(
                1, int(math.ceil(frame_length_ms * sample_rate / 1000.0))
            )
            if waveform.shape[1] < minimum_samples:
                waveform = torch.nn.functional.pad(
                    waveform, (0, minimum_samples - waveform.shape[1])
                )
            features = torchaudio.compliance.kaldi.fbank(
                waveform * args.waveform_scale,
                sample_frequency=float(sample_rate),
                frame_length=frame_length_ms,
                frame_shift=frame_shift_ms,
                num_mel_bins=num_mel_bins,
                dither=dither,
                energy_floor=0.0,
                use_energy=False,
            ).detach().cpu().numpy().astype(np.float32)
            if features.shape[0] < minimum_frames:
                padded = np.zeros((minimum_frames, num_mel_bins), dtype=np.float32)
                padded[:features.shape[0]] = features
                features = padded

            offset = initial_offset
            chunks_run = 0
            caches = {
                str(spec["input"]): np.zeros(
                    tuple(int(value) for value in spec["shape"]),
                    dtype=np.dtype(str(spec.get("dtype", "float32"))),
                )
                for spec in cache_specs
            }
            all_ctc = []
            starts = range(0, features.shape[0] - minimum_frames + 1,
                           stride_frames)
            for chunk_start in starts:
                chunk = features[chunk_start:chunk_start + chunk_frames]
                feed = {
                    feature_name: np.asarray(
                        chunk[np.newaxis, ...],
                        dtype=_ort_numpy_dtype(inputs[feature_name].type, np),
                    )
                }
                if offset_name:
                    name = str(offset_name)
                    feed[name] = np.asarray(
                        offset, dtype=_ort_numpy_dtype(inputs[name].type, np)
                    )
                for name, value in constants.items():
                    feed[name] = np.asarray(
                        value, dtype=_ort_numpy_dtype(inputs[name].type, np)
                    )
                if attention:
                    name = str(attention["input"])
                    cache_frames = int(attention["cache_frames"])
                    encoder_frames = int(attention["chunk_frames"])
                    mask = np.ones(
                        (1, 1, cache_frames + encoder_frames), dtype=np.bool_
                    )
                    if chunks_run == 0:
                        mask[:, :, :cache_frames] = False
                    feed[name] = mask.astype(
                        _ort_numpy_dtype(inputs[name].type, np), copy=False
                    )
                for spec in cache_specs:
                    name = str(spec["input"])
                    feed[name] = caches[name].astype(
                        _ort_numpy_dtype(inputs[name].type, np), copy=False
                    )
                output_names = [ctc_name, *(str(spec["output"]) for spec in cache_specs)]
                values = session.run(output_names, feed)
                ctc = np.asarray(values[0])
                if ctc.ndim == 3 and ctc.shape[0] == 1:
                    ctc = ctc[0]
                if ctc.ndim != 2 or ctc.shape[0] < 1:
                    raise PipelineError(
                        f"unexpected CTC output shape for {key}: {ctc.shape}"
                    )
                if not np.isfinite(ctc).all():
                    raise PipelineError(f"CTC output contains NaN/Inf for {key}")
                all_ctc.append(ctc)
                for spec, value in zip(cache_specs, values[1:]):
                    caches[str(spec["input"])] = np.asarray(value)
                offset += int(ctc.shape[0])
                chunks_run += 1
            if not all_ctc:
                raise PipelineError(f"no ONNX chunks were generated for {key}")
            frame_ids = np.concatenate(all_ctc, axis=0).argmax(axis=-1).tolist()
            token_ids = ctc_greedy_ids(frame_ids, blank_id)
            missing_tokens = sorted({item for item in token_ids if item not in tokens})
            if missing_tokens:
                raise PipelineError(
                    f"token table has no entries for predicted IDs {missing_tokens}"
                )
            hypotheses[key] = args.token_separator.join(tokens[item] for item in token_ids)
            log_lines.append(
                f"{key} audio_sec={waveform.shape[1] / sample_rate:.3f} "
                f"fbank_frames={features.shape[0]} chunks={chunks_run} "
                f"ctc_frames={sum(value.shape[0] for value in all_ctc)}"
            )
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(f"single-ONNX decode failed for {key}: {exc}") from exc
        if index % 100 == 0:
            print(f"Decoded {index}/{len(keys)} utterances", flush=True)
    elapsed = time.monotonic() - start_time
    rtf = elapsed / total_audio if total_audio > 0 else math.nan
    log_lines.extend([
        f"decoded_utterances {len(keys)}",
        f"audio_sec {total_audio}",
        f"wall_sec {elapsed}",
        f"rtf {rtf}",
    ])
    return hypotheses, total_audio, elapsed, "\n".join(log_lines) + "\n"


def contains_cjk(text: str) -> bool:
    return any(
        "CJK" in unicodedata.name(character, "")
        or "HIRAGANA" in unicodedata.name(character, "")
        or "KATAKANA" in unicodedata.name(character, "")
        or "HANGUL" in unicodedata.name(character, "")
        for character in text
    )


def char_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            tokens.append("".join(current))
            current.clear()

    for character in text:
        category = unicodedata.category(character)
        name = unicodedata.name(character, "")
        is_cjk = any(tag in name for tag in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL"))
        if is_cjk:
            flush()
            tokens.append(character)
        elif character.isspace() or character == "▁":
            flush()
        elif category.startswith("P"):
            flush()
        else:
            current.append(character)
    flush()
    return tokens


def tokenize(text: str, unit: str, case_sensitive: bool) -> list[str]:
    normalized = text if case_sensitive else text.casefold()
    if unit == "word":
        return normalized.split()
    return char_tokens(normalized)


def edit_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    distance = [[0] * cols for _ in range(rows)]
    operation = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        distance[i][0] = i
        operation[i][0] = "del"
    for j in range(1, cols):
        distance[0][j] = j
        operation[0][j] = "ins"
    for i in range(1, rows):
        for j in range(1, cols):
            choices = [
                (distance[i - 1][j] + 1, 1, "del"),
                (distance[i][j - 1] + 1, 2, "ins"),
                (distance[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
                 0, "cor" if reference[i - 1] == hypothesis[j - 1] else "sub"),
            ]
            best = min(choices)
            distance[i][j] = best[0]
            operation[i][j] = best[2]
    substitutions = deletions = insertions = 0
    i, j = len(reference), len(hypothesis)
    while i or j:
        op = operation[i][j]
        if op == "del":
            deletions += 1
            i -= 1
        elif op == "ins":
            insertions += 1
            j -= 1
        else:
            substitutions += op == "sub"
            i -= 1
            j -= 1
    return deletions, insertions, substitutions


def parse_rtf(log_text: str, fallback_wall: float,
              wavs: dict[str, Path]) -> tuple[float, float, float, str]:
    def last_float(pattern: str) -> float | None:
        matches = re.findall(pattern, log_text, flags=re.MULTILINE | re.IGNORECASE)
        return float(matches[-1]) if matches else None

    audio = last_float(r"^audio_sec\s+([0-9.eE+-]+)\s*$")
    wall = last_float(r"^wall_sec\s+([0-9.eE+-]+)\s*$")
    reported_rtf = last_float(r"^(?:rtf\s+|.*\bRTF:\s*)([0-9.eE+-]+)\s*$")
    if audio is None:
        duration = 0.0
        try:
            for wav_path in wavs.values():
                with wave.open(str(wav_path), "rb") as wav_file:
                    duration += wav_file.getnframes() / wav_file.getframerate()
            audio = duration
        except (wave.Error, OSError, ZeroDivisionError):
            audio = math.nan
    if reported_rtf is not None:
        if wall is None:
            wall = (reported_rtf * audio
                    if audio and math.isfinite(audio) else fallback_wall)
        return audio, wall, reported_rtf, "decoder"
    if wall is None:
        wall = fallback_wall
    rtf = wall / audio if audio and math.isfinite(audio) else math.nan
    return audio, wall, rtf, "wall_clock"


def fmt_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if math.isfinite(value) else "N/A"


def write_outputs(output_dir: Path, keys: list[str], wavs: dict[str, Path],
                  refs: dict[str, str], hyps: dict[str, str], scoring_unit: str,
                  case_sensitive: bool, audio_sec: float, decode_sec: float,
                  rtf: float, rtf_source: str) -> None:
    total_del = total_ins = total_sub = total_ref = wrong = 0
    wrong_keys: list[str] = []
    jsonl_path = output_dir / "results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for key in keys:
            ref_tokens = tokenize(refs[key], scoring_unit, case_sensitive)
            hyp_tokens = tokenize(hyps[key], scoring_unit, case_sensitive)
            deletions, insertions, substitutions = edit_counts(ref_tokens, hyp_tokens)
            total_del += deletions
            total_ins += insertions
            total_sub += substitutions
            total_ref += len(ref_tokens)
            if deletions + insertions + substitutions:
                wrong += 1
                wrong_keys.append(key)
            row = {"key": key, "wav": str(wavs[key]),
                   "ref": refs[key], "hyp": hyps[key]}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    errors = total_del + total_ins + total_sub
    wer = errors / total_ref if total_ref else math.nan
    ser = wrong / len(keys) if keys else math.nan
    summary_path = output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as stream:
        stream.write("ASR evaluation summary\n")
        stream.write(f"utterances: {len(keys)}\n")
        stream.write(f"wrong_utterances: {wrong}\n")
        stream.write(f"reference_tokens: {total_ref}\n")
        stream.write(f"scoring_unit: {scoring_unit}\n")
        stream.write(f"case_sensitive: {str(case_sensitive).lower()}\n")
        stream.write(f"DEL: {total_del}\n")
        stream.write(f"INS: {total_ins}\n")
        stream.write(f"SUB: {total_sub}\n")
        stream.write(f"WER: {fmt_float(wer * 100, 2)}%\n")
        stream.write(f"SER: {fmt_float(ser * 100, 2)}%\n")
        stream.write(f"audio_seconds: {fmt_float(audio_sec, 3)}\n")
        stream.write(f"decode_seconds: {fmt_float(decode_sec, 3)}\n")
        stream.write(f"RTF: {fmt_float(rtf, 4)}\n")
        stream.write(f"rtf_source: {rtf_source}\n")
        stream.write("\nWrong cases\n\n")
        for key in wrong_keys:
            stream.write(f"wav: {wavs[key]}\n")
            stream.write(f"hyp: {hyps[key]}\n")
            stream.write(f"ref: {refs[key]}\n\n")


def main() -> int:
    args = parse_args()
    try:
        keys, wavs, refs = load_dataset(args.data_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = args.output_dir.resolve()
        wav_scp = output_dir / "input.wav.scp"
        hyp_path = output_dir / "hyp.txt"
        log_path = output_dir / "decoder.log"
        write_decoder_input(wav_scp, keys, wavs)
        if args.onnx_model is not None:
            model_path = ensure_file(args.onnx_model, "single ONNX model")
            contract_path = args.contract or default_contract_path(model_path)
            contract = load_single_contract(contract_path)
            if args.tokens is None:
                raise PipelineError(
                    "--tokens is required with --onnx-model because the ONNX "
                    "contains token IDs but not their text symbols"
                )
            tokens = read_token_table(args.tokens)
            hyps, audio_sec, decode_sec, log_text = run_single_onnx(
                model_path, contract, tokens, keys, wavs, args
            )
            log_path.write_text(log_text, encoding="utf-8")
            with hyp_path.open("w", encoding="utf-8") as stream:
                for key in keys:
                    stream.write(f"{key} {hyps[key]}\n")
            rtf = decode_sec / audio_sec if audio_sec > 0 else math.nan
            rtf_source = "wall_clock"
        else:
            assert args.model_dir is not None
            model_dir = args.model_dir.expanduser().resolve()
            decoder, decoder_kind = detect_decoder(model_dir, args.decoder_bin,
                                                    args.decoder_kind)
            unit_path = validate_model(model_dir, decoder_kind, args.unit_path)
            if decoder_kind == "sdk_unpacked":
                assert unit_path is not None
                model_dir = make_sdk_package(output_dir, model_dir, unit_path,
                                             args, refs)
            command = decoder_command(args, decoder, decoder_kind, model_dir,
                                      wav_scp, hyp_path, unit_path)
            stdout_path = hyp_path if decoder_kind == "sdk_v1" else None
            elapsed, log_text = run_decoder(command, log_path, stdout_path)
            hyps = read_keyed_text(hyp_path, "decoder hypotheses")
            missing = [key for key in keys if key not in hyps]
            extra = [key for key in hyps if key not in wavs]
            decode_errors = [
                key for key in keys if hyps.get(key, "").startswith("<ERROR:")
            ]
            if missing or extra or decode_errors:
                raise PipelineError(
                    "invalid decoder output: "
                    f"missing={missing[:5]}, extra={extra[:5]}, "
                    f"errors={decode_errors[:5]}"
                )
            audio_sec, decode_sec, rtf, rtf_source = parse_rtf(
                log_text, elapsed, wavs
            )
        unit = args.scoring_unit
        if unit == "auto":
            unit = "char" if any(contains_cjk(ref) for ref in refs.values()) else "word"
        write_outputs(output_dir, keys, wavs, refs, hyps, unit,
                      args.case_sensitive, audio_sec, decode_sec, rtf, rtf_source)
        print(f"Wrote {output_dir / 'results.jsonl'}")
        print(f"Wrote {output_dir / 'summary.txt'}")
        return 0
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
