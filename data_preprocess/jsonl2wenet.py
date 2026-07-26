#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def split_colon_arg(value: str, arg_name: str) -> List[str]:
    parts = [x.strip() for x in value.split(":") if x.strip()]
    if not parts:
        raise ValueError(f"{arg_name} is empty.")
    return parts


def read_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}, line {line_idx + 1}: {e}") from e

            yield line_idx, item


def get_audio_path(item: Dict[str, Any]) -> str:
    """
    Support both:
      - audiofile_path   # your current field
      - audio_filepath   # common NeMo field
    """
    if "audiofile_path" in item:
        return item["audiofile_path"]
    if "audio_filepath" in item:
        return item["audio_filepath"]
    if "path" in item:
        return item["path"]

    raise KeyError(
        "Missing audio path field. Expected `audiofile_path` or `audio_filepath`."
    )


def make_utt_id(audio_path: str, line_idx: int, prefix: str = "") -> str:
    """
    Example:
      wav/000000001.wav -> 000000001

    If duplicated names exist, the caller will add a suffix.
    """
    stem = Path(audio_path).stem

    if not stem:
        stem = f"utt_{line_idx:09d}"

    if prefix:
        return f"{prefix}_{stem}"

    return stem


def make_unique_utt_id(base_utt_id: str, seen_utt_ids: set, global_idx: int) -> str:
    if base_utt_id not in seen_utt_ids:
        seen_utt_ids.add(base_utt_id)
        return base_utt_id

    new_utt_id = f"{base_utt_id}_{global_idx:09d}"

    while new_utt_id in seen_utt_ids:
        global_idx += 1
        new_utt_id = f"{base_utt_id}_{global_idx:09d}"

    seen_utt_ids.add(new_utt_id)
    return new_utt_id


def normalize_text(text: str, lowercase: bool = False) -> str:
    text = text.strip()

    # Collapse repeated whitespace.
    text = " ".join(text.split())

    if lowercase:
        text = text.lower()

    return text


def parse_manifest_and_audio_bases(
    manifest_arg: str,
    audio_base_arg: str,
) -> List[Tuple[Path, Path]]:
    manifests = [Path(x) for x in split_colon_arg(manifest_arg, "--manifest")]
    audio_bases = [Path(x) for x in split_colon_arg(audio_base_arg, "--audio-base")]

    if len(audio_bases) == 1 and len(manifests) > 1:
        audio_bases = audio_bases * len(manifests)

    if len(manifests) != len(audio_bases):
        raise ValueError(
            "The number of manifests must match the number of audio bases, "
            "unless only one audio base is provided.\n"
            f"Got {len(manifests)} manifests and {len(audio_bases)} audio bases."
        )

    return list(zip(manifests, audio_bases))


def parse_output_plan(
    out_dir_arg: str,
    lines_arg: str,
    total_items: int,
) -> List[Tuple[Path, int]]:
    out_dirs = [Path(x) for x in split_colon_arg(out_dir_arg, "--out-dir")]

    if lines_arg is None:
        if len(out_dirs) != 1:
            raise ValueError(
                "When using multiple --out-dir values, you must also provide --lines."
            )
        line_specs = ["rest"]
    else:
        line_specs = split_colon_arg(lines_arg, "--lines")

    if len(out_dirs) != len(line_specs):
        raise ValueError(
            "--out-dir and --lines must have the same number of fields.\n"
            f"Got {len(out_dirs)} output dirs and {len(line_specs)} line specs."
        )

    counts: List[int] = []
    rest_index = None
    fixed_sum = 0

    for idx, spec in enumerate(line_specs):
        if spec == "rest":
            if rest_index is not None:
                raise ValueError("Only one `rest` is allowed in --lines.")
            rest_index = idx
            counts.append(-1)
            continue

        try:
            n = int(spec)
        except ValueError:
            raise ValueError(
                f"Invalid --lines value: {spec}. Use an integer or `rest`."
            )

        if n < 0:
            raise ValueError(f"Invalid --lines value: {spec}. It must be >= 0.")

        fixed_sum += n
        counts.append(n)

    if fixed_sum > total_items:
        raise ValueError(
            f"Requested {fixed_sum} fixed lines, but only {total_items} utterances exist."
        )

    if rest_index is not None:
        counts[rest_index] = total_items - fixed_sum
    else:
        unused = total_items - fixed_sum
        if unused > 0:
            print(
                f"[WARN] {unused} utterances will be unused because --lines has no `rest`."
            )

    return list(zip(out_dirs, counts))


def collect_items(
    manifest_audio_pairs: List[Tuple[Path, Path]],
    text_field: str,
    lowercase: bool,
    prefix: str,
    check_wav_exists: bool,
) -> Tuple[List[Dict[str, str]], int]:
    data_items: List[Dict[str, str]] = []
    seen_utt_ids = set()
    num_missing_audio = 0
    global_idx = 0

    for manifest_path, audio_base in manifest_audio_pairs:
        for line_idx, item in read_jsonl(manifest_path):
            rel_audio_path = get_audio_path(item)

            if text_field not in item:
                raise KeyError(
                    f"Missing `{text_field}` field at {manifest_path}, line {line_idx + 1}"
                )

            transcript = normalize_text(str(item[text_field]), lowercase=lowercase)

            audio_path = Path(rel_audio_path)

            if not audio_path.is_absolute():
                audio_path = audio_base / audio_path

            audio_path = audio_path.resolve()

            if check_wav_exists and not audio_path.exists():
                num_missing_audio += 1
                print(f"[WARN] Missing audio: {audio_path}")

            base_utt_id = make_utt_id(rel_audio_path, line_idx, prefix)
            utt_id = make_unique_utt_id(base_utt_id, seen_utt_ids, global_idx)

            data_items.append(
                {
                    "key": utt_id,
                    "wav": str(audio_path),
                    "txt": transcript,
                }
            )

            global_idx += 1

    return data_items, num_missing_audio


def write_wenet_dir(out_dir: Path, items: List[Dict[str, str]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_scp_path = out_dir / "wav.scp"
    text_path = out_dir / "text"
    data_list_path = out_dir / "data.list"

    with wav_scp_path.open("w", encoding="utf-8") as wav_scp, \
         text_path.open("w", encoding="utf-8") as text_file, \
         data_list_path.open("w", encoding="utf-8") as data_list:

        for item in items:
            utt_id = item["key"]
            audio_path = item["wav"]
            transcript = item["txt"]

            wav_scp.write(f"{utt_id} {audio_path}\n")
            text_file.write(f"{utt_id} {transcript}\n")
            data_list.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert one or more NeMo-style JSONL manifests to WeNet raw data format, "
            "then optionally randomly split into several output dirs."
        )
    )

    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help=(
            "Input JSONL manifest(s). Use ':' to merge several files, "
            "e.g. file1.jsonl:file2.jsonl"
        ),
    )

    parser.add_argument(
        "--audio-base",
        type=str,
        required=True,
        help=(
            "Base directory or directories for relative audio paths. "
            "Use ':' to match --manifest, e.g. dir1:dir2. "
            "If only one dir is given, it is used for all manifests."
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help=(
            "Output dir(s). Use ':' to split into several dirs, "
            "e.g. data/train:data/dev"
        ),
    )

    parser.add_argument(
        "--lines",
        type=str,
        default=None,
        help=(
            "Output line counts matching --out-dir. "
            "Use integers or one `rest`, e.g. rest:1000 or 1000:rest. "
            "If omitted, only one --out-dir is allowed and all data goes there."
        ),
    )

    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Transcript field in input JSONL. Default: text.",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional utterance ID prefix.",
    )

    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase transcript text.",
    )

    parser.add_argument(
        "--check-wav-exists",
        action="store_true",
        help="Check whether each audio file exists.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling before split. Default: 42.",
    )

    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Do not shuffle before split.",
    )

    args = parser.parse_args()

    manifest_audio_pairs = parse_manifest_and_audio_bases(
        args.manifest,
        args.audio_base,
    )

    data_items, num_missing_audio = collect_items(
        manifest_audio_pairs=manifest_audio_pairs,
        text_field=args.text_field,
        lowercase=args.lowercase,
        prefix=args.prefix,
        check_wav_exists=args.check_wav_exists,
    )

    if not data_items:
        raise RuntimeError("No valid utterances found.")

    if not args.no_shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(data_items)

    output_plan = parse_output_plan(
        out_dir_arg=args.out_dir,
        lines_arg=args.lines,
        total_items=len(data_items),
    )

    start = 0

    print("Input manifests:")
    for manifest_path, audio_base in manifest_audio_pairs:
        print(f"  manifest={manifest_path}, audio_base={audio_base}")

    print(f"Total utterances collected: {len(data_items)}")
    print(f"Shuffle: {not args.no_shuffle}")
    if not args.no_shuffle:
        print(f"Seed: {args.seed}")

    for out_dir, count in output_plan:
        end = start + count
        split_items = data_items[start:end]
        write_wenet_dir(out_dir, split_items)

        print(f"Written {len(split_items)} utterances to {out_dir}")
        print(f"  wav.scp:   {out_dir / 'wav.scp'}")
        print(f"  text:      {out_dir / 'text'}")
        print(f"  data.list: {out_dir / 'data.list'}")

        start = end

    if args.check_wav_exists:
        print(f"Missing audio files: {num_missing_audio}")


if __name__ == "__main__":
    main()