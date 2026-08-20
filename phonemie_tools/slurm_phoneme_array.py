"""Slurm array orchestration for JSONL phoneme conversion."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple


RUN_SCHEMA_VERSION = 1
ARRAY_PATTERN = re.compile(r"^0-(\d+)(?:%(\d+))?$")
JOB_ID_PATTERN = re.compile(r"^(\d+)(?:;([^\s;]+))?$")
RESERVED_LONG_OPTIONS = {
    "--chdir",
    "--error",
    "--output",
    "--parsable",
    "--wait",
    "--wrap",
}
RESERVED_SHORT_OPTIONS = {"-D", "-e", "-o", "-W"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write a JSON object atomically beside its final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2, sort_keys=True)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, path)


def load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as input_file:
            value = json.load(input_file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} {path} must contain a JSON object")
    return value


def _take_option_value(
    tokens: Sequence[str],
    index: int,
    option: str,
) -> Tuple[str, int]:
    token = tokens[index]
    if "=" in token:
        return token.split("=", 1)[1], index + 1
    if token.startswith("-a") and token != "-a":
        return token[2:], index + 1
    if index + 1 >= len(tokens):
        raise ValueError(f"{option} requires a value")
    return tokens[index + 1], index + 2


def parse_sbatch_args(value: str) -> Tuple[list[str], int, Optional[int], str]:
    """Parse safe sbatch arguments and derive the zero-based array layout."""
    try:
        tokens = shlex.split(value)
    except ValueError as error:
        raise ValueError(f"invalid --sbatch-args quoting: {error}") from error
    if not tokens:
        raise ValueError("--sbatch-args must not be empty")

    array_spec: Optional[str] = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        option = token.split("=", 1)[0]
        if option in RESERVED_LONG_OPTIONS or option in RESERVED_SHORT_OPTIONS:
            raise ValueError(
                f"{option} is managed by the phoneme Slurm controller and "
                "must not be included in --sbatch-args"
            )
        if any(
            token.startswith(prefix) and token != prefix
            for prefix in ("-D", "-e", "-o", "-W")
        ):
            raise ValueError(
                f"{token!r} conflicts with controller-managed sbatch options"
            )

        if option in {"--array", "-a"} or token.startswith("-a"):
            if array_spec is not None:
                raise ValueError("--sbatch-args must contain exactly one array option")
            array_spec, index = _take_option_value(tokens, index, option)
            continue
        index += 1

    if array_spec is None:
        raise ValueError(
            "--sbatch-args must include a zero-based contiguous array such as "
            "--array=0-31%8"
        )
    match = ARRAY_PATTERN.fullmatch(array_spec)
    if match is None:
        raise ValueError(
            "Slurm array must use the form 0-N or 0-N%M; lists, steps, and "
            "nonzero starting indices are unsupported"
        )
    last_task = int(match.group(1))
    shard_count = last_task + 1
    max_parallel = int(match.group(2)) if match.group(2) else None
    if max_parallel is not None and max_parallel < 1:
        raise ValueError("Slurm array concurrency must be at least 1")
    return tokens, shard_count, max_parallel, array_spec


def get_slurm_max_array_size(
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int:
    result = run_command(
        ["scontrol", "show", "config"],
        check=True,
        text=True,
        capture_output=True,
    )
    match = re.search(
        r"(?m)^\s*MaxArraySize\s*=\s*(\d+)\s*$",
        result.stdout,
    )
    if match is None:
        raise RuntimeError("scontrol output did not contain MaxArraySize")
    return int(match.group(1))


def validate_manifest(input_path: Path, text_key: str) -> int:
    """Validate the JSONL records before any Slurm job is submitted."""
    record_count = 0
    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object")
            if text_key not in record:
                raise ValueError(f"Line {line_number} has no {text_key!r} field")
            if not isinstance(record[text_key], str):
                raise ValueError(
                    f"Line {line_number} field {text_key!r} must be a string"
                )
            record_count += 1
    return record_count


def balanced_shard_sizes(total_records: int, shard_count: int) -> list[int]:
    if shard_count < 1:
        raise ValueError("shard count must be at least 1")
    if total_records < shard_count:
        raise ValueError(
            f"Slurm array has {shard_count} tasks but the input contains only "
            f"{total_records} records"
        )
    base, extra = divmod(total_records, shard_count)
    return [base + (1 if task_id < extra else 0) for task_id in range(shard_count)]


def split_manifest(
    input_path: Path,
    inputs_dir: Path,
    shard_sizes: Sequence[int],
) -> None:
    """Write balanced contiguous input shards while omitting blank lines."""
    inputs_dir.mkdir(parents=True, exist_ok=True)
    task_id = 0
    records_in_task = 0
    output_file = None
    try:
        with input_path.open("r", encoding="utf-8") as input_file:
            for raw_line in input_file:
                if not raw_line.strip():
                    continue
                while task_id < len(shard_sizes) and records_in_task >= shard_sizes[task_id]:
                    if output_file is not None:
                        output_file.close()
                    task_id += 1
                    records_in_task = 0
                    output_file = None
                if task_id >= len(shard_sizes):
                    raise RuntimeError("input record count changed while creating shards")
                if output_file is None:
                    output_file = (inputs_dir / f"part-{task_id:05d}.jsonl").open(
                        "w", encoding="utf-8"
                    )
                output_file.write(raw_line.rstrip("\r\n"))
                output_file.write("\n")
                records_in_task += 1
    finally:
        if output_file is not None:
            output_file.close()

    if task_id != len(shard_sizes) - 1 or records_in_task != shard_sizes[-1]:
        raise RuntimeError("input record count changed while creating shards")


def create_run_directory(output_path: Path, requested: Optional[Path]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if requested is not None:
        run_dir = requested.resolve()
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise ValueError(f"--slurm-work-dir already exists: {run_dir}") from error
        return run_dir
    return Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.slurm-",
            dir=output_path.parent,
        )
    ).resolve()


def write_worker_script(
    run_dir: Path,
    python_executable: Path,
    script_path: Path,
    base_prefix: Path,
) -> Path:
    script = run_dir / "run-array-task.sh"
    base_library = base_prefix / "lib"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    if base_library.is_dir():
        quoted_library = shlex.quote(str(base_library))
        lines.append(
            f"export LD_LIBRARY_PATH={quoted_library}${{LD_LIBRARY_PATH:+:${{LD_LIBRARY_PATH}}}}"
        )
    command = " ".join(
        shlex.quote(str(value))
        for value in (
            python_executable,
            script_path,
            "--slurm-worker",
            run_dir,
        )
    )
    lines.append(f"exec {command}")
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script.chmod(0o755)
    return script


def submit_array(
    run_dir: Path,
    sbatch_tokens: Sequence[str],
    worker_script: Path,
    output_name: str,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Tuple[str, Optional[str]]:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    has_job_name = any(
        token == "-J"
        or token.startswith("-J")
        or token == "--job-name"
        or token.startswith("--job-name=")
        for token in sbatch_tokens
    )
    command = ["sbatch", *sbatch_tokens]
    if not has_job_name:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", output_name)[:80]
        command.append(f"--job-name=phoneme-{safe_name}")
    command.extend(
        [
            "--parsable",
            f"--chdir={run_dir}",
            f"--output={logs_dir}/task-%A_%a.out",
            f"--error={logs_dir}/task-%A_%a.err",
            str(worker_script),
        ]
    )
    try:
        result = run_command(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or str(error)).strip()
        raise RuntimeError(f"sbatch submission failed: {detail}") from error
    submitted = result.stdout.strip()
    match = JOB_ID_PATTERN.fullmatch(submitted)
    if match is None:
        raise RuntimeError(f"unexpected sbatch --parsable output: {submitted!r}")
    return match.group(1), match.group(2)


def prepare_slurm_run(
    *,
    input_path: Path,
    output_path: Path,
    text_key: str,
    conversion: Mapping[str, Any],
    sbatch_args: str,
    requested_run_dir: Optional[Path],
    python_executable: Path,
    script_path: Path,
    base_prefix: Path,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output JSONL paths must be different")
    sbatch_tokens, shard_count, max_parallel, array_spec = parse_sbatch_args(
        sbatch_args
    )
    total_records = validate_manifest(input_path, text_key)
    if total_records == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output_path.with_name(
            f".{output_path.name}.{os.getpid()}.tmp"
        )
        temporary_output.write_text("", encoding="utf-8")
        os.replace(temporary_output, output_path)
        return {"empty": True, "total_records": 0}

    max_array_size = get_slurm_max_array_size(run_command)
    if shard_count > max_array_size:
        raise ValueError(
            f"Slurm array requests {shard_count} tasks but MaxArraySize is "
            f"{max_array_size}"
        )
    shard_sizes = balanced_shard_sizes(total_records, shard_count)
    run_dir = create_run_directory(output_path, requested_run_dir)
    for directory in ("inputs", "outputs", "logs", "status"):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    split_manifest(input_path, run_dir / "inputs", shard_sizes)

    run_manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": utc_now(),
        "input_jsonl": str(input_path.resolve()),
        "output_jsonl": str(output_path.resolve()),
        "python_executable": str(python_executable.resolve()),
        "script_path": str(script_path.resolve()),
        "text_key": text_key,
        "total_records": total_records,
        "shard_count": shard_count,
        "shard_sizes": shard_sizes,
        "array_spec": array_spec,
        "max_parallel": max_parallel,
        "sbatch_tokens": list(sbatch_tokens),
        "conversion": dict(conversion),
    }
    atomic_write_json(run_dir / "run.json", run_manifest)
    worker_script = write_worker_script(
        run_dir,
        python_executable,
        script_path,
        base_prefix,
    )
    job_id, cluster = submit_array(
        run_dir,
        sbatch_tokens,
        worker_script,
        output_path.name,
        run_command,
    )
    submission = {
        "submitted_at": utc_now(),
        "job_id": job_id,
        "cluster": cluster,
    }
    atomic_write_json(run_dir / "submission.json", submission)
    return {
        "empty": False,
        "job_id": job_id,
        "cluster": cluster,
        "run_dir": run_dir,
        "total_records": total_records,
        "shard_count": shard_count,
    }


def _task_status_path(run_dir: Path, task_id: int) -> Path:
    return run_dir / "status" / f"task-{task_id:05d}.json"


def run_array_worker(
    run_dir: Path,
    task_id: int,
    convert_manifest: Callable[..., int],
) -> int:
    """Run one array task and atomically publish its shard output/status."""
    run_dir = run_dir.resolve()
    run = load_json_object(run_dir / "run.json", "Slurm run manifest")
    shard_count = int(run["shard_count"])
    if task_id < 0 or task_id >= shard_count:
        raise ValueError(
            f"SLURM_ARRAY_TASK_ID {task_id} is outside 0-{shard_count - 1}"
        )
    expected_records = int(run["shard_sizes"][task_id])
    input_path = run_dir / "inputs" / f"part-{task_id:05d}.jsonl"
    output_path = run_dir / "outputs" / f"part-{task_id:05d}.jsonl"
    temporary_output = output_path.with_name(f".{output_path.name}.tmp")
    started = time.monotonic()
    status = {
        "task_id": task_id,
        "started_at": utc_now(),
        "expected_records": expected_records,
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
    }
    try:
        conversion = dict(run["conversion"])
        conversion["input_path"] = input_path
        conversion["output_path"] = temporary_output
        records_written = convert_manifest(**conversion)
        if records_written != expected_records:
            raise RuntimeError(
                f"worker wrote {records_written} records; expected "
                f"{expected_records}"
            )
        os.replace(temporary_output, output_path)
        status.update(
            {
                "state": "COMPLETED",
                "records_written": records_written,
                "finished_at": utc_now(),
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        atomic_write_json(_task_status_path(run_dir, task_id), status)
        return records_written
    except BaseException as error:
        status.update(
            {
                "state": "FAILED",
                "error": str(error),
                "finished_at": utc_now(),
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        atomic_write_json(_task_status_path(run_dir, task_id), status)
        raise


def _completed_status_count(run_dir: Path, shard_count: int) -> int:
    completed = 0
    for task_id in range(shard_count):
        path = _task_status_path(run_dir, task_id)
        if not path.is_file():
            continue
        try:
            status = load_json_object(path, "task status")
        except RuntimeError:
            continue
        if status.get("state") == "COMPLETED":
            completed += 1
    return completed


def wait_for_array(
    run_dir: Path,
    job_id: str,
    shard_count: int,
    poll_interval: float = 30.0,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    started = time.monotonic()
    last_completed = -1
    while True:
        result = run_command(
            ["squeue", "-h", "-j", job_id, "-o", "%i|%T"],
            check=True,
            text=True,
            capture_output=True,
        )
        completed = _completed_status_count(run_dir, shard_count)
        if completed != last_completed or result.stdout.strip():
            elapsed = max(time.monotonic() - started, 0.0)
            message = (
                f"Slurm array {job_id}: {completed}/{shard_count} tasks "
                f"completed; elapsed {elapsed:.0f}s"
            )
            if completed > 0 and elapsed > 0:
                eta = (shard_count - completed) / (completed / elapsed)
                message += f"; approximate ETA {eta:.0f}s"
            print(message, flush=True)
            last_completed = completed
        if not result.stdout.strip():
            return
        time.sleep(poll_interval)


def read_sacct_tasks(
    job_id: str,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[int, Tuple[str, str]]:
    result = run_command(
        [
            "sacct",
            "-n",
            "-X",
            "-j",
            job_id,
            "--format=JobIDRaw,State,ExitCode",
            "--parsable2",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    tasks: dict[int, Tuple[str, str]] = {}
    prefix = f"{job_id}_"
    for raw_line in result.stdout.splitlines():
        fields = raw_line.strip().split("|")
        if len(fields) < 3 or not fields[0].startswith(prefix):
            continue
        suffix = fields[0][len(prefix) :]
        if suffix.isdigit():
            tasks[int(suffix)] = (fields[1].rstrip("+"), fields[2])
    return tasks


def verify_array(
    run_dir: Path,
    job_id: str,
    shard_count: int,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    tasks: dict[int, Tuple[str, str]] = {}
    for attempt in range(6):
        tasks = read_sacct_tasks(job_id, run_command)
        if len(tasks) >= shard_count:
            break
        if attempt < 5:
            time.sleep(2)

    failures = []
    for task_id in range(shard_count):
        state, exit_code = tasks.get(task_id, ("MISSING", "unknown"))
        status_path = _task_status_path(run_dir, task_id)
        status_state = "MISSING"
        if status_path.is_file():
            status_state = load_json_object(status_path, "task status").get(
                "state", "INVALID"
            )
        if state != "COMPLETED" or exit_code != "0:0" or status_state != "COMPLETED":
            failures.append(
                f"task {task_id}: Slurm={state}/{exit_code}, status={status_state}, "
                f"logs={run_dir / 'logs' / f'task-{job_id}_{task_id}.out'} and "
                f"{run_dir / 'logs' / f'task-{job_id}_{task_id}.err'}"
            )
    if failures:
        raise RuntimeError(
            "Slurm array did not complete successfully; final output was not "
            "merged:\n" + "\n".join(failures)
        )


def count_nonblank_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as input_file:
        return sum(1 for line in input_file if line.strip())


def merge_outputs(run_dir: Path, run: Mapping[str, Any]) -> Path:
    output_path = Path(str(run["output_jsonl"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    merged_records = 0
    try:
        with temporary_file:
            for task_id, expected_records in enumerate(run["shard_sizes"]):
                shard_path = run_dir / "outputs" / f"part-{task_id:05d}.jsonl"
                if not shard_path.is_file():
                    raise RuntimeError(f"missing shard output: {shard_path}")
                actual_records = count_nonblank_lines(shard_path)
                if actual_records != int(expected_records):
                    raise RuntimeError(
                        f"shard {task_id} contains {actual_records} records; "
                        f"expected {expected_records}"
                    )
                with shard_path.open("r", encoding="utf-8") as shard_file:
                    shutil.copyfileobj(shard_file, temporary_file)
                merged_records += actual_records
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        if merged_records != int(run["total_records"]):
            raise RuntimeError(
                f"merged {merged_records} records; expected {run['total_records']}"
            )
        os.replace(temporary_path, output_path)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return output_path


def finalize_slurm_run(
    run_dir: Path,
    *,
    poll_interval: float = 30.0,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path:
    run_dir = run_dir.resolve()
    run = load_json_object(run_dir / "run.json", "Slurm run manifest")
    if run.get("schema_version") != RUN_SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported Slurm run schema: {run.get('schema_version')!r}"
        )
    submission = load_json_object(
        run_dir / "submission.json", "Slurm submission metadata"
    )
    job_id = str(submission["job_id"])
    shard_count = int(run["shard_count"])
    wait_for_array(run_dir, job_id, shard_count, poll_interval, run_command)
    verify_array(run_dir, job_id, shard_count, run_command)
    return merge_outputs(run_dir, run)


def finalize_command(python_executable: Path, script_path: Path, run_dir: Path) -> str:
    return " ".join(
        shlex.quote(str(value))
        for value in (
            python_executable,
            script_path,
            "--slurm-finalize",
            run_dir,
        )
    )
