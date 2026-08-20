import json
import subprocess
import sys
from pathlib import Path

import pytest


TOOLS_DIR = Path(__file__).resolve().parents[2] / "phonemie_tools"
sys.path.insert(0, str(TOOLS_DIR))

import slurm_phoneme_array as slurm_array  # noqa: E402


class FakeSlurm:
    def __init__(self, job_id="12345", shard_count=3, failed_task=None):
        self.job_id = job_id
        self.shard_count = shard_count
        self.failed_task = failed_task
        self.commands = []

    def __call__(self, command, **arguments):
        self.commands.append(command)
        assert arguments["check"] is True
        assert arguments["text"] is True
        assert arguments["capture_output"] is True
        if command[:3] == ["scontrol", "show", "config"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="ClusterName = test\nMaxArraySize = 75000\n",
                stderr="",
            )
        if command[0] == "sbatch":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=f"{self.job_id};test-cluster\n",
                stderr="",
            )
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[0] == "sacct":
            lines = []
            for task_id in range(self.shard_count):
                state = "FAILED" if task_id == self.failed_task else "COMPLETED"
                exit_code = "1:0" if task_id == self.failed_task else "0:0"
                lines.append(f"{self.job_id}_{task_id}|{state}|{exit_code}|")
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="\n".join(lines) + "\n",
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")


@pytest.mark.parametrize(
    "value,shards,parallel,spec",
    [
        ("--array=0-31%8 --cpus-per-task=2 --mem=8G", 32, 8, "0-31%8"),
        ("--array 0-3 --partition cpu", 4, None, "0-3"),
        ("-a0-0", 1, None, "0-0"),
    ],
)
def test_parse_sbatch_args(value, shards, parallel, spec):
    tokens, actual_shards, actual_parallel, actual_spec = (
        slurm_array.parse_sbatch_args(value)
    )
    assert tokens
    assert actual_shards == shards
    assert actual_parallel == parallel
    assert actual_spec == spec


@pytest.mark.parametrize(
    "value,error",
    [
        ("--cpus-per-task=2", "must include"),
        ("--array=1-4", "form 0-N"),
        ("--array=0,2", "form 0-N"),
        ("--array=0-3 --output=custom.log", "managed"),
        ("--array=0-3 -emy.err", "conflicts"),
        ("--array=0-3 --wait", "managed"),
    ],
)
def test_parse_sbatch_args_rejects_unsafe_or_unsupported_values(value, error):
    with pytest.raises(ValueError, match=error):
        slurm_array.parse_sbatch_args(value)


def test_validate_and_split_manifest_balances_contiguous_records(tmp_path):
    input_path = tmp_path / "input.jsonl"
    records = [
        {"audio_filepath": f"/{index}.wav", "text": str(index), "duration": 1.0}
        for index in range(7)
    ]
    input_path.write_text(
        "\n\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    assert slurm_array.validate_manifest(input_path, "text") == 7
    sizes = slurm_array.balanced_shard_sizes(7, 3)
    assert sizes == [3, 2, 2]
    slurm_array.split_manifest(input_path, tmp_path / "inputs", sizes)

    shard_texts = []
    for task_id in range(3):
        path = tmp_path / "inputs" / f"part-{task_id:05d}.jsonl"
        shard_texts.append(
            [json.loads(line)["text"] for line in path.read_text().splitlines()]
        )
    assert shard_texts == [["0", "1", "2"], ["3", "4"], ["5", "6"]]


def test_prepare_workers_and_finalize_preserve_order(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    records = [
        {
            "audio_filepath": f"/{index}.wav",
            "text": f"text-{index}",
            "duration": 1.0,
            "index": index,
        }
        for index in range(5)
    ]
    input_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    fake_slurm = FakeSlurm(shard_count=3)

    submitted = slurm_array.prepare_slurm_run(
        input_path=input_path,
        output_path=output_path,
        text_key="text",
        conversion={"text_key": "text"},
        sbatch_args="--array=0-2%2 --cpus-per-task=2 --mem=8G",
        requested_run_dir=tmp_path / "run",
        python_executable=Path(sys.executable),
        script_path=TOOLS_DIR / "nemo_jsonl_to_phonemes.py",
        base_prefix=Path(sys.base_prefix),
        run_command=fake_slurm,
    )
    run_dir = submitted["run_dir"]
    assert submitted["job_id"] == "12345"
    sbatch_command = next(command for command in fake_slurm.commands if command[0] == "sbatch")
    assert "--array=0-2%2" in sbatch_command
    assert any("task-%A_%a.out" in argument for argument in sbatch_command)
    assert any("task-%A_%a.err" in argument for argument in sbatch_command)
    subprocess.run(
        ["bash", "-n", str(run_dir / "run-array-task.sh")],
        check=True,
    )

    def fake_convert_manifest(**arguments):
        source_records = [
            json.loads(line)
            for line in arguments["input_path"].read_text().splitlines()
        ]
        with arguments["output_path"].open("w", encoding="utf-8") as output_file:
            for record in source_records:
                record["text"] += "-ipa"
                output_file.write(json.dumps(record) + "\n")
        return len(source_records)

    for task_id in range(3):
        slurm_array.run_array_worker(run_dir, task_id, fake_convert_manifest)

    merged_path = slurm_array.finalize_slurm_run(
        run_dir,
        poll_interval=0,
        run_command=fake_slurm,
    )
    merged = [json.loads(line) for line in merged_path.read_text().splitlines()]
    assert [record["index"] for record in merged] == list(range(5))
    assert [record["text"] for record in merged] == [
        f"text-{index}-ipa" for index in range(5)
    ]
    assert all(
        (run_dir / "status" / f"task-{task_id:05d}.json").is_file()
        for task_id in range(3)
    )


def test_failed_array_does_not_replace_final_output(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "status").mkdir(parents=True)
    (run_dir / "logs").mkdir()
    slurm_array.atomic_write_json(
        run_dir / "run.json",
        {
            "schema_version": 1,
            "output_jsonl": str(tmp_path / "output.jsonl"),
            "total_records": 2,
            "shard_count": 2,
            "shard_sizes": [1, 1],
        },
    )
    slurm_array.atomic_write_json(
        run_dir / "submission.json",
        {"job_id": "12345"},
    )
    for task_id, state in enumerate(("COMPLETED", "FAILED")):
        slurm_array.atomic_write_json(
            run_dir / "status" / f"task-{task_id:05d}.json",
            {"task_id": task_id, "state": state},
        )
    fake_slurm = FakeSlurm(shard_count=2, failed_task=1)

    with pytest.raises(RuntimeError, match="task 1"):
        slurm_array.finalize_slurm_run(
            run_dir,
            poll_interval=0,
            run_command=fake_slurm,
        )
    assert not (tmp_path / "output.jsonl").exists()


def test_empty_manifest_writes_output_without_sbatch(tmp_path):
    input_path = tmp_path / "empty.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text("\n\n", encoding="utf-8")
    fake_slurm = FakeSlurm(shard_count=2)

    result = slurm_array.prepare_slurm_run(
        input_path=input_path,
        output_path=output_path,
        text_key="text",
        conversion={},
        sbatch_args="--array=0-1",
        requested_run_dir=None,
        python_executable=Path(sys.executable),
        script_path=TOOLS_DIR / "nemo_jsonl_to_phonemes.py",
        base_prefix=Path(sys.base_prefix),
        run_command=fake_slurm,
    )

    assert result["empty"] is True
    assert output_path.read_text(encoding="utf-8") == ""
    assert not fake_slurm.commands


def test_slurm_mode_rejects_same_input_and_output(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text(
        '{"audio_filepath":"/one.wav","text":"中国","duration":1.0}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be different"):
        slurm_array.prepare_slurm_run(
            input_path=path,
            output_path=path,
            text_key="text",
            conversion={},
            sbatch_args="--array=0-0",
            requested_run_dir=None,
            python_executable=Path(sys.executable),
            script_path=TOOLS_DIR / "nemo_jsonl_to_phonemes.py",
            base_prefix=Path(sys.base_prefix),
            run_command=FakeSlurm(shard_count=1),
        )
