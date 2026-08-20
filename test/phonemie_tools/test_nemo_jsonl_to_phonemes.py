import json
import os
import sys
import types
from pathlib import Path

import pytest


TOOLS_DIR = Path(__file__).resolve().parents[2] / "phonemie_tools"
sys.path.insert(0, str(TOOLS_DIR))

import nemo_jsonl_to_phonemes as converter  # noqa: E402


class ImmediateFuture:
    def __init__(self, function, *args):
        try:
            self.value = function(*args)
            self.error = None
        except Exception as error:  # pragma: no cover - asserted via result()
            self.value = None
            self.error = error

    def result(self):
        if self.error is not None:
            raise self.error
        return self.value


class InlineExecutor:
    def __init__(self, max_workers, initializer, initargs):
        self.max_workers = max_workers
        initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def submit(self, function, *args):
        return ImmediateFuture(function, *args)


def install_fake_g2p_mix(monkeypatch, outputs=None, failure_text=None):
    calls = []
    outputs = outputs or {}

    class FakeG2P:
        def __init__(self, **arguments):
            calls.append(arguments)
            self.output = arguments["output"]

        def __call__(self, text):
            if text == failure_text:
                raise ValueError("unsupported transcript")
            phones = outputs.get(text, ())
            return types.SimpleNamespace(phones=phones)

    class FakeG2PWBackend:
        name = "g2pw"

        def __init__(self, **arguments):
            self.arguments = arguments

        def _convert(self, text):
            return ()

    module = types.ModuleType("g2p_mix")
    module.G2P = FakeG2P
    backends_module = types.ModuleType("g2p_mix.backends")
    backends_module.G2PWBackend = FakeG2PWBackend
    monkeypatch.setitem(sys.modules, "g2p_mix", module)
    monkeypatch.setitem(sys.modules, "g2p_mix.backends", backends_module)
    return calls


@pytest.fixture(autouse=True)
def reset_worker_state(monkeypatch):
    monkeypatch.setattr(converter, "_worker_backend", converter.DEFAULT_BACKEND)
    monkeypatch.setattr(converter, "_worker_g2p", None)


def test_g2pw_worker_uses_surface_tone_ipa_configuration(monkeypatch):
    calls = install_fake_g2p_mix(
        monkeypatch,
        outputs={
            "中国 idea": ("ʈ͡ʂ", "ʊ", "ŋ˥˥", "k", "w", "o˧˥", "a", "ɪ", "d", "ˈi", "ə"),
        },
    )

    converter.initialize_worker("unused", "unused", "▁", backend="g2pw")
    result = converter.phonemize_batch(["中国 idea", ""])

    assert len(calls) == 1
    assert calls[0]["backend"].name == "g2pw"
    assert calls[0]["backend"].arguments == {"unknown_policy": "strict"}
    assert {key: value for key, value in calls[0].items() if key != "backend"} == {
        "mode": "mandarin",
        "output": "ipa",
        "unknown": "strict",
        "tone_sandhi": True,
    }
    assert result == ["ʈ͡ʂ ʊ ŋ˥˥ k w o˧˥ a ɪ d ˈi ə", ""]


def test_g2pw_override_mapping_is_character_specific(tmp_path):
    mapping_path = tmp_path / "overrides.json"
    mapping_path.write_text(
        '{"崖":{"yai2":"ya2"}}\n',
        encoding="utf-8",
    )

    overrides = converter.load_g2pw_pinyin_overrides(mapping_path)

    assert converter.apply_g2pw_pinyin_overrides(
        "悬崖",
        ("xuan2", "yai2"),
        overrides,
    ) == ("xuan2", "ya2")
    assert converter.apply_g2pw_pinyin_overrides(
        "呀",
        ("yai2",),
        overrides,
    ) == ("yai2",)


@pytest.mark.parametrize(
    "contents,error",
    [
        ("[]", "must contain a JSON object"),
        ('{"悬崖":{"yai2":"ya2"}}', "must be one character"),
        ('{"崖":{"yai2":"yai2"}}', "distinct, non-empty"),
    ],
)
def test_g2pw_override_mapping_rejects_invalid_entries(
    tmp_path,
    contents,
    error,
):
    mapping_path = tmp_path / "overrides.json"
    mapping_path.write_text(contents, encoding="utf-8")

    with pytest.raises(RuntimeError, match=error):
        converter.load_g2pw_pinyin_overrides(mapping_path)


def test_g2pw_rejects_invalid_phone_sequences(monkeypatch):
    install_fake_g2p_mix(monkeypatch, outputs={"中国": ("bad phone",)})
    converter.initialize_worker("unused", "unused", None, backend="g2pw")

    with pytest.raises(RuntimeError, match="phone containing whitespace"):
        converter.phonemize_batch(["中国"])


def test_g2pw_manifest_preserves_jsonl_fields_and_order(tmp_path, monkeypatch):
    install_fake_g2p_mix(
        monkeypatch,
        outputs={
            "中国": ("ʈ͡ʂ", "ʊ", "ŋ˥˥", "k", "w", "o˧˥"),
            "银行": ("i", "n˧˥", "x", "a", "ŋ˧˥"),
        },
    )
    monkeypatch.setattr(
        converter.concurrent.futures,
        "ProcessPoolExecutor",
        InlineExecutor,
    )
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    records = [
        {
            "audio_filepath": "/data/one.wav",
            "text": "中国",
            "duration": 1.234,
            "lang": "zh",
        },
        {
            "audio_filepath": "/data/two.wav",
            "text": "",
            "duration": 0.5,
        },
        {
            "audio_filepath": "/data/three.wav",
            "text": "银行",
            "duration": 2.0,
            "speaker": "speaker-1",
        },
    ]
    input_path.write_text(
        "\n\n".join(json.dumps(record, ensure_ascii=False) for record in records)
        + "\n",
        encoding="utf-8",
    )

    count = converter.convert_manifest(
        input_path=input_path,
        output_path=output_path,
        espeak_library="unused",
        language="",
        word_token="invalid token is ignored",
        text_key="text",
        workers=1,
        batch_size=2,
        pending_batches=2,
        progress_every=0,
        backend="g2pw",
    )

    output_records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert count == 3
    assert [record["text"] for record in output_records] == [
        "ʈ͡ʂ ʊ ŋ˥˥ k w o˧˥",
        "",
        "i n˧˥ x a ŋ˧˥",
    ]
    for source, output in zip(records, output_records):
        assert {key: value for key, value in output.items() if key != "text"} == {
            key: value for key, value in source.items() if key != "text"
        }


def test_g2pw_failure_reports_jsonl_line_range(tmp_path, monkeypatch):
    install_fake_g2p_mix(monkeypatch, failure_text="坏")
    monkeypatch.setattr(
        converter.concurrent.futures,
        "ProcessPoolExecutor",
        InlineExecutor,
    )
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        '{"audio_filepath":"/one.wav","text":"好","duration":1.0}\n'
        '{"audio_filepath":"/two.wav","text":"坏","duration":1.0}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"input lines 1-2: g2p-mix failed for batch item 2: "
            r"unsupported transcript; text='坏'"
        ),
    ):
        converter.convert_manifest(
            input_path=input_path,
            output_path=tmp_path / "output.jsonl",
            espeak_library="unused",
            language="unused",
            word_token=None,
            text_key="text",
            workers=1,
            batch_size=2,
            pending_batches=1,
            progress_every=0,
            backend="g2pw",
        )


def test_parallel_g2pw_warms_shared_resources_before_workers(
    tmp_path,
    monkeypatch,
):
    events = []
    install_fake_g2p_mix(monkeypatch, outputs={"中国": ("ipa",)})

    def fake_warm_up():
        events.append("warm-up")

    class RecordingExecutor(InlineExecutor):
        def __init__(self, max_workers, initializer, initargs):
            events.append("executor")
            super().__init__(max_workers, initializer, initargs)

    monkeypatch.setattr(converter, "warm_up_g2pw_resources", fake_warm_up)
    monkeypatch.setattr(
        converter.concurrent.futures,
        "ProcessPoolExecutor",
        RecordingExecutor,
    )
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text(
        '{"audio_filepath":"/one.wav","text":"中国","duration":1.0}\n',
        encoding="utf-8",
    )

    converter.convert_manifest(
        input_path=input_path,
        output_path=output_path,
        espeak_library="unused",
        language="unused",
        word_token=None,
        text_key="text",
        workers=4,
        batch_size=1,
        pending_batches=1,
        progress_every=0,
        backend="g2pw",
    )

    assert events == ["warm-up", "executor"]


def test_missing_or_old_g2p_mix_has_install_guidance(monkeypatch):
    old_module = types.ModuleType("g2p_mix")
    monkeypatch.setitem(sys.modules, "g2p_mix", old_module)

    with pytest.raises(RuntimeError, match="requirements-g2pw.txt"):
        converter.create_g2pw_converter()


def test_cli_uses_one_g2pw_worker_by_default(tmp_path, monkeypatch):
    captured = {}

    def fake_convert_manifest(**arguments):
        captured.update(arguments)
        return 0

    monkeypatch.setattr(converter, "convert_manifest", fake_convert_manifest)

    status = converter.main([
        str(tmp_path / "input.jsonl"),
        str(tmp_path / "output.jsonl"),
        "--backend",
        "g2pw",
    ])

    assert status == 0
    assert captured["workers"] == 1
    assert captured["pending_batches"] == 2


@pytest.fixture(scope="module")
def real_g2pw():
    if os.environ.get("WENET_TEST_G2PW") != "1":
        pytest.skip(
            "set WENET_TEST_G2PW=1 to run the real G2PW model regression"
        )
    return converter.create_g2pw_converter()


@pytest.mark.parametrize(
    "text,target,source_phones,tone",
    [
        ("银行办理业务", "行", ("h", "ang"), "2"),
        ("他行走在路上", "行", ("x", "ing"), "2"),
        ("重庆市", "重", ("ch", "ong"), "2"),
        ("这个物体的重量很大", "重", ("zh", "ong"), "4"),
        ("我喜欢听音乐", "乐", ("y", "ue"), "4"),
        ("他今天很快乐", "乐", ("l", "e"), "4"),
        ("孩子已经长大了", "长", ("zh", "ang"), "3"),
        ("这条路很长", "长", ("ch", "ang"), "2"),
        ("请按时还款", "还", ("h", "uan"), "2"),
        ("他还有一本书", "还", ("h", "ai"), "2"),
    ],
)
def test_real_g2pw_polyphone_context(
    real_g2pw,
    text,
    target,
    source_phones,
    tone,
):
    result = real_g2pw(text)
    matching_units = [unit for unit in result.units if unit.text == target]

    assert len(matching_units) == 1
    assert matching_units[0].source_phones == source_phones
    assert matching_units[0].tone == tone
