#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
import wave


SCRIPT = Path(__file__).with_name("test_onnx.py")
SPEC = importlib.util.spec_from_file_location("test_onnx", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TestScoring(unittest.TestCase):
    def test_edit_counts(self):
        deletion, insertion, substitution = MODULE.edit_counts(
            "a b c d".split(), "a x c d e".split()
        )
        self.assertEqual((deletion, insertion, substitution), (0, 1, 1))

    def test_cjk_character_tokens_keep_latin_word(self):
        self.assertEqual(MODULE.char_tokens("你 好 ASR!"), ["你", "好", "ASR"])

    def test_ctc_greedy_collapse_across_blanks(self):
        self.assertEqual(
            MODULE.ctc_greedy_ids([0, 3, 3, 0, 3, 4, 4, 0], blank_id=0),
            [3, 3, 4],
        )

    def test_int8_model_contract_name(self):
        model = Path("/tmp/stage1-wuw.int8.onnx")
        self.assertEqual(
            MODULE.default_contract_path(model),
            Path("/tmp/stage1-wuw.contract.json"),
        )


class TestEndToEnd(unittest.TestCase):
    def test_fake_sdk_decoder_outputs_requested_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            data = root / "data"
            output = root / "output"
            model.mkdir()
            data.mkdir()
            for name in ("encoder.onnx", "ctc.onnx", "decoder.onnx"):
                (model / name).write_bytes(b"fake test graph")
            (model / "units.txt").write_text(
                "<blank> 0\n<unk> 1\n", encoding="utf-8"
            )

            wav_paths = []
            for index in range(2):
                wav_path = root / f"{index}.wav"
                with wave.open(str(wav_path), "wb") as stream:
                    stream.setnchannels(1)
                    stream.setsampwidth(2)
                    stream.setframerate(16000)
                    stream.writeframes(b"\0\0" * 1600)
                wav_paths.append(wav_path)
            (data / "wav.scp").write_text(
                f"utt1 {wav_paths[0]}\nutt2 {wav_paths[1]}\n", encoding="utf-8"
            )
            (data / "text").write_text(
                "utt1 hello world\nutt2 good day\n", encoding="utf-8"
            )

            decoder = root / "asr_batch_decode"
            decoder.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib, sys\n"
                "args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
                "pathlib.Path(args['--result']).write_text("
                "'utt1 hello world\\nutt2 bad day\\n', encoding='utf-8')\n"
                "print('audio_sec 0.2', file=sys.stderr)\n"
                "print('wall_sec 0.1', file=sys.stderr)\n"
                "print('rtf 0.5', file=sys.stderr)\n",
                encoding="utf-8",
            )
            decoder.chmod(decoder.stat().st_mode | stat.S_IXUSR)

            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--model-dir", str(model),
                 "--data-dir", str(data), "--output-dir", str(output),
                 "--decoder-bin", str(decoder), "--decoder-kind", "sdk"],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            generated_manifest = json.loads(
                (output / "runtime_model" / "sdk_model.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(generated_manifest["postprocess"]["language_type"], "en")
            rows = [json.loads(line) for line in
                    (output / "results.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["wav"], str(wav_paths[0]))
            self.assertEqual(rows[1]["hyp"], "bad day")
            summary = (output / "summary.txt").read_text(encoding="utf-8")
            self.assertIn("DEL: 0\nINS: 0\nSUB: 1\n", summary)
            self.assertIn("WER: 25.00%", summary)
            self.assertIn("SER: 50.00%", summary)
            self.assertIn("RTF: 0.5000", summary)
            self.assertIn(f"wav: {wav_paths[1]}\nhyp: bad day\nref: good day\n\n", summary)


if __name__ == "__main__":
    unittest.main()
