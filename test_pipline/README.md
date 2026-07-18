# WeNet ONNX Test Pipeline

This tool decodes a WeNet-format test data directory and creates:

- `results.jsonl`: one `{key, wav, ref, hyp}` object per utterance. `wav` is an
  absolute path.
- `summary.txt`: DEL, INS, SUB, WER, SER, RTF, followed by every wrong case in
  `wav` / `hyp` / `ref` format and an empty line between cases.
- `hyp.txt`, `decoder.log`, and `input.wav.scp`: intermediate files retained for
  debugging and reproducibility.

## Input layout

The preferred test directory is the normal WeNet/Kaldi layout:

```text
test_data/
  wav.scp   # utterance_id /absolute/or/relative/path.wav
  text      # utterance_id reference transcription
```

A raw JSON `data.list` containing `key`, `wav`, and `txt` is accepted when
`wav.scp` is absent. A shard `data.list` alone is not enough because it does not
provide directly readable WAV paths.

The model can be either:

1. A single streaming WeNet CTC model that exposes `ctc_log_probs`, such as
   `stage1-wuw.int8.onnx`. This mode also needs its generated contract JSON and
   the model's token table.
2. An SDK runtime package containing `sdk_model.json` and its referenced model
   assets. The tool auto-detects a built `SDK/*/build/asr_batch_decode`. Newer
   SDK packages containing `manifest.json` are decoded with
   `SDK/*/build/batch_files`.
3. A standard WeNet CPU ONNX directory containing `encoder.onnx`, `ctc.onnx`,
   `decoder.onnx`, and `units.txt` (or `tokens.txt`). The tool uses a built
   `build/bin/decoder_main` when available. Otherwise, it creates a no-LM SDK
   package under `OUTPUT_DIR/runtime_model` and uses `asr_batch_decode`.

The ONNX graphs alone are not sufficient: decoding also needs a token table and
frontend/decode settings. An SDK package keeps these in `sdk_model.json`.

## Run the WUW single-ONNX model on server `u`

The WUW virtual environment contains ONNX Runtime, PyTorch, and torchaudio:

```bash
PYTHON=/mnt/users/jinyang_wang/WUW/.wuw/bin/python \
./test_pipline/run.sh \
  --onnx-model /mnt/users/jinyang_wang/WUW/model/base_expts3/output/stage1-wuw.int8.onnx \
  --tokens /mnt/users/jinyang_wang/ASR_wenet/training/expts_3/ipa_units.txt \
  --data-dir /path/to/test_data \
  --output-dir /path/to/eval_output \
  --device cpu \
  --scoring-unit word
```

The contract is auto-detected as `stage1-wuw.contract.json` beside the model.
Pass `--contract PATH` only when it is stored elsewhere. `--device auto` uses
CUDA when the installed ONNX Runtime exposes a CUDA provider; otherwise it uses
CPU.

This particular model emits space-separated IPA phones. Its `text` references
must therefore also contain IPA phones for the error rate to be valid. The
requested `WER` field is then a token error rate—technically phone error rate
(PER). Comparing these hypotheses directly with ordinary word transcripts is
not valid without a phone-to-word decoder.

## Run

```bash
./test_pipline/run.sh \
  --model-dir /path/to/model_package \
  --data-dir /path/to/test_data \
  --output-dir /path/to/eval_output
```

Useful options:

```text
--decoder-bin PATH          Use a specific asr_batch_decode or decoder_main
--decoder-kind sdk|wenet    Override executable auto-detection
--scoring-unit auto|word|char
--case-sensitive
--unit-path PATH            Token table override for decoder_main
--chunk-size N
--num-left-chunks N
--thread-num N
--sample-rate N
--num-bins N
--chunk-ms N              SDK 1.x batch audio chunk size
--contract PATH           Contract for a single streaming ONNX
--tokens PATH             Token table for a single streaming ONNX
--device auto|cpu|gpu
--waveform-scale FLOAT    WeNet audio scale before fbank; default 32768
--token-separator TEXT    Separator used to render CTC tokens
```

`--scoring-unit auto` uses character scoring when the references contain CJK
text, and word scoring otherwise. The selected unit is recorded in
`summary.txt`, so the reported WER is unambiguous.
