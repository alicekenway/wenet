# WeNet SDK 0.1.0: Flashlight Lexicon + KenLM Rescore Decoder

This SDK uses a Flashlight-Text CTC lexicon decoder for first-pass word N-best
generation, then applies word-level KenLM rescoring after finalization. The old
WFST comparison tool can still be built with
`ASR_SDK_ENABLE_LEGACY_WFST=ON`, but the new 0.1.0 package does not require
`TLG.fst`.

The Flashlight package path selects the acoustic backend from the single ONNX
model's `model_type` metadata. 0.1.0 supports both `zipformer2` and
`wenet_ctc`.

Public headers remain under `include/asr_sdk` and do not expose Flashlight,
KenLM, WeNet, or ONNX Runtime headers.

## Local ONNX Runtime

0.1.0 expects ONNX Runtime `1.25.1` under this SDK tree:

```text
third_party/onnxruntime/include
third_party/onnxruntime/lib
```

Recreate it on a clean machine:

```bash
ORT_VERSION=1.25.1
SDK_ROOT=/home/jinyang_wang/Dev/ASR/ASR_wenet/wenet/SDK/0.1.0

mkdir -p "${SDK_ROOT}/third_party/onnxruntime"
cd /tmp
curl -L -o onnxruntime.tgz \
  "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"
tar -xzf onnxruntime.tgz
cp -a "onnxruntime-linux-x64-${ORT_VERSION}/include" \
  "${SDK_ROOT}/third_party/onnxruntime/"
cp -a "onnxruntime-linux-x64-${ORT_VERSION}/lib" \
  "${SDK_ROOT}/third_party/onnxruntime/"
test -f "${SDK_ROOT}/third_party/onnxruntime/lib/libonnxruntime.so"
```

Use a different ORT root only by passing:

```bash
cmake -S . -B build \
  -DASR_SDK_ONNXRUNTIME_ROOT=/path/to/onnxruntime
```

The build fails early if headers or `libonnxruntime.so` are missing, or if the
version is not `1.25.1`.

## Build

```bash
cd /home/jinyang_wang/Dev/ASR/ASR_wenet/wenet/SDK/0.1.0
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DASR_SDK_BUILD_EXAMPLES=OFF
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

Important options:

```text
ASR_SDK_ENABLE_FLASHLIGHT_DECODER=ON
ASR_SDK_ENABLE_LEGACY_WFST=OFF
ASR_SDK_FLASHLIGHT_TEXT_ROOT=third_party/flashlight-text
ASR_SDK_KENLM_ROOT=third_party/kenlm
ASR_SDK_KENLM_INSTALL_ROOT=third_party/kenlm-install
```

KenLM library sources are built from `third_party/kenlm`. The `kenlm-install`
directory supplies reusable `lmplz` and `build_binary` tools.

## Build the Runtime Package

Reusable LM/package commands live in:

```text
/home/jinyang_wang/Dev/ASR/ASR_wenet/LM/kenlm_lm
```

Typical flow:

```bash
cd /home/jinyang_wang/Dev/ASR/ASR_wenet
python3 LM/kenlm_lm/tools/sample_eval_metadata.py
LM/kenlm_lm/tools/train_kenlm_arpa.sh
LM/kenlm_lm/tools/build_kenlm_binary.sh
python3 LM/kenlm_lm/tools/build_words_from_lm.py
python3 LM/kenlm_lm/tools/build_lexicon_for_am.py
wenet/SDK/0.1.0/package_workflows/prepare_flashlight_runtime_package.sh
```

The default package is written to:

```text
test/0.1.0/model_flashlight
```

`prepare_flashlight_runtime_package.sh` writes the Flashlight decoder settings
into `sdk_model.json`. In 0.1.0, `lm_weight`, `word_score`, and `unk_score`
apply to final KenLM rescoring, not online beam-search shallow fusion. Override
them during package creation when you want the public SDK path to use the same
settings as a tuning run:

```bash
LM_WEIGHT=0.5 WORD_SCORE=0.0 BEAM_SIZE=50 BEAM_SIZE_TOKEN=20 \
  wenet/SDK/0.1.0/package_workflows/prepare_flashlight_runtime_package.sh
```

Supported manifest fields are `beam_size`, `beam_size_token`,
`beam_threshold`, `lm_weight`, `word_score`, `unk_score`, `sil_score`,
`log_add`, `allow_unk`, `smearing`, `nbest`, `feature_type`, `blank_token`,
`sil_token`, `unk_word`, `sample_rate`, `mapping`, `final_mapping`, and `debug`.
Use `feature_type=kaldi` for standard sherpa/icefall Zipformer CTC and exported
WeNet CTC models; the default `whisper` mode is kept for packages that were
built with the earlier SDK frontend.

`mapping` is the AM-stage mapping and is applied before KenLM rescoring.
`final_mapping` is applied after rescoring and only changes the displayed output.
Both files may be empty, which means identity mapping.

The package script exposes the manifest token names as environment variables.
The defaults are `BLANK_TOKEN=<blk>`, `SIL_TOKEN=▁`, `UNK_WORD=<unk>`, and
`SAMPLE_RATE=16000`. For models whose `tokens.txt` uses `<blank>` instead of
`<blk>`, build the package with:

```bash
BLANK_TOKEN="<blank>" FEATURE_TYPE=kaldi \
  wenet/SDK/0.1.0/package_workflows/prepare_flashlight_runtime_package.sh
```

The package script accepts symlinked input files for the acoustic model, tokens,
KenLM files, lexicon, words, optional AM-stage mapping, and optional final-stage
mapping. Symlinks are dereferenced during copy, so the output package contains
real files instead of links to files outside the package.

## Run

Standalone decoder:

```bash
wenet/SDK/0.1.0/build/zipformer_ctc_flashlight_main \
  --model test/0.1.0/model_flashlight/model.onnx \
  --tokens test/0.1.0/model_flashlight/tokens.txt \
  --words test/0.1.0/model_flashlight/words.txt \
  --lexicon test/0.1.0/model_flashlight/lexicon.txt \
  --lm test/0.1.0/model_flashlight/lm.bin \
  --mapping test/0.1.0/model_flashlight/output_mapping.txt \
  --wav model/sherpa-onnx-streaming-zipformer-ctc-zh-2025-06-30/test_wavs/0.wav
```

This standalone tool remains Zipformer-specific. Use `asr_stream_file` or
`asr_package_eval` for `model_type=wenet_ctc` packages.

Public SDK final-result path:

```bash
wenet/SDK/0.1.0/build/asr_stream_file \
  --model_dir test/0.1.0/model_flashlight \
  --wav model/sherpa-onnx-streaming-zipformer-ctc-zh-2025-06-30/test_wavs/0.wav \
  --print_partial false \
  --debug false
```

For public partials, feed chunked audio and enable printing:

```bash
wenet/SDK/0.1.0/build/asr_stream_file \
  --model_dir test/0.1.0/model_flashlight \
  --wav model/sherpa-onnx-streaming-zipformer-ctc-zh-2025-06-30/test_wavs/0.wav \
  --chunk_ms 200 \
  --print_partial true
```

Acceptance comparison:

```bash
LM/kenlm_lm/tools/run_003_acceptance_eval.sh
```

It writes `eval.tsv` with `ref`, `greedy`, `lm`, CER columns, and RTF fields.

Package evaluation with JSONL output and a summary:

```bash
wenet/SDK/0.1.0/build/asr_package_eval \
  --model_dir test/0.1.0/model_flashlight \
  --decode_mode lm \
  --metadata data/hf_wenetspeech_test_net/wenetspeech_test_net_sample_2000/metadata.jsonl \
  --wav_parent data/hf_wenetspeech_test_net/wenetspeech_test_net_sample_2000 \
  --output_json test/0.1.0/package_eval/output.jsonl

python3 wenet/SDK/0.1.0/cli/summarize_asr_package_eval.py \
  --input_json test/0.1.0/package_eval/output.jsonl \
  --summary test/0.1.0/package_eval/summary.txt
```

Use `--decode_mode greedy` to evaluate AM-only CTC greedy decoding without
loading the lexicon or KenLM.

Use `--debug true` with LM mode to write a separate readable N-best rescoring
log. `asr_package_eval` keeps the JSONL clean and writes blocks like
`ref#audio_path#reference` followed by `hypN#text#am_score#lm_score`.

See `cli/README.md` for field definitions and metric behavior.

## Current Caveats

- The final rescoring path can only rank hypotheses kept by the first-pass
  Flashlight lexicon decoder. Tune `BEAM_SIZE`, `BEAM_SIZE_TOKEN`,
  `BEAM_THRESHOLD`, `LM_WEIGHT`, and `WORD_SCORE` together.
- Full acceptance-set partial latency and revision-rate metrics are not yet
  collected.
