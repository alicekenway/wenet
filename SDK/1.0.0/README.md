# WeNet Lite C++ Streaming ASR SDK

This project is a standalone C++17 streaming ASR SDK inspired by WeNet runtime
design. It is meant to be small enough to study and embed, while still shaped
like a deployable runtime:

```text
PCM audio
  -> streaming fbank + optional global CMVN
  -> streaming CTC model boundary
  -> optional blank skipping
  -> greedy / CTC prefix / WFST decoder
  -> partial and final ASR results
  -> C++ API and C ABI
```

The default build is self-contained. It uses a deterministic model stub so the
SDK, examples, tools, and tests work without ONNX Runtime or OpenFst installed.
For a real runtime build, enable ONNX Runtime and OpenFst with CMake options.

## What Is Included

- Public C++ API: `include/wenet_sdk/asr_engine.h`,
  `include/wenet_sdk/stream.h`, `include/wenet_sdk/result.h`.
- Stable C ABI: `include/wenet_sdk/c_api.h`.
- Streaming frontend: resampling, frame extraction, fbank, CMVN.
- Model package loader and validator.
- Model backend boundary for ONNX Runtime.
- Greedy CTC, CTC prefix, and WFST-shaped decoder interfaces.
- OpenFst-backed WFST token passing when `WENETSDK_ENABLE_OPENFST=ON`.
- Tools for validation, inspection, streaming WAV decode, microphone demo, and
  benchmarking.
- C++, C, batch, and embedded-style examples.

## Repository Layout

```text
include/wenet_sdk/        Public C++ and C API headers
src/core/                 Engine, stream session, recognizer orchestration
src/frontend/             PCM, resampling, fbank, CMVN
src/model/                Model metadata and ONNX model boundary
src/decoder/              Greedy, prefix, WFST, symbols, blank skipping
src/postprocess/          Text, timestamp, endpoint helpers
src/io/                   WAV reader/writer helpers
tools/                    CLI tools
examples/                 Minimal C++, C, batch, and embedded examples
scripts/                  Package/export/validation helper scripts
model_example/            Small toy model package for smoke tests
tests/                    Unit and integration tests
docs/                     Additional design notes
```

## Quick Start: Self-Contained Build

Use this first. It does not need ONNX Runtime or OpenFst.

```bash
cmake -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Validate and run the toy package:

```bash
./build/validate_model_package --model_dir model_example
./build/inspect_model --model_dir model_example
./build/minimal_streaming model_example
```

Expected `minimal_streaming` output is similar to:

```text
partial: hello world sdk
final: hello world sdk
```

In this mode `runtime_backend` is `deterministic-onnx-stub`. It is useful for
SDK integration tests and API development, not real ASR accuracy.

## Full Backend Build: ONNX Runtime + OpenFst

Use this when you want real ONNX Runtime model execution and binary OpenFst graph
loading. On this machine the verified configuration is:

```bash
cmake -B build_full \
  -DWENETSDK_ENABLE_ONNX=ON \
  -DWENETSDK_ENABLE_OPENFST=ON \
  -DONNXRuntime_INCLUDE_DIR=/home/jinyang_wang/Dev/ASR/ASR_embedded/SDK/1.0.0/_deps/onnxruntime/include \
  -DONNXRuntime_LIBRARY=/home/jinyang_wang/miniforge3/envs/wenet/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime.so.1.23.2 \
  -DCMAKE_PREFIX_PATH=/home/jinyang_wang/miniforge3/envs/wenet

cmake --build build_full -j
LD_LIBRARY_PATH=/home/jinyang_wang/miniforge3/envs/wenet/lib \
  ctest --test-dir build_full --output-on-failure
```

Useful inspection commands:

```bash
LD_LIBRARY_PATH=/home/jinyang_wang/miniforge3/envs/wenet/lib \
  ./build_full/inspect_model --model_dir model_example

LD_LIBRARY_PATH=/home/jinyang_wang/miniforge3/envs/wenet/lib \
  ./build_full/inspect_fst --model_dir model_example
```

When OpenFst is disabled, the WFST decoder object falls back to CTC prefix
decoding so the SDK remains runnable. When OpenFst is enabled, `TLG.fst` is read
as an OpenFst `StdVectorFst` and decoded with streaming token passing.

## Create a Small WAV for Tool Testing

If you do not already have a WAV file, generate a 0.5 second 16 kHz sine wave:

```bash
python3 - <<'PY'
import math
import struct
import wave

path = "/tmp/wenet_sdk_demo.wav"
sample_rate = 16000
samples = int(sample_rate * 0.5)
with wave.open(path, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sample_rate)
    for i in range(samples):
        value = int(12000 * math.sin(2 * math.pi * 440 * i / sample_rate))
        w.writeframes(struct.pack("<h", value))
print(path)
PY
```

## Decode a WAV File

`asr_stream_file` feeds a WAV file in streaming chunks and prints partials,
final text, RTF, and first partial latency.

```bash
LD_LIBRARY_PATH=/home/jinyang_wang/miniforge3/envs/wenet/lib \
  ./build_full/asr_stream_file \
    --model_dir model_example \
    --wav /tmp/wenet_sdk_demo.wav \
    --chunk_ms 100 \
    --decoder ctc_wfst \
    --print_partial true
```

Decoder choices:

```text
auto         Use the decoder from manifest.json
greedy       Greedy CTC
prefix       CTC prefix beam
wfst         CTC WFST decoder
```

Output looks like:

```text
[partial] hello world sdk
[final] hello world sdk
rtf: 0.23
first_partial_latency_ms: 42.8
partial_updates: 3
decode_chunks: 4
audio_ms: 500
elapsed_ms: 116
```

## Benchmark a WAV File

`benchmark` runs the same streaming path and reports timing counters.

```bash
LD_LIBRARY_PATH=/home/jinyang_wang/miniforge3/envs/wenet/lib \
  ./build_full/benchmark \
    --model_dir model_example \
    --wav /tmp/wenet_sdk_demo.wav \
    --chunk_ms 100
```

Fields:

```text
chunks                     Number of model/decoder chunks processed
partial_updates            Number of non-empty partial results
first_partial_latency_ms   Time from start of decode to first non-empty partial
elapsed_ms                 Wall-clock runtime
audio_ms                   Audio duration
rtf                        elapsed_ms / audio_ms
```

## Decode a Batch of WAV Files

Create a WAV list. Each line may be either `path.wav` or `utt_id path.wav`.

```bash
printf 'demo /tmp/wenet_sdk_demo.wav\n' > /tmp/wenet_sdk_wav.scp

LD_LIBRARY_PATH=/home/jinyang_wang/miniforge3/envs/wenet/lib \
  ./build_full/batch_files \
    --model_dir model_example \
    --wav_list /tmp/wenet_sdk_wav.scp \
    --chunk_ms 100
```

Output format:

```text
demo    hello world sdk
```

## Use the C++ API

The minimal example is `examples/cpp/minimal_streaming.cc`.

Core pattern:

```cpp
#include "wenet_sdk/asr_engine.h"

wenet_sdk::EngineConfig config;
config.model_dir = "model_example";
config.enable_timestamps = true;
config.decoder_type = wenet_sdk::DecoderType::kAuto;

auto engine = wenet_sdk::AsrEngine::Create(config);
auto stream = engine->CreateStream();

stream->AcceptWaveform(sample_rate, pcm_float.data(), pcm_float.size());
while (stream->DecodeReady()) {
  stream->Decode();
  auto partial = stream->GetResult();
}

stream->SetInputFinished();
while (stream->DecodeReady()) {
  stream->Decode();
}
auto final_result = stream->GetFinalResult();
```

Build and run:

```bash
cmake --build build_full -j
LD_LIBRARY_PATH=/home/jinyang_wang/miniforge3/envs/wenet/lib \
  ./build_full/minimal_streaming model_example
```

Useful `EngineConfig` fields:

```text
model_dir             Model package directory
num_threads           Runtime thread hint
enable_endpoint       Enable endpoint detection
enable_timestamps     Fill token timestamps when supported
decoder_type          auto, greedy, prefix, or WFST
chunk_size            Feature chunk size override
num_left_chunks       Left context override
blank_skip_thresh     Blank-frame skip threshold
```

## Use the C API

The C example is `examples/c/simple_c_api.c`.

Run it:

```bash
LD_LIBRARY_PATH=/home/jinyang_wang/miniforge3/envs/wenet/lib \
  ./build_full/simple_c_api model_example
```

Typical output is JSON:

```json
{"text":"hello world sdk","is_final":true,"confidence":0.9,"tokens":[...]}
```

The C ABI is useful for firmware, JNI, Rust, Python, and Go bindings. Returned
JSON strings are owned by the stream and remain valid until the next result call
on that stream.

Important C API calls:

```c
wenet_sdk_create_engine
wenet_sdk_create_stream
wenet_sdk_accept_pcm16
wenet_sdk_accept_float32
wenet_sdk_decode_ready
wenet_sdk_decode
wenet_sdk_get_result_json
wenet_sdk_get_final_result_json
wenet_sdk_reset_stream
wenet_sdk_destroy_stream
wenet_sdk_destroy_engine
```

## Embedded-Style Usage

The example `examples/embedded/pseudo_audio_callback.cc` shows the intended
shape for audio callback integration:

```bash
LD_LIBRARY_PATH=/home/jinyang_wang/miniforge3/envs/wenet/lib \
  ./build_full/pseudo_audio_callback model_example
```

The key rule is: do not run model inference or WFST search inside an audio
callback. Copy PCM into a bounded queue in the callback, then decode from a
worker loop.

For real microphone input, build the PortAudio demo:

```bash
cmake -B build_mic \
  -DWENETSDK_ENABLE_PORTAUDIO=ON \
  -DCMAKE_PREFIX_PATH=/path/to/portaudio/prefix
cmake --build build_mic -j
./build_mic/asr_stream_mic --model_dir model_example --seconds 10
```

Without `WENETSDK_ENABLE_PORTAUDIO`, `asr_stream_mic` still builds and exits
with a clear message that PortAudio support is disabled.

## Model Package Tutorial

A runtime package should look like:

```text
model_dir/
  manifest.json
  config.yaml
  encoder.onnx
  ctc.onnx          # optional when CTC is integrated into encoder.onnx
  tokens.txt
  words.txt
  TLG.fst
  global_cmvn       # optional
  checksum.sha256   # optional
```

Minimum validation:

```bash
scripts/validate_package.py --model_dir model_dir --require-onnx
./build_full/validate_model_package --model_dir model_dir
```

The C++ validator checks:

```text
manifest.json exists and parses
tokens.txt and words.txt load
blank_id exists in tokens.txt
TLG.fst exists and is readable
FST labels fit token/word tables when OpenFst is enabled
encoder/ctc ONNX files exist when required
global_cmvn dimension matches feature_dim when present
checksum.sha256 passes when present
```

### Create Starter Metadata

If you already copied model files into `model_dir`, write a starter manifest and
config:

```bash
scripts/package_model.py \
  --model_dir model_dir \
  --sample_rate 16000 \
  --feature_dim 80 \
  --subsampling_rate 4 \
  --encoder encoder.onnx \
  --ctc "" \
  --tokens tokens.txt \
  --words words.txt \
  --graph TLG.fst \
  --decoder ctc_wfst \
  --chunk_size 16 \
  --num_left_chunks 4 \
  --write_checksum
```

### Token and Word Tables

`tokens.txt` and `words.txt` may use any of these forms:

```text
HELLO 1
1 HELLO
HELLO
```

For CTC models, `blank_id` in `manifest.json` must exist in `tokens.txt`.

### CMVN Formats

`global_cmvn` is optional. The loader accepts:

WeNet JSON stats:

```json
{"mean_stat":[...],"var_stat":[...],"frame_num":12345}
```

Kaldi text global CMVN stats:

```text
[ mean_0 mean_1 ... frame_count var_0 var_1 ... 0 ]
```

Legacy flat SDK format:

```text
mean_0 mean_1 ... mean_N inv_std_0 inv_std_1 ... inv_std_N
```

The final CMVN transform is:

```text
normalized = (feature - mean) * inv_std
```

### FST Graph

The runtime expects a prebuilt `TLG.fst`. It does not build the graph on device.
For small text-format fixtures:

```bash
scripts/build_tlg.sh --from-text TLG.txt TLG.fst
```

For production, build `T`, `L`, `G`, compose `TLG.fst`, optimize/prune offline,
then copy only the final binary graph into the package.

## Tests

Default tests:

```bash
cmake -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Full backend tests:

```bash
cmake --build build_full -j
LD_LIBRARY_PATH=/home/jinyang_wang/miniforge3/envs/wenet/lib \
  ctest --test-dir build_full --output-on-failure
```

Focused tests:

```bash
ctest --test-dir build -R cmvn --output-on-failure
ctest --test-dir build -R model_package --output-on-failure
ctest --test-dir build_full -R wfst --output-on-failure
```

## Troubleshooting

### `failed to create engine`

Run:

```bash
./build_full/validate_model_package --model_dir model_dir
./build_full/inspect_model --model_dir model_dir
```

Common causes:

```text
missing manifest.json
missing ONNX model file
blank_id not present in tokens.txt
bad global_cmvn dimension
checksum.sha256 mismatch
TLG.fst label ids exceed token/word table size
```

### ONNX Runtime library cannot be found

Set `LD_LIBRARY_PATH` to the directory that contains `libonnxruntime.so` and the
OpenFst libraries:

```bash
export LD_LIBRARY_PATH=/home/jinyang_wang/miniforge3/envs/wenet/lib:$LD_LIBRARY_PATH
```

If ONNX Runtime lives under a Python package, also include its `capi` directory.

### WFST output is empty or strange

Check:

```bash
./build_full/inspect_fst --model_dir model_dir
./build_full/inspect_model --model_dir model_dir
```

Then verify:

```text
token ids match ONNX CTC output ids
blank_id matches the model
FST input labels use token ids consistently
FST output labels use word ids consistently
decoder graph was built for the same vocabulary
```

### Real audio gives poor recognition

This SDK does not fix model/package mismatch. Check:

```text
sample rate
feature_dim
frame_length_ms and frame_shift_ms
global_cmvn source and dimension
chunk_size and num_left_chunks
whether ONNX output is log_prob, probability, or logits
decoder type and LM/FST graph
```

### Microphone tool says PortAudio is disabled

Reconfigure with:

```bash
cmake -B build_mic -DWENETSDK_ENABLE_PORTAUDIO=ON -DCMAKE_PREFIX_PATH=/path/to/portaudio
```

## More Documentation

- `docs/architecture.md`: ownership and runtime structure.
- `docs/model_package.md`: package contract and validation.
- `docs/decoder.md`: decoder notes.
- `docs/embedded_porting.md`: embedded integration guidance.
- `docs/performance_tuning.md`: benchmarking and tuning notes.
- `wenet_cpp_streaming_asr_sdk_plan.md`: original implementation plan.
