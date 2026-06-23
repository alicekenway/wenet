# WeNet-Style C++ Streaming ASR SDK Development Plan

> **Purpose for Codex:** Use this document as the working implementation spec for a C++ streaming ASR SDK inspired by WeNet runtime design. The SDK should run a streaming CTC ONNX model and decode with an n-gram WFST graph (`TLG.fst`). It should be useful for learning and also shaped like an industrial embedded runtime.

---

## 1. Project Summary

Build a standalone **C++ streaming ASR SDK** with this pipeline:

```text
PCM audio stream
  -> feature extraction
  -> streaming ONNX CTC model
  -> optional blank-frame skipping
  -> CTC WFST decoder using n-gram TLG.fst
  -> partial/final ASR results
  -> C++ API + C ABI for embedded integration
```

The project should **refer to WeNet's framework and runtime design**, but should not become a hard dependency on the entire WeNet repository.

The intended runtime model package contains:

```text
model_dir/
  manifest.json
  config.yaml
  encoder.onnx              # or model.onnx, depending export style
  ctc.onnx                  # optional if CTC is separate
  global_cmvn               # optional, depending model export
  tokens.txt
  words.txt
  TLG.fst
  checksum.sha256           # optional but recommended
```

Primary runtime use case:

```text
embedded device / industrial controller / edge box
  -> microphone or PCM stream
  -> low-latency ASR
  -> command text or transcription result
```

---

## 2. Important Implementation Decision: What to Build vs What to Reuse from WeNet

From the earlier design discussion:

> Reuse WeNet ideas/components around **model inference and decoding infrastructure**, but implement the **SDK API layer, streaming session interface, model package contract, and integration layer** as our own clean C++ project.

### 2.1 Build Ourselves

Implement these parts in this SDK:

| Area | Reason |
|---|---|
| Public C++ SDK API | Needs to be clean, stable, embedded-friendly, and independent from WeNet internals. |
| C ABI wrapper | Required for industrial integration with C firmware, JNI, Rust, Python, Go, etc. |
| `AsrEngine` / `StreamSession` lifecycle | We need explicit ownership, memory control, reset behavior, and multi-stream readiness. |
| Model package loader | Avoid runtime guessing. Validate all model files and dimensions at startup. |
| ONNX Runtime wrapper | Keep ONNX details private and configurable for embedded deployment. |
| Frontend wrapper/pipeline | We can implement or adapt fbank/CMVN, but the SDK should own the interface. |
| Result builder and postprocess | Product-specific text formatting, timestamps, confidence, endpointing. |
| Benchmark, inspection, and validation tools | Needed for deployment and debugging. |
| Tests | Needed to learn and to prevent regressions. |

### 2.2 Reuse or Closely Reference from WeNet

Use WeNet as the main reference for:

| WeNet area | How to use it |
|---|---|
| Streaming CTC/U2 runtime design | Reuse the concepts: chunking, caches, partial CTC result, finalization. |
| ONNX streaming model handling | Study how WeNet handles `offset`, attention cache, CNN cache, and chunk input/output. |
| Decoder parameter names | Reuse familiar names such as `chunk_size`, `num_left_chunks`, `beam`, `max_active`, `blank_id`, `blank_skip_thresh`, `nbest`. |
| CTC WFST search | Use the same conceptual decoding path: CTC log-probs + `TLG.fst` + beam search. |
| LM/FST graph preparation flow | Use WeNet/Kaldi/OpenFst-style offline graph building for `T`, `L`, `G`, and `TLG`. |
| Optional decoder code | If license and integration constraints allow, vendor only the minimal decoder subset rather than importing all WeNet. |

### 2.3 Avoid

Do **not** start by copying the full WeNet runtime into this project.

Avoid these in the MVP:

```text
full WeNet server stack
attention rescoring
GPU runtime
hotword/context biasing
complex ITN
multi-mic processing
noise suppression
large dynamic plugin system
```

These can be added later after the CTC + WFST streaming path is stable.

---

## 3. External References

Use these as engineering references while implementing:

- [WeNet repository](/home/jinyang_wang/Dev/ASR/ASR_wenet/wenet)
- [WeNet LM / CTC WFST documentation](/home/jinyang_wang/Dev/ASR/ASR_wenet/wenet/blob/main/docs/lm.md)
- [WeNet ONNX runtime model implementation](/home/jinyang_wang/Dev/ASR/ASR_wenet/wenet/blob/main/runtime/core/decoder/onnx_asr_model.cc)
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/)
- [ONNX Runtime C++ API notes](https://onnxruntime.ai/docs/get-started/with-cpp.html)
- [OpenFst](https://www.openfst.org/)
- [KenLM](https://kheafield.com/code/kenlm/)

---

## 4. High-Level Architecture

```text
+------------------------+
| Application             |
| C++ API / C API         |
+-----------+------------+
            |
            v
+------------------------+
| AsrEngine               |
| shared model resources  |
+-----------+------------+
            |
            v
+------------------------+
| StreamSession           |
| per-stream state        |
+-----------+------------+
            |
            v
+------------------------+
| FeaturePipeline         |
| PCM -> fbank -> CMVN    |
+-----------+------------+
            |
            v
+------------------------+
| OnnxCtcModel            |
| streaming ONNX forward  |
+-----------+------------+
            |
            v
+------------------------+
| BlankSkipper            |
| optional speedup        |
+-----------+------------+
            |
            v
+------------------------+
| CtcWfstDecoder          |
| TLG.fst + beam search   |
+-----------+------------+
            |
            v
+------------------------+
| ResultBuilder           |
| tokens/words -> result  |
+------------------------+
```

### 4.1 Ownership Model

```text
AsrEngine
  owns immutable/shared resources:
    - parsed model metadata
    - ONNX Runtime environment/session options
    - ONNX model/session handles or session pool
    - FST graph
    - token and word symbol tables
    - global config
    - logger

StreamSession
  owns mutable per-stream resources:
    - audio ring buffer
    - feature cache
    - model streaming cache tensors
    - decoder active states/tokens
    - endpoint state
    - latest partial result
    - final result
```

This is critical for embedded and industrial systems: load heavy resources once, then create/reset lightweight streams.

---

## 5. Repository Layout

```text
wenet-lite-sdk/
  CMakeLists.txt
  cmake/
    FindONNXRuntime.cmake
    FindOpenFst.cmake
    Toolchain-aarch64-linux.cmake
    Sanitizers.cmake

  include/
    wenet_sdk/
      asr_engine.h
      stream.h
      result.h
      config.h
      c_api.h
      version.h

  src/
    core/
      asr_engine.cc
      stream_session.h
      stream_session.cc
      recognizer.h
      recognizer.cc
      result_builder.h
      result_builder.cc

    frontend/
      feature_pipeline.h
      feature_pipeline.cc
      fbank.h
      fbank.cc
      cmvn.h
      cmvn.cc
      window.h
      window.cc
      ring_buffer.h
      ring_buffer.cc
      resampler.h
      resampler.cc

    model/
      asr_model.h
      onnx_ctc_model.h
      onnx_ctc_model.cc
      onnx_session_pool.h
      onnx_session_pool.cc
      model_metadata.h
      model_metadata.cc
      tensor_utils.h
      tensor_utils.cc

    decoder/
      decoder_interface.h
      greedy_ctc_decoder.h
      greedy_ctc_decoder.cc
      ctc_prefix_decoder.h
      ctc_prefix_decoder.cc
      ctc_wfst_decoder.h
      ctc_wfst_decoder.cc
      fst_loader.h
      fst_loader.cc
      blank_skipper.h
      blank_skipper.cc
      symbol_table.h
      symbol_table.cc
      nbest.h
      nbest.cc
      token.h
      token.cc

    postprocess/
      text_normalizer.h
      text_normalizer.cc
      timestamp_estimator.h
      timestamp_estimator.cc
      endpoint.h
      endpoint.cc

    io/
      wav_reader.h
      wav_reader.cc
      wav_writer.h
      wav_writer.cc
      microphone_portaudio.cc

    utils/
      status.h
      status.cc
      logging.h
      logging.cc
      timer.h
      timer.cc
      thread_pool.h
      thread_pool.cc
      checksum.h
      checksum.cc
      json.h

  tools/
    asr_stream_file.cc
    asr_stream_mic.cc
    benchmark.cc
    inspect_model.cc
    inspect_fst.cc
    validate_model_package.cc

  examples/
    cpp/
      minimal_streaming.cc
      batch_files.cc
    c/
      simple_c_api.c
    embedded/
      pseudo_audio_callback.cc

  scripts/
    export_wenet_to_onnx.py
    prepare_lm.sh
    build_tlg.sh
    quantize_onnx.py
    package_model.py
    validate_package.py

  tests/
    unit/
      test_ctc_collapse.cc
      test_symbol_table.cc
      test_feature_pipeline.cc
      test_blank_skipper.cc
      test_endpoint.cc
    integration/
      test_decode_toy_fst.cc
      test_decode_wav.cc
      test_model_forward.cc
      test_model_package.cc
      test_streaming_boundaries.cc
    golden/
      wav/
      expected_text/

  model_example/
    manifest.json
    config.yaml
    tokens.txt
    words.txt
    TLG.fst
    encoder.onnx
    ctc.onnx
    global_cmvn

  docs/
    architecture.md
    model_package.md
    decoder.md
    embedded_porting.md
    performance_tuning.md
```

---

## 6. File Responsibilities

### 6.1 Public SDK Headers

| File | Goal |
|---|---|
| `include/wenet_sdk/asr_engine.h` | Main SDK entry. Loads model package and creates streams. Should expose no ONNX/OpenFst types. |
| `include/wenet_sdk/stream.h` | Streaming session interface: accept audio, decode chunks, retrieve partial/final results, reset. |
| `include/wenet_sdk/result.h` | Defines `AsrResult`, `TokenResult`, timestamps, confidence, final flag, JSON-friendly fields. |
| `include/wenet_sdk/config.h` | Defines stable config structs. Avoid exposing YAML/JSON library types in public API. |
| `include/wenet_sdk/c_api.h` | Stable C ABI for embedded systems and language bindings. |
| `include/wenet_sdk/version.h` | SDK version, ABI version, build flags, dependency versions. |

### 6.2 Core Runtime

| File | Goal |
|---|---|
| `src/core/asr_engine.cc` | Loads metadata, config, symbols, FST graph, and model sessions. Owns shared resources. |
| `src/core/stream_session.*` | Implements one streaming recognition session. Owns per-stream mutable state. |
| `src/core/recognizer.*` | Orchestrates feature extraction, model forward, decoder advance, and result generation. |
| `src/core/result_builder.*` | Converts decoder paths into SDK result objects. Handles partial/final formatting. |

### 6.3 Frontend

| File | Goal |
|---|---|
| `feature_pipeline.*` | Converts audio chunks into feature chunks. Caches leftover samples and frames. |
| `fbank.*` | Computes log-Mel/filterbank features. Must be deterministic and tested. |
| `cmvn.*` | Loads/applies global CMVN. Validate dimension equals feature dim. |
| `window.*` | Handles frame slicing, Hamming window, pre-emphasis if needed. |
| `ring_buffer.*` | Bounded PCM buffer for streaming input. Useful for real-time audio. |
| `resampler.*` | Optional sample-rate conversion. Keep outside the critical decode path if possible. |

### 6.4 ONNX Model Runtime

| File | Goal |
|---|---|
| `asr_model.h` | Abstract model interface: `Reset`, `ForwardChunk`, `VocabSize`, `SubsamplingRate`. |
| `onnx_ctc_model.*` | ONNX Runtime wrapper for streaming CTC model. Handles input/output names, tensors, cache state. |
| `onnx_session_pool.*` | Optional multi-stream session pool. Embedded MVP may use one session per engine or stream. |
| `model_metadata.*` | Parses `manifest.json` and `config.yaml`. Validates all files/dimensions. |
| `tensor_utils.*` | Tensor creation, shape checking, output copying, log-softmax utilities if needed. |

### 6.5 Decoders

| File | Goal |
|---|---|
| `decoder_interface.h` | Abstract streaming decoder API. Allows greedy, prefix beam, and WFST decoders. |
| `greedy_ctc_decoder.*` | Educational/debug decoder: argmax, remove blanks, collapse repeats. |
| `ctc_prefix_decoder.*` | Optional learning decoder without WFST. Useful before full WFST integration. |
| `ctc_wfst_decoder.*` | Production decoder using `TLG.fst`, acoustic log-probs, and beam pruning. |
| `fst_loader.*` | Loads FST graph and validates symbols. May memory-map large graphs later. |
| `blank_skipper.*` | Skips blank-heavy frames to reduce WFST compute. |
| `symbol_table.*` | Loads `tokens.txt` and `words.txt`; maps ids to strings. |
| `nbest.*` | Represents best hypotheses, acoustic score, LM score, total score. |
| `token.*` | Internal decoder token/state structure for educational/custom decoder implementation. |

### 6.6 Postprocess

| File | Goal |
|---|---|
| `text_normalizer.*` | Converts tokens/words into display text. Handles BPE markers, spaces, case options. |
| `timestamp_estimator.*` | Estimates token/word timestamps from frame indexes or CTC spikes. |
| `endpoint.*` | Detects end of utterance using silence/blank duration and max utterance length. |

### 6.7 Tools

| File | Goal |
|---|---|
| `tools/asr_stream_file.cc` | Decode WAV files as streaming chunks. Main debugging tool. |
| `tools/asr_stream_mic.cc` | Microphone demo. Keep optional because PortAudio/ALSA may not exist on target. |
| `tools/benchmark.cc` | Reports RTF, first partial latency, memory, blank skip ratio, decode timings. |
| `tools/inspect_model.cc` | Prints ONNX input/output names, shapes, vocab size, cache names. |
| `tools/inspect_fst.cc` | Checks FST graph size, start/final states, symbol compatibility. |
| `tools/validate_model_package.cc` | Validates model dir before deployment. |

---

## 7. Runtime Dependencies

### 7.1 Required Runtime Dependencies

| Dependency | Purpose | Notes |
|---|---|---|
| C++17 | SDK implementation | Use C++17 for embedded compatibility. |
| CMake | Build system | Add ARM/aarch64 toolchain files. |
| ONNX Runtime C/C++ | Run streaming CTC ONNX model | Keep inside `src/model`; do not leak into public headers. |
| OpenFst | Load/search WFST graphs | Used by `ctc_wfst_decoder` and `fst_loader`. |
| nlohmann/json or yaml-cpp | Parse manifest/config | Parse only at initialization. |
| GoogleTest | Unit/integration tests | Build only when tests are enabled. |

### 7.2 Optional Runtime Dependencies

| Dependency | Purpose |
|---|---|
| spdlog | Logging; can be replaced by small custom logger for embedded. |
| PortAudio / ALSA | Microphone demo only, not core SDK. |
| dr_wav | Tiny WAV reader for tools/tests. |

### 7.3 Offline / Build-Time Dependencies

| Dependency | Purpose |
|---|---|
| Python + WeNet | Export WeNet model to ONNX and verify behavior. |
| KenLM | Train n-gram language model from domain text. |
| OpenGrm / Kaldi / WeNet scripts | Convert ARPA to FST and build `T`, `L`, `G`, `TLG`. |
| ONNX tools | Inspect, simplify, quantize, or validate ONNX. |

### 7.4 Embedded Optimization Dependencies

| Tool | Purpose |
|---|---|
| ONNX Runtime reduced build | Smaller runtime binary. |
| ONNX Runtime static build | Easier deployment on some devices. |
| cross compiler | ARM/aarch64 target build. |
| perf / heaptrack / valgrind | Linux profiling. |

---

## 8. Model Package Contract

### 8.1 `manifest.json`

```json
{
  "sdk_model_version": 1,
  "model_type": "wenet_ctc_streaming_onnx",
  "sample_rate": 16000,
  "feature_dim": 80,
  "frame_length_ms": 25,
  "frame_shift_ms": 10,
  "subsampling_rate": 4,

  "onnx": {
    "encoder": "encoder.onnx",
    "ctc": "ctc.onnx",
    "output_type": "log_prob",
    "input_names": {
      "chunk": "chunk",
      "offset": "offset",
      "att_cache": "att_cache",
      "cnn_cache": "cnn_cache"
    },
    "output_names": {
      "encoder_out": "output",
      "att_cache": "r_att_cache",
      "cnn_cache": "r_cnn_cache",
      "log_probs": "log_probs"
    }
  },

  "vocab": {
    "tokens": "tokens.txt",
    "words": "words.txt",
    "blank_id": 0,
    "sos_id": -1,
    "eos_id": -1
  },

  "decoder": {
    "type": "ctc_wfst",
    "graph": "TLG.fst",
    "beam": 16.0,
    "lattice_beam": 10.0,
    "max_active": 7000,
    "min_active": 200,
    "acoustic_scale": 1.0,
    "lm_scale": 1.0,
    "length_penalty": 0.0,
    "blank_skip_thresh": 0.98,
    "nbest": 1
  }
}
```

### 8.2 `config.yaml`

```yaml
streaming:
  chunk_size: 16
  num_left_chunks: 4
  endpoint_silence_ms: 800
  max_segment_ms: 20000

runtime:
  intra_op_num_threads: 2
  inter_op_num_threads: 1
  enable_profiling: false

postprocess:
  lowercase: true
  remove_bpe_marker: true
  language_type: indo_european
```

### 8.3 Validation Rules

At engine creation, validate:

```text
[ ] model directory exists
[ ] manifest.json exists and parses
[ ] ONNX model files exist
[ ] TLG.fst exists
[ ] tokens.txt and words.txt exist
[ ] sample_rate is supported
[ ] feature_dim matches CMVN and model input
[ ] ONNX output vocab size equals token count
[ ] blank_id is within vocab range
[ ] FST input labels are compatible with token ids
[ ] FST output labels are compatible with word ids
[ ] checksum file passes if provided
```

Fail early with a useful error message.

---

## 9. Public C++ API

### 9.1 Header Design

Public headers should be stable and dependency-light.

Do **not** expose these types publicly:

```cpp
Ort::Session*
fst::StdVectorFst*
kaldi::LatticeFasterDecoder*
yaml::Node
nlohmann::json
```

Use PIMPL or abstract interfaces.

### 9.2 Example API

```cpp
namespace wenet_sdk {

struct EngineConfig {
  std::string model_dir;
  int num_threads = 1;
  bool enable_endpoint = true;
  bool enable_timestamps = false;
};

struct TokenResult {
  std::string token;
  int token_id = -1;
  float start_ms = -1.0f;
  float end_ms = -1.0f;
  float confidence = 0.0f;
};

struct AsrResult {
  std::string text;
  bool is_final = false;
  float confidence = 0.0f;
  std::vector<TokenResult> tokens;
};

class Stream {
 public:
  virtual ~Stream() = default;

  virtual void AcceptWaveform(
      int sample_rate,
      const float* samples,
      size_t n) = 0;

  virtual bool DecodeReady() const = 0;
  virtual void Decode() = 0;

  virtual AsrResult GetResult() const = 0;
  virtual AsrResult GetFinalResult() = 0;

  virtual void SetInputFinished() = 0;
  virtual void Reset() = 0;
};

class AsrEngine {
 public:
  static std::unique_ptr<AsrEngine> Create(const EngineConfig& config);

  virtual ~AsrEngine() = default;
  virtual std::unique_ptr<Stream> CreateStream() = 0;
};

}  // namespace wenet_sdk
```

### 9.3 Example Usage

```cpp
#include "wenet_sdk/asr_engine.h"

int main() {
  wenet_sdk::EngineConfig config;
  config.model_dir = "model_dir";
  config.num_threads = 2;

  auto engine = wenet_sdk::AsrEngine::Create(config);
  auto stream = engine->CreateStream();

  std::vector<float> pcm = ReadWavAsFloat("test.wav");
  const int sample_rate = 16000;
  const int chunk_samples = sample_rate * 100 / 1000;

  for (size_t offset = 0; offset < pcm.size(); offset += chunk_samples) {
    size_t n = std::min<size_t>(chunk_samples, pcm.size() - offset);
    stream->AcceptWaveform(sample_rate, pcm.data() + offset, n);

    while (stream->DecodeReady()) {
      stream->Decode();
      auto partial = stream->GetResult();
      if (!partial.text.empty()) {
        std::cout << "partial: " << partial.text << "\n";
      }
    }
  }

  stream->SetInputFinished();

  while (stream->DecodeReady()) {
    stream->Decode();
  }

  auto final_result = stream->GetFinalResult();
  std::cout << "final: " << final_result.text << "\n";

  return 0;
}
```

---

## 10. C API for Embedded Integration

Use a stable C ABI at system boundaries.

```c
typedef struct WenetSdkEngine WenetSdkEngine;
typedef struct WenetSdkStream WenetSdkStream;

WenetSdkEngine* wenet_sdk_create_engine(const char* model_dir);
void wenet_sdk_destroy_engine(WenetSdkEngine* engine);

WenetSdkStream* wenet_sdk_create_stream(WenetSdkEngine* engine);
void wenet_sdk_destroy_stream(WenetSdkStream* stream);

int wenet_sdk_accept_pcm16(
    WenetSdkStream* stream,
    int sample_rate,
    const int16_t* samples,
    int num_samples);

int wenet_sdk_accept_float32(
    WenetSdkStream* stream,
    int sample_rate,
    const float* samples,
    int num_samples);

int wenet_sdk_decode(WenetSdkStream* stream);
int wenet_sdk_decode_ready(WenetSdkStream* stream);

const char* wenet_sdk_get_result_json(WenetSdkStream* stream);
const char* wenet_sdk_get_final_result_json(WenetSdkStream* stream);

void wenet_sdk_set_input_finished(WenetSdkStream* stream);
void wenet_sdk_reset_stream(WenetSdkStream* stream);

int wenet_sdk_last_error_code(void* handle);
const char* wenet_sdk_last_error_message(void* handle);
```

Rules:

```text
- Never throw exceptions across the C ABI.
- Return integer status codes.
- Keep returned strings owned by the stream/engine until the next API call.
- Document thread-safety explicitly.
```

---

## 11. Internal Interfaces

### 11.1 Model Interface

```cpp
class AsrModel {
 public:
  virtual ~AsrModel() = default;

  virtual void Reset() = 0;

  virtual void ForwardChunk(
      const std::vector<std::vector<float>>& features,
      std::vector<std::vector<float>>* log_probs) = 0;

  virtual int VocabSize() const = 0;
  virtual int FeatureDim() const = 0;
  virtual int SubsamplingRate() const = 0;
};
```

### 11.2 Decoder Interface

```cpp
class StreamingDecoder {
 public:
  virtual ~StreamingDecoder() = default;

  virtual void Reset() = 0;

  virtual void Advance(
      const std::vector<std::vector<float>>& log_probs) = 0;

  virtual DecodeResult PartialResult() const = 0;
  virtual DecodeResult Finalize() = 0;
};
```

### 11.3 Recognizer Orchestration

```cpp
void StreamSession::Decode() {
  std::vector<std::vector<float>> feats;
  feature_pipeline_->ReadChunk(&feats);

  if (feats.empty()) {
    return;
  }

  std::vector<std::vector<float>> log_probs;
  model_->ForwardChunk(feats, &log_probs);

  blank_skipper_->Filter(&log_probs);
  decoder_->Advance(log_probs);

  latest_result_ = result_builder_->BuildPartial(decoder_->PartialResult());

  if (endpoint_->IsEndpoint(latest_result_)) {
    final_result_ = result_builder_->BuildFinal(decoder_->Finalize());
    ResetForNextSegment();
  }
}
```

---

## 12. Decoder Design

### 12.1 Decoder Types

```cpp
enum class DecoderType {
  kGreedyCtc,
  kCtcPrefixBeam,
  kCtcWfst
};
```

Start with greedy CTC for debugging. Then add WFST.

### 12.2 Greedy CTC Decoder

Purpose:

```text
- educational
- fast sanity check
- helps debug ONNX output and token mapping
- no LM/FST dependency
```

Algorithm:

```text
1. For each frame, choose argmax token id.
2. Remove blank ids.
3. Collapse repeated ids unless separated by blank.
4. Map ids to tokens.
5. Convert tokens to text.
```

Example:

```cpp
std::vector<int> CollapseCtc(
    const std::vector<int>& frame_best_ids,
    int blank_id) {
  std::vector<int> output;
  int prev = -1;

  for (int id : frame_best_ids) {
    if (id == blank_id) {
      prev = id;
      continue;
    }

    if (id != prev) {
      output.push_back(id);
    }

    prev = id;
  }

  return output;
}
```

Unit test:

```cpp
TEST(CtcCollapseTest, RemovesBlankAndRepeats) {
  int blank = 0;

  std::vector<int> best_path = {
      0, 1, 1, 0, 2, 2, 0, 2, 3, 0
  };

  std::vector<int> expected = {
      1, 2, 2, 3
  };

  EXPECT_EQ(CollapseCtc(best_path, blank), expected);
}
```

### 12.3 CTC WFST Decoder

Purpose:

```text
- production decoder
- uses n-gram language model through TLG.fst
- supports beam pruning and n-best output
```

Input:

```text
log_probs shape: [num_output_frames, vocab_size]
```

Per frame:

```text
1. Optionally skip blank-heavy frame.
2. Convert log-prob to acoustic cost.
3. Advance active WFST states.
4. Apply beam pruning.
5. Keep active state count bounded.
6. Update best partial path.
```

Finalization:

```text
1. Push final weights.
2. Extract best path or n-best.
3. Convert word ids to text.
4. Return final result.
```

### 12.4 Decoder Config

```yaml
decoder:
  type: ctc_wfst
  beam: 16.0
  lattice_beam: 10.0
  max_active: 7000
  min_active: 200
  acoustic_scale: 1.0
  lm_scale: 1.0
  blank_id: 0
  blank_skip_thresh: 0.98
  length_penalty: 0.0
  nbest: 1
```

### 12.5 Tuning Tips

| Symptom | Possible Fix |
|---|---|
| Too slow | Lower `max_active`, lower `beam`, increase blank skipping. |
| Too many insertions | Tune `lm_scale`, `length_penalty`, or graph weights. |
| Too many deletions | Reduce blank skipping, tune acoustic/LM balance. |
| Partial results unstable | Delay partial output or expose only stable prefix. |
| Good offline but bad streaming | Check chunk size, cache shapes, left context, and endpointing. |
| Garbage text | Check token order, blank id, FST labels, and log-prob/logit mismatch. |

---

## 13. Offline LM / FST Preparation

The embedded runtime should **not** build `TLG.fst` on device.

Offline pipeline:

```text
domain text
  -> text normalization
  -> train n-gram LM
  -> ARPA LM
  -> G.fst
  -> build lexicon L.fst
  -> build CTC/token topology T.fst
  -> compose TLG.fst
  -> optimize/prune/minimize
  -> package with ONNX model
```

Example script goal:

```bash
#!/usr/bin/env bash
set -euo pipefail

TEXT=data/domain_text.txt
ORDER=3
OUT=build_lm

mkdir -p "${OUT}"

# Train n-gram ARPA LM.
kenlm/build/bin/lmplz -o ${ORDER} < "${TEXT}" > "${OUT}/lm.arpa"

# Convert ARPA to G.fst.
# Exact command depends on selected toolchain:
#   - WeNet/Kaldi arpa2fst
#   - OpenGrm
#   - custom wrapper
arpa2fst --read-symbol-table=words.txt \
         --keep-symbols=true \
         "${OUT}/lm.arpa" \
         "${OUT}/G.fst"

# Build T, L, and compose TLG.
./build_tlg.sh tokens.txt lexicon.txt "${OUT}/G.fst" "${OUT}/TLG.fst"
```

Only ship these to the device:

```text
TLG.fst
tokens.txt
words.txt
```

---

## 14. ONNX Runtime Design

### 14.1 Goals

`OnnxCtcModel` should:

```text
- load ONNX model files
- inspect input/output names and shapes
- create input tensors
- manage streaming cache tensors
- run per-chunk inference
- return CTC log-probs
- hide ONNX Runtime details from public API
```

### 14.2 Important Runtime Checks

```text
[ ] Is output already log-softmax/log-prob?
[ ] Does output dimension equal token count?
[ ] Are cache tensor shapes correct?
[ ] Does chunk size match model export assumptions?
[ ] Does model require offset input?
[ ] Does model require attention/CNN caches?
[ ] Is batch size always 1 for embedded MVP?
```

### 14.3 Output Type Handling

```cpp
if (output_type == OutputType::kLogProb) {
  cost = -logp;
} else if (output_type == OutputType::kLogit) {
  logp = LogSoftmax(logits);
  cost = -logp;
} else if (output_type == OutputType::kProbability) {
  logp = std::log(std::max(prob, epsilon));
  cost = -logp;
}
```

Never apply `log()` to an already-log value.

---

## 15. Feature Pipeline Design

### 15.1 Input

Support:

```text
- float32 PCM in range [-1.0, 1.0]
- int16 PCM through C API
- mono audio
- default sample rate: 16000 Hz
```

### 15.2 Processing

```text
PCM samples
  -> optional resampling
  -> frame slicing
  -> windowing
  -> fbank/log-Mel
  -> global CMVN
  -> chunk assembly
```

### 15.3 Important Details

```text
frame_length_ms = 25
frame_shift_ms = 10
feature_dim = 80
sample_rate = 16000
```

The values must come from the model package, not hardcoded globally.

### 15.4 Testing

Compare C++ features against a Python/WeNet reference.

Acceptance target:

```text
max_abs_diff(cpp_features, reference_features) < selected_tolerance
```

---

## 16. Endpointing

Endpointing should be simple in MVP.

Inputs:

```text
- consecutive blank-heavy frames
- silence duration estimate
- input finished flag
- max utterance duration
```

Config:

```yaml
streaming:
  endpoint_silence_ms: 800
  max_segment_ms: 20000
```

Behavior:

```text
- partial result can update during speech
- endpoint finalizes current decoder state
- stream resets decoder/model/frontend state for next segment
- final result remains accessible
```

---

## 17. Embedded System Design Rules

### 17.1 Do Not Decode in the Audio Callback

Bad:

```cpp
void AudioCallback(...) {
  stream->AcceptWaveform(...);
  stream->Decode();  // bad: ONNX + WFST in real-time callback
}
```

Better:

```text
Audio callback:
  copy PCM into bounded ring buffer

Worker thread:
  read PCM
  call AcceptWaveform
  call Decode
  publish result
```

### 17.2 Avoid Allocations in the Hot Path

Preallocate after stream creation:

```text
- audio buffers
- feature matrices
- ONNX input buffers
- model cache tensors
- log-prob buffers
- decoder token pools
- result buffers
```

### 17.3 Keep Logs Bounded

Log levels:

```text
ERROR
WARN
INFO
DEBUG
TRACE
```

Counters:

```text
frames_decoded
chunks_decoded
blank_frames_skipped
blank_skip_ratio
rtf
first_partial_latency_ms
last_decode_ms
active_wfst_states
endpoint_count
```

### 17.4 Public ABI Stability

Use C ABI for external integration.

Rules:

```text
- no exceptions across ABI
- no STL types across C ABI
- no ONNX/OpenFst/Kaldi types in public headers
- version the ABI
- expose a last-error string
```

---

## 18. CMake Skeleton

```cmake
cmake_minimum_required(VERSION 3.18)
project(wenet_lite_sdk LANGUAGES C CXX)

option(WENETSDK_BUILD_TOOLS "Build CLI tools" ON)
option(WENETSDK_BUILD_TESTS "Build tests" ON)
option(WENETSDK_BUILD_EXAMPLES "Build examples" ON)
option(WENETSDK_USE_STATIC_RUNTIME "Prefer static deps" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ONNXRuntime REQUIRED)
find_package(OpenFst REQUIRED)

add_library(wenet_sdk
  src/core/asr_engine.cc
  src/core/stream_session.cc
  src/core/recognizer.cc
  src/core/result_builder.cc

  src/frontend/feature_pipeline.cc
  src/frontend/fbank.cc
  src/frontend/cmvn.cc
  src/frontend/window.cc
  src/frontend/ring_buffer.cc

  src/model/onnx_ctc_model.cc
  src/model/onnx_session_pool.cc
  src/model/model_metadata.cc
  src/model/tensor_utils.cc

  src/decoder/greedy_ctc_decoder.cc
  src/decoder/ctc_wfst_decoder.cc
  src/decoder/fst_loader.cc
  src/decoder/blank_skipper.cc
  src/decoder/symbol_table.cc
  src/decoder/nbest.cc

  src/postprocess/text_normalizer.cc
  src/postprocess/timestamp_estimator.cc
  src/postprocess/endpoint.cc

  src/utils/status.cc
  src/utils/logging.cc
)

target_include_directories(wenet_sdk
  PUBLIC include
  PRIVATE src
)

target_link_libraries(wenet_sdk
  PRIVATE
    ONNXRuntime::ONNXRuntime
    OpenFst::fst
)

if(WENETSDK_BUILD_TOOLS)
  add_executable(asr_stream_file tools/asr_stream_file.cc)
  target_link_libraries(asr_stream_file PRIVATE wenet_sdk)

  add_executable(benchmark tools/benchmark.cc)
  target_link_libraries(benchmark PRIVATE wenet_sdk)
endif()
```

---

## 19. Development Phases

### Phase 0: Project Skeleton

Goal:

```text
Create buildable project structure with public headers, empty implementation, examples, and tests.
```

Deliverables:

```text
CMakeLists.txt
include/wenet_sdk/*.h
src/core/asr_engine.cc
src/core/stream_session.cc
examples/cpp/minimal_streaming.cc
```

Acceptance:

```bash
cmake -B build
cmake --build build -j
```

---

### Phase 1: Model Package Loader and Inspection

Goal:

```text
Load and validate model package metadata before real decoding.
```

Deliverables:

```text
src/model/model_metadata.*
tools/validate_model_package.cc
tools/inspect_model.cc
docs/model_package.md
```

Tasks:

```text
[ ] Parse manifest.json
[ ] Parse config.yaml or JSON runtime config
[ ] Validate file existence
[ ] Load token and word symbol tables
[ ] Print model package summary
[ ] Add helpful errors
```

Acceptance:

```bash
./build/validate_model_package --model_dir model_example
./build/inspect_model --model_dir model_example
```

---

### Phase 2: SDK Skeleton with Dummy Decoder

Goal:

```text
Make the public API usable before implementing ONNX/WFST.
```

Deliverables:

```text
include/wenet_sdk/asr_engine.h
include/wenet_sdk/stream.h
include/wenet_sdk/result.h
src/core/asr_engine.cc
src/core/stream_session.cc
examples/cpp/minimal_streaming.cc
```

Tasks:

```text
[ ] Implement AsrEngine::Create
[ ] Implement CreateStream
[ ] Implement AcceptWaveform
[ ] Implement DecodeReady
[ ] Implement dummy Decode
[ ] Implement GetResult/GetFinalResult
```

Acceptance:

```text
minimal_streaming compiles and prints a dummy result.
```

---

### Phase 3: Symbol Tables and Greedy CTC Decoder

Goal:

```text
Understand and test CTC decoding without ONNX or WFST complexity.
```

Deliverables:

```text
src/decoder/symbol_table.*
src/decoder/greedy_ctc_decoder.*
tests/unit/test_ctc_collapse.cc
tests/unit/test_symbol_table.cc
```

Tasks:

```text
[ ] Load tokens.txt
[ ] Load words.txt
[ ] Implement CTC collapse
[ ] Implement token-to-text conversion
[ ] Add unit tests
```

Acceptance:

```bash
ctest -R ctc
ctest -R symbol
```

---

### Phase 4: Feature Pipeline

Goal:

```text
Convert streaming PCM into model-compatible fbank features.
```

Deliverables:

```text
src/frontend/feature_pipeline.*
src/frontend/fbank.*
src/frontend/cmvn.*
src/frontend/window.*
tests/unit/test_feature_pipeline.cc
```

Tasks:

```text
[ ] Implement PCM buffering
[ ] Implement frame extraction
[ ] Implement windowing
[ ] Implement fbank
[ ] Implement CMVN
[ ] Cache leftover samples between chunks
[ ] Compare with Python/WeNet reference
```

Acceptance:

```text
C++ features approximately match reference features.
```

---

### Phase 5: ONNX Streaming CTC Forward

Goal:

```text
Run streaming ONNX model and produce CTC log-probs.
```

Deliverables:

```text
src/model/onnx_ctc_model.*
src/model/tensor_utils.*
tests/integration/test_model_forward.cc
```

Tasks:

```text
[ ] Create ONNX Runtime environment/session
[ ] Inspect input/output names
[ ] Allocate tensors
[ ] Manage streaming cache tensors
[ ] Run chunk inference
[ ] Return log_probs
[ ] Compare C++ output with Python ONNX output
```

Acceptance:

```text
same WAV + same chunking -> same output shape and numerically close log_probs
```

---

### Phase 6: End-to-End Greedy Streaming Tool

Goal:

```text
Decode real WAV with fbank + ONNX + greedy CTC.
```

Deliverables:

```text
tools/asr_stream_file.cc
examples/cpp/minimal_streaming.cc
```

Tasks:

```text
[ ] Read WAV
[ ] Feed audio in streaming chunks
[ ] Run feature pipeline
[ ] Run ONNX
[ ] Run greedy CTC
[ ] Print partial/final text
```

Acceptance:

```bash
./build/asr_stream_file \
  --model_dir ./model_dir \
  --wav ./test.wav \
  --decoder greedy \
  --chunk_ms 100
```

---

### Phase 7: WFST Decoder Integration

Goal:

```text
Decode with TLG.fst and n-gram language model.
```

Deliverables:

```text
src/decoder/fst_loader.*
src/decoder/ctc_wfst_decoder.*
tools/inspect_fst.cc
tests/integration/test_decode_toy_fst.cc
```

Tasks:

```text
[ ] Load TLG.fst
[ ] Validate symbols
[ ] Implement or integrate WFST beam search
[ ] Feed CTC log-prob frames
[ ] Apply beam/max-active pruning
[ ] Extract best path
[ ] Return partial/final result
```

Acceptance:

```text
toy FST + synthetic log_probs -> expected command text
real model + TLG.fst -> readable ASR output
```

Implementation note:

```text
Plan A: vendor a minimal license-compatible WeNet/Kaldi-style decoder subset.
Plan B: implement a simplified token-passing WFST decoder for learning, then replace with optimized decoder.
```

---

### Phase 8: Blank Skipping, Endpointing, and Result Stability

Goal:

```text
Improve latency and make streaming behavior production-like.
```

Deliverables:

```text
src/decoder/blank_skipper.*
src/postprocess/endpoint.*
src/core/result_builder.*
```

Tasks:

```text
[ ] Skip frames with blank probability above threshold
[ ] Track skipped frame count
[ ] Add endpoint detection
[ ] Emit partial results
[ ] Emit final results
[ ] Reset state after endpoint
```

Acceptance:

```text
silence triggers final result; next utterance starts cleanly.
```

---

### Phase 9: C API and Embedded Demo

Goal:

```text
Expose stable C ABI and show safe embedded usage.
```

Deliverables:

```text
include/wenet_sdk/c_api.h
src/core/c_api.cc
examples/c/simple_c_api.c
examples/embedded/pseudo_audio_callback.cc
docs/embedded_porting.md
```

Tasks:

```text
[ ] Wrap engine creation/destruction
[ ] Wrap stream creation/destruction
[ ] Add PCM16 input
[ ] Add JSON result output
[ ] Add last-error API
[ ] Add audio callback demo
```

Acceptance:

```text
C example compiles and decodes test WAV.
```

---

### Phase 10: Benchmarking and Industrial Hardening

Goal:

```text
Measure and optimize target behavior.
```

Deliverables:

```text
tools/benchmark.cc
docs/performance_tuning.md
cmake/Toolchain-aarch64-linux.cmake
scripts/package_model.py
scripts/validate_package.py
```

Tasks:

```text
[ ] Measure RTF
[ ] Measure first partial latency
[ ] Measure feature/model/decoder time
[ ] Measure peak memory
[ ] Report active WFST states
[ ] Add cross-compilation docs
[ ] Add model checksum validation
```

Acceptance:

```text
runs on target board and reports stable performance metrics.
```

---

## 20. CLI Tools

### 20.1 `asr_stream_file`

Command:

```bash
./asr_stream_file \
  --model_dir ./model_dir \
  --wav ./test.wav \
  --chunk_ms 100 \
  --decoder ctc_wfst \
  --print_partial true
```

Expected output:

```text
[partial] turn
[partial] turn on
[partial] turn on the light
[final] turn on the light
RTF: 0.42
first_partial_latency_ms: 180
peak_active_states: 3480
blank_skip_ratio: 0.61
```

### 20.2 `benchmark`

Command:

```bash
./benchmark \
  --model_dir ./model_dir \
  --wav_list ./wav.scp \
  --chunk_ms 100 \
  --decoder ctc_wfst
```

Output:

```text
audio_sec: 120.0
wall_sec: 42.0
rtf: 0.35
feature_ms: 2100
onnx_ms: 31000
decoder_ms: 7800
post_ms: 200
peak_rss_mb: 186
blank_skip_ratio: 0.58
```

---

## 21. Testing Plan

### 21.1 Unit Tests

| Test | Goal |
|---|---|
| `test_ctc_collapse.cc` | Validate blank/repeat behavior. |
| `test_symbol_table.cc` | Validate token/word id mapping. |
| `test_feature_pipeline.cc` | Compare C++ features with reference. |
| `test_blank_skipper.cc` | Ensure blank frame skipping is safe. |
| `test_endpoint.cc` | Validate endpoint behavior. |

### 21.2 Integration Tests

| Test | Goal |
|---|---|
| `test_model_package.cc` | Validate model directory and metadata. |
| `test_model_forward.cc` | Compare ONNX output with Python reference. |
| `test_decode_toy_fst.cc` | Decode synthetic scores with tiny FST. |
| `test_decode_wav.cc` | Decode known WAV and compare expected text. |
| `test_streaming_boundaries.cc` | Different chunk splits should produce similar output. |

### 21.3 Performance Tests

Measure:

```text
RTF
first partial latency
final latency
peak memory
average active WFST states
max active WFST states
blank skip ratio
feature time
ONNX time
decoder time
postprocess time
```

---

## 22. Industrial Production Checklist

```text
[ ] C API stable and documented
[ ] public headers hide implementation dependencies
[ ] model package versioned
[ ] model checksum verified
[ ] deterministic test set exists
[ ] no large allocations after stream start
[ ] bounded logs
[ ] endpoint tested with noise and silence
[ ] power-loss-safe config loading
[ ] watchdog-friendly decode loop
[ ] sample-rate mismatch handled
[ ] token/FST mismatch rejected at init
[ ] benchmarked on real target hardware
[ ] cross-compile CI added
[ ] third-party license report prepared
```

---

## 23. Common Pitfalls

### 23.1 Token Order Mismatch

`tokens.txt` order must match ONNX CTC output ids.

Bad:

```text
ONNX output id 10 = "a"
tokens.txt line 10 = "b"
```

This produces unreadable recognition errors.

### 23.2 Blank ID Mismatch

The decoder must use the same blank id as the model.

```text
manifest blank_id == ONNX blank id == decoder blank id == FST topology assumption
```

### 23.3 Log-Prob vs Logit Confusion

Check whether the ONNX output is:

```text
log_prob
probability
logit
```

The decoder cost must be computed correctly.

### 23.4 Chunk Size Confusion

`chunk_size` may mean:

```text
raw audio samples
feature frames
encoder frames
```

Use explicit names internally:

```text
audio_chunk_samples
feature_chunk_frames
encoder_output_frames
```

### 23.5 FST Too Large for Embedded Memory

For embedded devices:

```text
- use domain-specific text
- prune the LM
- limit vocabulary if possible
- optimize TLG.fst offline
- consider memory mapping
- do not ship graph-building tools
```

---

## 24. Learning Path

### Exercise 1: Greedy CTC

Implement:

```text
frame argmax
remove blanks
collapse repeats
map token ids to text
```

Learning goal:

```text
understand CTC alignment and why blank exists
```

### Exercise 2: Prefix Beam Search

Implement a small prefix beam search without LM.

Learning goal:

```text
understand blank/non-blank probability and prefix recombination
```

### Exercise 3: Tiny Command Grammar FST

Create a tiny grammar:

```text
turn on light
turn off light
start motor
stop motor
```

Decode synthetic log-probs through it.

Learning goal:

```text
understand why WFST helps command recognition
```

### Exercise 4: Real TLG Decoder

Replace the toy FST with real `TLG.fst`.

Learning goal:

```text
understand T, L, G, TLG, acoustic score, LM score, and beam pruning
```

### Exercise 5: Profile and Tune

Add timers around:

```text
feature extraction
ONNX inference
WFST search
postprocess
```

Tune:

```text
beam
max_active
blank_skip_thresh
chunk_size
num_threads
```

---

## 25. Codex Implementation Instructions

When implementing from this plan:

```text
1. Keep each phase buildable.
2. Add tests before adding complex decoder behavior.
3. Do not expose third-party runtime types in public headers.
4. Prefer small files with clear ownership.
5. Add validation and error messages early.
6. Implement greedy decoder before WFST decoder.
7. Make the WAV streaming tool the main integration test.
8. Keep embedded constraints in mind: bounded memory, stable ABI, predictable runtime.
```

Suggested commit order:

```text
001_project_skeleton
002_public_cpp_api
003_model_package_loader
004_symbol_tables
005_greedy_ctc_decoder
006_feature_pipeline
007_onnx_ctc_model
008_streaming_file_tool
009_fst_loader
010_ctc_wfst_decoder
011_blank_skipper_endpoint
012_c_api
013_benchmark_and_embedded_docs
```

---

## 26. MVP Definition

The MVP is complete when this works:

```bash
./asr_stream_file \
  --model_dir ./model_dir \
  --wav ./test.wav \
  --chunk_ms 100 \
  --decoder ctc_wfst \
  --print_partial true
```

And produces:

```text
partial results during audio
one final result at the end
RTF and latency metrics
no model/token/FST validation errors
```

The MVP should include:

```text
[ ] C++ API
[ ] model package loader
[ ] symbol tables
[ ] feature pipeline
[ ] ONNX CTC forward
[ ] greedy CTC decoder
[ ] CTC WFST decoder
[ ] WAV streaming tool
[ ] benchmark tool
[ ] C API wrapper
```

---

## 27. Final Design Principle

The SDK should feel like this:

```text
WeNet-inspired inside,
clean C++ SDK outside,
offline FST/model preparation,
small embedded runtime,
fast debugging through greedy CTC,
production decoding through CTC WFST.
```
