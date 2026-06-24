# Development Plan: Sherpa Zipformer CTC ONNX in WeNet SDK

## Objective

Integrate a Sherpa/Icefall streaming Zipformer CTC ONNX model into the WeNet SDK by writing an independent ONNX runner and passing frame-level CTC log probabilities into WeNet's existing WFST decoder.

The safest implementation order is:

> **Feature parity → ONNX log-probability parity → greedy-decoding parity → WeNet WFST → SDK integration**

For the WFST decoding, directly use the wenet dependency, dont invent it by yourself!
You need to write the sherpa onnx runner/backend which is independent to the wenet wfst. 
So that it provide the ctc logit of each frame for the wenet wfst.
also keep the original wenet ctc path, user can choose between them.
In the future will provide also the nemo onnx model, so please make the implementation generic. 

Do not debug the ONNX runner and the WFST decoder at the same time.

here is the implementation of last version
/home/jinyang_wang/Dev/ASR/ASR_wenet/wenet/SDK/0.0.1

you can add feature on it, so the implementation is more identical.

---

## Target architecture

```text
PCM audio
   │
   ▼
Model-specific feature pipeline
   │  [feature_frames, 80]
   ▼
StreamingCtcBackend
   ├── OriginalWenetCtcOnnxBackend
   ├── Zipformer2CtcOnnxBackend
   ├── NeMoCtcOnnxBackend        (later)
   └── Other CTC backends        (later)
   │
   ▼
CTC log probabilities [T, vocab_size]
   │
   ├── Greedy decoder            (validation/debugging)
   └── WeNet CtcWfstBeamSearch   (production)
```

The stable framework boundary should be **frame-level CTC log probabilities**, not a model-specific ONNX interface.

---

# PR 1 — Establish a reference implementation

## 1. Select the initial model

Start with the non-quantized model:

```text
sherpa-onnx-streaming-zipformer-ctc-zh-2025-06-30
```

Use FP32 during development. Switch to `model.int8.onnx` only after the FP32 implementation matches Sherpa.

Initially support only:

```text
16 kHz
mono
PCM16 input
batch size 1
CPU ONNX Runtime
```

Do not add resampling, CUDA, batching, Android, or dynamic execution-provider selection yet.

I downloaded the model here, so you can use it directly
/home/jinyang_wang/Dev/ASR/ASR_wenet/model/sherpa-onnx-streaming-zipformer-ctc-zh-2025-06-30

## 2. Create golden reference outputs

Run the official Sherpa executable on:

- Every WAV included with the model
- Approximately 10 clean Mandarin utterances
- Several long utterances
- Several noisy utterances
- Several utterances containing repeated characters and English words

Save the following reference artifacts outside Git, or under a small test-fixture directory:

```text
reference/
├── wav/
├── sherpa_greedy_results.json
├── features/
│   └── utterance_001.npy
├── log_probs/
│   ├── utterance_001_chunk_000.npy
│   ├── utterance_001_chunk_001.npy
│   └── ...
└── metadata.json
```

Write a temporary reference dumper against upstream `sherpa-onnx` that exports:

1. Fbank features
2. Input chunks passed to ONNX
3. CTC log probabilities
4. Greedy token IDs
5. State-tensor shapes after every chunk

This reference dumper is only a development oracle. It should not become a runtime dependency.

## Acceptance gate

Do not proceed until the official Sherpa executable produces stable, reproducible results on the selected test set.

---

# PR 2 — Introduce the CTC backend contract

Avoid designing a universal “any ONNX ASR model” runner. Zipformer and NeMo have different cache conventions. Generalize only the output boundary.

Add:

```text
runtime/core/decoder/streaming_ctc_backend.h
runtime/core/decoder/streaming_ctc_model_info.h
```

Suggested interface:

```cpp
struct StreamingCtcModelInfo {
  int sample_rate = 16000;
  int feature_dim = 80;

  // Number of feature frames passed to one model invocation.
  int input_window_frames = 0;

  // Number of new feature frames consumed per invocation.
  int input_shift_frames = 0;

  int vocab_size = 0;
  int blank_id = 0;

  // Usually 40 ms for this Zipformer CTC family.
  float output_frame_shift_ms = 0.0f;

  bool output_is_log_probs = true;
};

class StreamingCtcBackend {
 public:
  virtual ~StreamingCtcBackend() = default;

  virtual const StreamingCtcModelInfo& Info() const = 0;

  virtual void Reset() = 0;

  // features must contain exactly input_window_frames * feature_dim values.
  virtual void Forward(
      const float* features,
      int num_frames,
      std::vector<std::vector<float>>* log_probs) = 0;

  // A clone shares immutable model resources but owns independent stream state.
  virtual std::unique_ptr<StreamingCtcBackend> CloneStream() const = 0;
};
```

## Resource ownership

Separate immutable resources from per-stream state:

```cpp
class Zipformer2CtcOnnxResource {
  Ort::Session session;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  ZipformerMetadata metadata;
};

class Zipformer2CtcOnnxBackend : public StreamingCtcBackend {
  std::shared_ptr<Zipformer2CtcOnnxResource> resource_;
  std::vector<Ort::Value> states_;
};
```

Each recognizer stream must have private cache tensors. The ONNX session can be shared after concurrency tests pass.

## Why this fits WeNet

The current WeNet search layer already consumes:

```cpp
std::vector<std::vector<float>> logp
```

The decoder also already separates model execution from search:

```cpp
model_->ForwardEncoder(chunk_feats, &ctc_log_probs);
searcher_->Search(ctc_log_probs);
```

That is the correct extension boundary.

---

# PR 3 — Implement the Zipformer2 ONNX runner

Add:

```text
runtime/core/decoder/zipformer2_ctc_onnx_backend.h
runtime/core/decoder/zipformer2_ctc_onnx_backend.cc
```

## 1. Load and inspect the ONNX model

At startup, read:

```text
model_type
version
T
decode_chunk_len
encoder_dims
query_head_dims
value_head_dims
num_heads
num_encoder_layers
cnn_module_kernels
left_context_len
```

Validate:

```text
model_type == "zipformer2"
input[0].name == "x"
output[0].name == "log_probs"
input feature dimension == 80
batch size is dynamic or 1
number of output states == number of input states
```

Never hardcode `T=45` or `decode_chunk_len=32`, even though those are common values. Read them from ONNX metadata.

## 2. Initialize cache tensors

The model uses six state tensors for every encoder layer:

```text
cached_key
cached_nonlin_attn
cached_val1
cached_val2
cached_conv1
cached_conv2
```

It also has:

```text
embed_states
processed_lens
```

Initialize all of these to zero and replace them with the corresponding ONNX outputs after each invocation.

Implement a typed state descriptor:

```cpp
struct TensorSpec {
  std::string name;
  ONNXTensorElementDataType type;
  std::vector<int64_t> shape;
};
```

For the first implementation, follow the upstream state construction exactly. Do not attempt to infer arbitrary future model cache semantics.

## 3. Run one chunk

The execution path should be:

```cpp
inputs = {
    feature_tensor,   // [1, T, 80]
    state_0,
    state_1,
    ...
};

outputs = session.Run(inputs);

log_probs = outputs[0];
states_ = outputs[1:];
```

Output zero should have shape:

```text
[batch, output_frames, vocab_size]
```

The remaining outputs are the next streaming states.

## 4. Validate output numerically

For several output frames, calculate:

```cpp
logsumexp(log_probs[t])
```

Expected:

```text
approximately 0
```

Suggested tolerance:

```text
abs(logsumexp) < 1e-2
```

Also validate:

```text
output vocab dimension == max_token_id + 1
all values are finite
output length is stable between chunks
```

## 5. Runner unit tests

Add:

```text
runtime/core/test/zipformer2_ctc_onnx_backend_test.cc
```

Test:

- Metadata parsing
- State-tensor count
- State-tensor type and shape
- Reset determinism
- Two consecutive chunks with persistent state
- Difference between persistent-state and reset-every-chunk outputs
- Invalid model metadata
- Invalid feature dimension
- Invalid vocabulary size
- Non-finite output detection

## Acceptance gate

Using reference feature tensors, the FP32 runner's log probabilities should match the upstream runner closely:

```text
mean absolute error < 1e-5
maximum absolute error < 1e-4
```

Relax these slightly only if different ONNX Runtime builds produce harmless numerical variation.

---

# PR 4 — Implement feature-extraction parity

This is likely the highest-risk part.

## Do not use the current WeNet frontend unchanged

The target Sherpa feature configuration is approximately:

```text
sample_rate       = 16000
feature_dim       = 80
low_freq          = 20
high_freq         = -400   # 7600 Hz at 16 kHz
dither            = 0
normalize_samples = true
snip_edges        = false
frame_length      = 25 ms
frame_shift       = 10 ms
window            = povey
preemphasis       = 0.97
remove_dc_offset  = true
```

The current WeNet frontend differs in important ways:

- It defaults to raw int16-scale samples rather than normalized samples.
- Its high-frequency limit is effectively Nyquist.
- Its frame calculation only emits complete frames, similar to `snip_edges=true`.
- It does not flush reflected boundary frames when input ends.

## Recommended implementation

Add a separate frontend first:

```text
runtime/core/frontend/zipformer_feature_pipeline.h
runtime/core/frontend/zipformer_feature_pipeline.cc
```

Do not alter existing WeNet behavior until the new frontend passes parity tests.

The new implementation must support:

1. PCM16-to-`[-1, 1]` normalization
2. Configurable high frequency
3. `snip_edges=false`
4. Correct left and right boundary reflection
5. Streaming residual-sample handling
6. Final feature flush
7. Dither zero
8. Exact Povey-window behavior
9. Correct FFT size and mel-filter construction

## Reference tests

Add a fixed short WAV fixture and compare:

```text
number of frames
first five frames
middle frames
last five frames
full feature tensor
```

Suggested tolerances:

```text
mean absolute feature error < 1e-5
maximum feature error < 1e-4
```

Then feed both feature tensors into the same ONNX runner and compare log probabilities.

## Acceptance gate

The following must all match the upstream reference:

```text
feature frame count
feature values
chunk boundaries
per-chunk log probabilities
```

Do not proceed merely because the final transcript looks similar. A feature mismatch can remain hidden on easy audio and fail badly later.

---

# PR 5 — Implement the streaming scheduler and greedy validation

Sherpa does not pass arbitrary chunks to the model. It:

1. Reads `T` feature frames
2. Runs ONNX
3. Advances by `decode_chunk_len`
4. Keeps the remaining overlap
5. Preserves model states

## 1. Implement overlap handling

Suppose metadata says:

```text
T = 45
decode_chunk_len = 32
```

Then:

```text
first inference:
  read 45 new frames

later inference:
  retain 13 overlapping frames
  read 32 new frames
  concatenate into 45 frames
```

Keep this logic inside the model adapter rather than making the main decoder understand Zipformer.

## 2. Add an `AsrModel` adapter

Add:

```text
runtime/core/decoder/streaming_ctc_asr_model.h
runtime/core/decoder/streaming_ctc_asr_model.cc
```

Responsibilities:

```cpp
class StreamingCtcAsrModel : public AsrModel {
 public:
  int num_frames_for_chunk(bool start) const override;
  void ForwardEncoder(...) override;
  void Reset() override;
  std::shared_ptr<AsrModel> Copy() const override;

  void set_chunk_size(int) override;
  void set_num_left_chunks(int) override;

  void AttentionRescoring(...) override;
};
```

For Zipformer:

```cpp
num_frames_for_chunk(false) = input_window_frames;
num_frames_for_chunk(true)  = input_shift_frames;
```

Override `ForwardEncoder()` directly so that the base WeNet feature-cache implementation does not perform a second, incompatible caching operation.

`set_chunk_size()` should reject or ignore external WeNet chunk-size values. The ONNX model metadata determines valid chunk geometry.

## 3. Determine the CTC output-frame shift

After the first inference:

```cpp
output_frame_shift_ms =
    input_shift_frames * feature_frame_shift_ms / output_frames_per_chunk;
```

Validate that this remains constant. It will likely be 40 ms for this model family, but derive and validate it rather than assuming it.

This value is required for:

- Endpoint timing
- Token timestamps
- Global frame offsets

## 4. Implement a debug greedy decoder

Before WFST, implement:

```cpp
argmax each frame
remove blank
collapse repeated token IDs
```

Expose:

```text
--decoding_method=greedy
--dump_log_probs=...
--dump_topk=5
```

Compare exact token IDs with upstream Sherpa.

## 5. Handle beginning and end padding

For parity testing, add:

```text
300 ms zero padding before speech
800 ms zero padding after speech
```

before declaring input finished.

For production, make this configurable:

```json
{
  "initial_padding_ms": 300,
  "final_padding_ms": 800
}
```

Subtract initial padding from user-visible timestamps, clamping at zero.

Later, evaluate whether the full 300/800 ms is necessary. Do not change it during parity debugging.

## Acceptance gate

For FP32, require exact greedy token-ID agreement on the official model WAVs.

Then run the same gate using INT8 against the official INT8 result.

---

# PR 6 — Connect the output to WeNet WFST

Only begin this after greedy parity passes.

## 1. Feed log probabilities directly

Use the existing:

```cpp
searcher_->Search(ctc_log_probs);
```

Do not:

- Convert log probabilities back to probabilities
- Apply softmax again
- Collapse repetitions
- Remove blank frames before WFST
- Pass greedy token IDs

## 2. Configure blank consistently

Parse the blank ID from `tokens.txt`:

```text
<blk>
<blank>
or <eps>, depending on model
```

Set it in both:

```cpp
decode_options.ctc_wfst_search_opts.blank
decode_options.ctc_endpoint_config.blank
```

## 3. Disable optimizations for the first WFST test

Initially use:

```text
blank_skip_thresh = 1.0
blank_scale       = 1.0
acoustic_scale    = 1.0
nbest             = 1
rescoring_weight  = 0
ctc_weight        = 1
```

This removes blank-frame skipping as a possible source of disagreement.

Enable and tune blank skipping later.

## 4. Build graph resources from Sherpa tokens

Create:

```text
units.txt
TLG.fst
words.txt
```

from the Sherpa model vocabulary, not from an unrelated WeNet model.

The critical mapping is:

```text
FST acoustic input label = CTC token ID + 1
```

Add a graph validator that checks:

```text
blank ID exists
max token ID + 1 == ONNX vocab dimension
every non-epsilon graph input label is in [1, vocab_size]
units.txt IDs match tokens.txt
no old WeNet vocabulary is present
```

## 5. Add a bridge executable

Create:

```text
zipformer_ctc_wfst_main
```

Usage:

```bash
zipformer_ctc_wfst_main \
  --model model.onnx \
  --tokens tokens.txt \
  --fst TLG.fst \
  --words words.txt \
  --wav test.wav \
  --print-greedy \
  --print-wfst
```

Its output should show:

```text
greedy text
WFST text
number of CTC frames
forward RTF
search RTF
total RTF
```

## Acceptance gate

The WFST result should:

- Produce valid Mandarin output
- Not show systematic one-token shifts
- Not emit special symbols
- Preserve streaming partial results
- Improve, or at least not materially degrade, CER on an LM-relevant evaluation set

---

# PR 7 — Integrate runtime model selection into the SDK

The current WeNet API selects Torch or ONNX at compile time and constructs the existing WeNet `OnnxAsrModel`. It also constructs the feature pipeline before reading model-specific configuration.

That should become runtime model selection.

## 1. Add a model manifest

Example:

```json
{
  "model_type": "zipformer2_ctc_onnx",
  "model": "model.int8.onnx",
  "tokens": "tokens.txt",

  "feature": {
    "sample_rate": 16000,
    "num_bins": 80,
    "frame_length_ms": 25,
    "frame_shift_ms": 10,
    "low_freq": 20,
    "high_freq": -400,
    "dither": 0,
    "snip_edges": false,
    "normalize_samples": true
  },

  "streaming": {
    "initial_padding_ms": 300,
    "final_padding_ms": 800
  },

  "decoder": {
    "type": "wfst",
    "graph": "TLG.fst",
    "words": "words.txt",
    "acoustic_scale": 1.0,
    "blank_skip_threshold": 1.0
  }
}
```

Keep `T`, `decode_chunk_len`, state shapes, and vocabulary size out of the manifest. These should come from ONNX metadata.

## 2. Add a model factory

```cpp
std::shared_ptr<AsrModel> CreateAsrModel(
    const ModelManifest& manifest);
```

Possible types:

```text
wenet_torch
wenet_onnx
zipformer2_ctc_onnx
nemo_ctc_onnx          # later
```

## 3. Construct the model before the frontend

New initialization sequence:

```text
read manifest
load model metadata
validate model
construct matching feature pipeline
load token table
load WFST
construct decoder
```

## 4. Preserve backward compatibility

When `model.json` is absent:

```text
use the current WeNet model-directory behavior
```

Do not break existing:

```text
final.zip
encoder.onnx + ctc.onnx + decoder.onnx
units.txt
TLG.fst
words.txt
```

## 5. Remove attention rescoring from pure CTC mode

For Zipformer CTC:

```text
rescoring_weight = 0
ctc_weight = 1
reverse_weight = 0
```

The backend should return a clear error if attention rescoring is requested.

---

# PR 8 — Score calibration and production evaluation

Once the data path is correct, tune the decoder.

## Evaluation matrix

| Mode | Purpose |
|---|---|
| Official Sherpa greedy | External reference |
| Your runner greedy | Runner correctness |
| Your runner + WeNet prefix beam | Decoder-independent baseline |
| Your runner + WeNet WFST | Target system |
| Existing WeNet model + WFST | Existing SDK baseline |

Measure:

```text
CER
English-token error rate
proper-noun CER
number error rate
RTF
peak memory
first-partial latency
finalization latency
partial-result revision rate
```

## Parameters to tune

Tune independently:

```text
acoustic_scale
WFST beam
lattice_beam
max_active
min_active
blank_skip_thresh
blank_scale
endpoint blank threshold
LM weight during graph construction
word insertion penalty
```

Recommended order:

1. Disable blank skipping
2. Tune acoustic/LM balance
3. Tune beam and active states
4. Tune endpoint thresholds
5. Enable blank skipping
6. Compare FP32 and INT8

Do not tune CER using only AISHELL-style clean audio. Include long-form, spontaneous, noisy, code-switched, and domain-specific data.

---

# PR 9 — Hardening and CI

## Required tests
For testing, here are test datasets you can use
/home/jinyang_wang/Dev/ASR/ASR_wenet/data/hf_wenetspeech_test_net/wenetspeech_test_net_sample_2000

you can test it under this dir
/home/jinyang_wang/Dev/ASR/ASR_wenet/test/0.0.2

And we test please compare the result, if the result is strange or even return nothing, which means the implemenation may have problem, less likely to be the error of data.
But dont always test all data, test small subset to make sure the implementation work, then test the whole data.
test the greedy to make sure the onnx runner work, then test the full path, make sure the wfst also work. 


### Feature tests

```text
PCM chunk boundaries do not change features
single-call and streaming feature extraction agree
InputFinished flushes correctly
reset removes all residual state
```

### Runner tests

```text
reset determinism
state persistence
long-stream numerical stability
no memory growth over many chunks
invalid ONNX metadata
mismatched tokens
wrong feature dimension
```

### Decoder tests

```text
nonzero blank ID
repeated labels around blank
blank-only audio
very short audio
empty input
final partial chunk
long silence
continuous-decoding reset
timestamp monotonicity
```

### Concurrency tests

```text
multiple recognizers share model resource
each recognizer has independent ONNX states
one stream reset does not affect another
parallel Session::Run produces stable results
```

## Model artifacts

Do not commit the large model into the repository.

Add:

```text
tools/download_zipformer_ctc_zh.sh
models/zipformer_ctc_zh/model.json
models/zipformer_ctc_zh/SHA256SUMS
```

CI can use either:

- A tiny synthetic streaming CTC ONNX model
- An optional integration job that downloads the real model
- A manually triggered accuracy benchmark

---

# File-level change map

## New files

```text
runtime/core/decoder/streaming_ctc_backend.h
runtime/core/decoder/streaming_ctc_model_info.h
runtime/core/decoder/streaming_ctc_asr_model.h
runtime/core/decoder/streaming_ctc_asr_model.cc
runtime/core/decoder/zipformer2_ctc_onnx_backend.h
runtime/core/decoder/zipformer2_ctc_onnx_backend.cc

runtime/core/frontend/zipformer_feature_pipeline.h
runtime/core/frontend/zipformer_feature_pipeline.cc

runtime/core/utils/model_manifest.h
runtime/core/utils/model_manifest.cc
runtime/core/decoder/asr_model_factory.h
runtime/core/decoder/asr_model_factory.cc

runtime/core/test/zipformer_feature_pipeline_test.cc
runtime/core/test/zipformer2_ctc_onnx_backend_test.cc
runtime/core/test/zipformer_wfst_integration_test.cc
```

## Modified files

```text
runtime/core/decoder/CMakeLists.txt
runtime/core/frontend/CMakeLists.txt
runtime/core/test/CMakeLists.txt
runtime/core/api/wenet_api.cc
runtime/core/decoder/asr_decoder.cc
runtime/core/decoder/asr_decoder.h
```

---

# Mandatory go/no-go gates

## Gate 1 — Feature parity

```text
your features ≈ Sherpa features
```

## Gate 2 — Posterior parity

```text
your ONNX log_probs ≈ Sherpa log_probs
```

## Gate 3 — Greedy parity

```text
your token IDs == Sherpa token IDs
```

## Gate 4 — WFST viability

```text
WFST improves domain accuracy without unacceptable latency
```

If Gate 2 fails, do not tune the WFST.

If Gate 3 fails while Gate 2 passes, inspect blank ID, CTC collapse, padding, and token loading.

If Gate 3 passes but Gate 4 fails, the problem is graph construction or score calibration rather than model inference.

---

# Recommended implementation order

1. Download and validate the FP32 Sherpa Zipformer CTC model.
2. Build a reference dumper from upstream Sherpa.
3. Introduce the generic streaming CTC backend interface.
4. Implement ONNX metadata parsing and cache initialization.
5. Match per-chunk ONNX log probabilities using reference features.
6. Implement a Sherpa-compatible feature pipeline.
7. Match feature tensors and greedy token IDs exactly.
8. Add the WeNet `AsrModel` adapter and streaming scheduler.
9. Feed unmodified CTC log probabilities into `CtcWfstBeamSearch`.
10. Rebuild `TLG.fst` from the Sherpa token vocabulary.
11. Add runtime model selection and a model manifest.
12. Tune decoder scores and endpointing.
13. Validate INT8 accuracy and performance.
14. Add CI, concurrency tests, and production benchmarks.

---

# Reference repositories

- WeNet fork: https://github.com/alicekenway/wenet
- Sherpa ONNX: https://github.com/k2-fsa/sherpa-onnx
- Icefall Zipformer streaming CTC exporter: https://github.com/k2-fsa/icefall


