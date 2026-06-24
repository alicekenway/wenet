# Development Plan: Flashlight-Text + KenLM Shallow Fusion for WeNet SDK 0.0.2

## 1. Scope and baseline

This plan extends the SDK at:

- Repository: `alicekenway/wenet`
- Baseline commit: `bbfb78f3c34c4542ee4638b657abf1dfe495ae48`
- Baseline commit message and SDK project version: `0.0.2`
- Proposed next version: `0.0.3`

The `0.0.2` SDK already contains:

- A shared `libasr_sdk.so`.
- A dynamically linked ONNX Runtime.
- Statically linked WeNet runtime components.
- A sherpa-compatible streaming Zipformer2 CTC ONNX backend.
- A token table, CTC greedy decoder, and feature extractor.
- A standalone `zipformer_ctc_wfst_main` path that sends CTC posteriors to the WeNet WFST decoder.
- Model-package parsing and validation infrastructure.
- C and C++ SDK APIs with hidden internal symbols.

The new production decoding path will replace WFST decoding with Flashlight-Text lexicon decoding and a word-level KenLM:

```text
PCM
  ↓
feature extraction
  ↓
streaming Zipformer2 CTC ONNX AM
  ↓
frame-level CTC log scores [T, V]
  ↓
Flashlight CTC LexiconDecoder
  + multilingual lexicon
  + word-level KenLM shallow fusion
  ↓
lexicon output word IDs
  ↓
optional output-sequence mapping
  ↓
text formatting / timestamps / SDK result
```

The initial implementation should keep the old WFST executable, user can choose between the wfst and the new pathway. The new SDK package must not require `TLG.fst`, OpenFST graph generation, or Kaldi WFST search.

---

## 2. Goals

### Functional goals

1. Decode sherpa-compatible streaming CTC ONNX output with Flashlight-Text.
2. Use a strict lexicon and a **word-level KenLM** during beam search.
3. Support multilingual and code-switched output through one language-agnostic decoder.
4. Preserve streaming partial results, final results, n-best output, reset, and concurrent streams.
5. Add an optional post-lexicon mapping file.
6. Accept a missing or empty mapping file as an identity transform.
7. Keep Flashlight, KenLM, tokenizer, and decoder internals out of public SDK headers.
8. Integrate the new resources into the existing model-package validator.
9. Make decoder scoring and pruning configurable through the package manifest.
10. Retain the current greedy path as a debug fallback.

### Non-goals for the first release

1. Do not implement a custom lexicon decoder.
2. Do not train an LM inside the SDK.
3. Do not perform language identification inside the decoder.
4. Do not add a neural LM.
5. Do not implement open-vocabulary BPE fallback in the first release.
6. Do not make the post-decoder mapping affect the AM/LM beam score.
7. Do not remove the old WFST code until the new path passes accuracy and latency gates.

---

## 3. Important vocabulary distinction

There are two dictionaries, and they must never be mixed.

### AM token dictionary

This is the CTC output vocabulary:

```text
tokens.txt
```

It contains BPE pieces, characters, special symbols, blank, and possibly a word separator.

The token ID must equal the corresponding ONNX output column.

### Lexicon output dictionary

This is the decoder's word vocabulary:

```text
words.txt
```

It contains words or word-like units scored by KenLM and emitted by the Flashlight lexicon decoder.

The new mapping file operates on **this dictionary**, not on `tokens.txt`.

For the rule:

```text
牛 乃 -> 牛 奶
```

`牛`, `乃`, and `奶` are lexicon output words in `words.txt`. They may each be encoded by one or several AM/BPE tokens in `lexicon.txt`.

---

## 4. Multilingual model contract

One decoder implementation can serve all languages. Language-specific behavior belongs in data preparation, not in C++ branches.

The decoder requires:

1. A multilingual, word-segmented text corpus for the word-level LM.
2. One word-level KenLM trained on those multilingual word sequences.
3. A lexicon mapping every supported output word to the exact BPE/CTC token sequence used by the AM.
4. A stable output word dictionary.
5. Consistent normalization between AM transcripts, LM text, lexicon words, and runtime output.

Example segmented LM text:

```text
打开 MTV
turn on the music
播放 Taylor Swift
使用 GPU 模式
```

Possible lexicon entries:

```text
打开 ▁打 开
MTV ▁M T V
turn ▁turn
music ▁mu sic
GPU ▁G P U
模式 ▁模 式
```

The exact token sequence must be produced by the tokenizer used to train the acoustic model. It must not be approximated manually.

### Strict lexicon limitation

The first implementation is a strict lexicon decoder. Words outside the lexicon either:

- emit `<unk>` with `unk_score`, or
- are rejected when unknown-word decoding is disabled.

For multilingual production quality, lexicon coverage and normalization are therefore critical. Open-vocabulary fallback can be a later extension.

---

## 5. Model package layout

A Flashlight/KenLM package should contain:

```text
model_package/
├── model.onnx
├── tokens.txt
├── words.txt
├── lexicon.txt
├── lm.bin
├── output_mapping.txt
├── tokenizer.model          # recommended for reproducibility/tooling
└── manifest.json            # use the existing package-manifest mechanism
```

The exact ONNX filename may remain configurable.

### `tokens.txt`

Requirements:

- Explicit integer IDs.
- IDs are unique and contiguous unless the existing token loader supports sparse IDs.
- `max_token_id + 1 == ONNX vocabulary dimension`.
- Contains the configured CTC blank token.
- Contains the configured silence/word-separator token required by the Flashlight decoder.
- All lexicon spelling tokens exist here.

### `words.txt`

Suggested format:

```text
<unk> 0
打开 1
MTV 2
turn 3
music 4
牛 5
乃 6
奶 7
```

Requirements:

- Stable IDs across package rebuilds whenever possible.
- Unique strings and IDs.
- Contains `<unk>`.
- Excludes `<s>` and `</s>` from emitted results even if KenLM uses them internally.
- Every lexicon output label is present.
- Every source and target item in `output_mapping.txt` is present in version 1.

### `lexicon.txt`

Suggested format:

```text
打开 ▁打 开
MTV ▁M T V
turn ▁turn
music ▁mu sic
牛 ▁牛
乃 ▁乃
奶 ▁奶
```

Allow multiple spellings by repeating the word:

```text
read ▁read
read ▁re ad
```

Validation requirements:

- The first field is in `words.txt`.
- Every remaining field is in `tokens.txt`.
- Every line contains at least one spelling token.
- Duplicate identical entries are rejected or deduplicated deterministically.
- Empty spellings are rejected.

### `lm.bin`

Use a KenLM binary rather than loading ARPA in production.


Requirements:

- Word strings use exactly the same normalization as `words.txt`.
- `<unk>` behavior is defined.
- Start/end sentence symbols are present as required by KenLM.
- The model is trained on word-separated multilingual text.
- Package preparation records the KenLM order and checksum.

### `output_mapping.txt`

This file may be:

- absent, when the manifest does not name it;
- zero bytes;
- comments and blank lines only; or
- populated with rewrite rules.

All three empty cases mean identity mapping.

---

## 6. Output mapping specification

### 6.1 Purpose

The mapping is a post-decoder correction layer for controlled future modifications.

It is applied after:

```text
Flashlight lexicon + KenLM decoding
```

and before:

```text
text joining, punctuation/ITN, result JSON, and public SDK output
```

It does not change the decoder score and does not influence beam search.

### 6.2 Canonical syntax

One UTF-8 rule per line:

```text
source_token_1 source_token_2 -> target_token_1 target_token_2
```

Example:

```text
牛 乃 -> 牛 奶
```

Comments and blank lines:

```text
# Common recognition correction
牛 乃 -> 牛 奶

# Empty lines are allowed
```

Parsing rules:

- The delimiter is the exact ASCII sequence ` -> `.
- Leading and trailing whitespace is ignored.
- Fields on each side are split by ASCII whitespace.
- Both sides must be non-empty.
- `#` starts a full-line comment after leading whitespace.
- Inline comments are not supported in version 1.
- Files are UTF-8 without required BOM.

### 6.3 Dictionary validation

Mandatory:

- Every source token must exist in the lexicon output dictionary, as requested.
- Every target token also must exist in the lexicon output dictionary in version 1.

Requiring targets to exist keeps result IDs, timestamps, JSON, and downstream consumers consistent. A future version may permit literal target strings, but that should be a separate format version.

Invalid example:

```text
not_in_words 牛 -> 牛 奶
```

Package loading fails with a line-numbered error.

### 6.4 Matching semantics

Use deterministic, single-pass, left-to-right, longest-source matching.

Given:

```text
A B -> X
A B C -> Y
```

and input:

```text
A B C D
```

the result is:

```text
Y D
```

Rules:

1. At each input position, select the longest matching source sequence.
2. Equal source sequences are duplicates and are rejected at load time.
3. If nothing matches, copy one input token.
4. Replacements are not fed back into the mapper.
5. The mapper is non-recursive.
6. Cycles are harmless because only one pass is performed.
7. Rule order does not affect matches except that exact duplicate sources are errors.

Store rules in a trie keyed by output word IDs.

### 6.5 Partial results

Apply the same mapper to the entire current best word sequence whenever a partial result is requested.

A rule spanning an unfinished tail will not fire until all source words are present. Partial revisions are acceptable and already normal in streaming ASR.

Example:

```text
partial 1: 牛
partial 2 raw: 牛 乃
partial 2 mapped: 牛 奶
```

### 6.6 N-best results

Apply mapping independently to every hypothesis.

If multiple raw hypotheses map to the same final word sequence:

- keep the hypothesis with the highest decoder score;
- optionally preserve the number of merged hypotheses in debug metadata.

### 6.7 Timestamp policy

Keep raw decoder word spans internally.

For one-to-one and equal-length replacements:

- preserve timestamps positionally.

For length-changing replacements:

1. Compute the union interval from the first source start to the last source end.
2. Divide that interval into contiguous target intervals in target order.
3. Mark generated timestamps as mapped/derived internally.
4. Keep the decoder score unchanged.

If the public API cannot mark derived timestamps without an ABI change, document this behavior and keep the marker internal.

---

## 7. Proposed decoder configuration

Extend the existing package manifest rather than adding public ABI fields wherever possible.

Logical example:

```json
{
  "model": {
    "type": "zipformer2_ctc_onnx",
    "path": "model.onnx",
    "tokens": "tokens.txt",
    "blank_token": "<blk>",
    "sil_token": "|"
  },
  "decoder": {
    "type": "flashlight_lexicon_kenlm",
    "words": "words.txt",
    "lexicon": "lexicon.txt",
    "lm": "lm.bin",
    "mapping": "output_mapping.txt",

    "beam_size": 50,
    "beam_size_token": 20,
    "beam_threshold": 25.0,

    "lm_weight": 1.5,
    "word_score": -0.5,
    "unk_score": -5.0,
    "sil_score": 0.0,

    "log_add": false,
    "allow_unk": true,
    "smearing": "max",
    "nbest": 1
  }
}
```

Initial defaults are only starting points. Tune them on held-out multilingual data.

### Decoder construction settings

Use Flashlight with:

```text
criterionType = CTC
isLmToken     = false
```

The LM is word-level and is scored when a lexicon word is completed.

Do not manually convert KenLM log bases outside the Flashlight KenLM adapter. Treat the adapter's score as the decoder's LM score source.

---

## 8. Target internal architecture

### 8.1 Shared immutable resources

One engine-level object shared by streams:

```cpp
struct FlashlightDecoderResource {
  TokenDictionary am_tokens;
  WordDictionary output_words;

  std::shared_ptr<Trie> lexicon_trie;
  std::shared_ptr<LM> word_lm;

  int blank_id;
  int sil_id;
  int unk_word_id;

  FlashlightDecoderOptions options;
  OutputSequenceMapper output_mapper;
};
```

The resource must be immutable after construction.

### 8.2 Per-stream decoder state

Each ASR stream owns:

```cpp
class FlashlightCtcStreamDecoder {
 public:
  Status Start();
  Status DecodeChunk(const float* log_scores, int frames, int vocab_size);
  StatusOr<DecodedHypothesis> PartialResult() const;
  StatusOr<std::vector<DecodedHypothesis>> Finalize();
  Status Reset();

 private:
  std::unique_ptr<LexiconDecoder> decoder_;
  bool started_ = false;
  bool finalized_ = false;
};
```

Never share a mutable Flashlight decoder instance between streams.

### 8.3 Decoder factory

Introduce an internal decoder interface:

```cpp
class CtcDecoderBackend {
 public:
  virtual ~CtcDecoderBackend() = default;
  virtual Status Start() = 0;
  virtual Status DecodeChunk(const float* data, int t, int v) = 0;
  virtual StatusOr<DecodedHypothesis> PartialResult() const = 0;
  virtual StatusOr<std::vector<DecodedHypothesis>> Finalize() = 0;
  virtual Status Reset() = 0;
};
```

Backends:

```text
CtcGreedyDecoderBackend
FlashlightLexiconKenLmDecoderBackend
LegacyWfstDecoderBackend       # temporary comparison only
```

### 8.4 Result mapping

Create a decoder-independent internal representation:

```cpp
struct DecodedWord {
  int word_id;
  std::string text;
  int start_frame;
  int end_frame;
};

struct DecodedHypothesis {
  double total_score;
  double am_score;
  double lm_score;
  std::vector<int> token_ids;
  std::vector<DecodedWord> raw_words;
  std::vector<DecodedWord> mapped_words;
};
```

The mapper consumes `raw_words` and produces `mapped_words`.

---

## 9. Build and dependency strategy

### 9.1 Pin dependencies

Use pinned source revisions, not floating branches.

Recommended repository layout:

```text
SDK/third_party/
├── flashlight-text/
└── kenlm/
```

Use Git submodules or a reproducible archive with checksums.

Record:

- Flashlight-Text commit.
- KenLM commit.
- Build options.
- License files.
- Compiler version used for release artifacts.

### 9.2 Linkage

Preferred SDK distribution:

```text
libasr_sdk.so        shared
ONNX Runtime         shared, as in 0.0.2
Flashlight-Text      static and private
KenLM                static and private
```

Requirements:

- Build static dependencies with position-independent code.
- Hide their symbols behind `libasr_sdk.so`.
- Preserve the existing hidden-visibility policy.
- Use `--exclude-libs,ALL` where supported.
- Do not include Flashlight or KenLM headers from public SDK headers.
- Verify dependency licensing before distributing statically linked binaries.

### 9.3 CMake options

Add:

```cmake
option(ASR_SDK_ENABLE_FLASHLIGHT_DECODER
       "Build Flashlight lexicon + KenLM decoder" ON)

option(ASR_SDK_ENABLE_LEGACY_WFST
       "Build legacy WFST comparison decoder" OFF)

set(ASR_SDK_FLASHLIGHT_TEXT_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/flashlight-text"
    CACHE PATH "Pinned Flashlight-Text source")

set(ASR_SDK_KENLM_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/kenlm"
    CACHE PATH "Pinned KenLM source")
```

Add dedicated modules:

```text
cmake/FlashlightText.cmake
cmake/KenLM.cmake
```

Exact target names must be verified against the pinned Flashlight-Text revision. Wrap external targets behind SDK-owned aliases so future dependency changes do not spread through the tree.

### 9.4 Internal core library

In `0.0.2`, sherpa/Zipformer sources are grouped for the standalone WFST tool rather than the main SDK library.

Create:

```cmake
add_library(asr_sdk_ctc_core STATIC
  src/sherpa_onnx_wenet/ctc_greedy_decoder.cc
  src/sherpa_onnx_wenet/token_table.cc
  src/sherpa_onnx_wenet/whisper_feature_extractor.cc
  src/sherpa_onnx_wenet/zipformer2_ctc_onnx_backend.cc

  src/flashlight_decoder/flashlight_decoder_resource.cc
  src/flashlight_decoder/flashlight_ctc_stream_decoder.cc
  src/flashlight_decoder/lexicon_loader.cc
  src/flashlight_decoder/word_dictionary.cc
  src/flashlight_decoder/output_sequence_mapper.cc
  src/flashlight_decoder/flashlight_result_mapper.cc
)
```

Link this internal library into:

- `asr_sdk`;
- the new standalone validation tool;
- tests.

This avoids compiling the same backend sources independently into multiple binaries.

---

## 10. File-level implementation map

### New files

```text
SDK/cmake/FlashlightText.cmake
SDK/cmake/KenLM.cmake

SDK/src/decoder/ctc_decoder_backend.h
SDK/src/decoder/ctc_decoder_factory.h
SDK/src/decoder/ctc_decoder_factory.cc

SDK/src/flashlight_decoder/flashlight_decoder_options.h
SDK/src/flashlight_decoder/flashlight_decoder_resource.h
SDK/src/flashlight_decoder/flashlight_decoder_resource.cc
SDK/src/flashlight_decoder/flashlight_ctc_stream_decoder.h
SDK/src/flashlight_decoder/flashlight_ctc_stream_decoder.cc
SDK/src/flashlight_decoder/lexicon_loader.h
SDK/src/flashlight_decoder/lexicon_loader.cc
SDK/src/flashlight_decoder/word_dictionary.h
SDK/src/flashlight_decoder/word_dictionary.cc
SDK/src/flashlight_decoder/output_sequence_mapper.h
SDK/src/flashlight_decoder/output_sequence_mapper.cc
SDK/src/flashlight_decoder/flashlight_result_mapper.h
SDK/src/flashlight_decoder/flashlight_result_mapper.cc

SDK/tools/zipformer_ctc_flashlight_main.cc

SDK/scripts/build_multilingual_lexicon.py
SDK/scripts/prepare_flashlight_runtime_package.sh

SDK/test/flashlight_decoder_test.cc
SDK/test/lexicon_loader_test.cc
SDK/test/output_sequence_mapper_test.cc
SDK/test/flashlight_streaming_test.cc
SDK/test/flashlight_package_validation_test.cc
```

### Modified files

```text
SDK/CMakeLists.txt
SDK/cmake/Options.cmake
SDK/cmake/InstallRules.cmake

SDK/src/package/model_package.cc
SDK/src/package/model_package_validator.cc

SDK/src/sdk/asr_engine.cc
SDK/src/sdk/asr_stream.cc
SDK/src/sdk/result_json.cc

SDK/include/asr_sdk/version.h
SDK/README.md
SDK/guide.md
```

Modify public configuration headers only when unavoidable. Prefer manifest-driven settings to preserve ABI version 1. If a public struct layout changes, bump the ABI version deliberately.

---

## 11. Step-by-step delivery plan

# Phase 0 — Freeze the 0.0.2 baseline

### Tasks

1. Create a development branch from commit:
   ```text
   bbfb78f3c34c4542ee4638b657abf1dfe495ae48
   ```
2. Archive the exact 0.0.2 build command and dependency versions.
3. Select a fixed multilingual/code-switching evaluation set.
4. Record for greedy and WFST:
   - final transcript;
   - partial transcripts;
   - CER/WER;
   - decoder-only latency;
   - total RTF;
   - peak RSS;
   - first-partial latency;
   - finalization latency.
5. Save raw CTC log-score fixtures for a small number of utterances.

### Required fixtures

Include cases such as:

```text
打开 MTV
牛奶
牛乃
turn on the music
播放 Taylor Swift
使用 GPU 模式
```

### Exit criteria

- The 0.0.2 binaries are reproducible.
- Golden CTC score fixtures are available.
- WFST performance is measured rather than estimated.

---

# Phase 1 — Integrate Flashlight-Text and KenLM into CMake

### Tasks

1. Pin both dependencies.
2. Disable unneeded examples, Python bindings, tests, and training components.
3. Build Flashlight-Text and KenLM as private static dependencies.
4. Enable PIC.
5. Add symbol-hiding and link-order checks.
6. Add dependency versions to `print_build_info`.
7. Add a CI build with the Flashlight decoder enabled.
8. Keep `ASR_SDK_ENABLE_LEGACY_WFST` available for comparison.

### Exit criteria

- `libasr_sdk.so` links successfully.
- Public headers do not include dependency headers.
- A consumer program links only against the SDK and its documented runtime libraries.
- No unexpected Flashlight/KenLM shared-library dependency appears when static linkage is selected.

---

# Phase 2 — Add package schema and validators

### Tasks

1. Extend `ModelPackage` with:
   - decoder type;
   - lexicon path;
   - output word dictionary path;
   - KenLM path;
   - mapping path;
   - Flashlight decoder parameters;
   - blank/silence/unknown symbol names.
2. Update `ModelPackageValidator`.
3. Validate ONNX vocabulary dimension against `tokens.txt`.
4. Validate all lexicon spelling pieces against the AM token dictionary.
5. Validate all lexicon labels against the output word dictionary.
6. Validate KenLM special symbols and output vocabulary compatibility.
7. Validate mapping syntax and dictionary membership.
8. Permit an absent or empty mapping file.
9. Update `inspect_package` to print:
   - token count;
   - word count;
   - lexicon entry count;
   - LM order if available;
   - mapping rule count;
   - blank/sil/unk IDs;
   - estimated trie size.

### Failure policy

All resource mismatches fail at engine initialization, not during streaming.

Errors must include:

- resource filename;
- line number where relevant;
- offending token or word;
- expected dictionary.

### Exit criteria

A malformed package cannot create an engine. A valid package is fully validated before any audio is accepted.

---

# Phase 3 — Build the immutable Flashlight decoder resource

### Tasks

1. Load AM token dictionary while preserving IDs.
2. Load output word dictionary.
3. Load the KenLM binary through the Flashlight LM adapter.
4. Parse lexicon spellings.
5. Build a Flashlight trie.
6. Insert each word spelling with the output word ID.
7. Initialize trie look-ahead/smearing scores.
8. Use `MAX` smearing initially.
9. Store blank, silence, and unknown IDs.
10. Store decoder options.
11. Load and compile the output mapping into an ID trie.
12. Share the completed resource across decoder streams.

### Exit criteria

- Resource construction is deterministic.
- Two loads of the same package produce identical IDs and trie counts.
- Resource state is immutable and thread-safe after construction.

---

# Phase 4 — Implement a standalone Flashlight CTC decoder tool

Add:

```text
zipformer_ctc_flashlight_main
```

Suggested CLI:

```bash
zipformer_ctc_flashlight_main \
  --model model.onnx \
  --tokens tokens.txt \
  --words words.txt \
  --lexicon lexicon.txt \
  --lm lm.bin \
  --mapping output_mapping.txt \
  --wav test.wav \
  --beam_size 50 \
  --beam_size_token 20 \
  --beam_threshold 25 \
  --lm_weight 1.5 \
  --word_score -0.5
```

### Tasks

1. Reuse the 0.0.2 Zipformer ONNX backend.
2. Feed contiguous `[T, V]` CTC log scores to `LexiconDecoder::decodeStep`.
3. Use:
   ```text
   CriterionType::CTC
   isLmToken=false
   ```
4. Call `decodeBegin` once per utterance.
5. Call `decodeStep` for each model output chunk.
6. Retrieve best partial hypotheses.
7. Call `decodeEnd` exactly once.
8. Retrieve final n-best hypotheses.
9. Map decoder output IDs to words.
10. Apply `OutputSequenceMapper`.
11. Print:
    - raw words;
    - mapped words;
    - AM score;
    - LM score;
    - total score;
    - search time;
    - total RTF.

### Numerical contract

- The ONNX backend outputs log scores in the same order as `tokens.txt`.
- Do not apply softmax again.
- Do not CTC-collapse before Flashlight.
- Do not remove blank frames.
- Do not perform manual KenLM log-base conversion outside the Flashlight adapter.

### Exit criteria

- A toy LM changes the winner in a known ambiguous case.
- Single-chunk and multi-chunk final results agree.
- The tool no longer needs `TLG.fst`.

---

# Phase 5 — Implement the output sequence mapper

### Tasks

1. Implement parser and line-numbered errors.
2. Resolve strings to output word IDs at load time.
3. Build a longest-match source trie.
4. Implement single-pass rewriting.
5. Handle empty file as identity.
6. Apply to partial, final, and n-best hypotheses.
7. Deduplicate mapped n-best outputs.
8. Implement timestamp transfer.
9. Add optional debug logging:
   ```text
   raw:    牛 乃
   mapped: 牛 奶
   rule:   line 1
   ```
10. Keep the decoder score unchanged.

### Required tests

```text
empty file                         -> identity
comments only                      -> identity
牛 乃 -> 牛 奶                    -> basic replacement
A B -> X and A B C -> Y            -> longest match
duplicate source                   -> load error
unknown source word                -> load error
unknown target word                -> load error in v1
empty source or target             -> load error
cyclic rules                       -> safe because non-recursive
two raw n-best paths same mapped   -> keep highest score
UTF-8 content                      -> correct
```

### Exit criteria

The exact example:

```text
raw decoder output: 牛 乃
mapping rule:       牛 乃 -> 牛 奶
public output:      牛 奶
```

passes in both standalone and SDK tests.

---

# Phase 6 — Integrate the decoder into `libasr_sdk.so`

### Tasks

1. Introduce the decoder backend interface and factory.
2. Make the engine select the decoder from the package manifest.
3. Move the Zipformer backend and Flashlight decoder into a shared internal static core library.
4. Create one immutable Flashlight resource per engine.
5. Create one mutable decoder instance per `AsrStream`.
6. Feed each ONNX output chunk directly into the stream decoder.
7. Convert partial/final Flashlight results into the existing SDK result type.
8. Apply mapping before public result formatting.
9. Preserve reset and continuous-decoding behavior.
10. Preserve C and C++ APIs.
11. Avoid public ABI changes by keeping tuning parameters in the package manifest.
12. If ABI changes are unavoidable, bump ABI version and add compatibility tests.

### Result order

```text
Flashlight raw word output
  ↓
output mapping
  ↓
spacing/joining
  ↓
optional punctuation/ITN
  ↓
SDK result JSON
```

Do not apply the mapping after ITN, because mapped source words may no longer be visible.

### Exit criteria

- Existing SDK examples work with the new package.
- Multiple streams can share one engine safely.
- Reset produces the same result as a fresh stream.
- Mapping is visible through C, C++, and JSON result paths.

---

# Phase 7 — Build lexicon and package preparation tools

Add a reproducible script:

```text
build_multilingual_lexicon.py
```

Inputs:

```text
word vocabulary
exact AM tokenizer model
AM tokens.txt
normalization config
```

Outputs:

```text
words.txt
lexicon.txt
coverage report
rejected words report
```

Algorithm:

1. Read the desired word vocabulary, usually from the KenLM vocabulary plus domain additions.
2. Normalize each word exactly as in AM training.
3. Encode each word with the exact AM tokenizer.
4. Verify every resulting piece exists in `tokens.txt`.
5. Emit the word-to-piece lexicon.
6. Emit stable word IDs.
7. Report words that cannot be encoded.
8. Add special words according to policy.
9. Compute checksums.
10. Run `inspect_package`.

Add:

```text
prepare_flashlight_runtime_package.sh
```

It should assemble:

- ONNX AM;
- AM token dictionary;
- output word dictionary;
- lexicon;
- KenLM binary;
- empty mapping file by default;
- manifest;
- checksums.

### Empty mapping creation

The preparation script should always create:

```bash
: > output_mapping.txt
```

unless the caller supplies a populated mapping.

### Exit criteria

A package can be rebuilt from source resources without manual file editing.

---

# Phase 8 — Accuracy and decoder tuning

Tune in this order:

1. Confirm all dictionaries and token IDs.
2. Disable mapping during baseline accuracy tuning.
3. Tune `beam_size_token`.
4. Tune `beam_threshold`.
5. Tune `beam_size`.
6. Tune `lm_weight`.
7. Tune `word_score`.
8. Tune `unk_score`.
9. Tune `sil_score`.
10. Compare `log_add=false` and `true`.
11. Evaluate trie smearing choices if needed.
12. Re-enable mapping and measure only intended corrections.

Grid example:

```text
lm_weight:      0.5, 1.0, 1.5, 2.0, 2.5
word_score:    -2.0, -1.0, -0.5, 0.0, 0.5
beam_size:      20, 50, 100
beam_token:     10, 20, 40
beam_threshold: 10, 20, 25, 40
```

Evaluate by language and code-switch category, not only aggregate metrics.

Metrics:

- CER for CJK-heavy sets.
- WER for whitespace-delimited languages.
- Mixed error rate or token error rate for code-switch sets.
- Named-entity accuracy.
- Acronym accuracy.
- `<unk>` rate.
- Decoder-only RTF.
- End-to-end RTF.
- Peak memory.
- First partial latency.
- Finalization latency.
- Partial revision rate.

### Exit criteria

Set project-specific gates before release. Suggested relative gates:

- Flashlight search is materially faster than the 0.0.2 WFST search on the same CTC fixtures.
- No unacceptable aggregate accuracy regression after tuning.
- Code-switching examples decode without language-specific C++ branches.
- `<unk>` rate is within the agreed lexicon-coverage target.
- Mapping changes only listed patterns.

---

# Phase 9 — Remove production WFST dependency

Only after the Flashlight path passes all gates:

1. Make `flashlight_lexicon_kenlm` the default decoder for new packages.
2. Stop shipping TLG resources.
3. Compile the WFST comparison tool only when:
   ```text
   ASR_SDK_ENABLE_LEGACY_WFST=ON
   ```
4. Remove WeNet/Kaldi/OpenFST decoder archives from the production SDK linkage if no other SDK path uses them.
5. Re-measure binary size and exported symbols.
6. Keep a migration document for old packages.

Do not remove the old path in the same PR that first introduces Flashlight. A short comparison period reduces risk.

---

## 12. Test plan

### Unit tests

#### Dictionary tests

- Stable ID loading.
- Duplicate IDs.
- Duplicate strings.
- Missing blank.
- Missing silence token.
- Sparse IDs.
- ONNX vocabulary mismatch.

#### Lexicon tests

- Single-piece word.
- Multi-piece word.
- Multiple spellings.
- Unknown spelling token.
- Unknown output word.
- Empty spelling.
- UTF-8 words.
- Code-switched lexicon.

#### KenLM tests

- Start state.
- Word transition.
- End state.
- Unknown word.
- Deterministic score.
- LM changes best path in a toy case.

#### CTC tests

- Blank-only audio.
- Repeated token without blank.
- Repeated token separated by blank.
- Word boundary behavior.
- Multiple words.
- Chunk boundary inside a BPE spelling.
- Final partial word.
- Empty utterance.

#### Mapping tests

All cases listed in Phase 5.

### Integration tests

1. Saved CTC emissions decoded without running ONNX.
2. Full WAV through Zipformer ONNX and Flashlight.
3. Same WAV with multiple audio chunk sizes.
4. Partial and final output.
5. Stream reset.
6. Two simultaneous streams.
7. N-best output.
8. Empty mapping package.
9. Populated mapping package.
10. Multilingual/code-switch examples.

### Regression tests

- Greedy output remains unchanged.
- Public C and C++ examples still compile.
- Old 0.0.2 package behavior is either preserved or rejected with a clear migration error.
- Build-info output reports dependency revisions.
- Hidden symbol policy remains effective.

---

## 13. Performance considerations

### Avoid unnecessary copies

The 0.0.2 backend may expose nested vectors. Flashlight expects contiguous frame-major emissions.

Prefer:

```cpp
struct CtcLogScores {
  std::vector<float> data;  // [T * V]
  int frames;
  int vocab_size;
};
```

If changing the backend is risky, flatten once per model chunk, not once per frame.

### Token pruning

`beam_size_token` is a major speed control. Start conservatively and tune on multilingual data.

### Resource sharing

Share:

- KenLM model;
- token dictionary;
- word dictionary;
- lexicon trie;
- mapping trie.

Do not share:

- Flashlight decoder hypothesis state;
- stream counters;
- partial-result buffers.

### Memory

A multilingual word lexicon can dominate memory. Record:

- number of words;
- number of spellings;
- trie nodes;
- LM file size;
- resident memory after engine creation;
- per-stream incremental memory.

---

## 14. Risks and mitigations

### Risk: lexicon coverage is insufficient

Effect:

- high `<unk>` rate;
- names and new terms fail;
- code-switch words disappear.

Mitigation:

- coverage reports during package creation;
- domain lexicon additions;
- frequent lexicon/LM refresh;
- explicit `<unk>` policy;
- later open-vocabulary fallback if required.

### Risk: tokenizer mismatch

Effect:

- lexicon paths do not correspond to AM output;
- decoder accuracy collapses.

Mitigation:

- generate lexicon only with the original tokenizer;
- store tokenizer checksum;
- validate all lexicon pieces;
- include tokenizer metadata in the package.

### Risk: output word segmentation differs between AM and LM data

Effect:

- correct acoustic sequences are disfavored or unreachable.

Mitigation:

- one shared normalization and segmentation pipeline;
- version the pipeline;
- regression tests on code-switch text.

### Risk: no suitable silence/word-separator AM token

Effect:

- Flashlight lexicon boundary handling is ambiguous.

Mitigation:

- make `sil_token` an explicit package contract;
- fail validation when missing;
- ensure future multilingual CTC training includes the required separator/boundary representation.

### Risk: mapping hides decoder errors

Effect:

- metrics look better without improving search;
- excessive rules accumulate.

Mitigation:

- always retain raw output in debug logs;
- report mapping hit counts;
- evaluate accuracy both before and after mapping;
- review rule growth.

### Risk: mapping timestamps are synthetic

Mitigation:

- preserve raw spans internally;
- document derived timestamp policy;
- prefer equal-length corrections when timestamps matter.

### Risk: static dependency licensing or binary size

Mitigation:

- perform a dependency-license review;
- record notices;
- compare static and dynamic packaging;
- make linkage mode configurable if required.

---

## 15. Pull request sequence

### PR 1 — Dependency and build foundation

- Pin Flashlight-Text and KenLM.
- Add CMake modules and options.
- Add dependency build-info.
- No decoder behavior change.

### PR 2 — Package schema and validation

- Add lexicon, words, LM, and mapping fields.
- Extend `inspect_package`.
- Add malformed-package tests.

### PR 3 — Flashlight resource loader

- Token dictionary.
- Word dictionary.
- Lexicon trie.
- KenLM loader.
- Unit tests.

### PR 4 — Standalone streaming decoder

- `zipformer_ctc_flashlight_main`.
- Partial/final/n-best.
- Saved-emission tests.
- Performance instrumentation.

### PR 5 — Output sequence mapper

- Parser.
- Dictionary validation.
- Longest-match rewrite trie.
- Timestamp and n-best behavior.
- Exact `牛 乃 -> 牛 奶` integration test.

### PR 6 — SDK integration

- Decoder backend/factory.
- Engine and stream integration.
- C/C++/JSON results.
- Concurrency and reset tests.

### PR 7 — Package-generation tooling

- Multilingual lexicon generator.
- Runtime package script.
- Coverage reports.
- Empty default mapping file.

### PR 8 — Tuning and migration

- Multilingual tuning.
- WFST comparison.
- Documentation.
- Make Flashlight default for new packages.

### PR 9 — Optional WFST removal

- Compile legacy path only behind an option.
- Remove unused runtime dependencies after verification.

---

## 16. Definition of done

The feature is complete when all of the following are true:

- [ ] The SDK builds from the pinned 0.0.2 baseline with Flashlight-Text and KenLM.
- [ ] The production decoder uses Flashlight CTC lexicon beam search.
- [ ] KenLM is word-level and scored at lexicon word completion.
- [ ] The decoder supports multilingual/code-switched lexicon entries without language-specific C++ logic.
- [ ] ONNX output IDs, AM token IDs, lexicon spellings, word IDs, and LM words are fully validated.
- [ ] A package no longer requires `TLG.fst`.
- [ ] An absent or empty `output_mapping.txt` is an identity mapping.
- [ ] `牛 乃 -> 牛 奶` works exactly as specified.
- [ ] Every mapping source token is validated against the lexicon output dictionary.
- [ ] Every mapping target token is also validated against that dictionary in version 1.
- [ ] Mapping is deterministic, longest-match, single-pass, and non-recursive.
- [ ] Partial, final, and n-best outputs pass mapping tests.
- [ ] Multiple streams are thread-safe.
- [ ] Stream reset is deterministic.
- [ ] Search latency is measured against the 0.0.2 WFST baseline.
- [ ] Accuracy is measured by language and code-switch category.
- [ ] Public SDK headers do not expose Flashlight or KenLM types.
- [ ] Dependency versions and licenses are recorded.
- [ ] The SDK version is bumped to `0.0.3`.

---

## 17. Recommended first implementation slice

The smallest useful end-to-end slice is:

1. Build Flashlight-Text and KenLM.
2. Load `tokens.txt`, `words.txt`, `lexicon.txt`, and `lm.bin`.
3. Feed saved CTC emissions into Flashlight.
4. Produce one final best word sequence.
5. Load an empty mapping file.
6. Add:
   ```text
   牛 乃 -> 牛 奶
   ```
7. Verify raw and mapped output.
8. Only then connect streaming ONNX and the public SDK.

This isolates decoder correctness from feature extraction, ONNX cache handling, audio streaming, and API integration.
