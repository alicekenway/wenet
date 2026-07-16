# WeNet SDK 0.0.6: Flashlight + KenLM CTC Decoder

This SDK adds a Flashlight-Text CTC lexicon decoder with a word-level KenLM
shallow-fusion score. The old WFST comparison tool can still be built with
`ASR_SDK_ENABLE_LEGACY_WFST=ON`, but the 0.0.6 package does not require
`TLG.fst`.

The Flashlight package path selects the acoustic backend from the single ONNX
model's `model_type` metadata. 0.0.6 supports both `zipformer2` and
`wenet_ctc`.

Public headers remain under `include/asr_sdk` and do not expose Flashlight,
KenLM, WeNet, or ONNX Runtime headers.

## Runtime contact lists and handmade contact bias

0.0.6 recognizes runner-provided contacts without putting private contact data
in a reusable model package. A contact-capable package contains the normal
main LM plus a small, handmade **pattern-bias** contact LM. It is not a
statistical LM trained from contact sentences. Its rules are written as TSV,
for example:

```text
CALL <CONTACT>    9
DIAL <CONTACT>    9
<CONTACT>         5
```

`wenet/kenlm_lm/tools/generate_contact_bias_arpa.py` converts those positive
bonuses into a deliberately sparse ARPA and writes `contact_lm.meta.json`.
The SDK accepts only this `pattern_bias_v1` format for runtime contacts; an old
statistical contact LM is rejected during package validation.

The staged, shared `words.txt` contains `<CONTACT>`. Neither `tokens.txt` nor
the static `lexicon.txt` may contain it: a complete runtime name is a virtual
lexicon label whose AM spelling is supplied at runtime. The package manifest
declares the contact LM, metadata, and its independent lambda:

```json
{
  "lm": "lm.bin",
  "lm_weight": 0.5,
  "contact_lm": "contact_lm.bin",
  "contact_lm_weight": 1.5,
  "contact_lm_mode": "pattern_bias_v1",
  "contact_lm_metadata": "contact_lm.meta.json",
  "contact_lm_accumulation_factor": 0.5,
  "contact_class_word": "<CONTACT>",
  "smearing": "max"
}
```

The SDK runs the AM once. Normal words receive only the weighted main-LM
score. A name may expand as one virtual `<CONTACT>` label: its acoustic tokens
are consumed normally, but literal name words are not sent to either LM. The
contact branch is retained only when `<CONTACT>` completes one configured
pattern. For a rule with bonus `S`, a name of `n` logical words receives:

```text
contact_lm_weight * S * (1 + a + ... + a^(n-1))
```

where `a` is `contact_lm_accumulation_factor` in `[0, 1]`. The ordinary
decoder word-score correction is also applied for the extra words represented
by a multiword virtual label. After the name, main-LM history is reset rather
than connecting the word after the name to the word before it.

Ordinary words not explicitly present in either LM do not discard an acoustic
hypothesis: that LM contributes zero and its history resets. In contrast, an
unmatched `<CONTACT>` path is discarded. This keeps the handcrafted contact LM
from affecting normal words while preventing a contact expansion outside a
configured rule.

Dynamic contact labels use neutral trie lookahead. The contact bonus is added
only after a name completes a verified `<CONTACT>` rule, so shared AM prefixes
and the size of the supplied contact list cannot add a hidden extra reward.

A contact-capable package requires `smearing=max`. A main-only package remains
valid for ordinary decoding; compiling a non-empty contact context against it
returns `FailedPrecondition`.

The package workflow takes the main LM from `MAIN_LM_BIN` (default
`${LM_DIR}/models/lm.bin`) and enables contact mode only when `CONTACT_LM_BIN`
is set. Supply `WORDS_FILE` and `LEXICON_FILE` from the main-LM package; the
staged copy of `words.txt` receives the virtual class label automatically.
For example:

```bash
MAIN_LM_BIN=/path/to/main_lm.bin \
CONTACT_LM_BIN=/path/to/contact_lm.bin \
CONTACT_LM_METADATA=/path/to/contact_lm.meta.json \
WORDS_FILE=/path/to/shared_words.txt \
LEXICON_FILE=/path/to/shared_lexicon.txt \
SMEARING=max \
wenet/SDK/0.0.6/package_workflows/prepare_flashlight_runtime_package.sh
```

When contact mode is enabled, the workflow copies `contact_lm.bin`, writes
the contact-LM mode/metadata/factor into the manifest, validates the
class-word placement and metadata, and includes both contact files in
checksums.

### C++ API

```cpp
#include "asr_sdk/asr_engine.h"
#include "asr_sdk/decode_context.h"

asr_sdk::DecodeContextConfig contacts;
contacts.contacts.push_back({
    "ada-42", "Ada Wong", {{"Ada Wong", {"▁Ada", "▁Wong"}, 2}}});
contacts.contact_list.max_contacts = 10000;

auto context_or = engine->CompileDecodeContext(contacts);
auto stream_or = engine->CreateStream(std::move(context_or).value());
```

`DecodeContext` is immutable after compilation, can be shared by concurrent
streams from the same engine, and contains copies of runner strings.  Calling
the existing `engine->CreateStream()` continues to use only the original base
trie and main-LM path.

### C API

```c
AsrSdkContext* context = NULL;
asr_sdk_create_context(engine, &context);
asr_sdk_context_add_contact_form(
    context, "ada-42", "Ada Wong", "Ada Wong", NULL, 0, 2);
asr_sdk_context_compile(context);
asr_sdk_create_stream_with_context(engine, context, &stream);
/* The input strings may now be freed. */
```

Builder calls fail after compilation.  A context must be compiled before it is
used, and it cannot be used with a different engine. Removing the old
score/trigger setter functions is an ABI break; `asr_sdk_abi_version()` now
returns 3.

### Contact scoring and ambiguity

The model package, not the caller, controls the two LM weights, the handmade
patterns, their bonuses, and the accumulation factor. There are no per-contact
scores or caller-provided trigger words in the C++ or C APIs. All runtime
contacts in the same matched pattern receive the same LM bonus; their acoustic
evidence can still differ.

The initial rules should be tuned on a held-out contact set. In particular, a
short or common runtime name can create a false contact hypothesis in any
supported pattern if `contact_lm_weight` is too high. Lower the lambda,
lower/remove broad pattern bonuses, or restrict the supplied name list before
changing decoder code.

Identical AM token spellings are intentionally ambiguous.  The SDK returns all
candidate contact IDs and display names rather than silently selecting one.
The visible result never contains an internal virtual label.  Multiword
contact timestamps are deterministically split across visible display words;
the entity timestamp covers the complete virtual span.

### Result JSON and privacy

Best and n-best JSON now contain optional `entities` arrays:

```json
{
  "entities": [{
    "type": "contact",
    "text": "Ada Wong",
    "start_ms": 840.0,
    "end_ms": 1320.0,
    "score": 13.5,
    "ambiguous": false,
    "candidates": [{"contact_id": "ada-42", "display_name": "Ada Wong"}]
  }]
}
```

For a contact entity, `score` is its positive weighted handmade contact bonus:
`contact_lm_weight * S * (1 + a + ... + a^(n-1))`. It excludes the AM score
and internal word-score correction.

Contact data lives only in the in-memory context and stream resources.  It is
not written to the model directory, package checksums, or a compiled-context
cache.  Debug mode logs context counts rather than the contact list; recognized
contact text can appear in a result/debug hypothesis because it is ASR output.

`asr_package_eval --contact_names names.txt --debug true` accepts one contact
display name per line and writes N-best debug rows as `hypN#text#AM#LM`. If
`--contact_names` is omitted, it creates no contact context and uses only the
main LM. A contact scored by the handmade LM is marked only in the debug text,
for example `CALL Ada_Wong<CONTACT>`; result JSON keeps the normal display
name, for example `CALL Ada Wong`. `--contacts_tsv` remains available when
explicit contact IDs, spoken forms, or AM-token spellings are required.

## Local ONNX Runtime

0.0.6 expects ONNX Runtime `1.25.1` under this SDK tree:

```text
third_party/onnxruntime/include
third_party/onnxruntime/lib
```

Recreate it on a clean machine:

```bash
ORT_VERSION=1.25.1
SDK_ROOT=/home/jinyang_wang/Dev/ASR/ASR_wenet/wenet/SDK/0.0.6

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
cd /home/jinyang_wang/Dev/ASR/ASR_wenet/wenet/SDK/0.0.6
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
LM/kenlm_lm/tools/convert_kenlm_binary.sh
python3 LM/kenlm_lm/tools/generate_words_from_lm.py
python3 LM/kenlm_lm/tools/generate_lexicon_for_am.py
wenet/SDK/0.0.6/package_workflows/prepare_flashlight_runtime_package.sh
```

The default package is written to:

```text
test/0.0.6/model_flashlight
```

`prepare_flashlight_runtime_package.sh` writes the Flashlight decoder settings
into `sdk_model.json`. Override them during package creation when you want the
public SDK path to use the same settings as a tuning run:

```bash
LM_WEIGHT=0.5 WORD_SCORE=0.0 BEAM_SIZE=50 BEAM_SIZE_TOKEN=20 \
  wenet/SDK/0.0.6/package_workflows/prepare_flashlight_runtime_package.sh
```

Supported manifest fields are `beam_size`, `beam_size_token`,
`beam_threshold`, `lm_weight`, `word_score`, `unk_score`, `sil_score`,
`log_add`, `allow_unk`, `smearing`, `nbest`, `feature_type`, `blank_token`,
`sil_token`, `unk_word`, and `sample_rate`. Contact-capable packages also use
`contact_lm`, `contact_lm_weight`, `contact_lm_mode`, `contact_lm_metadata`,
`contact_lm_accumulation_factor`, and `contact_class_word`; these are emitted
together only when a handmade contact LM is supplied.
Use `feature_type=kaldi` for standard sherpa/icefall Zipformer CTC and exported
WeNet CTC models; the default `whisper` mode is kept for packages that were
built with the earlier SDK frontend.

The package script exposes the manifest token names as environment variables.
The defaults are `BLANK_TOKEN=<blk>`, `SIL_TOKEN=▁`, `UNK_WORD=<unk>`, and
`SAMPLE_RATE=16000`. For models whose `tokens.txt` uses `<blank>` instead of
`<blk>`, build the package with:

```bash
BLANK_TOKEN="<blank>" FEATURE_TYPE=kaldi \
  wenet/SDK/0.0.6/package_workflows/prepare_flashlight_runtime_package.sh
```

The package script accepts symlinked input files for the acoustic model, tokens,
KenLM files, lexicon, words, and optional mapping. Symlinks are dereferenced
during copy, so the output package contains real files instead of links to files
outside the package.

## Run

Standalone decoder:

```bash
wenet/SDK/0.0.6/build/zipformer_ctc_flashlight_main \
  --model test/0.0.6/model_flashlight/model.onnx \
  --tokens test/0.0.6/model_flashlight/tokens.txt \
  --words test/0.0.6/model_flashlight/words.txt \
  --lexicon test/0.0.6/model_flashlight/lexicon.txt \
  --lm test/0.0.6/model_flashlight/lm.bin \
  --mapping test/0.0.6/model_flashlight/output_mapping.txt \
  --wav model/sherpa-onnx-streaming-zipformer-ctc-zh-2025-06-30/test_wavs/0.wav
```

This standalone tool remains Zipformer-specific. Use `asr_stream_file` or
`asr_package_eval` for `model_type=wenet_ctc` packages.

Public SDK final-result path:

```bash
wenet/SDK/0.0.6/build/asr_stream_file \
  --model_dir test/0.0.6/model_flashlight \
  --wav model/sherpa-onnx-streaming-zipformer-ctc-zh-2025-06-30/test_wavs/0.wav \
  --print_partial false
```

For public partials, feed chunked audio and enable printing:

```bash
wenet/SDK/0.0.6/build/asr_stream_file \
  --model_dir test/0.0.6/model_flashlight \
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
wenet/SDK/0.0.6/build/asr_package_eval \
  --model_dir test/0.0.6/model_flashlight \
  --decode_mode lm \
  --metadata data/hf_wenetspeech_test_net/wenetspeech_test_net_sample_2000/metadata.jsonl \
  --wav_parent data/hf_wenetspeech_test_net/wenetspeech_test_net_sample_2000 \
  --output_json test/0.0.6/package_eval/output.jsonl

python3 wenet/SDK/0.0.6/cli/summarize_asr_package_eval.py \
  --input_json test/0.0.6/package_eval/output.jsonl \
  --summary test/0.0.6/package_eval/summary.txt
```

Use `--decode_mode greedy` to evaluate AM-only CTC greedy decoding without
loading the lexicon or KenLM.

See `cli/README.md` for field definitions and metric behavior.

## Current Caveats

- The first 100-sample acceptance run shows current LM settings are not
  accuracy-ready: greedy CER is lower than LM CER. Tune `LM_WEIGHT`,
  `WORD_SCORE`, beam settings, and lexicon coverage before release.
- Full acceptance-set partial latency and revision-rate metrics are not yet
  collected.
