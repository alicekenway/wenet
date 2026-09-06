# WeNet SDK 0.0.16: Compact Lexicon for Linux x86-64 and Android ARM64

SDK 0.0.16 retains the flexible multi-LM decoder and optional inverse text
normalization while replacing the package-sized heap trie with a memory-mapped
compact lexicon.

This version keeps the 0.0.15 decoder scoring behavior. Runtime contact contexts share
the package-sized token/word dictionaries, compact lexicon, and fixed KenLMs.
A context owns only its dynamic contact
metadata, a small overlay trie, and an LM wrapper over the shared fixed models.
The contact decoder traverses the base and overlay tries as one logical union
inside a single beam search. Both contact and no-contact decoding use indexed
states in the compact static trie.

Both the normal and strict runtime-contact lookahead score are precomputed for
every compact state during packaging. Engine startup therefore does not build
the old trie, smear it, or walk it to prepare contact lookahead.

One engine may create multiple streams and those streams may run concurrently.
Each individual stream remains sequential and must be owned by one thread at a
time. A compiled decode context is immutable and may be shared. ITN access is
internally synchronized so concurrent streams do not race in the shared normalizer.
ITN runs after final output mapping and is applied on every partial and
final best-hypothesis snapshot, so a later snapshot may replace `100` with
`101`. The original best text is available as `raw_text`; n-best, token,
entity, confidence, and score fields remain unchanged.

Packages without ITN keys preserve the previous behavior and are recommended
for spoken-form WER/CER evaluation. See `../../ITN/README.md` for grammar,
export, reference conversion, and package instructions.

The SDK also retains hypothesis-local pre-LM mapping with score rollback and
mapped-history replay. The decoder can load any number of fixed ngram and
pattern-bias KenLM models. The AM
runs once. At each completed output event, exactly one eligible LM contribution
is selected by maximum score.

## Package configuration

### Automatic word-based contact spellings

Automatic DCC now generates up to three distinct paths **per word**: canonical
SentencePiece, globally shortest AM-token spelling (token-ID tie break), and
character fallback. This matches the static lexicon builder with `--ignore-case`,
`--max-spellings 3`, character fallback enabled, and standalone boundary disabled.
Case-insensitive fallback uses a compiled Unicode 13.0.0 full-casefold table,
matching the Python environment used by the static builder; it is not a runtime
Unicode scan. Canonical encoding receives the original word unchanged.

All combinations are retained across words: up to 9 paths for two words and
27 for three. This replaces exhaustive whole-name enumeration; it does not
claim to cover every possible tokenization. Names absent from `words.txt` or
the static lexicon use exactly the same generator. They remain full-name
dynamic labels, not new static vocabulary entries.

Set `"sentencepiece_model": "sentencepiece.model"` in the manifest and package
the same model used by the static builder. The packaging workflow accepts
`SENTENCEPIECE_MODEL` and includes the copied model in checksums. SentencePiece
0.2.1 is linked statically (inference only); neither Python nor a SentencePiece
shared library is required on Linux or Android. The same `.model` and
`lexicon.bin` work on both architectures.

The tokenizer and lookup index initialize lazily once per engine. Per-context
word caching avoids repeating work for shared name components. Contexts build
only their overlay, sharing the base compact lexicon and fixed LMs. Stream
creation reuses a compiled context without rebuilding either trie.

Packages without the model still support ordinary ASR and explicit `am_tokens`.
Automatic text forms require it, even if their words exist in the main lexicon.
Model load/compatibility errors fail compilation, including in skip mode.
Individual unavailable spelling strategies are diagnosed; a word with no valid
strategy fails by default. `skip_unencodable_forms=true` explicitly permits
skipping such forms, with diagnostics. Resource limits are never skipped or
silently truncated. Debug engine logs report unique-word and contact path counts.

Both `max_total_dynamic_forms` and `max_token_segmentations_per_form` now default
to 100,000; the 64-token spelling limit remains. C++ applications must rebuild
against the new header to pick up changed defaults. Public structure layouts
and ABI version are unchanged. Explicitly configured smaller limits remain
effective.

Tests: `word_spelling_test` checks combination/limit invariants;
`test/word_spelling_parity.py` compares C++ paths against the Python generator
and static lexicon. `dcc_automatic_test PACKAGE CONTACT_NAMES [1000|2000]`
checks automatic contexts, sharing, concurrency, errors, and scale. Its scale
names are generated from three-path words, not supplied token paths.

`sdk_model.json` points to a separate LM configuration and uses a positive
length-penalty parameter:

```json
{
  "decoder_type": "flashlight_compact_lexicon_kenlm",
  "model_path": "model.onnx",
  "tokens": "tokens.txt",
  "words": "words.txt",
  "lexicon": "lexicon.bin",
  "lexicon_format": "compact_trie_v1",
  "lm_search": "lm_search.json",
  "mapping": "output_mapping.txt",
  "final_mapping": "final_output_mapping.txt",
  "length_penalty": 0.2,
  "smearing": "max"
}
```

`lm_search.json` is keyed by `.bin` basenames in the package root:

```json
{
  "general.bin": {
    "type": "ngram",
    "weight": 0.5,
    "clip": true,
    "clip_lower": 0.0,
    "clip_upper": 8.0
  },
  "domain.bin": {
    "type": "ngram",
    "weight": 0.8,
    "clip": false
  },
  "slot_bias.bin": {
    "type": "bias",
    "weight": 1.5,
    "contact_lm_accumulation_factor": 0.5,
    "slots": ["<CONTACT>", "<ADDRESS>", "<APP>"]
  }
}
```

Weights must be finite and positive. With clipping enabled, both bounds are
required and must satisfy `0 <= clip_lower <= clip_upper`. With clipping
disabled, bounds are forbidden. Bias accumulation factors must be in `[0,1]`.
Slot class tokens belong in `words.txt` and each bias LM that declares them,
but never in the AM token table or static lexicon. Ordinary ngram LMs may omit
slot tokens; they advance through their unknown state and do not contribute on
a dynamic-slot transition.

There is no `contact_lm.meta.json`. Bias bonuses are encoded directly as
negative terminal scores in the handmade pattern ARPA/binary.

## Score formula

KenLM's native base-10 scores are retained. For normal word `w` and ngram `k`:

```text
raw_k      = log10 P_k(w | history)
reference_k = log10 P_k(<unk> | history)
relative_k = raw_k - reference_k
adjusted_k = clip(relative_k, lower_k, upper_k)  # when enabled
candidate_k = weight_k * adjusted_k
word_lm_score = max_k(candidate_k)
```

Every ngram observes the complete word history, even when it does not win.
An OOV advances that model as `<unk>`, so its un-clipped relative score is zero.

For a dynamic value of slot `c`, containing `m` logical words, a matching bias
model with stored terminal score `s < 0`, weight `beta`, and accumulation `a`
contributes:

```text
raw_bonus = -s
multiplier = 1 + a + ... + a^(m-1)
bias_candidate = beta * raw_bonus * multiplier
slot_lm_score = max_matching_bias(bias_candidate)
```

No ngram contribution is added on the slot transition. Every ngram still
advances with the class token, so a later word is conditioned on `<CONTACT>`,
`<ADDRESS>`, or the relevant class rather than the private slot value.

For an event with AM contribution `A`, LM contribution `M`, logical word count
`m`, and package penalty `lambda >= 0`:

```text
event_score = A + M - lambda * m
```

The final score is cumulative. It is never divided by word or token count.
EOS uses the same ngram `score(</s>) - score(<unk>)`, clipping, weighting, and
maximum rule, without a length penalty.

With clipping and a nonnegative lower bound, increasing a model weight cannot
lower the decoder score. With `clip: false`, a relative score may be negative;
in that expert mode a higher weight can lower a path again.

Per-word maximum fusion is a decoder feature, not a normalized probability
model. Adding more LMs can increase scores simply because more models compete.
Tune weights, upper bounds, and the length penalty on held-out data.

## Pre-LM and final mappings

`output_mapping.txt` is the pre-LM mapping. `final_output_mapping.txt` is an
optional post-decoding text correction. Both use:

```text
A B C -> D F G
```

When the raw AM word sequence completes `A B C`, the decoder restores every
fixed LM to its state before `A`, replays `D F G`, and returns a correction for
the source contributions already accumulated at `A` and `B`. Future words are
therefore conditioned on `D F G`. The AM score remains attached to the raw
`A B C` path.

For source contributions `l_A,l_B,l_C`, mapped contributions
`l_D,l_F,l_G`, source length `m`, target length `n`, and the decoder's
per-word score `word_score = -length_penalty`, the contribution returned when
the mapping completes is:

```text
completion = (l_D + l_F + l_G) - (l_A + l_B)
             + word_score * (n - m)
```

Together with the already accumulated `l_A+l_B` and the raw-word insertion
scores, this yields the mapped LM score and mapped logical-word penalty.

All pre-LM source and target words must exist in `words.txt`. Streaming pre-LM
rules do not support the `$` end anchor. The packaging workflow rejects a
source that is a strict prefix of another source, for example:

```text
A B -> X
A B C -> Y
```

This check makes the mapping completion point unambiguous. Mapping operates on
the raw AM word stream once; mapped targets are not fed back into the mapping
matcher.

Rollback repairs the accumulated LM score and future LM state, but it cannot
restore a hypothesis that the beam pruned before the complete source phrase
arrived. Short conservative mappings and a sufficiently large beam are still
recommended.

## Runtime slots

The C++ context accepts generic `SlotClass`/`SlotValueEntry` values. The C API
adds `asr_sdk_context_add_slot_form`. Existing contact APIs remain wrappers for
`<CONTACT>`.

The evaluation tools accept a simple section file:

```text
#<CONTACT>
Alice
Bob Smith

#<ADDRESS>
Beijing Station

#<APP>
Music
```

Use `--slots PATH`. Blank lines are ignored. Repeated sections merge and
duplicate values are removed. Each line supplies its ID, display text, and
spoken form. An absent or empty section is off. Values before the first header
and non-empty sections without a configured bias LM are errors.

Results expose generic `slot_token` and `value_id`. `<CONTACT>` results also
retain the legacy `contact_id` alias.

## Build and test

```bash
cmake -S . -B build_linux \
  -DASR_SDK_ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.16.3 \
  -DASR_SDK_REQUIRED_ORT_VERSION=1.16.3 \
  -DASR_SDK_BUILD_TESTS=ON \
  -DASR_SDK_BUILD_TOOLS=ON \
  -DASR_SDK_BUILD_EXAMPLES=OFF
cmake --build build_linux -j2
ctest --test-dir build_linux --output-on-failure
```

For Android ARM64, configure the same source directory with the NDK toolchain,
`ANDROID_ABI=arm64-v8a`, `ANDROID_PLATFORM=android-28`, and an extracted ONNX
Runtime 1.16.3 AAR. The root CMake file selects `cmake/LinuxX86_64.cmake` or
`cmake/AndroidArm64.cmake`; shared source lists live in `cmake/Common.cmake`.
Linux retains all configured backends and supports the pinned ONNX Runtime
1.16.3 or 1.25.1 layouts. Android compiles the production Flashlight path only
and remains pinned to ONNX Runtime 1.16.3.

## Package workflow

```bash
OUT_DIR=/path/to/package \
AM_MODEL=/path/to/model.onnx \
TOKENS_FILE=/path/to/tokens.txt \
LM_SEARCH_JSON=/path/to/lm_search.json \
LM_BIN_DIR=/path/to/lm/binaries \
WORDS_FILE=/path/to/words.txt \
LEXICON_FILE=/path/to/lexicon.txt \
MAPPING=/path/to/output_mapping.txt \
FINAL_MAPPING=/path/to/final_output_mapping.txt \
COMPACT_LEXICON_TOOL=/path/to/build/compact_lexicon_tool \
LENGTH_PENALTY=0.2 \
package_workflows/prepare_flashlight_runtime_package.sh
```

The workflow stages a text source package, compiles and verifies `lexicon.bin`,
then removes `lexicon.txt` before writing `checksums.sha256`. The compiler
report is written next to the package by default, not inside it. LM CPU work
and state memory grow approximately linearly with the number of configured
models; no arbitrary model-count limit is used.
