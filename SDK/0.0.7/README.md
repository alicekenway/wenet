# WeNet SDK 0.0.7: Multi-LM Maximum Fusion

SDK 0.0.7 replaces the fixed main/contact pair from 0.0.6 with one decoder
that can load any number of fixed ngram and pattern-bias KenLM models. The AM
runs once. At each completed output event, exactly one eligible LM contribution
is selected by maximum score.

## Package configuration

`sdk_model.json` points to a separate LM configuration and uses a positive
length-penalty parameter:

```json
{
  "decoder_type": "flashlight_lexicon_kenlm",
  "model_path": "model.onnx",
  "tokens": "tokens.txt",
  "words": "words.txt",
  "lexicon": "lexicon.txt",
  "lm_search": "lm_search.json",
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
Slot class tokens belong in `words.txt` and the fixed LMs, but never in the AM
token table or static lexicon.

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
cmake -S . -B build_007 \
  -DASR_SDK_BUILD_TESTS=ON \
  -DASR_SDK_BUILD_TOOLS=ON \
  -DASR_SDK_BUILD_EXAMPLES=OFF
cmake --build build_007 -j2
ctest --test-dir build_007 --output-on-failure
```

## Package workflow

```bash
OUT_DIR=/path/to/package \
AM_MODEL=/path/to/model.onnx \
TOKENS_FILE=/path/to/tokens.txt \
LM_SEARCH_JSON=/path/to/lm_search.json \
LM_BIN_DIR=/path/to/lm/binaries \
WORDS_FILE=/path/to/words.txt \
LEXICON_FILE=/path/to/lexicon.txt \
LENGTH_PENALTY=0.2 \
package_workflows/prepare_flashlight_runtime_package.sh
```

The workflow copies every referenced binary and includes all fixed resources
in `checksums.sha256`. LM CPU work and state memory grow approximately linearly
with the number of configured models; no arbitrary model-count limit is used.
