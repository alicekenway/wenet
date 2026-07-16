# Superseded

This document describes the retired manual-bias design. Do not implement it.
The active design is [dual_lm_contact_development_plan.md](dual_lm_contact_development_plan.md): it uses a separately trained contact LM, a single `<CONTACT>` class transition for each completed runtime name, and no manual partial/complete/command rewards.

---

# Historical Development Plan: Runtime Contact-List Bias for the CTC + Lexicon + KenLM Decoder

## Document status

- **Repository:** `alicekenway/wenet`
- **Target source tree:** `SDK/0.0.5`
- **Baseline branch:** `main`
- **Baseline inspected:** `9d3297d597d0a4704ecc06f8ddfe6b17ccd0adc3`
- **Primary decoder:** Flashlight `LexiconDecoder` with a word-level KenLM
- **Purpose:** implementation specification for Codex
- **Contact-data ownership:** runtime runner/application only; contact data is never part of the model package

---

## 1. High-level abstract

Extend the existing streaming CTC, lexicon, and KenLM shallow-fusion decoder with an immutable runtime contact context supplied by the runner.

The normal language model continues to score ordinary sentence spans. Each runtime contact spoken form is compiled into a dynamic lexicon path that shares the acoustic model's existing token vocabulary. A complete spoken form is represented internally as one virtual decoder label, even when its visible text contains multiple words such as `Ada Wong`.

While the acoustic token path matches only a prefix of a contact name, the dynamic lexicon supplies a small lookahead score so the path can remain in the beam. When the complete contact form is reached, the composite language model advances the main KenLM with the package-defined class word `<CONTACT>` instead of trying to score the unseen literal name. It then applies either:

1. a **complete-contact score**, or
2. a higher **command-contact score** when a configured command trigger such as `dial`, `message`, `call`, or `text` occurred within the configured word window.

The main LM therefore sees a normalized sequence such as:

```text
please call the phone of <CONTACT> tomorrow
```

while the acoustic decoder still consumes the real token sequence for:

```text
please call the phone of Ada Wong tomorrow
```

After the contact completes, later words are scored from the main-LM state after `<CONTACT>`, so an unseen user name does not damage the LM history.

The contact list, aliases, spoken forms, IDs, trigger configuration, and bias scores are provided by the runner at runtime. They are held only in an immutable compiled context and are not written into `sdk_model.json`, `words.txt`, `lexicon.txt`, `lm.bin`, package checksums, or any other model-package resource.

---

## 2. Final design decisions

These decisions are part of the implementation contract.

### 2.1 Runtime contacts are not model resources

The model package contains only reusable, user-independent resources:

```text
model.onnx
tokens.txt
words.txt
lexicon.txt
lm.bin
output_mapping.txt
sdk_model.json
```

The runner supplies:

- contact IDs;
- display names;
- one or more spoken forms;
- optional exact AM token spellings;
- contact-bias scores;
- trigger words;
- trigger-window length.

The same model package must work with no contacts, with one user's contacts, or with a different user's contacts.

### 2.2 The main LM contains `<CONTACT>`

`<CONTACT>` is a **word in the main-LM vocabulary**, not an acoustic-model token.

It must:

- exist in `words.txt`;
- occur in the training text used to build `lm.bin`;
- not exist in `tokens.txt` as a new CTC output;
- not have a pronunciation entry in the static `lexicon.txt`;
- not appear as visible output text;
- be configurable through the package manifest, with `<CONTACT>` as the default spelling.

### 2.3 One complete spoken form is one virtual decoder label

For the contact:

```text
contact_id = ada-42
display_name = Ada Wong
spoken_form = Ada Wong
```

the runtime context creates one internal label similar to:

```text
__runtime_contact_form_0__
```

Its dynamic lexicon spelling is the full AM-token sequence for `Ada Wong`.

This is intentionally different from adding `Ada` and `Wong` as unrelated contact unigrams. It ensures that:

- `Ada Wong` can receive a complete-match score;
- `Wong Ada` does not receive that score;
- `Ada Claire` does not receive that score;
- the normal path and contact path can coexist;
- the main LM advances exactly once with `<CONTACT>`;
- the decoder can continue with the normal LM after the name;
- no custom multi-output LM interface is required.

### 2.4 Contact scores are weighted bias scores, not normalized probabilities

The runtime contact component is a weighted bias automaton. It is not required to sum to one over the contact list.

All contacts receive the same default score unless the API is extended later with explicit per-contact priors.

The three score tiers are:

```text
partial_match_score < complete_match_score < command_match_score
```

Interpretation:

- **partial match:** temporary lexicon lookahead while only a name prefix is matched;
- **complete match:** final score for a complete contact outside command context;
- **command match:** final score for a complete contact within command context.

The complete and command scores are mutually exclusive final tiers. Do not accidentally add both.

### 2.5 Main-LM scoring is preserved

A normal path remains:

```text
AM + main LM + normal decoder penalties
```

A completed contact path becomes:

```text
AM
+ main LM over <CONTACT>
+ selected contact score
+ normal decoder penalties
+ any required word-count correction
```

It is **not**:

```text
AM + contact score only
```

### 2.6 Contexts are immutable during an utterance

The runner compiles a contact context before creating a stream.

The context may be shared by multiple streams, but it cannot be mutated after compilation. To change the active user or contact list, the runner creates a new context and uses it for subsequent streams.

Mid-utterance contact updates are a non-goal for the first implementation.

### 2.7 Empty context must preserve current behavior exactly

Calling the existing:

```cpp
engine->CreateStream()
```

must continue to use the current static decoder resource and current KenLM path.

No-contact decoding must not be routed through the new wrapper merely for convenience. This makes regression testing and backward compatibility much stronger.

---

## 3. Current implementation baseline

The current SDK has the following relevant behavior.

### 3.1 Engine-level shared resources

`src/sdk/asr_engine.cc` constructs one shared `FlashlightDecoderResource` containing:

- AM token table;
- output word dictionary;
- Flashlight token and word dictionaries;
- static lexicon trie;
- KenLM;
- output mapper;
- decoder options.

`FlashlightAsrResources` shares that decoder resource and an acoustic backend template across streams.

### 3.2 Stream-level decoder state

Each `FlashlightAsrStream`:

- clones the streaming acoustic backend;
- creates one `FlashlightCtcStreamDecoder`;
- owns feature and decoding state;
- uses the shared immutable decoder resource.

### 3.3 Current decoder construction

`FlashlightCtcStreamDecoder::Start()` constructs:

```cpp
fl::lib::text::LexiconDecoder(
    options,
    resource->LexiconTrie(),
    resource->WordLm(),
    silence_id,
    blank_id,
    unknown_word_id,
    transitions,
    false);
```

The last argument is `false`, so the LM is word-level. The LM is queried when a lexicon word completes.

### 3.4 Current lexicon lookahead

The static trie is populated with one score per lexicon entry and then smeared. During token-by-token decoding, the smeared score provides word-prefix lookahead. At word completion, the decoder replaces the lookahead approximation with the real LM score.

This behavior is the correct place to implement the low partial-contact score without changing the CTC model.

### 3.5 Current public API limitations

The current public API has:

```cpp
AsrEngine::CreateStream()
```

but no runtime context object.

The result API contains text, tokens, n-best results, scores, and debug JSON, but no contact entity metadata.

The C API likewise has no context handle.

### 3.6 Current constraints that the implementation must account for

1. `tokens.txt` IDs must match ONNX output columns. Runtime contacts must never add AM tokens.
2. `WordDictionary` is currently file-loaded and immutable.
3. `FlashlightDecoderResource` currently discards the loaded base lexicon entries after building the trie.
4. `ConvertFlashlightResult()` assumes every emitted word ID exists in the static output dictionary.
5. n-best deduplication currently relies primarily on mapped word IDs.
6. the vendored Flashlight trie silently refuses more than six labels on one spelling path.
7. trie `LOGADD` smearing would make shared-prefix bias depend on the number of contacts; runtime contacts therefore require `MAX` smearing in version 1.

---

## 4. Goals

### 4.1 Functional goals

1. Accept a user contact list from the runner at runtime.
2. Compile the contact list into a reusable immutable decode context.
3. Preserve ordinary main-LM decoding outside contact spans.
4. Keep a contact path alive with a low partial-match lookahead.
5. Apply a higher score when a complete contact spoken form is recognized.
6. Apply the highest score when a complete form follows a configured command trigger within a configurable window.
7. Advance the main KenLM with `<CONTACT>` when a contact completes.
8. Continue normal main-LM scoring after the contact span.
9. Return the opaque contact ID and display name in structured result metadata.
10. Support aliases and multiple spoken forms per contact.
11. Detect and expose ambiguous contacts with the same acoustic spelling.
12. Keep separate contexts isolated across concurrent streams.
13. Preserve the existing no-context API and output behavior.
14. Add C++, C, CLI, unit-test, and integration-test coverage.
15. Keep contact data out of the model package and out of persistent runtime caches.

### 4.2 Quality goals

1. No measurable no-context accuracy regression.
2. Controlled false-contact insertion rate outside command context.
3. Bounded context compilation time and memory.
4. Small runtime RTF overhead after context compilation.
5. Deterministic context compilation and deterministic label assignment.
6. Actionable validation errors for unencodable names and malformed configuration.

---

## 5. Non-goals for version 1

1. Do not retrain or modify the CTC acoustic model.
2. Do not add contact names to `tokens.txt`.
3. Do not store contact data in the model package.
4. Do not implement a neural contextual LM.
5. Do not implement a general-purpose G2P system.
6. Do not update a context after a stream has started.
7. Do not automatically choose between two contacts with identical spoken forms.
8. Do not execute calls or messages inside the ASR SDK.
9. Do not hard-code language-specific text normalization beyond explicit runner configuration.
10. Do not replace the Flashlight decoder with a new decoder in the first implementation.
11. Do not make per-contact usage or recency priors part of the first release.
12. Do not port this feature to `SDK/0.0.5_android` until the desktop implementation passes all gates.

---

## 6. Algorithm specification

## 6.1 Search paths

For an utterance such as:

```text
please call the phone of Ada Wong tomorrow
```

the decoder can maintain ordinary and contact interpretations.

### Ordinary path

The static lexicon emits ordinary words:

```text
please | call | the | phone | of | Ada | Wong | tomorrow
```

The main KenLM scores every emitted word.

### Contact path

The static lexicon emits ordinary words before the name:

```text
please | call | the | phone | of
```

The combined lexicon then follows the complete acoustic-token path of the runtime contact form:

```text
Ada Wong
```

but emits one internal virtual contact label at the end.

The composite LM responds to that virtual label by scoring:

```text
<CONTACT>
```

in the main KenLM and adding the selected contact score.

The next ordinary word, `tomorrow`, is scored by the main KenLM from the state after `<CONTACT>`.

## 6.2 Scoring objective

The current decoder conceptually optimizes:

```text
S =
  AM
  + lm_weight * LM
  + word_score * number_of_emitted_words
  + unk_score * number_of_unknown_words
  + sil_score * separator_count
```

For an ordinary emitted word `w`:

```text
LM_wrapper(w, state) = LM_main(w, state)
```

For a completed runtime contact label `c`:

```text
LM_wrapper(c, state) =
  LM_main(<CONTACT>, state)
  + contact_bias(state) / lm_weight
  + word_score_correction(c) / lm_weight
```

where:

```text
contact_bias(state) =
  command_match_score   if command context is active
  complete_match_score  otherwise
```

The public runtime scores are specified in **final decoder-score units**, not raw KenLM units. Dividing by `lm_weight` inside the LM wrapper prevents the decoder from multiplying the configured contact score twice.

When a non-empty contact context is used:

```text
lm_weight must be finite and > 0
```

If it is zero or negative, context compilation must fail with a clear `FailedPrecondition` or `InvalidArgument` status.

## 6.3 Partial-match lookahead

A contact path receives the low partial score before the complete virtual label is emitted.

For a dynamic contact lexicon entry, use a trie insertion/lookahead score equivalent to:

```text
main_lm_start_score(<CONTACT>)
+ partial_match_score / lm_weight
```

The smeared trie score is used only as lookahead while the path is incomplete. At completion, Flashlight's existing telescoping lookahead logic replaces it with the real context-dependent LM score.

This means:

- the incomplete path temporarily receives the low partial score;
- the final path receives the complete or command score;
- the partial score is not permanently double-counted.

Version 1 requires `MAX` trie smearing for contact-enabled contexts.

Do not use `LOGADD` for runtime contacts, because the prefix score would increase merely because more contacts share the same prefix.

## 6.4 Complete-match tier

A full contact form outside the command window receives:

```text
complete_match_score
```

This allows names to be recognized even without a command trigger. The design therefore does not rely entirely on context.

## 6.5 Command-match tier

A full contact form receives:

```text
command_match_score
```

when a configured trigger occurred within the preceding `trigger_window_words` ordinary main-LM words.

Initial runner example:

```text
triggers = dial,message,call,text
trigger_window_words = 5
```

This lets the following receive the highest tier:

```text
call Ada Wong
please call Ada Wong
please call the phone of Ada Wong
message Claire
text Leon tomorrow
```

The trigger state resets after a contact completes.

Version 1 supports single-word trigger entries. Multiword trigger phrases can be added later without changing the public contact model.

## 6.6 Trigger tracker state

Each composite LM state stores:

- underlying main-LM state;
- whether a trigger is active;
- number of ordinary words since the most recent trigger;
- enough information to create deterministic cached child states.

Transition rules:

```text
normal word is a trigger:
    trigger_active = true
    words_since_trigger = 0

normal non-trigger word while active:
    words_since_trigger += 1
    deactivate if words_since_trigger > trigger_window_words

contact completion:
    choose command tier if active
    advance main LM with <CONTACT>
    deactivate trigger

finish:
    finish underlying main LM
```

Trigger matching uses raw static word IDs before output mapping.

## 6.7 Word-score correction

A multiword name is represented by one virtual decoder label, so Flashlight applies `word_score` once.

The equivalent ordinary path may contain multiple logical words. Without correction, a negative `word_score` would unfairly favor the one-label contact path.

For a contact spoken form with `logical_word_count = n`, apply:

```text
word_score_correction = (n - 1) * word_score
```

This makes the total insertion score approximately equal to an `n`-word ordinary path.

The correction is enabled by default and covered by exact-score unit tests.

For languages or package segmentations where whitespace is not the output-word boundary, the runner may explicitly provide `logical_word_count`. Otherwise:

- whitespace-separated spoken forms use their non-empty field count;
- unsegmented forms default to one.

## 6.8 Main-LM state after the name

After a completed contact, the underlying main LM state must be exactly the state produced by:

```text
LM_main.score(previous_main_state, contact_class_word_id)
```

The literal name words must never be sent to the main KenLM on the contact path.

Consequently:

```text
P(tomorrow | call <CONTACT>)
```

is available even if `Ada Wong` never appeared in LM training.

## 6.9 Overlapping contacts

The dynamic token trie must support:

```text
Ada
Ada Wong
Ada Lovelace
Ashley
```

At the end of `Ada`, the trie may simultaneously:

- emit a complete contact label for `Ada`;
- keep an incomplete path for `Ada Wong`;
- keep an incomplete path for `Ada Lovelace`.

The current Flashlight lexicon semantics already support a node containing labels and children.

## 6.10 Duplicate and homophonic contacts

Group runtime entries by exact AM-token spelling.

For one token spelling, store all matching runtime candidates:

```text
spoken surface: Alex Lee
candidate IDs:
  contact-17
  contact-93
```

Emit one dynamic decoder label for the spelling group. The final entity is marked ambiguous and returns all candidate IDs and display names.

Do not silently select the first contact.

Different spoken forms for the same contact may use separate labels but should deduplicate to the same semantic contact in n-best output.

## 6.11 Partial-result behavior

An incomplete dynamic contact path has not emitted a word label yet.

Version 1 behavior:

- do not expose an incomplete contact as a final entity;
- partial text may temporarily end before the contact;
- emit the contact display text and entity only after the complete form label is emitted;
- optionally expose pending-prefix information only in debug traces, never in the stable public result schema.

## 6.12 No-match behavior

If no dynamic contact path matches the audio:

- ordinary paths continue to use the current static lexicon and main KenLM;
- no contact score is applied;
- no contact entity is returned.

If the context is empty, use the exact current decoder resource rather than a wrapper.

---

## 7. Worked example

Runtime contacts:

```text
ada-42     Ada Wong     spoken form: Ada Wong
ashley-7   Ashley       spoken form: Ashley
claire-9   Claire       spoken form: Claire
leon-11    Leon         spoken form: Leon
```

Runtime scores:

```text
partial_match_score  = 2.0
complete_match_score = 6.0
command_match_score  = 10.0
trigger_window_words = 5
triggers             = dial,message,call,text
```

Utterance:

```text
please call the phone of Ada Wong tomorrow
```

### Step A: normal words

The decoder emits:

```text
please call the phone of
```

The wrapper forwards each word to the main KenLM.

When `call` is emitted, command state becomes active.

### Step B: contact prefix

The AM begins producing the token sequence of `Ada Wong`.

The combined lexicon contains:

- ordinary static word paths;
- the runtime full-form path for `Ada Wong`.

While the runtime path is incomplete, trie lookahead gives it the low partial tier. The path can remain in the beam without yet claiming a complete contact.

### Step C: contact completion

At the end of the full token sequence, the dynamic label for `Ada Wong` is emitted.

Because `call` remains within the configured five-word window, the wrapper:

1. scores `<CONTACT>` in the main KenLM;
2. applies `command_match_score`;
3. applies the word-score correction;
4. resets command state;
5. records the runtime contact metadata.

Conceptual normalized main-LM history:

```text
please call the phone of <CONTACT>
```

### Step D: continuation

The decoder emits `tomorrow`.

The wrapper scores:

```text
P_main(tomorrow | ... of <CONTACT>)
```

not:

```text
P_main(tomorrow | ... Ada Wong)
```

### Competing paths

The beam may compare:

```text
please call the phone of Ada Wong tomorrow
please call the phone of a wrong tomorrow
please call the phone home tomorrow
```

The contact path is not forced. It wins only if the acoustic evidence, normalized main-LM score, and contact tier together outrank the alternatives.

---

## 8. Runtime public data model

Add a new public header:

```text
include/asr_sdk/decode_context.h
```

Recommended C++ API types:

```cpp
namespace asr_sdk {

struct ContactSpokenForm {
  // Visible normalized spoken form, for example "Ada Wong".
  std::string text;

  // Optional exact AM token strings. When present, these are authoritative.
  // Example: {"▁Ada", "▁Wong"}.
  std::vector<std::string> am_tokens;

  // Optional override. Zero means infer from text.
  int logical_word_count = 0;
};

struct ContactEntry {
  // Opaque runner-owned identifier. The SDK must not interpret it.
  std::string contact_id;

  // Name to return to the application.
  std::string display_name;

  // One or more aliases/pronunciations.
  std::vector<ContactSpokenForm> spoken_forms;
};

struct ContactBiasOptions {
  float partial_match_score = 2.0f;
  float complete_match_score = 6.0f;
  float command_match_score = 10.0f;

  std::vector<std::string> trigger_words = {
      "dial", "message", "call", "text"};

  int trigger_window_words = 5;

  // Safety and complexity limits.
  int max_contacts = 10000;
  int max_spoken_forms_per_contact = 8;
  int max_tokens_per_spoken_form = 64;
  int max_total_dynamic_forms = 50000;

  // Exact failure is safer than silently dropping a contact.
  bool skip_unencodable_forms = false;

  // Correct Flashlight word_score for a multiword virtual label.
  bool correct_word_score = true;
};

struct DecodeContextConfig {
  std::vector<ContactEntry> contacts;
  ContactBiasOptions contact_bias;
};

class ASR_SDK_API DecodeContext {
 public:
  virtual ~DecodeContext() = default;
};

}  // namespace asr_sdk
```

### 8.1 Engine API

Recommended additions:

```cpp
class AsrEngine {
 public:
  virtual StatusOr<std::shared_ptr<const DecodeContext>> CompileDecodeContext(
      const DecodeContextConfig& config) = 0;

  virtual StatusOr<std::unique_ptr<AsrStream>> CreateStream(
      std::shared_ptr<const DecodeContext> context) = 0;

  // Existing compatibility API remains.
  virtual StatusOr<std::unique_ptr<AsrStream>> CreateStream() = 0;
};
```

Behavior:

- `CompileDecodeContext()` validates and compiles contacts once.
- the returned object is immutable and thread-safe;
- multiple streams may share it;
- `CreateStream()` uses no context;
- a context compiled by one engine cannot be used with a different engine or model package;
- passing a null context is equivalent to `CreateStream()`.

### 8.2 C API

Add an opaque context handle:

```c
typedef struct AsrSdkContext AsrSdkContext;
```

Recommended builder API:

```c
int asr_sdk_create_context(
    AsrSdkEngine* engine,
    AsrSdkContext** out_context);

int asr_sdk_context_set_scores(
    AsrSdkContext* context,
    float partial_score,
    float complete_score,
    float command_score);

int asr_sdk_context_set_trigger_window(
    AsrSdkContext* context,
    int trigger_window_words);

int asr_sdk_context_add_trigger(
    AsrSdkContext* context,
    const char* trigger_word);

int asr_sdk_context_add_contact_form(
    AsrSdkContext* context,
    const char* contact_id,
    const char* display_name,
    const char* spoken_form,
    const char* const* am_tokens,
    int num_am_tokens,
    int logical_word_count);

int asr_sdk_context_compile(AsrSdkContext* context);

int asr_sdk_create_stream_with_context(
    AsrSdkEngine* engine,
    const AsrSdkContext* context,
    AsrSdkStream** out_stream);

void asr_sdk_destroy_context(AsrSdkContext* context);
```

Rules:

- builder methods fail after successful compilation;
- `asr_sdk_create_stream_with_context()` requires a compiled context;
- string inputs are copied;
- the runner may free its input strings immediately after each call;
- repeated `contact_id` values add aliases/forms to the same contact;
- existing C functions remain available.

A JSON convenience API may be added later, but it should not block the first implementation or require a fragile nested-JSON parser.

---

## 9. Runner contact-file format

Add a simple UTF-8 TSV format for CLI tools.

One spoken form per row:

```text
contact_id<TAB>display_name<TAB>spoken_form<TAB>optional_space_separated_am_tokens<TAB>optional_logical_word_count
```

Example:

```text
ada-42	Ada Wong	Ada Wong	▁Ada ▁Wong	2
ashley-7	Ashley	Ashley	▁Ashley	1
claire-9	Claire	Claire	▁Claire	1
leon-11	Leon	Leon	▁Leon	1
```

Requirements:

- blank lines and full-line `#` comments are allowed;
- first three fields are mandatory;
- token field may be empty to request automatic compilation;
- duplicate rows are deduplicated deterministically;
- malformed rows produce line-numbered errors;
- the file is loaded by the runner and passed through the runtime context API;
- the file is not copied into the model directory.

Add CLI arguments to `asr_stream_file`:

```text
--contacts_tsv PATH
--contact_partial_score FLOAT
--contact_complete_score FLOAT
--contact_command_score FLOAT
--contact_triggers dial,message,call,text
--contact_trigger_window 5
```

The runner compiles the context once before creating the stream.

---

## 10. Contact spelling compilation

Runtime contact names must be converted into existing AM token IDs. No new AM token may be introduced.

Create:

```text
src/flashlight_decoder/contact_spelling_compiler.h
src/flashlight_decoder/contact_spelling_compiler.cc
```

### 10.1 Compilation order

For each `ContactSpokenForm`:

#### Method 1: explicit AM tokens

When `am_tokens` is non-empty:

1. validate every token exists in `tokens.txt`;
2. reject blank and other forbidden special tokens where inappropriate;
3. map token strings to IDs;
4. accept the sequence as authoritative.

This is the preferred production path because it gives the runner exact control.

#### Method 2: compose static lexicon spellings

When explicit tokens are absent:

1. normalize whitespace without changing case;
2. split the spoken form into lexical units;
3. look up all static lexicon spellings for each unit;
4. concatenate compatible spellings;
5. cap the Cartesian product at a configured maximum;
6. deduplicate token sequences.

Example:

```text
Ada -> ▁A da
Wong -> ▁W ong
```

becomes:

```text
Ada Wong -> ▁A da ▁W ong
```

#### Method 3: token-table segmentation fallback

If a lexical unit has no static lexicon spelling:

1. transform spaces according to the package's separator convention;
2. run deterministic dynamic programming over token strings in `tokens.txt`;
3. generate up to a small configured number of complete segmentations;
4. prefer fewer tokens, then deterministic token-ID order;
5. reject paths containing blank or non-emitting special tokens.

This fallback creates valid AM-token paths without claiming to be a general G2P system.

### 10.2 Failure behavior

Default behavior is strict:

- if a contact has no encodable spoken form, context compilation fails;
- the error identifies the contact ID and spoken form;
- no partial context is published.

When `skip_unencodable_forms=true`:

- skip only the invalid form;
- keep the contact if another form remains;
- record diagnostics in the compiled-context report;
- never silently skip the entire contact without reporting it.

### 10.3 Determinism

Given the same:

- model package;
- contact configuration;
- spelling options;

the compiler must produce identical:

- token sequences;
- dynamic labels;
- metadata ordering;
- context fingerprint.

---

## 11. Model-package changes

The actual contact list remains external, but the main LM must be contact-class aware.

### 11.1 Manifest field

Add an optional manifest field:

```json
{
  "contact_class_word": "<CONTACT>"
}
```

Add to `ModelPackage`:

```cpp
std::string contact_class_word = "<CONTACT>";
bool supports_runtime_contacts = false;
```

Recommended capability logic:

```text
supports_runtime_contacts =
    manifest explicitly contains contact_class_word
    and words.txt contains that word
```

Old packages remain valid for no-context decoding.

If a non-empty runtime contact context is requested from a package that is not contact-ready, return:

```text
FailedPrecondition:
runtime contacts require a package with contact_class_word
```

### 11.2 Main-LM training data

Prepare class-normalized LM text.

Examples:

```text
please call <CONTACT>
please call the phone of <CONTACT>
message <CONTACT>
text <CONTACT> tomorrow
did <CONTACT> answer
```

Mix these examples with the ordinary LM corpus.

The main LM should learn both:

```text
P(<CONTACT> | call)
P(<CONTACT> | phone of)
```

and continuation probabilities such as:

```text
P(tomorrow | <CONTACT>)
P(answer | did <CONTACT>)
```

### 11.3 `words.txt`

Add `<CONTACT>` as a normal dense word ID.

Do not add runtime names.

### 11.4 Static `lexicon.txt`

Do not add a pronunciation for `<CONTACT>`.

The class word is never acoustically emitted. It is sent to KenLM only by the composite LM wrapper when a runtime contact label completes.

Add a validator rule that rejects a static pronunciation for the configured contact class word.

### 11.5 Package workflow

Update:

```text
package_workflows/prepare_flashlight_runtime_package.sh
```

Add:

```text
CONTACT_CLASS_WORD="${CONTACT_CLASS_WORD:-<CONTACT>}"
```

Write it into `sdk_model.json`.

The script must not accept or copy a contact-list file into the package.

### 11.6 Package inspection

Update `inspect_package` output:

```text
runtime contacts: supported|unsupported
contact class word/id: <CONTACT>/<id>
contact class has static pronunciation: no
```

---

## 12. Internal architecture

## 12.1 Static model resource

Refactor the current `FlashlightDecoderResource` so the engine-owned base resource retains:

- token table;
- base output dictionary;
- Flashlight word dictionary;
- base KenLM;
- base lexicon entries;
- base lexicon trie;
- static output mapper;
- decoder options;
- special IDs;
- contact class word ID when supported;
- a word-to-spellings index for runtime contact compilation.

The no-context path continues to use its existing base trie and base KenLM directly.

## 12.2 Compiled runtime context

Create an internal immutable type:

```text
src/flashlight_decoder/compiled_decode_context.h
src/flashlight_decoder/compiled_decode_context.cc
```

Suggested contents:

```cpp
struct RuntimeContactCandidate {
  std::string contact_id;
  std::string display_name;
};

struct RuntimeContactForm {
  int dynamic_word_id;
  std::string spoken_form;
  std::vector<int> token_ids;
  int logical_word_count;
  std::vector<RuntimeContactCandidate> candidates;
};

class CompiledDecodeContext {
 public:
  FlashlightDecoderResourcePtr decoder_resource;
  std::vector<RuntimeContactForm> forms;
  ContactBiasOptions options;
  std::string fingerprint;
  std::vector<std::string> diagnostics;
};
```

The context owns a contact-enabled decoder resource containing:

- combined lexicon trie;
- composite LM;
- dynamic output-label registry;
- runtime metadata.

It is shareable by concurrent streams.

## 12.3 Dynamic word-ID range

Use:

```text
base IDs:    [0, base_word_count)
dynamic IDs: [base_word_count, base_word_count + dynamic_form_count)
```

IDs are assigned deterministically after grouping and sorting forms.

Dynamic IDs are valid only inside the compiled context that created them.

Never expose a dynamic numeric ID as a stable application contact ID.

## 12.4 Combined trie

To compile a context:

1. create a new trie with the same AM-token vocabulary size and separator root;
2. insert every retained base lexicon entry using the current base lookahead score;
3. insert each unique runtime contact spelling with its dynamic label and contact lookahead score;
4. smear with `MAX`;
5. freeze the trie in the context.

Compilation happens once per runtime context, not once per utterance.

A later optimization may deep-clone a prebuilt base trie, but correctness comes first. Measure compilation cost before adding a complex overlay structure.

## 12.5 Composite LM

Create:

```text
src/flashlight_decoder/contextual_contact_lm.h
src/flashlight_decoder/contextual_contact_lm.cc
```

The class implements:

```cpp
fl::lib::text::LM
```

and wraps the engine's base KenLM.

Suggested state:

```cpp
struct ContextualContactLmState : fl::lib::text::LMState {
  fl::lib::text::LMStatePtr main_lm_state;
  bool trigger_active = false;
  int words_since_trigger = 0;
};
```

The implementation must use deterministic child-state caching so equivalent transitions reuse the same state pointers. Flashlight compares LM states by pointer identity.

### Normal-word transition

```text
input ID is a base word:
    next_main, main_score = base_lm.score(main_state, word_id)
    update trigger tracker
    return composite child and main_score
```

### Dynamic-contact transition

```text
input ID is a runtime contact label:
    tier = command if trigger_active else complete
    next_main, main_score =
        base_lm.score(main_state, contact_class_word_id)

    combined_score =
        main_score
        + tier_score / lm_weight
        + word_score_correction / lm_weight

    reset trigger tracker
    return composite child and combined_score
```

### Finish

```text
base_lm.finish(main_state)
```

### Cache update

Forward unique underlying main-LM states to the wrapped LM's `updateCache()`.

## 12.6 Dynamic label registry

Create an internal resolver with:

```cpp
bool IsDynamicContactId(int word_id) const;
const RuntimeContactForm& ContactFormForId(int word_id) const;
std::string InternalWordForId(int word_id) const;
```

Do not require dynamic IDs to exist in the package `WordDictionary`.

---

## 13. Flashlight trie label-limit patch

The vendored trie currently stops adding labels after six labels on one acoustic spelling and only writes an error to stderr.

This is unsafe for:

- homophones;
- static words plus a runtime contact label;
- ambiguous contact spellings.

Make a minimal isolated patch in the vendored Flashlight source:

1. treat the existing constant as an initial reserve hint, not a hard correctness limit;
2. always append a valid label and score;
3. add a regression test inserting more than six labels;
4. document the local patch;
5. do not make unrelated third-party changes.

Grouping identical runtime spellings reduces label pressure but does not remove the need for this fix.

---

## 14. Result representation

Extend the public result schema.

Recommended structures:

```cpp
struct ContactCandidateResult {
  std::string contact_id;
  std::string display_name;
};

struct EntityResult {
  std::string type;              // "contact"
  std::string text;              // visible spoken/display text
  float start_ms = -1.0f;
  float end_ms = -1.0f;
  float score = 0.0f;
  bool ambiguous = false;
  std::vector<ContactCandidateResult> candidates;
};

struct NBestResult {
  std::string text;
  float score = 0.0f;
  std::vector<TokenResult> tokens;
  std::vector<EntityResult> entities;
};

struct AsrResult {
  ...
  std::vector<EntityResult> entities;
};
```

### 14.1 Mapping order

Do not pass dynamic contact IDs blindly through the current static `OutputSequenceMapper`.

Use this order:

1. convert raw Flashlight labels to typed base/contact items;
2. apply package output mapping only to contiguous base-word spans;
3. expand each contact item to visible display words;
4. preserve one entity span covering the complete contact;
5. build final text and public tokens.

### 14.2 Contact timestamps

The virtual contact label has one acoustic frame span.

For visible token output:

- split `display_name` by normalized whitespace;
- divide the contact span deterministically among visible words;
- mark derived timestamps internally;
- keep the entity start/end equal to the full contact span.

For unsegmented names, emit one visible token.

### 14.3 N-best deduplication

Replace the current ID-only dedup key.

The semantic dedup key must include:

- visible text;
- entity type;
- sorted candidate contact IDs;
- entity span order.

Two spoken forms of the same contact producing the same final semantic result may merge, keeping the highest-scoring hypothesis.

Two different contacts with the same display text must not be incorrectly merged into one unique contact.

### 14.4 JSON output

Add:

```json
{
  "entities": [
    {
      "type": "contact",
      "text": "Ada Wong",
      "start_ms": 840.0,
      "end_ms": 1320.0,
      "score": 10.0,
      "ambiguous": false,
      "candidates": [
        {
          "contact_id": "ada-42",
          "display_name": "Ada Wong"
        }
      ]
    }
  ]
}
```

Include entities inside each n-best item as well as at the top level for the best result.

Do not expose internal dynamic labels.

---

## 15. File-by-file implementation map

| File | Required work |
|---|---|
| `include/asr_sdk/decode_context.h` | New public context, contact, spoken-form, and score configuration types. |
| `include/asr_sdk/asr_engine.h` | Add context compilation and context-aware stream creation while retaining `CreateStream()`. |
| `include/asr_sdk/result.h` | Add entity and contact-candidate result structures; add entities to best and n-best results. |
| `include/asr_sdk/c_api.h` | Add opaque context handle and builder/compile/create-stream functions. |
| `src/sdk/asr_engine.cc` | Compile Flashlight contact contexts; reject non-empty contexts for unsupported decoder types; create streams from compiled resources. |
| `src/sdk/c_api.cc` | Own context handle, copy input data, validate lifecycle, map statuses. |
| `src/sdk/result_json.cc` | Serialize entities and n-best entity data. |
| `src/package/model_package.h` | Add contact class word and capability fields. |
| `src/package/model_package.cc` | Parse `contact_class_word`; preserve backward compatibility. |
| `src/package/model_package_validator.cc` | Validate class word, no static pronunciation, capability report, and contact-ready constraints. |
| `src/flashlight_decoder/flashlight_decoder_resource.h/.cc` | Retain base lexicon entries and spelling index; expose contact class ID; build or support contact-enabled resources. |
| `src/flashlight_decoder/contact_spelling_compiler.h/.cc` | Compile runtime spoken forms into valid AM token sequences. |
| `src/flashlight_decoder/compiled_decode_context.h/.cc` | Own immutable context resource, metadata, diagnostics, and fingerprint. |
| `src/flashlight_decoder/contextual_contact_lm.h/.cc` | Wrap KenLM; implement `<CONTACT>` transition, trigger window, score tiers, state caching, finish, and cache forwarding. |
| `src/flashlight_decoder/dynamic_contact_lexicon.h/.cc` | Group spellings, assign IDs, build combined trie, and compute lookahead scores. |
| `src/flashlight_decoder/decoded_hypothesis.h` | Add internal entity metadata and optional contact-score diagnostics. |
| `src/flashlight_decoder/flashlight_result_mapper.h/.cc` | Resolve dynamic IDs, protect contact spans from static mapping, expand visible names, attach entities. |
| `src/flashlight_decoder/flashlight_ctc_stream_decoder.h/.cc` | Accept the context-specific decoder resource without altering the no-context path. |
| `src/flashlight_decoder/flashlight_asr_stream.h/.cc` | Hold compiled context/resource and build context-aware results. |
| `src/flashlight_decoder/word_dictionary.h/.cc` | Add safe in-memory clone/add APIs if needed by diagnostics or extended output lookup. |
| `src/flashlight_decoder/lexicon_loader.h/.cc` | Retain/index word spellings for runtime form compilation. |
| `src/flashlight_decoder/debug_trace.h/.cc` | Add redacted context count, contact-bias score/tier, and entity metadata when debug is explicitly enabled. |
| `third_party/flashlight-text/.../Trie.h/.cpp` | Remove the six-label correctness cap; preserve an initial reserve only. |
| `cli/asr_stream_file.cc` | Add contact TSV and scoring flags; compile context once; print contact entities. |
| `cli/zipformer_ctc_flashlight_main.cc` | Add low-level contact-context test flags if maintained as a decoder diagnostic tool. |
| `package_workflows/prepare_flashlight_runtime_package.sh` | Add contact class manifest field; never package contacts. |
| `CMakeLists.txt` | Add sources, public header, tests, and optional context benchmark target. |
| `README.md` and `cli/README.md` | Document package preparation, runner usage, score semantics, limitations, and result schema. |

---

## 16. Implementation phases

## Phase 0: lock the baseline

### Tasks

1. Build current `SDK/0.0.5`.
2. Run all existing CTest targets.
3. Save a deterministic no-context decode result set:
   - final text;
   - n-best text;
   - raw word IDs;
   - total, AM, and LM scores;
   - RTF.
4. Add a small regression fixture if one does not exist.
5. Record the exact compiler, build flags, and package checksum.

### Exit criteria

- current tests pass;
- baseline result file is committed under `test/fixtures` or generated deterministically;
- Codex can demonstrate unchanged no-context behavior after every later phase.

### Suggested commit

```text
test: lock no-context Flashlight decoder baseline
```

---

## Phase 1: make the model package contact-class aware

### Tasks

1. Add `contact_class_word` parsing.
2. Add package capability reporting.
3. Validate:
   - class word exists in `words.txt`;
   - class word differs from `<unk>`;
   - class word has no static lexicon pronunciation;
   - contact-ready package uses `smearing=max`.
4. Update package preparation script.
5. Add a small contact-normalized LM training fixture or documented external training step.
6. Add package-validator tests for:
   - old package without contact support;
   - valid contact-ready package;
   - missing class word;
   - class word incorrectly present in lexicon;
   - unsupported smearing.

### Exit criteria

- old package loads for no-context decoding;
- valid contact-ready package reports support;
- invalid packages fail with specific messages;
- no contact data appears in package output.

### Suggested commit

```text
feat(package): add runtime contact class capability
```

---

## Phase 2: add runtime context API and validation

### Tasks

1. Add public C++ context types.
2. Add `CompileDecodeContext()`.
3. Add context-aware `CreateStream()`.
4. Keep current `CreateStream()` behavior.
5. Implement immutable internal context ownership.
6. Add C API context builder and lifecycle.
7. Validate:
   - non-empty IDs and display names;
   - at least one spoken form;
   - score ordering;
   - finite scores;
   - positive trigger window;
   - configured limits;
   - context belongs to the engine that compiled it;
   - no mutation after compile.
8. Add empty-context fast path.

### Exit criteria

- empty context is accepted and routes to the base decoder;
- non-empty context can be built but does not yet affect decoding;
- context handles are reusable across streams;
- invalid lifecycle operations fail cleanly.

### Suggested commit

```text
feat(api): add immutable runner-provided decode context
```

---

## Phase 3: implement contact spelling compilation

### Tasks

1. Retain base lexicon entries in the model resource.
2. Build a word-to-spellings index.
3. Implement explicit-token validation.
4. Implement static-lexicon composition.
5. Implement bounded token-table segmentation fallback.
6. Group identical token spellings.
7. Assign deterministic dynamic labels.
8. Produce diagnostics and context fingerprint.
9. Add safety limits.

### Exit criteria

- the example contacts compile;
- invalid tokens and unencodable forms fail as specified;
- duplicate forms are deterministic;
- no AM token table is modified.

### Suggested commit

```text
feat(context): compile contact spoken forms to AM tokens
```

---

## Phase 4: build the dynamic combined lexicon

### Tasks

1. Patch the Flashlight trie label cap.
2. Build a combined trie from base entries and dynamic contact entries.
3. Insert contact lookahead scores.
4. Enforce `MAX` smearing.
5. Support:
   - single-word contacts;
   - multiword contacts;
   - shared prefixes;
   - a contact that is also a prefix of another contact;
   - base-word and contact labels on the same spelling;
   - ambiguous duplicate contacts.
6. Verify lookahead telescoping does not alter final score.

### Exit criteria

- trie search finds every contact spelling;
- partial nodes have the intended lookahead;
- complete leaves expose dynamic labels;
- more than six labels on one path are preserved;
- no-context trie is untouched.

### Suggested commit

```text
feat(decoder): build runtime contact lexicon paths
```

---

## Phase 5: implement the composite contact LM

### Tasks

1. Implement wrapper and state type.
2. Forward normal words to KenLM.
3. Map dynamic labels to `<CONTACT>`.
4. Add complete and command tiers.
5. Implement trigger window.
6. Add word-score correction.
7. Implement `finish()` and `updateCache()`.
8. Use deterministic child-state caching.
9. Add exact-score tests with a fake main LM.
10. Verify two contact IDs remain distinct even when their underlying main-LM state is the same.

### Exit criteria

- normal word scores equal base LM scores;
- contact completion advances the base LM with `<CONTACT>`;
- continuation after contact uses the correct base state;
- command and non-command scores match exact expected values;
- expired trigger uses the complete tier;
- LM state merging is correct.

### Suggested commit

```text
feat(decoder): add contextual contact LM wrapper
```

---

## Phase 6: integrate streams and results

### Tasks

1. Let a stream select base or context decoder resource.
2. Resolve dynamic labels in partial and final hypotheses.
3. Add contact entities.
4. Protect contact spans from static output mapping.
5. Expand display words and derive timestamps.
6. Fix semantic n-best deduplication.
7. Add debug score/tier fields without exposing internal labels.
8. Test reset and finalization.

### Exit criteria

- final text displays the contact name;
- contact ID is returned;
- ambiguous contacts return multiple candidates;
- no internal labels appear in public output;
- reset reuses the same immutable context safely;
- no-context JSON remains backward compatible except for optional empty fields if intentionally added.

### Suggested commit

```text
feat(result): return runtime contact entities
```

---

## Phase 7: runner and C API integration

### Tasks

1. Implement TSV loader.
2. Add `asr_stream_file` flags.
3. Compile context once before stream creation.
4. Print entity results in debug or structured output.
5. Add C API example.
6. Update C++ example.
7. Decide release ABI policy:
   - additive functions can preserve binary compatibility;
   - changing public C++ result structure layout may require an ABI-version bump.
8. Update build-info JSON to advertise runtime-contact support.

### Exit criteria

A runner command can decode with contacts without modifying the model directory.

Example:

```bash
build/asr_stream_file \
  --model_dir test/0.0.5/model_flashlight_contact_ready \
  --wav test.wav \
  --contacts_tsv contacts.tsv \
  --contact_partial_score 2 \
  --contact_complete_score 6 \
  --contact_command_score 10 \
  --contact_triggers dial,message,call,text \
  --contact_trigger_window 5
```

### Suggested commit

```text
feat(cli): supply runtime contact context from runner
```

---

## Phase 8: evaluation, tuning, and documentation

### Tasks

1. Build a deterministic contact evaluation set.
2. Build a negative/general-speech set.
3. Tune score tiers and beam settings.
4. Measure context compilation time and memory.
5. Measure streaming RTF and partial stability.
6. Document known limits.
7. Add migration notes and package preparation instructions.
8. Produce a final validation report.

### Suggested commit

```text
docs: validate and document runtime contact bias
```

---

## 17. Test plan

## 17.1 Unit tests

Add the following test targets.

### `contact_context_config_test`

Cases:

- empty context;
- valid context;
- empty contact ID;
- empty display name;
- no spoken forms;
- duplicate form;
- invalid score ordering;
- NaN or infinity;
- zero trigger window;
- contact-count and token-count limits;
- mutation after compile.

### `contact_spelling_compiler_test`

Cases:

- explicit tokens;
- unknown token;
- forbidden blank token;
- static lexicon composition;
- multiple pronunciation combinations;
- token-table fallback;
- no possible segmentation;
- UTF-8 text;
- deterministic ordering;
- strict versus skip mode.

### `dynamic_contact_lexicon_test`

Cases:

- `Ada Wong`;
- `Ashley`;
- `Ada` and `Ada Wong`;
- `Ada Wong` and `Ada Lovelace`;
- base word and contact with identical spelling;
- duplicate contacts with identical spelling;
- more than six labels;
- partial-node score;
- completed dynamic label;
- MAX smearing requirement.

### `contextual_contact_lm_test`

Use a fake deterministic main LM.

Cases:

- start state;
- ordinary word forwarding;
- trigger activation;
- trigger distance increment;
- trigger expiration;
- contact completion outside context;
- contact completion inside context;
- `<CONTACT>` main-LM transition;
- continuation after contact;
- finish;
- word-score correction;
- `lm_weight` conversion;
- deterministic state caching;
- separate dynamic IDs remain distinct.

### `contact_result_mapper_test`

Cases:

- one contact;
- multiword display name;
- ambiguous candidates;
- static mapping before and after contact;
- derived timestamps;
- no internal labels;
- semantic n-best deduplication.

### `contact_c_api_test`

Cases:

- context lifecycle;
- copied strings;
- compile then create two streams;
- mutation after compile;
- invalid engine/context pairing;
- destroy order.

## 17.2 Synthetic decoder integration test

Create a tiny deterministic decoder fixture:

```text
tokens:
  <blk>, ▁, p, l, e, a, s, c, o, f, d, w, n, g, h, m, ...

base words:
  <unk>, please, call, the, phone, of, home, tomorrow, <CONTACT>

base lexicon:
  please ...
  call ...
  home ...
  tomorrow ...
```

Runtime contact:

```text
Ada Wong
```

Construct CTC emissions that make these acoustically competitive:

```text
call home
call Ada Wong
call a wrong
```

Assertions:

1. without context, the baseline candidate wins;
2. with contact context, `call Ada Wong` wins;
3. a partial `Ada` path remains in the beam;
4. `Ada Wong` receives complete tier without a trigger;
5. it receives command tier after `call`;
6. `tomorrow` is scored from the main state after `<CONTACT>`;
7. disabling the contact context reproduces the baseline score exactly.

## 17.3 Concurrency tests

1. One engine.
2. Context A contains `Ada Wong`.
3. Context B contains `Leon`.
4. Decode streams concurrently.
5. Verify:
   - A never returns B's contacts;
   - B never returns A's contacts;
   - empty context returns neither;
   - context destruction after stream creation is safe through shared ownership.

## 17.4 Package tests

- old package, no context: pass;
- old package, non-empty context: fail precondition;
- contact-ready package: pass;
- `<CONTACT>` missing from words: fail;
- `<CONTACT>` in lexicon: fail;
- contacts file accidentally placed in package: packaging script should neither copy nor reference it.

## 17.5 Regression tests

Run the existing evaluation path with no runtime context.

Require:

- identical best text;
- identical n-best text;
- identical score components within exact floating-point output where possible;
- no material RTF change;
- all existing CTest targets pass.

---

## 18. Evaluation and tuning plan

## 18.1 Datasets

Create three deterministic sets.

### A. Contact-command positives

Examples:

```text
call Ada Wong
please call Ada Wong
please call the phone of Ada Wong
message Claire
text Leon tomorrow
dial Ashley
```

Include:

- short and long names;
- overlapping names;
- code-switched names;
- accents;
- aliases;
- acoustically confusable ordinary phrases.

### B. Contact mentions without command triggers

Examples:

```text
I met Ada Wong yesterday
Claire answered the question
Leon will arrive tomorrow
```

This set validates the middle complete tier.

### C. Negative/general speech

Include phrases confusable with names:

```text
the weather is clear
I saw a lion
that was a wrong number
call home
```

This set measures false contact insertion.

## 18.2 Metrics

Report:

- contact entity Recall@1;
- contact entity Recall@N;
- exact contact-ID accuracy for unambiguous contacts;
- ambiguous-contact detection accuracy;
- false contact insertion rate;
- general WER/CER;
- command-context versus non-command accuracy;
- average partial revisions;
- first contact-entity latency;
- streaming RTF;
- context compilation wall time;
- context memory by number of contacts and token nodes.

## 18.3 Tuning grid

Start with a bounded grid:

```text
partial_match_score:  0.5, 1, 2, 3
complete_match_score: 3, 5, 7, 9
command_match_score:  6, 9, 12, 15
trigger_window_words: 1, 3, 5, 8
beam_size:            current, 2x current
beam_size_token:      current, 2x current
```

Always enforce:

```text
partial < complete < command
```

Tune on a development set and report once on the held-out set.

## 18.4 Provisional release gates

These are initial engineering gates and may be revised after baseline measurement.

1. No-context text regression: **zero** on the locked deterministic fixture.
2. No-context CTest regression: **zero failures**.
3. Synthetic command-contact tests: **100% pass**.
4. General WER/CER regression with an active contact list: no more than **0.2 absolute points** on the negative set.
5. False contact insertion rate: no more than **0.5%** on the negative set.
6. Streaming RTF overhead after context compilation: no more than **10%**.
7. Context compilation for 5,000 contacts: target under **2 seconds** on the reference machine.
8. Additional context memory for 5,000 contacts: target under **100 MB**.
9. No cross-context leakage in concurrency tests.
10. No raw contact data persisted to the model directory.

If a gate is missed, keep the feature behind the runtime context API and do not change the default no-context path.

---

## 19. Error handling and validation

Return explicit statuses for:

| Condition | Recommended status |
|---|---|
| Package lacks contact class support | `FailedPrecondition` |
| `lm_weight <= 0` with contacts | `FailedPrecondition` |
| Contact score is non-finite | `InvalidArgument` |
| Score ordering is invalid | `InvalidArgument` |
| Empty contact ID/display/form | `InvalidArgument` |
| Unknown explicit AM token | `InvalidArgument` |
| Spoken form cannot be encoded | `InvalidArgument` |
| Context exceeds configured limits | `InvalidArgument` |
| Context used with another engine | `FailedPrecondition` |
| Context modified after compile | `FailedPrecondition` |
| Context used before compile in C API | `FailedPrecondition` |
| Dynamic label lookup fails internally | `Internal` |

Do not:

- silently map unencodable contacts to `<unk>`;
- silently drop dynamic labels;
- silently choose one ambiguous contact;
- silently fall back to a different model package.

---

## 20. Security and privacy requirements

1. Contact data is supplied only through runtime APIs or runner input.
2. Never write contacts into the model package.
3. Never include contacts in package checksums.
4. Do not create an on-disk compiled context cache in version 1.
5. Copy runner strings into context-owned memory.
6. Use immutable shared context ownership for thread safety.
7. Apply strict size limits to prevent memory exhaustion.
8. Treat contact IDs as opaque strings.
9. Do not log raw contacts by default.
10. When debug logging is enabled:
    - log context counts and score tiers;
    - avoid dumping the whole contact list;
    - final recognized contact text may appear because it is part of the ASR result;
    - document this behavior.
11. Release all context-owned data when the last context/stream reference is destroyed.
12. Best-effort memory clearing may be added for sensitive strings, but it must not be advertised as guaranteed secure erasure.

---

## 21. Performance considerations

### 21.1 Compile once, decode many times

The runner should compile one context per active user/profile and reuse it across utterances.

Do not rebuild the combined trie for every stream when the same compiled context is available.

### 21.2 Context fingerprint

Compute a deterministic in-memory fingerprint from:

- package identity/checksum;
- normalized contacts;
- token spellings;
- bias options;
- trigger words.

Use it only for diagnostics and optional in-process cache lookup. Do not persist it with raw contact data.

### 21.3 Trie size

Track:

- unique contact forms;
- unique token spellings;
- total dynamic trie nodes;
- maximum spelling length;
- shared-prefix ratio.

Reject contexts exceeding safety limits before allocating unbounded memory.

### 21.4 Beam impact

Long contact forms receive only static prefix lookahead until completion. If they are still pruned too early:

1. tune partial score;
2. tune `beam_size` and `beam_size_token`;
3. measure false insertions;
4. only then consider a later context-dependent prefix-scoring hook in the decoder.

Do not begin by rewriting the decoder.

---

## 22. Backward compatibility and release policy

### 22.1 Source compatibility

Keep:

```cpp
AsrEngine::CreateStream()
```

and all existing stream methods.

New API is additive.

### 22.2 Binary compatibility

Adding fields to exported C++ structs changes their layout. Before release, choose one:

1. bump `ASR_SDK_ABI_VERSION`; or
2. expose entity data through new accessor interfaces rather than changing existing struct layout.

For this SDK, an ABI bump is the clearer choice if `AsrResult` and `NBestResult` are modified directly.

The C API can remain backward compatible by adding new functions and extending returned JSON with new optional fields.

### 22.3 Old packages

- old package + no context: supported;
- old package + non-empty context: explicit failure;
- contact-ready package + no context: supported;
- contact-ready package + context: supported.

### 22.4 Feature advertisement

Add to build-info or package inspection:

```json
{
  "runtime_contact_bias": true
}
```

Do not infer support merely from SDK version; verify the package capability too.

---

## 23. Documentation deliverables

Update documentation with:

1. architecture diagram;
2. distinction between model package and runtime contact context;
3. `<CONTACT>` LM-training instructions;
4. contact TSV format;
5. C++ API example;
6. C API example;
7. runner command example;
8. score units and ordering;
9. trigger-window semantics;
10. ambiguity behavior;
11. output JSON schema;
12. privacy behavior;
13. performance limits;
14. troubleshooting:
    - missing `<CONTACT>`;
    - unencodable spoken form;
    - contact path pruned;
    - false positives;
    - ambiguous contacts.

---

## 24. Codex implementation rules

Codex should follow these constraints while implementing.

1. Work from the current `SDK/0.0.5` code, not from an imagined generic decoder.
2. Preserve the existing no-context resource and execution path.
3. Do not add names or `<CONTACT>` to the acoustic-model token vocabulary.
4. Do not add a static pronunciation for `<CONTACT>`.
5. Do not store contact data in model-package files.
6. Use a complete spoken form as one dynamic virtual lexicon label in version 1.
7. Use `MAX` smearing for contact-enabled combined tries.
8. Treat public contact scores as final decoder-score units.
9. Correct `word_score` for multiword virtual labels.
10. Ensure main-LM state advances with `<CONTACT>` exactly once.
11. Use immutable context objects and shared ownership.
12. Do not silently skip invalid contacts unless explicitly configured.
13. Do not expose internal dynamic labels.
14. Preserve ambiguous candidate IDs.
15. Add tests before integrating the CLI.
16. Keep the third-party Flashlight patch minimal and documented.
17. Run formatting, build, CTest, synthetic decoder tests, and no-context regression tests after every phase.
18. Use small phase-aligned commits.
19. Do not claim performance or accuracy improvements without the recorded evaluation report.
20. If an implementation detail conflicts with this algorithm, stop that change and document the conflict rather than silently changing the agreed scoring semantics.

---

## 25. Definition of done

The feature is complete only when all items below are true.

### Package

- [ ] Contact-ready LM contains `<CONTACT>`.
- [ ] `words.txt` contains `<CONTACT>`.
- [ ] Static `lexicon.txt` does not contain `<CONTACT>`.
- [ ] Manifest declares `contact_class_word`.
- [ ] Contact list is absent from the package.
- [ ] Package validator and inspector cover contact capability.

### Runtime API

- [ ] Runner can compile contacts into an immutable context.
- [ ] Context can be shared by multiple streams.
- [ ] Existing `CreateStream()` still works.
- [ ] C API has a complete context lifecycle.
- [ ] Invalid contexts fail with actionable errors.

### Decoder

- [ ] Normal words use main KenLM.
- [ ] Partial contact path receives low lookahead.
- [ ] Complete contact receives higher score.
- [ ] Triggered complete contact receives highest score.
- [ ] Completed contact advances KenLM with `<CONTACT>`.
- [ ] Later words use the state after `<CONTACT>`.
- [ ] Multiword word-score correction is applied.
- [ ] Overlapping and ambiguous contacts work.
- [ ] No-contact path is unchanged.

### Results

- [ ] Visible text contains display names, not internal labels.
- [ ] Contact IDs are returned as structured entities.
- [ ] Ambiguity is represented explicitly.
- [ ] Timestamps cover the contact span.
- [ ] n-best semantic deduplication is correct.
- [ ] C API JSON includes entities.

### Testing

- [ ] Existing tests pass.
- [ ] New unit tests pass.
- [ ] Synthetic end-to-end decoder tests pass.
- [ ] Concurrency isolation tests pass.
- [ ] Old-package compatibility tests pass.
- [ ] No-context regression fixture is unchanged.
- [ ] Performance and accuracy report is produced.

### Documentation and privacy

- [ ] Runner usage is documented.
- [ ] LM preparation is documented.
- [ ] Score tuning is documented.
- [ ] Contact data is not persisted.
- [ ] Debug privacy behavior is documented.

---

## 26. Expected final architecture

```text
                         reusable model package
                ┌─────────────────────────────────┐
                │ model.onnx                      │
                │ tokens.txt                      │
                │ words.txt, including <CONTACT>  │
                │ lexicon.txt, excluding <CONTACT>│
                │ lm.bin trained with <CONTACT>   │
                │ sdk_model.json                  │
                └─────────────────────────────────┘
                                  │
                                  ▼
                       engine-owned base resources
                                  │
              runner supplies contacts and bias configuration
                                  │
                                  ▼
                    compile immutable DecodeContext
                ┌─────────────────────────────────┐
                │ dynamic contact spellings       │
                │ combined lexicon trie           │
                │ virtual contact labels          │
                │ contextual LM wrapper           │
                │ trigger tracker                 │
                │ contact metadata                │
                └─────────────────────────────────┘
                                  │
                                  ▼
                           streaming decode
                ┌─────────────────────────────────┐
                │ CTC AM scores                   │
                │ normal path: main LM            │
                │ contact prefix: low lookahead   │
                │ contact complete: higher score  │
                │ command contact: highest score  │
                │ main LM receives <CONTACT>      │
                └─────────────────────────────────┘
                                  │
                                  ▼
                  visible text + structured contact entity
```

This architecture implements the agreed behavior while preserving the current acoustic model, the existing normal-language path, and the separation between reusable model resources and private runtime user data.
