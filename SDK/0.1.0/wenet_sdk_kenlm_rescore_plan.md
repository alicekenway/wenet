# Codex Development Plan: Replace Online KenLM Shallow Fusion with Final KenLM Rescoring

## Goal

Modify `SDK/0.1.0` so the Flashlight/KenLM path works as:

```text
CTC AM logits
  ↓
Flashlight LexiconDecoder first pass
  - keep lexicon
  - do not use KenLM shallow fusion during beam search
  ↓
raw word N-best hypotheses
  ↓
AM-stage mapping, optional/identity when empty
  ↓
KenLM final rescoring on complete AM-mapped word sequences
  ↓
final-stage mapping, optional/identity when empty
  ↓
deduplicate by final displayed words
  ↓
return final N-best + optional debug log
```

This plan intentionally **keeps the lexicon**. The lexicon remains the first-pass word candidate generator. KenLM is removed from online beam scoring and used only after first-pass decoding is finished.

---

## Non-goals

Do **not** replace `LexiconDecoder` with a token-level CTC beam decoder in this patch.

Do **not** implement WFST/lattice rescoring in this patch.

Do **not** try to score partial words with KenLM.

Do **not** run KenLM rescoring for partial results.

---

## Current relevant files

Expected main files to modify:

```text
SDK/0.1.0/src/flashlight_decoder/flashlight_ctc_stream_decoder.cc
SDK/0.1.0/src/flashlight_decoder/flashlight_decoder_resource.h
SDK/0.1.0/src/flashlight_decoder/flashlight_decoder_resource.cc
SDK/0.1.0/src/flashlight_decoder/flashlight_result_mapper.cc
SDK/0.1.0/src/flashlight_decoder/decoded_hypothesis.h
SDK/0.1.0/src/flashlight_decoder/decoded_hypothesis.cc
SDK/0.1.0/src/flashlight_decoder/output_sequence_mapper.h
SDK/0.1.0/src/flashlight_decoder/output_sequence_mapper.cc
SDK/0.1.0/src/flashlight_decoder/flashlight_asr_stream.cc
SDK/0.1.0/src/flashlight_decoder/flashlight_asr_stream.h
SDK/0.1.0/src/flashlight_decoder/flashlight_decoder_options.h
SDK/0.1.0/src/package/model_package.h
SDK/0.1.0/src/package/model_package.cc
SDK/0.1.0/src/package/model_package_validator.cc
SDK/0.1.0/include/asr_sdk/config.h
SDK/0.1.0/include/asr_sdk/result.h
SDK/0.1.0/src/sdk/result_json.cc
SDK/0.1.0/cli/asr_stream_file.cc
SDK/0.1.0/cli/asr_package_eval.cc
SDK/0.1.0/package_workflows/prepare_flashlight_runtime_package.sh
SDK/0.1.0/CMakeLists.txt
```

Expected new files:

```text
SDK/0.1.0/src/flashlight_decoder/kenlm_rescorer.h
SDK/0.1.0/src/flashlight_decoder/kenlm_rescorer.cc
SDK/0.1.0/src/flashlight_decoder/debug_trace.h
SDK/0.1.0/src/flashlight_decoder/debug_trace.cc
```

`debug_trace.*` can be skipped if Codex chooses a smaller implementation, but there should still be one central helper for building debug JSON.

---

## Terminology

Use three word sequences in `DecodedHypothesis`:

```cpp
raw_words
```

Words emitted directly by Flashlight `LexiconDecoder`.

```cpp
am_mapped_words
```

Words after the first mapping stage. This is the sequence scored by KenLM.

```cpp
mapped_words
```

Final displayed words after the final mapping stage. Keep this name for compatibility with existing output code.

Recommended naming:

```text
raw_words         = raw decoder words
am_mapped_words   = after AM-stage mapping; used by KenLM
mapped_words      = after final-stage mapping; returned to user
```

---

## Mapping design

There are now two mapping stages.

### 1. AM-stage mapping

Applied immediately after first-pass AM/lexicon decoding.

Purpose:

```text
raw_words -> am_mapped_words
```

KenLM scores this sequence.

This mapping is useful when AM/lexicon output contains forms that should be normalized before word-LM scoring.

### 2. Final-stage mapping

Applied after KenLM rescoring/reranking.

Purpose:

```text
am_mapped_words -> mapped_words
```

The SDK returns this sequence to the user.

This mapping is useful for final text correction, command normalization, or display normalization.

Important behavior:

```text
KenLM must score am_mapped_words, not mapped_words.
```

If a rule should affect LM scoring, put it in the AM-stage mapping. If a rule should only affect the final displayed result, put it in the final-stage mapping.

---

## Mapping file names and manifest fields

Keep backward compatibility with the current `mapping` manifest field.

Recommended manifest fields:

```json
{
  "mapping": "output_mapping.txt",
  "final_mapping": "final_output_mapping.txt"
}
```

Interpretation:

```text
mapping       = AM-stage mapping path
final_mapping = final-stage mapping path
```

Both files may be empty. Empty means identity mapping.

If `mapping` is missing, default to `output_mapping.txt` if present, otherwise identity.

If `mapping` is explicitly `""`, use identity.

If `final_mapping` is missing, use identity by default. Alternatively, the packaging script may create an empty `final_output_mapping.txt` and point to it.

---

## Mapping replacement semantics

The mapper must replace **all occurrences** across a sentence, not only the first occurrence.

Example:

```text
mapping rule:
A B -> X

input:
A B D A B

output:
X D X
```

A sentence may contain more than one mapping match. Apply the mapping repeatedly while scanning left-to-right.

Current `OutputSequenceMapper::RewriteWords()` already scans the whole sentence and applies every non-overlapping longest match. Keep and test this behavior.

### Longest-match behavior

If two rules can match at the same position, use the longest source sequence.

Example:

```text
rules:
A B -> X
A B C -> Y

input:
A B C D

output:
Y D
```

### Empty mapping behavior

If the mapping file is empty or the mapping path is disabled, output must be identical to input.

---

## First-pass decoder change: remove online shallow fusion

### File

```text
src/flashlight_decoder/flashlight_ctc_stream_decoder.cc
```

### Change `MakeDecoderOptions()`

Current behavior passes the package LM scores into Flashlight. Replace it with neutral online LM settings.

Target behavior:

```cpp
fl::lib::text::LexiconDecoderOptions MakeDecoderOptions(
    const FlashlightDecoderOptions& options) {
  fl::lib::text::LexiconDecoderOptions out;
  out.beamSize = options.beam_size;
  out.beamSizeToken = options.beam_size_token;
  out.beamThreshold = options.beam_threshold;

  // KenLM is final-only now. Do not use online shallow fusion.
  out.lmWeight = 0.0;
  out.wordScore = 0.0;
  out.unkScore = options.allow_unk
      ? 0.0
      : -std::numeric_limits<float>::infinity();
  out.silScore = 0.0;

  out.logAdd = options.log_add;
  out.criterionType = fl::lib::text::CriterionType::CTC;
  return out;
}
```

Keep using `LexiconDecoder`; do not remove the lexicon.

---

## Trie construction change: remove KenLM-derived insertion scores

### File

```text
src/flashlight_decoder/flashlight_decoder_resource.cc
```

The trie is currently built by scoring lexicon entries with KenLM. That must stop.

Target behavior:

```cpp
for (const LexiconEntry& entry : entries) {
  lexicon_trie_->insert(entry.token_ids, entry.word_id, 0.0f);
}
```

Keep:

```cpp
lexicon_trie_->smear(SmearingModeFromString(options_.smearing));
```

Smearing has no LM-score effect when all insertion scores are zero, but keeping it avoids unnecessary structural changes.

Keep loading KenLM into `word_lm_`, because the new final rescorer will use it.

---

## Resource changes for two mapping stages

### Files

```text
src/flashlight_decoder/flashlight_decoder_resource.h
src/flashlight_decoder/flashlight_decoder_resource.cc
```

### Constructor signature

Change from one mapping path to two mapping paths:

```cpp
FlashlightDecoderResource(
    const std::filesystem::path& tokens_path,
    const std::filesystem::path& words_path,
    const std::filesystem::path& lexicon_path,
    const std::filesystem::path& lm_path,
    const std::filesystem::path& am_mapping_path,
    const std::filesystem::path& final_mapping_path,
    FlashlightDecoderOptions options,
    std::string blank_token,
    std::string sil_token,
    std::string unk_word);
```

### Members

Replace:

```cpp
OutputSequenceMapper output_mapper_;
```

with:

```cpp
OutputSequenceMapper am_mapper_;
OutputSequenceMapper final_mapper_;
```

### Accessors

Add:

```cpp
const OutputSequenceMapper& AmMapper() const { return am_mapper_; }
const OutputSequenceMapper& FinalMapper() const { return final_mapper_; }
```

For backward compatibility inside the codebase, either remove `Mapper()` usages or temporarily keep:

```cpp
const OutputSequenceMapper& Mapper() const { return final_mapper_; }
```

But prefer to update all call sites to use `AmMapper()` and `FinalMapper()` explicitly.

### Loading behavior

```cpp
am_mapper_ = OutputSequenceMapper::Load(am_mapping_path, output_words_);
final_mapper_ = OutputSequenceMapper::Load(final_mapping_path, output_words_);
```

If either path is empty, `OutputSequenceMapper::Load()` should return identity.

---

## `OutputSequenceMapper::Load()` empty-file behavior

### File

```text
src/flashlight_decoder/output_sequence_mapper.cc
```

Current behavior already supports empty files: an empty or comment-only file gives identity behavior with `RuleCount() == 0`.

Keep this behavior.

Add or update tests to guarantee:

```text
empty AM mapping       -> identity
empty final mapping    -> identity
multiple matches       -> all replaced
longest match          -> longest source rule wins
UTF-8 words            -> still work
```

---

## `DecodedHypothesis` changes

### File

```text
src/flashlight_decoder/decoded_hypothesis.h
```

Current struct:

```cpp
struct DecodedHypothesis {
  double total_score = 0.0;
  double am_score = 0.0;
  double lm_score = 0.0;
  std::vector<int> token_ids;
  std::vector<DecodedWord> raw_words;
  std::vector<DecodedWord> mapped_words;
};
```

Target struct:

```cpp
struct DecodedHypothesis {
  // Final score used for sorting/ranking.
  double total_score = 0.0;

  // Acoustic score from Flashlight emitting model.
  double am_score = 0.0;

  // Final KenLM rescore on am_mapped_words.
  double lm_score = 0.0;

  // Optional first-pass score for debugging only.
  double first_pass_score = 0.0;

  std::vector<int> token_ids;

  // Direct Flashlight output.
  std::vector<DecodedWord> raw_words;

  // After AM-stage mapping; this is what KenLM scores.
  std::vector<DecodedWord> am_mapped_words;

  // After final-stage mapping; this is returned to the user.
  std::vector<DecodedWord> mapped_words;
};
```

Update helper functions if they assume only `mapped_words` exists.

---

## Convert Flashlight result

### File

```text
src/flashlight_decoder/flashlight_result_mapper.cc
```

Current behavior sets:

```cpp
hyp.total_score = result.score;
hyp.am_score = result.emittingModelScore;
hyp.lm_score = result.lmScore;
hyp.mapped_words = resource.Mapper().RewriteWords(hyp.raw_words);
```

Target behavior:

```cpp
hyp.first_pass_score = result.score;
hyp.total_score = result.emittingModelScore;
hyp.am_score = result.emittingModelScore;
hyp.lm_score = 0.0;

// Build raw_words as before.

hyp.am_mapped_words = resource.AmMapper().RewriteWords(hyp.raw_words);

// Before final rescoring, initialize final output to AM-mapped words.
// The final mapper will be applied after KenLM rescoring.
hyp.mapped_words = hyp.am_mapped_words;
```

Do not use `result.lmScore` as the final LM score anymore.

`result.lmScore` is from Flashlight online scoring. Since online LM is now disabled, treat it as unused debug-only data if needed.

---

## Add final KenLM rescorer

### New files

```text
src/flashlight_decoder/kenlm_rescorer.h
src/flashlight_decoder/kenlm_rescorer.cc
```

### Header sketch

```cpp
#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_KENLM_RESCORER_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_KENLM_RESCORER_H_

#include <vector>

#include "flashlight_decoder/decoded_hypothesis.h"
#include "flashlight_decoder/flashlight_decoder_resource.h"

namespace asr_sdk::internal::flashlight_decoder {

double ScoreWordsWithKenLm(
    const FlashlightDecoderResource& resource,
    const std::vector<DecodedWord>& words);

void RescoreAndApplyFinalMapping(
    const FlashlightDecoderResource& resource,
    std::vector<DecodedHypothesis>* hyps);

}  // namespace asr_sdk::internal::flashlight_decoder

#endif
```

### Implementation sketch

```cpp
#include "flashlight_decoder/kenlm_rescorer.h"

#include <tuple>

namespace asr_sdk::internal::flashlight_decoder {
namespace {

int CountWords(const std::vector<DecodedWord>& words) {
  int count = 0;
  for (const auto& word : words) {
    if (word.word_id >= 0) {
      ++count;
    }
  }
  return count;
}

int CountWordId(const std::vector<DecodedWord>& words, int word_id) {
  int count = 0;
  for (const auto& word : words) {
    if (word.word_id == word_id) {
      ++count;
    }
  }
  return count;
}

}  // namespace

double ScoreWordsWithKenLm(
    const FlashlightDecoderResource& resource,
    const std::vector<DecodedWord>& words) {
  const auto& lm = resource.WordLm();

  auto state = lm->start(false);
  double score = 0.0;

  for (const auto& word : words) {
    if (word.word_id < 0) {
      continue;
    }

    fl::lib::text::LMStatePtr next_state;
    float word_score = 0.0f;
    std::tie(next_state, word_score) = lm->score(state, word.word_id);
    state = std::move(next_state);
    score += static_cast<double>(word_score);
  }

  // If the Flashlight LM wrapper exposes an EOS/final score API, add it here.
  // If not, leave this consistent with existing online-decoder conventions.

  return score;
}

void RescoreAndApplyFinalMapping(
    const FlashlightDecoderResource& resource,
    std::vector<DecodedHypothesis>* hyps) {
  if (hyps == nullptr) {
    return;
  }

  const auto& options = resource.Options();

  for (auto& hyp : *hyps) {
    // KenLM scores the AM-stage mapped words.
    hyp.lm_score = ScoreWordsWithKenLm(resource, hyp.am_mapped_words);

    const int word_count = CountWords(hyp.am_mapped_words);
    const int unk_count = CountWordId(
        hyp.am_mapped_words, resource.UnknownWordId());
    const int sil_count = CountWordId(
        hyp.am_mapped_words, resource.SilenceWordIdIfAvailable());

    // If there is no SilenceWordIdIfAvailable() helper, either add one or
    // skip sil_score for the first patch. Do not confuse AM silence token id
    // with output word id.

    hyp.total_score =
        hyp.am_score
        + options.lm_weight * hyp.lm_score
        + options.word_score * static_cast<double>(word_count)
        + options.unk_score * static_cast<double>(unk_count);

    // Optional: add sil_score only if a valid output silence-word id exists.
    // hyp.total_score += options.sil_score * static_cast<double>(sil_count);

    // Final mapping changes the returned text, not the LM scoring text.
    hyp.mapped_words = resource.FinalMapper().RewriteWords(hyp.am_mapped_words);
  }
}

}  // namespace asr_sdk::internal::flashlight_decoder
```

Important detail:

```text
Do not use AM silence token id as an output word id.
```

For the first implementation, it is acceptable to ignore `sil_score` unless the code has a reliable output-word silence id.

---

## Finalize path

### File

```text
src/flashlight_decoder/flashlight_ctc_stream_decoder.cc
```

Add include:

```cpp
#include "flashlight_decoder/kenlm_rescorer.h"
```

Change `Finalize()` to:

```cpp
StatusOr<std::vector<DecodedHypothesis>> FlashlightCtcStreamDecoder::Finalize() {
  if (impl_->finalized) {
    return Status::FailedPrecondition("Finalize called twice");
  }

  Status status = Start();
  if (!status.ok()) {
    return status;
  }

  try {
    impl_->decoder->decodeEnd();
    impl_->finalized = true;

    std::vector<DecodedHypothesis> hyps;
    const auto results = impl_->decoder->getAllFinalHypothesis();
    hyps.reserve(results.size());

    for (const auto& result : results) {
      hyps.push_back(ConvertFlashlightResult(result, *impl_->resource));
    }

    RescoreAndApplyFinalMapping(*impl_->resource, &hyps);

    return DeduplicateMapped(std::move(hyps),
                             impl_->resource->Options().nbest);
  } catch (const std::exception& e) {
    return ExceptionStatus("Flashlight final result failed", e);
  }
}
```

Partial results should continue to use first-pass output only:

```text
raw_words -> am_mapped_words -> mapped_words initialized as am_mapped_words
```

Do not run KenLM or final mapping for partials unless explicitly needed later.

If partials should show the same normalization as final output, applying final mapping to partials is allowed, but do not use final mapping for LM scoring.

---

## Deduplication behavior

Keep deduplication, but make sure it runs **after final mapping**.

Dedup key:

```text
mapped_words
```

not:

```text
raw_words
am_mapped_words
```

Reason:

```text
Different raw or AM-mapped hypotheses may become the same final displayed result.
The final N-best should contain distinct final results.
```

Example:

```text
raw hyp 1 -> am_mapped X -> final Y
raw hyp 2 -> am_mapped Z -> final Y
```

Only one `Y` should be returned.

For this patch, keep the existing simple behavior:

```text
group by final mapped words
keep the hypothesis with the highest total_score
sort by total_score
return top nbest
```

Do not implement log-sum duplicate merging in this patch unless it is very easy.

---

## Debug mode requirement

Add a debug mode so inference can return and print useful logs.

Debug mode should be available in both:

```text
public SDK path
CLI tools
```

### Public SDK config

### File

```text
include/asr_sdk/config.h
```

Add:

```cpp
bool debug = false;
```

### Manifest config

### Files

```text
src/package/model_package.h
src/package/model_package.cc
src/package/model_package_validator.cc
```

Add package-level debug config:

```cpp
bool debug = false;
```

Parse from manifest:

```json
{
  "debug": false
}
```

Resolved behavior:

```cpp
resolved.debug = config.debug || package.debug;
```

This allows either user code or package manifest to enable debug.

---

## Debug output carrier

The public `AsrResult` already has:

```cpp
std::string raw_backend_json;
```

Use this field to return debug details without changing `NBestResult` immediately.

When debug is disabled:

```text
raw_backend_json may remain empty or keep existing backend payload behavior.
```

When debug is enabled:

```text
raw_backend_json must contain structured JSON.
```

Recommended JSON shape:

```json
{
  "debug": true,
  "mode": "lm",
  "is_final": true,
  "error": "",
  "logs": [
    "decode_begin",
    "decode_step frames=... vocab=...",
    "decode_end",
    "rescore_nbest count=..."
  ],
  "final_nbest": [
    {
      "rank": 1,
      "text": "final displayed text",
      "raw_text": "raw decoder text",
      "am_mapped_text": "text scored by KenLM",
      "am_score": -123.45,
      "lm_score": -18.7,
      "total_score": -132.8
    }
  ]
}
```

For partial results, either omit `final_nbest` or use:

```json
{
  "debug": true,
  "mode": "lm",
  "is_final": false,
  "logs": [...],
  "partial_best": {
    "text": "...",
    "raw_text": "...",
    "am_mapped_text": "...",
    "am_score": -12.3,
    "lm_score": 0.0,
    "total_score": -12.3
  }
}
```

---

## Debug implementation details

### Add helper

Recommended new files:

```text
src/flashlight_decoder/debug_trace.h
src/flashlight_decoder/debug_trace.cc
```

Minimum functionality:

```cpp
class DebugTrace {
 public:
  void Add(std::string line);
  void SetError(std::string error);
  std::string ToResultJson(
      const std::vector<DecodedHypothesis>& hyps,
      bool is_final) const;
};
```

If Codex wants less structure, implement free functions:

```cpp
std::string BuildDebugJson(
    const std::vector<std::string>& logs,
    const std::string& error,
    const std::vector<DecodedHypothesis>& hyps,
    bool is_final);
```

Use existing JSON escaping helper:

```text
src/utils/json.h
src/utils/json.cc
```

### Store logs per stream

### File

```text
src/flashlight_decoder/flashlight_asr_stream.cc
```

Add to `FlashlightAsrStream::Impl`:

```cpp
std::vector<std::string> debug_logs;
std::string debug_error;
```

Add helper:

```cpp
bool DebugEnabled() const {
  return shared && shared->config.debug;
}
```

Log high-level events only:

```text
stream_init
accept_pcm samples=... sample_rate=...
forward_window final_padding=... frames=...
decode_chunk frames=... vocab=...
partial_result text=...
finalize_start
finalize_done nbest=...
error ...
```

Do not log per-frame logits or huge token probability arrays.

### Attach debug JSON to `AsrResult`

Modify `BuildAsrResult()` signature:

```cpp
AsrResult BuildAsrResult(
    const std::vector<DecodedHypothesis>& hyps,
    bool is_final,
    bool debug,
    const std::vector<std::string>& logs,
    const std::string& error);
```

If `debug == true`, set:

```cpp
result.raw_backend_json = BuildDebugJson(logs, error, hyps, is_final);
```

Make sure final result includes the final N-best after rescoring and final mapping.

---

## Debug printing in CLI tools

### `asr_stream_file`

### File

```text
cli/asr_stream_file.cc
```

Add argument:

```text
--debug true|false
```

When enabled:

```cpp
config.debug = true;
```

After final result, print:

```text
[final] <text>
[debug][lm][final_nbest]
rank=1 text="..." am_score=-123.45 lm_score=-18.70 total_score=-132.80
rank=2 text="..." am_score=-124.10 lm_score=-17.90 total_score=-133.05
...
```

The structured debug JSON should still be available in:

```cpp
final_result.raw_backend_json
```

Simple implementation:

```text
print final_result.raw_backend_json to stderr when --debug true
```

Better implementation:

```text
parse or build a readable table from the same debug data
```

For the first patch, printing JSON is acceptable if the JSON contains final N-best with AM and LM scores.

### `asr_package_eval`

### File

```text
cli/asr_package_eval.cc
```

Add argument:

```text
--debug true|false
```

When `decode_mode == lm` and debug is true:

```cpp
config.debug = true;
```

Add debug JSON into output JSONL:

```json
{
  "hyp": "...",
  "debug": { ... }
}
```

or, simpler:

```json
{
  "hyp": "...",
  "debug_json": "{...escaped...}"
}
```

Also print per-utterance final N-best to stderr:

```text
[debug][utt=123][lm][final_nbest]
rank=1 text="..." am_score=... lm_score=... total_score=...
rank=2 text="..." am_score=... lm_score=... total_score=...
```

Do this only under `--debug true`, because it can be verbose.

---

## Error reporting in debug mode

Status return values remain the primary error channel.

When debug mode is enabled and an error occurs during inference:

1. Return the normal `Status` error.
2. Add the error string into the stream debug log if the stream object is still alive.
3. CLI tools should print the error to stderr.
4. `asr_package_eval` should continue writing the existing `error` field to JSONL.

Do not suppress or replace existing `Status` errors.

Recommended helper inside `FlashlightAsrStream`:

```cpp
Status RecordError(Status status) {
  if (DebugEnabled() && !status.ok()) {
    impl_->debug_error = status.ToString();
    impl_->debug_logs.push_back("error " + status.ToString());
  }
  return status;
}
```

Use this helper around failures where practical, without overcomplicating the code.

---

## Public result fields

Current `NBestResult` only exposes:

```cpp
std::string text;
float score;
std::vector<TokenResult> tokens;
```

For minimal API change, keep this unchanged and put AM/LM scores in `AsrResult.raw_backend_json` when debug is enabled.

Optional future extension, not required in this patch:

```cpp
struct NBestResult {
  std::string text;
  float score = 0.0f;
  float am_score = 0.0f;
  float lm_score = 0.0f;
  std::string raw_text;
  std::string am_mapped_text;
  std::vector<TokenResult> tokens;
};
```

Do not make this optional extension unless API/ABI changes are acceptable.

---

## Build result conversion

### File

```text
src/flashlight_decoder/flashlight_asr_stream.cc
```

`BuildAsrResult()` currently sets `NBestResult.text` from `hyp.mapped_words` and `NBestResult.score` from `hyp.total_score`.

Keep that.

Add debug JSON generation when enabled.

For debug final N-best, include for each hypothesis:

```text
rank
final text = JoinWords(hyp.mapped_words, " ")
raw text = JoinWords(hyp.raw_words, " ")
am_mapped text = JoinWords(hyp.am_mapped_words, " ")
am_score = hyp.am_score
lm_score = hyp.lm_score
total_score = hyp.total_score
```

---

## Package parsing changes

### Files

```text
src/package/model_package.h
src/package/model_package.cc
```

Add paths:

```cpp
std::filesystem::path output_mapping_txt;        // existing, AM-stage mapping
std::filesystem::path final_output_mapping_txt;  // new, final-stage mapping
```

Parse manifest:

```cpp
const std::string mapping = FindJsonStringValue(
    manifest_json, "mapping", "output_mapping.txt");
package.output_mapping_txt = mapping.empty()
    ? std::filesystem::path()
    : ResolveUnder(package.root, mapping);

const std::string final_mapping = FindJsonStringValue(
    manifest_json, "final_mapping", "");
package.final_output_mapping_txt = final_mapping.empty()
    ? std::filesystem::path()
    : ResolveUnder(package.root, final_mapping);
```

Alternative default:

```cpp
FindJsonStringValue(manifest_json, "final_mapping", "final_output_mapping.txt")
```

If using this default, the packaging script should always create an empty `final_output_mapping.txt`.

Recommended simpler default:

```text
missing final_mapping -> identity
```

This avoids requiring an extra file.

---

## Package validator changes

### File

```text
src/package/model_package_validator.cc
```

Validate both mapping files if configured and present.

```cpp
if (!package.output_mapping_txt.empty() && FileExists(package.output_mapping_txt)) {
  (void)flashlight_decoder::OutputSequenceMapper::Load(
      package.output_mapping_txt, words);
}

if (!package.final_output_mapping_txt.empty() &&
    FileExists(package.final_output_mapping_txt)) {
  (void)flashlight_decoder::OutputSequenceMapper::Load(
      package.final_output_mapping_txt, words);
}
```

Inspect report should show:

```text
am_mapping: identity or path
am_mapping rule count: N
final_mapping: identity or path
final_mapping rule count: N
```

Also update decoder option report wording:

```text
decoder lm_weight: final KenLM rescore weight
decoder word_score: final word insertion score
decoder unk_score: final unk penalty
```

---

## Package workflow changes

### File

```text
package_workflows/prepare_flashlight_runtime_package.sh
```

Existing env:

```bash
MAPPING="${MAPPING:-}"
```

Keep it as AM-stage mapping.

Add:

```bash
FINAL_MAPPING="${FINAL_MAPPING:-}"
DEBUG="${DEBUG:-false}"
```

Copy behavior:

```bash
if [[ -n "${MAPPING}" ]]; then
  require_file "${MAPPING}"
  copy_runtime_file "${MAPPING}" "${OUT_DIR}/output_mapping.txt"
else
  : > "${OUT_DIR}/output_mapping.txt"
fi

if [[ -n "${FINAL_MAPPING}" ]]; then
  require_file "${FINAL_MAPPING}"
  copy_runtime_file "${FINAL_MAPPING}" "${OUT_DIR}/final_output_mapping.txt"
else
  : > "${OUT_DIR}/final_output_mapping.txt"
fi
```

Manifest should include:

```json
"mapping": "output_mapping.txt",
"final_mapping": "final_output_mapping.txt",
"debug": false
```

Note: `debug` can default false even if the package contains the field. CLI or SDK config can override it.

---

## Engine creation changes

### File

```text
src/sdk/asr_engine.cc
```

When constructing `FlashlightDecoderResource`, pass two mapping paths:

```cpp
auto decoder_resource = std::make_shared<...>(
    package.tokens_txt,
    package.words_txt,
    package.lexicon_txt,
    package.kenlm_bin,
    package.output_mapping_txt,
    package.final_output_mapping_txt,
    options,
    package.blank_token,
    package.sil_token,
    package.unk_word);
```

When resolving config:

```cpp
resolved.debug = config.debug || package.debug;
```

---

## CMake changes

### File

```text
CMakeLists.txt
```

Add new files to `ASR_SDK_FLASHLIGHT_DECODER_SOURCES`:

```cmake
src/flashlight_decoder/kenlm_rescorer.cc
src/flashlight_decoder/debug_trace.cc
```

If `debug_trace.cc` is not created, only add `kenlm_rescorer.cc`.

---

## Tests to add/update

### 1. Mapper applies multiple replacements

### File

```text
test/output_sequence_mapper_test.cc
```

Add case:

```text
words:
A B C D X Y

rule:
A B -> X

input:
A B C A B D

expected:
X C X D
```

This confirms “replace all cases” behavior.

### 2. Two-stage mapping test

Add a new test file if useful:

```text
test/two_stage_mapping_test.cc
```

Fake hypothesis:

```text
raw_words: A B D A B
AM mapping: A B -> X
final mapping: X D -> Y
```

Expected pipeline:

```text
raw:        A B D A B
am_mapped:  X D X
final:      Y X
```

### 3. Rescore uses AM mapping, not final mapping

Use a fake rescore helper if KenLM is hard to instantiate in unit tests.

Test concept:

```text
raw_words -> am_mapped_words is what scorer receives
final mapping should not change lm_score input
```

If a real tiny KenLM fixture exists, add an integration test. Otherwise, test the pipeline with a fake scoring function or split the mapping part into a pure helper.

### 4. Dedup after final mapping

Fake hypotheses:

```text
hyp1 am_mapped=A X final=Y total=-5
hyp2 am_mapped=B X final=Y total=-6
hyp3 am_mapped=C X final=Z total=-7
```

Expected final list:

```text
Y once, from hyp1
Z once
```

### 5. Debug JSON test

Create a small helper test:

```text
test/debug_trace_test.cc
```

Expected debug JSON contains:

```text
final_nbest
text
raw_text
am_mapped_text
am_score
lm_score
total_score
```

### 6. Package options test

### File

```text
test/model_package_decoder_options_test.cc
```

Add manifest fields:

```json
"final_mapping": "final_output_mapping.txt",
"debug": true
```

Expected:

```text
package.final_output_mapping_txt points to final_output_mapping.txt
package.debug == true
resolved config debug follows config.debug || package.debug
```

---

## Acceptance criteria

### Functional

1. `decode_mode=lm` still uses Flashlight `LexiconDecoder`.
2. First-pass decoding no longer uses KenLM shallow fusion:
   - `LexiconDecoderOptions.lmWeight == 0.0`
   - `wordScore == 0.0`
   - `silScore == 0.0`
   - `unkScore == 0.0` if `allow_unk=true`, otherwise `-inf`
   - trie insertion scores are `0.0f`
3. KenLM is used only after `decoder->decodeEnd()` and `getAllFinalHypothesis()`.
4. KenLM scores `am_mapped_words`.
5. The SDK returns `mapped_words`, which means after final-stage mapping.
6. Both mapping stages can be empty and behave as identity.
7. Mapping replaces all non-overlapping matches across a sentence.
8. Final deduplication groups by final `mapped_words`.
9. Partial results do not use KenLM rescoring.
10. Final results are sorted by:

```text
am_score + lm_weight * lm_score + word_score * word_count + unk_score * unk_count
```

### Debug

When debug mode is enabled under LM mode:

1. The SDK returns structured debug information in `AsrResult.raw_backend_json`.
2. The final debug JSON includes final N-best results.
3. Each final N-best item includes:

```text
final text
raw text
AM-mapped text
AM score
LM score
total score
```

4. CLI tools can print per-utterance debug output.
5. Inference errors are still returned through `Status`, and debug mode additionally records the error string in the debug log when possible.

### Build/test

1. `cmake --build build -j` succeeds.
2. `ctest --test-dir build --output-on-failure` succeeds.
3. Existing `output_sequence_mapper_test` passes.
4. New tests for multi-replacement and two-stage mapping pass.
5. `asr_stream_file --debug true` prints final result and debug information.
6. `asr_package_eval --decode_mode lm --debug true` writes normal JSONL plus debug details.

---

## Suggested implementation order for Codex

### Step 1: Add data fields

- Add `am_mapped_words` and `first_pass_score` to `DecodedHypothesis`.
- Add `debug` to `EngineConfig`.
- Add `debug` and `final_output_mapping_txt` to `ModelPackage`.

### Step 2: Add two mapping resources

- Change `FlashlightDecoderResource` constructor and members.
- Load `am_mapper_` and `final_mapper_`.
- Update `AsrEngine::CreateFlashlightEngine()` call site.

### Step 3: Neutralize online KenLM

- Set `lmWeight`, `wordScore`, `silScore`, and `unkScore` to neutral values in `MakeDecoderOptions()`.
- Insert lexicon entries into the trie with `0.0f` scores.

### Step 4: Convert result into raw and AM-mapped words

- Update `ConvertFlashlightResult()`.
- Stop using `result.lmScore` as final `lm_score`.

### Step 5: Add final KenLM rescorer

- Add `kenlm_rescorer.h/cc`.
- Score `am_mapped_words`.
- Apply final mapping into `mapped_words` after scoring.

### Step 6: Wire rescorer into finalization

- Call `RescoreAndApplyFinalMapping()` inside `Finalize()` before `DeduplicateMapped()`.
- Keep partial results unrescored.

### Step 7: Add debug JSON

- Add debug helper.
- Store per-stream logs.
- Attach debug JSON to `raw_backend_json`.
- Add CLI `--debug` flags.

### Step 8: Update package scripts and docs

- Add `FINAL_MAPPING` and `DEBUG` to package workflow.
- Update README wording from “KenLM shallow fusion” to “KenLM final rescoring”.
- Document two mapping stages.

### Step 9: Add tests

- Multi-replacement mapping.
- Two-stage mapping.
- Dedup after final mapping.
- Package parsing for `final_mapping` and `debug`.
- Debug JSON content.

---

## Example final debug output

Readable CLI output may look like:

```text
[final] open the window
[debug][lm][final_nbest]
rank=1 text="open the window" raw="open the widow" am_mapped="open the window" am_score=-83.214 lm_score=-5.928 total_score=-86.178
rank=2 text="open window" raw="open window" am_mapped="open window" am_score=-84.005 lm_score=-6.711 total_score=-87.360
rank=3 text="open the windows" raw="open the windows" am_mapped="open the windows" am_score=-83.906 lm_score=-8.219 total_score=-88.016
```

Structured `raw_backend_json` may look like:

```json
{
  "debug": true,
  "mode": "lm",
  "is_final": true,
  "error": "",
  "logs": [
    "stream_init",
    "decode_begin",
    "decode_end",
    "rescore_nbest count=3"
  ],
  "final_nbest": [
    {
      "rank": 1,
      "text": "open the window",
      "raw_text": "open the widow",
      "am_mapped_text": "open the window",
      "am_score": -83.214,
      "lm_score": -5.928,
      "total_score": -86.178
    }
  ]
}
```

---

## Important caution

Final-stage mapping happens **after** LM scoring. Therefore, if final-stage mapping changes the sentence meaning substantially, KenLM has not scored the changed sentence.

Rule of thumb:

```text
Need LM to prefer/penalize the corrected form? Put the rule in AM-stage mapping.
Only need display/command normalization? Put the rule in final-stage mapping.
```

