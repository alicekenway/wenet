#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_DECODED_HYPOTHESIS_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_DECODED_HYPOTHESIS_H_

#include <string>
#include <vector>

namespace asr_sdk::internal::flashlight_decoder {

enum class DecoderSource {
  kMain,
  kContact,
};

struct DecodedWord {
  int word_id = -1;
  std::string text;
  int start_frame = -1;
  int end_frame = -1;
  bool timestamp_derived = false;
  bool is_contact = false;
  int contact_entity_index = -1;
};

struct DecodedContactCandidate {
  std::string contact_id;
  std::string display_name;
};

struct DecodedEntity {
  std::string type = "contact";
  std::string text;
  int start_frame = -1;
  int end_frame = -1;
  float score = 0.0f;
  bool ambiguous = false;
  std::vector<DecodedContactCandidate> candidates;
};

struct DecodedHypothesis {
  double total_score = 0.0;
  double am_score = 0.0;
  double lm_score = 0.0;
  // Actual contribution to the decoder objective after LM weights are
  // applied.  It differs from lm_score for the main decoder and is the value
  // printed in human-facing contact debug logs.
  double weighted_lm_score = 0.0;
  double first_pass_score = 0.0;
  std::vector<int> token_ids;
  std::vector<DecodedWord> raw_words;
  std::vector<DecodedWord> am_mapped_words;
  std::vector<DecodedWord> mapped_words;
  std::vector<DecodedEntity> entities;
  DecoderSource source = DecoderSource::kMain;
};

std::vector<int> WordIds(const std::vector<DecodedWord>& words);
std::string JoinWords(const std::vector<DecodedWord>& words,
                      const std::string& separator);

// Merge independently searched main/contact hypotheses.  Contact-aware
// callers retain distinct semantic entities even when their visible text is
// identical; ordinary no-context decoding keeps its historical word-ID key.
std::vector<DecodedHypothesis> DeduplicateDecodedHypotheses(
    std::vector<DecodedHypothesis> hyps, int limit, bool contact_aware);

bool HasCompletedContact(const DecodedHypothesis& hyp);

// Contact-LM hypotheses are eligible to compete with the main beam only after
// a runtime contact terminal has been emitted.  This keeps the narrow contact
// LM from winning an ordinary utterance before a name has completed.
std::vector<DecodedHypothesis> MergeMainAndEligibleContactHypotheses(
    std::vector<DecodedHypothesis> main_hyps,
    std::vector<DecodedHypothesis> contact_hyps, int limit);

}  // namespace asr_sdk::internal::flashlight_decoder

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_DECODED_HYPOTHESIS_H_
