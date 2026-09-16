#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_DECODED_HYPOTHESIS_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_DECODED_HYPOTHESIS_H_

#include <string>
#include <vector>

namespace asr_sdk::internal::flashlight_decoder {

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
  std::string value_id;
  std::string contact_id;
  std::string display_name;
};

struct DecodedEntity {
  std::string type = "contact";
  std::string slot_token = "<CONTACT>";
  std::string text;
  int start_frame = -1;
  int end_frame = -1;
  float score = 0.0f;
  bool ambiguous = false;
  std::vector<DecodedContactCandidate> candidates;
};

struct LmEventTrace {
  std::string event;
  std::string filename;
  std::string type;
  double raw_score = 0.0;
  double reference_score = 0.0;
  double adjusted_score = 0.0;
  double weighted_score = 0.0;
};

struct DecodedHypothesis {
  double total_score = 0.0;
  double am_score = 0.0;
  double lm_score = 0.0;
  // Actual contribution to the decoder objective after all fixed-LM weights
  // and max fusion are applied.
  double weighted_lm_score = 0.0;
  double first_pass_score = 0.0;
  std::vector<int> token_ids;
  std::vector<DecodedWord> raw_words;
  std::vector<DecodedWord> am_mapped_words;
  std::vector<DecodedWord> mapped_words;
  std::vector<DecodedEntity> entities;
  std::vector<LmEventTrace> lm_events;
};

std::vector<int> WordIds(const std::vector<DecodedWord>& words);
std::string JoinWords(const std::vector<DecodedWord>& words,
                      const std::string& separator);

// Slot-aware callers retain distinct semantic entities even when their
// visible text is identical; ordinary decoding keeps its word-ID key.
std::vector<DecodedHypothesis> DeduplicateDecodedHypotheses(
    std::vector<DecodedHypothesis> hyps, int limit, bool contact_aware);

}  // namespace asr_sdk::internal::flashlight_decoder

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_DECODED_HYPOTHESIS_H_
