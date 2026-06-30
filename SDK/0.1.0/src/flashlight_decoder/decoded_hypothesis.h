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
};

struct DecodedHypothesis {
  double total_score = 0.0;
  double am_score = 0.0;
  double lm_score = 0.0;
  std::vector<int> token_ids;
  std::vector<DecodedWord> raw_words;
  std::vector<DecodedWord> mapped_words;
};

std::vector<int> WordIds(const std::vector<DecodedWord>& words);
std::string JoinWords(const std::vector<DecodedWord>& words,
                      const std::string& separator);

}  // namespace asr_sdk::internal::flashlight_decoder

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_DECODED_HYPOTHESIS_H_
