#include "flashlight_decoder/decoded_hypothesis.h"

namespace asr_sdk::internal::flashlight_decoder {

std::vector<int> WordIds(const std::vector<DecodedWord>& words) {
  std::vector<int> ids;
  ids.reserve(words.size());
  for (const DecodedWord& word : words) {
    ids.push_back(word.word_id);
  }
  return ids;
}

std::string JoinWords(const std::vector<DecodedWord>& words,
                      const std::string& separator) {
  std::string text;
  for (size_t i = 0; i < words.size(); ++i) {
    if (i != 0) {
      text += separator;
    }
    text += words[i].text;
  }
  return text;
}

}  // namespace asr_sdk::internal::flashlight_decoder
