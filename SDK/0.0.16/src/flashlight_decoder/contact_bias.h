#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_CONTACT_BIAS_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_CONTACT_BIAS_H_

#include <algorithm>
#include <cmath>

namespace asr_sdk::internal::flashlight_decoder {

// A runtime contact is one lexical label even when its spoken form contains
// multiple words.  The geometric factor lets users give longer names a
// controlled amount of extra contact bias without multiplying the bonus
// linearly by word count.
inline float ContactBonusMultiplier(int logical_word_count,
                                    double accumulation_factor) {
  const int count = std::max(1, logical_word_count);
  float multiplier = 0.0f;
  float term = 1.0f;
  for (int index = 0; index < count; ++index) {
    multiplier += term;
    term *= static_cast<float>(accumulation_factor);
  }
  return multiplier;
}

inline float ContactWordScoreCorrection(int logical_word_count,
                                        float word_score) {
  return static_cast<float>(std::max(0, logical_word_count - 1)) *
         word_score;
}

inline bool IsValidContactAccumulationFactor(double value) {
  return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

}  // namespace asr_sdk::internal::flashlight_decoder

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_CONTACT_BIAS_H_
