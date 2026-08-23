#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_FLASHLIGHT_DECODER_OPTIONS_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_FLASHLIGHT_DECODER_OPTIONS_H_

#include <string>

namespace asr_sdk::internal::flashlight_decoder {

struct FlashlightDecoderOptions {
  int beam_size = 50;
  int beam_size_token = 20;
  double beam_threshold = 25.0;
  double lm_weight = 1.5;
  double word_score = -0.5;
  double unk_score = -5.0;
  double sil_score = 0.0;
  bool log_add = false;
  bool allow_unk = true;
  int nbest = 1;
  std::string smearing = "max";
};

}  // namespace asr_sdk::internal::flashlight_decoder

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_FLASHLIGHT_DECODER_OPTIONS_H_
