#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_KENLM_RESCORER_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_KENLM_RESCORER_H_

#include <vector>

#include "flashlight_decoder/decoded_hypothesis.h"
#include "flashlight_decoder/flashlight_decoder_resource.h"

namespace asr_sdk::internal::flashlight_decoder {

double ScoreWordsWithKenLm(const FlashlightDecoderResource& resource,
                           const std::vector<DecodedWord>& words);

void RescoreAndApplyFinalMapping(const FlashlightDecoderResource& resource,
                                 std::vector<DecodedHypothesis>* hyps);

}  // namespace asr_sdk::internal::flashlight_decoder

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_KENLM_RESCORER_H_
