#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_DEBUG_TRACE_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_DEBUG_TRACE_H_

#include <string>
#include <vector>

#include "flashlight_decoder/decoded_hypothesis.h"

namespace asr_sdk::internal::flashlight_decoder {

std::string BuildDebugJson(const std::vector<std::string>& logs,
                           const std::string& error,
                           const std::vector<DecodedHypothesis>& hyps,
                           bool is_final);

}  // namespace asr_sdk::internal::flashlight_decoder

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_DEBUG_TRACE_H_
