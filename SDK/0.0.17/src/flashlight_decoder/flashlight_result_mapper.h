#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_FLASHLIGHT_RESULT_MAPPER_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_FLASHLIGHT_RESULT_MAPPER_H_

#include "flashlight/lib/text/decoder/Utils.h"
#include "flashlight_decoder/decoded_hypothesis.h"
#include "flashlight_decoder/flashlight_decoder_resource.h"

namespace asr_sdk::internal::flashlight_decoder {

DecodedHypothesis ConvertFlashlightResult(
    const fl::lib::text::DecodeResult& result,
    const FlashlightDecoderResource& resource);

}  // namespace asr_sdk::internal::flashlight_decoder

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_FLASHLIGHT_RESULT_MAPPER_H_
