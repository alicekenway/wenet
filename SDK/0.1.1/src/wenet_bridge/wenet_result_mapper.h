#ifndef ASR_SDK_SRC_WENET_BRIDGE_WENET_RESULT_MAPPER_H_
#define ASR_SDK_SRC_WENET_BRIDGE_WENET_RESULT_MAPPER_H_

#include <string>

#include "asr_sdk/result.h"

namespace asr_sdk::internal {

AsrResult MapWenetJsonResult(const std::string& json);

}  // namespace asr_sdk::internal

#endif  // ASR_SDK_SRC_WENET_BRIDGE_WENET_RESULT_MAPPER_H_
