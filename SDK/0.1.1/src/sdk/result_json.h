#ifndef ASR_SDK_SRC_SDK_RESULT_JSON_H_
#define ASR_SDK_SRC_SDK_RESULT_JSON_H_

#include <string>

#include "asr_sdk/result.h"

namespace asr_sdk::internal {

std::string AsrResultToJson(const AsrResult& result);

}  // namespace asr_sdk::internal

#endif  // ASR_SDK_SRC_SDK_RESULT_JSON_H_
