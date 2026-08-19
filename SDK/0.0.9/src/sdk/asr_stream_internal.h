#ifndef ASR_SDK_SRC_SDK_ASR_STREAM_INTERNAL_H_
#define ASR_SDK_SRC_SDK_ASR_STREAM_INTERNAL_H_

#include <memory>

#include "asr_sdk/stream.h"
#include "wenet_bridge/wenet_stream_adapter.h"

namespace asr_sdk::internal {

std::unique_ptr<AsrStream> MakeAsrStream(
    std::unique_ptr<WenetStreamAdapter> adapter);

}  // namespace asr_sdk::internal

#endif  // ASR_SDK_SRC_SDK_ASR_STREAM_INTERNAL_H_
