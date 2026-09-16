#ifndef ASR_SDK_SRC_SHERPA_ONNX_WENET_CTC_ONNX_BACKEND_FACTORY_H_
#define ASR_SDK_SRC_SHERPA_ONNX_WENET_CTC_ONNX_BACKEND_FACTORY_H_

#include <memory>
#include <string>

#include "asr_sdk/config.h"
#include "sherpa_onnx_wenet/streaming_ctc_backend.h"

namespace asr_sdk::internal::sherpa_onnx_wenet {

std::unique_ptr<StreamingCtcBackend> CreateStreamingCtcBackend(
    const std::string& model_path, int num_threads, int blank_id,
    OrtLogLevel log_level = OrtLogLevel::kError,
    bool prepacking = false, bool device_initializers = false);

}  // namespace asr_sdk::internal::sherpa_onnx_wenet

#endif  // ASR_SDK_SRC_SHERPA_ONNX_WENET_CTC_ONNX_BACKEND_FACTORY_H_
