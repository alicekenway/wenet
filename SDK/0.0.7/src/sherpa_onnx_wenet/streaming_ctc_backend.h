#ifndef ASR_SDK_SRC_SHERPA_ONNX_WENET_STREAMING_CTC_BACKEND_H_
#define ASR_SDK_SRC_SHERPA_ONNX_WENET_STREAMING_CTC_BACKEND_H_

#include <memory>
#include <vector>

#include "sherpa_onnx_wenet/streaming_ctc_model_info.h"

namespace asr_sdk::internal::sherpa_onnx_wenet {

class StreamingCtcBackend {
 public:
  virtual ~StreamingCtcBackend() = default;

  virtual const StreamingCtcModelInfo& Info() const = 0;
  virtual void Reset() = 0;
  virtual void Forward(const float* features, int num_frames,
                       std::vector<std::vector<float>>* log_probs) = 0;
  virtual std::unique_ptr<StreamingCtcBackend> CloneStream() const = 0;
};

}  // namespace asr_sdk::internal::sherpa_onnx_wenet

#endif  // ASR_SDK_SRC_SHERPA_ONNX_WENET_STREAMING_CTC_BACKEND_H_
