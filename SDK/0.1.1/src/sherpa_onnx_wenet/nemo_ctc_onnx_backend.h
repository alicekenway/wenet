#ifndef ASR_SDK_SRC_SHERPA_ONNX_WENET_NEMO_CTC_ONNX_BACKEND_H_
#define ASR_SDK_SRC_SHERPA_ONNX_WENET_NEMO_CTC_ONNX_BACKEND_H_

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

#include "sherpa_onnx_wenet/streaming_ctc_backend.h"

namespace asr_sdk::internal::sherpa_onnx_wenet {

class NemoCtcOnnxResource {
 public:
  NemoCtcOnnxResource(const std::string& model_path, int num_threads,
                      int blank_id);

  const StreamingCtcModelInfo& Info() const { return info_; }

 private:
  friend class NemoCtcOnnxBackend;

  void ReadMetadata(int blank_id);
  void ReadNodes();
  std::vector<std::string> GetNodeNames(bool input) const;

  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  StreamingCtcModelInfo info_;
  std::vector<int64_t> cache_last_channel_shape_;
  std::vector<int64_t> cache_last_time_shape_;
};

class NemoCtcOnnxBackend final : public StreamingCtcBackend {
 public:
  explicit NemoCtcOnnxBackend(std::shared_ptr<NemoCtcOnnxResource> resource);

  const StreamingCtcModelInfo& Info() const override {
    return resource_->Info();
  }
  void Reset() override;
  void Forward(const float* features, int num_frames,
               std::vector<std::vector<float>>* log_probs) override;
  std::unique_ptr<StreamingCtcBackend> CloneStream() const override;

 private:
  std::shared_ptr<NemoCtcOnnxResource> resource_;
  std::vector<float> cache_last_channel_;
  std::vector<float> cache_last_time_;
  std::vector<int64_t> cache_last_channel_len_;
};

}  // namespace asr_sdk::internal::sherpa_onnx_wenet

#endif  // ASR_SDK_SRC_SHERPA_ONNX_WENET_NEMO_CTC_ONNX_BACKEND_H_
