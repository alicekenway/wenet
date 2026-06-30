#ifndef ASR_SDK_SRC_SHERPA_ONNX_WENET_WENET_CTC_ONNX_BACKEND_H_
#define ASR_SDK_SRC_SHERPA_ONNX_WENET_WENET_CTC_ONNX_BACKEND_H_

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

#include "sherpa_onnx_wenet/streaming_ctc_backend.h"

namespace asr_sdk::internal::sherpa_onnx_wenet {

class WenetCtcOnnxResource {
 public:
  WenetCtcOnnxResource(const std::string& model_path, int num_threads,
                       int blank_id);

  const StreamingCtcModelInfo& Info() const { return info_; }

 private:
  friend class WenetCtcOnnxBackend;

  void ReadMetadata(int blank_id);
  void ReadNodes();
  std::vector<std::string> GetNodeNames(bool input) const;

  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  StreamingCtcModelInfo info_;
  int chunk_size_ = 0;
  int left_chunks_ = 0;
  int required_cache_size_ = 0;
  int num_blocks_ = 0;
  int head_ = 0;
  int output_size_ = 0;
  int cnn_module_kernel_ = 0;
  int subsampling_factor_ = 0;
  int right_context_ = 0;
};

class WenetCtcOnnxBackend final : public StreamingCtcBackend {
 public:
  explicit WenetCtcOnnxBackend(std::shared_ptr<WenetCtcOnnxResource> resource);

  const StreamingCtcModelInfo& Info() const override {
    return resource_->Info();
  }
  void Reset() override;
  void Forward(const float* features, int num_frames,
               std::vector<std::vector<float>>* log_probs) override;
  std::unique_ptr<StreamingCtcBackend> CloneStream() const override;

 private:
  std::shared_ptr<WenetCtcOnnxResource> resource_;
  int64_t offset_ = 0;
  std::vector<float> att_cache_;
  std::vector<float> conv_cache_;
  Ort::Value att_cache_value_{nullptr};
  Ort::Value conv_cache_value_{nullptr};
};

}  // namespace asr_sdk::internal::sherpa_onnx_wenet

#endif  // ASR_SDK_SRC_SHERPA_ONNX_WENET_WENET_CTC_ONNX_BACKEND_H_
