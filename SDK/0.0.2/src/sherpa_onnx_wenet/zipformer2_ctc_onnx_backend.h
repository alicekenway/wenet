#ifndef ASR_SDK_SRC_SHERPA_ONNX_WENET_ZIPFORMER2_CTC_ONNX_BACKEND_H_
#define ASR_SDK_SRC_SHERPA_ONNX_WENET_ZIPFORMER2_CTC_ONNX_BACKEND_H_

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

#include "sherpa_onnx_wenet/streaming_ctc_backend.h"

namespace asr_sdk::internal::sherpa_onnx_wenet {

struct TensorSpec {
  std::string name;
  ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  std::vector<int64_t> shape;
};

class Zipformer2CtcOnnxResource {
 public:
  Zipformer2CtcOnnxResource(const std::string& model_path, int num_threads,
                            int blank_id);

  const StreamingCtcModelInfo& Info() const { return info_; }
  const std::vector<TensorSpec>& StateInputSpecs() const {
    return state_input_specs_;
  }
  const std::vector<TensorSpec>& StateOutputSpecs() const {
    return state_output_specs_;
  }

 private:
  friend class Zipformer2CtcOnnxBackend;

  void ReadMetadata(int blank_id);
  void ReadNodes();
  std::vector<std::string> GetNodeNames(bool input) const;

  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<TensorSpec> state_input_specs_;
  std::vector<TensorSpec> state_output_specs_;
  StreamingCtcModelInfo info_;
};

class Zipformer2CtcOnnxBackend final : public StreamingCtcBackend {
 public:
  explicit Zipformer2CtcOnnxBackend(
      std::shared_ptr<Zipformer2CtcOnnxResource> resource);

  const StreamingCtcModelInfo& Info() const override {
    return resource_->Info();
  }
  void Reset() override;
  void Forward(const float* features, int num_frames,
               std::vector<std::vector<float>>* log_probs) override;
  std::unique_ptr<StreamingCtcBackend> CloneStream() const override;

 private:
  std::vector<Ort::Value> MakeZeroStates() const;

  std::shared_ptr<Zipformer2CtcOnnxResource> resource_;
  std::vector<Ort::Value> states_;
};

}  // namespace asr_sdk::internal::sherpa_onnx_wenet

#endif  // ASR_SDK_SRC_SHERPA_ONNX_WENET_ZIPFORMER2_CTC_ONNX_BACKEND_H_
