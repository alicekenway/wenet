#include "sherpa_onnx_wenet/ctc_onnx_backend_factory.h"

#include <algorithm>
#include <stdexcept>

#include "onnxruntime_cxx_api.h"  // NOLINT

#include "sherpa_onnx_wenet/wenet_ctc_onnx_backend.h"
#include "sherpa_onnx_wenet/zipformer2_ctc_onnx_backend.h"

namespace asr_sdk::internal::sherpa_onnx_wenet {
namespace {

std::string ReadModelType(const std::string& model_path, int num_threads) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ctc_onnx_backend_factory");
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(std::max(1, num_threads));
  options.SetInterOpNumThreads(1);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  Ort::Session session(env, model_path.c_str(), options);
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::ModelMetadata metadata = session.GetModelMetadata();
  Ort::AllocatedStringPtr value =
      metadata.LookupCustomMetadataMapAllocated("model_type", allocator);
  if (value == nullptr) {
    throw std::runtime_error("missing ONNX metadata: model_type");
  }
  return value.get();
}

}  // namespace

std::unique_ptr<StreamingCtcBackend> CreateStreamingCtcBackend(
    const std::string& model_path, int num_threads, int blank_id) {
  const std::string model_type = ReadModelType(model_path, num_threads);
  if (model_type == "zipformer2") {
    auto resource = std::make_shared<Zipformer2CtcOnnxResource>(
        model_path, num_threads, blank_id);
    return std::make_unique<Zipformer2CtcOnnxBackend>(std::move(resource));
  }
  if (model_type == "wenet_ctc") {
    auto resource = std::make_shared<WenetCtcOnnxResource>(
        model_path, num_threads, blank_id);
    return std::make_unique<WenetCtcOnnxBackend>(std::move(resource));
  }
  throw std::runtime_error("unsupported CTC ONNX model_type: " + model_type);
}

}  // namespace asr_sdk::internal::sherpa_onnx_wenet
