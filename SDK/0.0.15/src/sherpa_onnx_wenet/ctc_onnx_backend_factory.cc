#include "sherpa_onnx_wenet/ctc_onnx_backend_factory.h"

#include <stdexcept>

#include "sherpa_onnx_wenet/ctc_onnx_session.h"
#include "sherpa_onnx_wenet/wenet_ctc_onnx_backend.h"
#include "sherpa_onnx_wenet/zipformer2_ctc_onnx_backend.h"

namespace asr_sdk::internal::sherpa_onnx_wenet {
namespace {

std::string ReadModelType(const std::shared_ptr<CtcOnnxSession>& session) {
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::ModelMetadata metadata = session->Session().GetModelMetadata();
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
  auto session = std::make_shared<CtcOnnxSession>(model_path, num_threads);
  const std::string model_type = ReadModelType(session);
  if (model_type == "zipformer2") {
    auto resource = std::make_shared<Zipformer2CtcOnnxResource>(
        std::move(session), blank_id);
    return std::make_unique<Zipformer2CtcOnnxBackend>(std::move(resource));
  }
  if (model_type == "wenet_ctc") {
    auto resource = std::make_shared<WenetCtcOnnxResource>(
        std::move(session), blank_id);
    return std::make_unique<WenetCtcOnnxBackend>(std::move(resource));
  }
  throw std::runtime_error("unsupported CTC ONNX model_type: " + model_type);
}

}  // namespace asr_sdk::internal::sherpa_onnx_wenet
