#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "decoder/symbol_table.h"
#include "model/model_metadata.h"

#ifdef WENETSDK_ENABLE_ONNX
#include "onnxruntime_cxx_api.h"  // NOLINT
#endif

namespace {

std::string ArgValue(int argc, char** argv, const std::string& name,
                     const std::string& fallback = "") {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) {
      return argv[i + 1];
    }
  }
  return fallback;
}

#ifdef WENETSDK_ENABLE_ONNX
std::string ShapeString(const std::vector<int64_t>& shape) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      os << ",";
    }
    os << shape[i];
  }
  os << "]";
  return os.str();
}

void PrintNodeInfo(Ort::Session* session, bool inputs,
                   const std::string& header) {
  Ort::AllocatorWithDefaultOptions allocator;
  const size_t count =
      inputs ? session->GetInputCount() : session->GetOutputCount();
  std::cout << header << ": " << count << "\n";
  for (size_t i = 0; i < count; ++i) {
    Ort::AllocatedStringPtr name =
        inputs ? session->GetInputNameAllocated(i, allocator)
               : session->GetOutputNameAllocated(i, allocator);
    Ort::TypeInfo type_info =
        inputs ? session->GetInputTypeInfo(i) : session->GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::cout << "  " << i << ": name=" << name.get()
              << " elem_type=" << tensor_info.GetElementType()
              << " shape=" << ShapeString(tensor_info.GetShape()) << "\n";
  }
}

void InspectOnnxIfPossible(const wenet_sdk::internal::ModelMetadata& metadata) {
  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "wenet_lite_inspect_model");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);

    const auto encoder_path = metadata.Resolve(metadata.onnx.encoder);
    Ort::Session encoder(env, encoder_path.c_str(), options);
    std::cout << "onnx_encoder: " << encoder_path.string() << "\n";
    PrintNodeInfo(&encoder, true, "onnx_encoder_inputs");
    PrintNodeInfo(&encoder, false, "onnx_encoder_outputs");

    if (!metadata.onnx.ctc.empty()) {
      const auto ctc_path = metadata.Resolve(metadata.onnx.ctc);
      Ort::Session ctc(env, ctc_path.c_str(), options);
      std::cout << "onnx_ctc: " << ctc_path.string() << "\n";
      PrintNodeInfo(&ctc, true, "onnx_ctc_inputs");
      PrintNodeInfo(&ctc, false, "onnx_ctc_outputs");
    }
  } catch (const std::exception& e) {
    std::cout << "onnx_inspect_error: " << e.what() << "\n";
  }
}
#endif

}  // namespace

int main(int argc, char** argv) {
  const std::string model_dir = ArgValue(argc, argv, "--model_dir");
  if (model_dir.empty()) {
    std::cerr << "usage: inspect_model --model_dir MODEL_DIR\n";
    return 2;
  }
  wenet_sdk::internal::ModelMetadata metadata;
  auto status = wenet_sdk::internal::LoadModelMetadata(model_dir, &metadata);
  if (!status.ok()) {
    std::cerr << status.message() << "\n";
    return 1;
  }
  wenet_sdk::internal::SymbolTable tokens;
  status = tokens.Load(metadata.Resolve(metadata.vocab.tokens));
  if (!status.ok()) {
    std::cerr << status.message() << "\n";
    return 1;
  }
  std::cout << wenet_sdk::internal::ModelSummary(metadata);
  std::cout << "vocab_size: " << tokens.Size() << "\n";
#ifdef WENETSDK_ENABLE_ONNX
  std::cout << "runtime_backend: onnxruntime";
#else
  std::cout << "runtime_backend: deterministic-onnx-stub";
#endif
  std::cout << "\n";
#ifdef WENETSDK_ENABLE_ONNX
  InspectOnnxIfPossible(metadata);
#endif
  return 0;
}
