#include <cassert>
#include <vector>

#include "decoder/symbol_table.h"
#include "model/model_metadata.h"
#include "model/onnx_ctc_model.h"

int main() {
#ifdef WENETSDK_ENABLE_ONNX
  const char* model_dir = "tests/fixtures/onnx_toy";
#else
  const char* model_dir = "model_example";
#endif
  wenet_sdk::internal::ModelMetadata metadata;
  auto status = wenet_sdk::internal::LoadModelMetadata(model_dir, &metadata);
  assert(status.ok());

  wenet_sdk::internal::SymbolTable tokens;
  status = tokens.Load(metadata.Resolve(metadata.vocab.tokens));
  assert(status.ok());

  wenet_sdk::internal::OnnxCtcModel model(metadata, tokens.Size());
  status = model.Init();
  assert(status.ok());

  std::vector<std::vector<float>> features(
      static_cast<size_t>(metadata.streaming.chunk_size),
      std::vector<float>(static_cast<size_t>(metadata.feature_dim), 2.0f));
  std::vector<std::vector<float>> log_probs;
  status = model.ForwardChunk(features, &log_probs);
  assert(status.ok());
  assert(!log_probs.empty());
  for (const auto& frame : log_probs) {
    assert(frame.size() == static_cast<size_t>(tokens.Size()));
  }
#ifdef WENETSDK_ENABLE_ONNX
  assert(log_probs.size() == 3);
#endif
  return 0;
}
