#include <cassert>
#include <vector>

#include "decoder/symbol_table.h"
#include "model/model_metadata.h"
#include "model/onnx_ctc_model.h"

int main() {
#ifndef WENETSDK_ENABLE_ONNX
  return 0;
#else
  wenet_sdk::internal::ModelMetadata metadata;
  auto status = wenet_sdk::internal::LoadModelMetadata(
      "tests/fixtures/onnx_toy", &metadata);
  assert(status.ok());

  wenet_sdk::internal::SymbolTable tokens;
  status = tokens.Load(metadata.Resolve(metadata.vocab.tokens));
  assert(status.ok());

  wenet_sdk::internal::OnnxCtcModel model(metadata, tokens.Size());
  status = model.Init();
  assert(status.ok());

  std::vector<std::vector<float>> features(
      4, std::vector<float>(static_cast<size_t>(metadata.feature_dim), 0.0f));
  std::vector<std::vector<float>> log_probs;
  status = model.ForwardChunk(features, &log_probs);
  assert(status.ok());
  assert(log_probs.size() == 3);
  assert(log_probs[0].size() == static_cast<size_t>(tokens.Size()));
  assert(log_probs[0][1] > log_probs[0][0]);
  assert(log_probs[1][2] > log_probs[1][0]);
  assert(log_probs[2][3] > log_probs[2][0]);
  return 0;
#endif
}
