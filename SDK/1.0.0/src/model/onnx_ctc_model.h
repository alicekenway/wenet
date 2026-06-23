#ifndef WENET_SDK_SRC_MODEL_ONNX_CTC_MODEL_H_
#define WENET_SDK_SRC_MODEL_ONNX_CTC_MODEL_H_

#include <memory>

#include "model/asr_model.h"
#include "model/model_metadata.h"

namespace wenet_sdk::internal {

class OnnxCtcModel final : public AsrModel {
 public:
  OnnxCtcModel(ModelMetadata metadata, int vocab_size);
  ~OnnxCtcModel() override;

  Status Init();
  void Reset() override;
  Status ForwardChunk(const std::vector<std::vector<float>>& features,
                      std::vector<std::vector<float>>* log_probs) override;

  int VocabSize() const override { return vocab_size_; }
  int FeatureDim() const override { return metadata_.feature_dim; }
  int SubsamplingRate() const override { return metadata_.subsampling_rate; }
  const char* BackendName() const;

 private:
  struct OrtState;

  Status InitOrtBackend();
  Status ForwardOrtChunk(const std::vector<std::vector<float>>& features,
                         std::vector<std::vector<float>>* log_probs);

  ModelMetadata metadata_;
  int vocab_size_ = 0;
  int chunk_index_ = 0;
  std::unique_ptr<OrtState> ort_;
};

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_MODEL_ONNX_CTC_MODEL_H_
