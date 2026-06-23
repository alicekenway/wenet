#ifndef WENET_SDK_SRC_MODEL_ASR_MODEL_H_
#define WENET_SDK_SRC_MODEL_ASR_MODEL_H_

#include <vector>

#include "utils/status.h"

namespace wenet_sdk::internal {

class AsrModel {
 public:
  virtual ~AsrModel() = default;

  virtual void Reset() = 0;
  virtual Status ForwardChunk(
      const std::vector<std::vector<float>>& features,
      std::vector<std::vector<float>>* log_probs) = 0;

  virtual int VocabSize() const = 0;
  virtual int FeatureDim() const = 0;
  virtual int SubsamplingRate() const = 0;
};

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_MODEL_ASR_MODEL_H_
