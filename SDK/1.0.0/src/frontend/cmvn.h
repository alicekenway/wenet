#ifndef WENET_SDK_SRC_FRONTEND_CMVN_H_
#define WENET_SDK_SRC_FRONTEND_CMVN_H_

#include <filesystem>
#include <vector>

#include "utils/status.h"

namespace wenet_sdk::internal {

class Cmvn {
 public:
  Status LoadIfPresent(const std::filesystem::path& path, int feature_dim);
  void Apply(std::vector<float>* frame) const;
  bool enabled() const { return !mean_.empty(); }

 private:
  std::vector<float> mean_;
  std::vector<float> inv_std_;
};

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_FRONTEND_CMVN_H_
