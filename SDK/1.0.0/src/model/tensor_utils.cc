#include "model/tensor_utils.h"

#include <algorithm>
#include <cmath>

namespace wenet_sdk::internal {

std::vector<float> LogSoftmax(const std::vector<float>& logits) {
  if (logits.empty()) {
    return {};
  }
  const float max_value = *std::max_element(logits.begin(), logits.end());
  double sum = 0.0;
  for (float v : logits) {
    sum += std::exp(v - max_value);
  }
  const float log_sum = max_value + static_cast<float>(std::log(sum));
  std::vector<float> output(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    output[i] = logits[i] - log_sum;
  }
  return output;
}

bool IsMatrixShape(const std::vector<std::vector<float>>& matrix, int cols) {
  if (cols < 0) {
    return false;
  }
  return std::all_of(matrix.begin(), matrix.end(), [cols](const auto& row) {
    return static_cast<int>(row.size()) == cols;
  });
}

}  // namespace wenet_sdk::internal
