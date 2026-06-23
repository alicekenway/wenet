#ifndef WENET_SDK_SRC_MODEL_TENSOR_UTILS_H_
#define WENET_SDK_SRC_MODEL_TENSOR_UTILS_H_

#include <vector>

namespace wenet_sdk::internal {

std::vector<float> LogSoftmax(const std::vector<float>& logits);
bool IsMatrixShape(const std::vector<std::vector<float>>& matrix, int cols);

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_MODEL_TENSOR_UTILS_H_
