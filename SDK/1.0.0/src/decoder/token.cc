#include "decoder/token.h"

#include <algorithm>

namespace wenet_sdk::internal {

std::vector<int> TraceLabels(const std::shared_ptr<Token>& token) {
  std::vector<int> labels;
  for (auto cur = token; cur != nullptr; cur = cur->prev) {
    if (cur->label >= 0) {
      labels.push_back(cur->label);
    }
  }
  std::reverse(labels.begin(), labels.end());
  return labels;
}

}  // namespace wenet_sdk::internal
