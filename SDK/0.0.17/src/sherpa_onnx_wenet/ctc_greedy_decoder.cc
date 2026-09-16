#include "sherpa_onnx_wenet/ctc_greedy_decoder.h"

#include <algorithm>

namespace asr_sdk::internal::sherpa_onnx_wenet {

std::vector<int> CtcGreedyDecode(
    const std::vector<std::vector<float>>& log_probs, int blank_id) {
  std::vector<int> ids;
  int prev = -1;
  for (const auto& frame : log_probs) {
    if (frame.empty()) {
      continue;
    }
    const int best =
        static_cast<int>(std::max_element(frame.begin(), frame.end()) -
                         frame.begin());
    if (best != blank_id && best != prev) {
      ids.push_back(best);
    }
    prev = best;
  }
  return ids;
}

}  // namespace asr_sdk::internal::sherpa_onnx_wenet
