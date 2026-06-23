#include "decoder/greedy_ctc_decoder.h"

#include <algorithm>
#include <cmath>

namespace wenet_sdk::internal {

std::vector<int> CollapseCtc(const std::vector<int>& frame_best_ids,
                             int blank_id) {
  std::vector<int> output;
  int prev = -1;
  for (int id : frame_best_ids) {
    if (id == blank_id) {
      prev = id;
      continue;
    }
    if (id != prev) {
      output.push_back(id);
    }
    prev = id;
  }
  return output;
}

GreedyCtcDecoder::GreedyCtcDecoder(GreedyCtcDecoderOptions options)
    : options_(options) {}

void GreedyCtcDecoder::Reset() {
  frame_best_ids_.clear();
  frame_best_logp_.clear();
}

void GreedyCtcDecoder::Advance(
    const std::vector<std::vector<float>>& log_probs) {
  for (const auto& frame : log_probs) {
    if (frame.empty()) {
      continue;
    }
    const auto it = std::max_element(frame.begin(), frame.end());
    frame_best_ids_.push_back(static_cast<int>(it - frame.begin()));
    frame_best_logp_.push_back(*it);
  }
}

DecodeResult GreedyCtcDecoder::PartialResult() const {
  return BuildResult();
}

DecodeResult GreedyCtcDecoder::Finalize() {
  return BuildResult();
}

DecodeResult GreedyCtcDecoder::BuildResult() const {
  DecodeResult result;
  int prev = -1;
  float sum_conf = 0.0f;
  int conf_count = 0;
  for (size_t i = 0; i < frame_best_ids_.size(); ++i) {
    const int id = frame_best_ids_[i];
    if (id == options_.blank_id) {
      prev = id;
      continue;
    }
    if (id != prev) {
      result.token_ids.push_back(id);
      result.frame_indexes.push_back(static_cast<int>(i));
      sum_conf += std::exp(frame_best_logp_[i]);
      ++conf_count;
    }
    prev = id;
  }
  result.confidence = conf_count > 0 ? sum_conf / conf_count : 0.0f;
  if (!result.token_ids.empty()) {
    NBestPath path;
    path.token_ids = result.token_ids;
    path.frame_indexes = result.frame_indexes;
    path.total_score = result.confidence;
    result.nbest.push_back(std::move(path));
  }
  return result;
}

}  // namespace wenet_sdk::internal
