#include "decoder/blank_skipper.h"

#include <cmath>

namespace wenet_sdk::internal {

BlankSkipper::BlankSkipper(BlankSkipperOptions options)
    : options_(options) {}

void BlankSkipper::Reset() {
  frames_seen_ = 0;
  frames_skipped_ = 0;
  skipped_previous_ = false;
  previous_skipped_.clear();
}

std::vector<std::vector<float>> BlankSkipper::Filter(
    const std::vector<std::vector<float>>& log_probs) {
  if (!options_.enabled || options_.blank_skip_thresh > 1.0f) {
    frames_seen_ += static_cast<int>(log_probs.size());
    return log_probs;
  }

  std::vector<std::vector<float>> kept;
  kept.reserve(log_probs.size());
  for (const auto& frame : log_probs) {
    ++frames_seen_;
    if (options_.blank_id < 0 ||
        options_.blank_id >= static_cast<int>(frame.size())) {
      kept.push_back(frame);
      skipped_previous_ = false;
      continue;
    }

    const float blank_prob = std::exp(frame[static_cast<size_t>(options_.blank_id)]);
    if (blank_prob > options_.blank_skip_thresh) {
      previous_skipped_ = frame;
      skipped_previous_ = true;
      ++frames_skipped_;
      continue;
    }

    if (skipped_previous_) {
      kept.push_back(previous_skipped_);
    }
    kept.push_back(frame);
    skipped_previous_ = false;
  }

  if (kept.empty() && !log_probs.empty()) {
    kept.push_back(log_probs.back());
    if (frames_skipped_ > 0) {
      --frames_skipped_;
    }
  }
  return kept;
}

}  // namespace wenet_sdk::internal
