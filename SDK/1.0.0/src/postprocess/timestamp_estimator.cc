#include "postprocess/timestamp_estimator.h"

#include <algorithm>

namespace wenet_sdk::internal {

TimestampEstimator::TimestampEstimator(int frame_shift_ms, int subsampling_rate)
    : frame_shift_ms_(std::max(1, frame_shift_ms)),
      subsampling_rate_(std::max(1, subsampling_rate)) {}

float TimestampEstimator::StartMs(int decoder_frame) const {
  return static_cast<float>(std::max(0, decoder_frame) * frame_shift_ms_ *
                            subsampling_rate_);
}

float TimestampEstimator::EndMs(int decoder_frame) const {
  return StartMs(decoder_frame + 1);
}

}  // namespace wenet_sdk::internal
