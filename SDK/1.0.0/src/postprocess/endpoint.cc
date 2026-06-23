#include "postprocess/endpoint.h"

#include <algorithm>

namespace wenet_sdk::internal {

Endpoint::Endpoint(EndpointOptions options) : options_(options) {}

void Endpoint::Reset() {
  total_frames_ = 0;
  trailing_blank_frames_ = 0;
}

void Endpoint::AdvanceFrames(int frames, bool blank_heavy) {
  frames = std::max(0, frames);
  total_frames_ += frames;
  if (blank_heavy) {
    trailing_blank_frames_ += frames;
  } else {
    trailing_blank_frames_ = 0;
  }
}

bool Endpoint::IsEndpoint(bool input_finished) const {
  if (input_finished) {
    return true;
  }
  const int shift = std::max(1, options_.frame_shift_ms);
  if (trailing_blank_frames_ * shift >= options_.endpoint_silence_ms) {
    return true;
  }
  return total_frames_ * shift >= options_.max_segment_ms;
}

}  // namespace wenet_sdk::internal
