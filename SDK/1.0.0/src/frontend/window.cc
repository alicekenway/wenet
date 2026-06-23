#include "frontend/window.h"

#include <cmath>

namespace wenet_sdk::internal {
namespace {

double Pi() { return std::acos(-1.0); }

}  // namespace

std::vector<float> BuildWindow(int frame_length, WindowType type) {
  std::vector<float> window(static_cast<size_t>(frame_length), 1.0f);
  if (frame_length <= 1 || type == WindowType::kRectangular) {
    return window;
  }
  for (int i = 0; i < frame_length; ++i) {
    const double ratio = 2.0 * Pi() * i / (frame_length - 1);
    double value = 1.0;
    switch (type) {
      case WindowType::kHamming:
        value = 0.54 - 0.46 * std::cos(ratio);
        break;
      case WindowType::kHanning:
        value = 0.5 - 0.5 * std::cos(ratio);
        break;
      case WindowType::kPovey:
        value = std::pow(0.5 - 0.5 * std::cos(ratio), 0.85);
        break;
      case WindowType::kRectangular:
        value = 1.0;
        break;
    }
    window[static_cast<size_t>(i)] = static_cast<float>(value);
  }
  return window;
}

void ApplyPreemphasis(std::vector<float>* frame, float coeff) {
  if (frame == nullptr || frame->empty() || coeff == 0.0f) {
    return;
  }
  for (int i = static_cast<int>(frame->size()) - 1; i >= 1; --i) {
    (*frame)[static_cast<size_t>(i)] -=
        coeff * (*frame)[static_cast<size_t>(i - 1)];
  }
  (*frame)[0] *= (1.0f - coeff);
}

void ApplyWindow(std::vector<float>* frame, const std::vector<float>& window) {
  if (frame == nullptr) {
    return;
  }
  const size_t n = std::min(frame->size(), window.size());
  for (size_t i = 0; i < n; ++i) {
    (*frame)[i] *= window[i];
  }
}

}  // namespace wenet_sdk::internal
