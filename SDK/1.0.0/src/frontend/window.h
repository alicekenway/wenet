#ifndef WENET_SDK_SRC_FRONTEND_WINDOW_H_
#define WENET_SDK_SRC_FRONTEND_WINDOW_H_

#include <vector>

namespace wenet_sdk::internal {

enum class WindowType {
  kHamming,
  kHanning,
  kPovey,
  kRectangular,
};

std::vector<float> BuildWindow(int frame_length, WindowType type);
void ApplyPreemphasis(std::vector<float>* frame, float coeff);
void ApplyWindow(std::vector<float>* frame, const std::vector<float>& window);

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_FRONTEND_WINDOW_H_
