#ifndef WENET_SDK_SRC_UTILS_TIMER_H_
#define WENET_SDK_SRC_UTILS_TIMER_H_

#include <chrono>

namespace wenet_sdk::internal {

class Timer {
 public:
  Timer();
  void Reset();
  double ElapsedMs() const;

 private:
  std::chrono::steady_clock::time_point start_;
};

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_UTILS_TIMER_H_
