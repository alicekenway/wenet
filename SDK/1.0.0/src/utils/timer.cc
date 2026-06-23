#include "utils/timer.h"

namespace wenet_sdk::internal {

Timer::Timer() { Reset(); }

void Timer::Reset() {
  start_ = std::chrono::steady_clock::now();
}

double Timer::ElapsedMs() const {
  const auto now = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(now - start_).count();
}

}  // namespace wenet_sdk::internal
